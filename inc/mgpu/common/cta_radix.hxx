#include "cta_scan.hxx"
#include <cstdio>

BEGIN_MGPU_NAMESPACE

template<typename type_t>
struct radix_permute_t {
  typedef unsigned_int_by_size_t<sizeof(type_t)> unsigned_type;
  typedef   signed_int_by_size_t<sizeof(type_t)>   signed_type;

  static unsigned_type to_radix_bits(type_t x) {
    if constexpr(std::is_unsigned_v<type_t>) {
      // Do nothing.
      return x;

    } else if constexpr(std::is_integral_v<type_t>) {
      // Flip the most significant bit.
      return x ^ (1<< (8 * sizeof(type_t) - 1));

    } else if constexpr(std::is_floating_point_v<type_t>) {
      // Always flip the most significant bit. Flip all other bits if the
      // most significant bit started flipped.
      unsigned_type y = *reinterpret_cast<const unsigned_type*>(&x);
      unsigned_type mask = 
        // Carry-in the sign bit to all lower bits
        ((signed_type)y>> (8 * sizeof(type_t) - 1)) | 
        // Always set the most significant bit
        ((unsigned_type)1<< (8 * sizeof(type_t) - 1));

      return y ^ mask;

    } else {
      static_assert("type cannot be converted to radix form");
    }
  }

  static type_t from_radix_bits(unsigned_type x) {
    if constexpr(std::is_unsigned_v<type_t>) {
      // Do nothing.
      return x;

    } else if constexpr(std::is_integral_v<type_t>) {
      // Flip the most significant bit.
      return x ^ (1<< (8 * sizeof(type_t) - 1));

    } else if constexpr(std::is_floating_point_v<type_t>) {
      // Flip the sign bit.
      x ^= (unsigned_type)1<< (8 * sizeof(type_t) - 1);

      // Flip the lower bits if the sign bit is set.
      unsigned_type mask = 
        // Carry-in the sign bit to all lower bits
        ((signed_type)x>> (8 * sizeof(type_t) - 1)) &
        // Always clear the most significant bit
        (((unsigned_type)1<< (8 * sizeof(type_t) - 1)) - 1);
      x ^= mask;

      return *reinterpret_cast<const type_t*>(&x);

    } else {
      static_assert("type cannot be converted from radix form");
    }
  }
};

////////////////////////////////////////////////////////////////////////////////
// Use shared memory histogram to rank digits.

template<int nt, int num_bits>
struct cta_radix_rank_t {
  enum { num_bins = 1<< num_bits, num_slots = num_bins / 2 + 1 };
  typedef cta_scan_t<nt, uint> scan_t;

  template<int vt>
  struct result_t {
    // All threads return scatter indices for each value.
    std::array<uint, vt> indices;

    // The first num_bins threads return the corresponding digit count.
    uint digit_scan;
  };

  union storage_t {
    uint16_t hist16[nt * num_bins];
    uint32_t hist32[nt * num_slots];
    typename scan_t::storage_t scan;
  };

  // Return the cta-wide reduction for each digit in the first num_bins
  // threads.
  template<int vt>
  uint reduce(std::array<uint, vt> x, storage_t& shared) {
    int tid = glcomp_LocalInvocationID.x;

    // Cooperatively zero out the histogram smem.
    @meta for(int i = 0; i < num_slots; ++i)
      shared.hist32[nt * i + tid] = 0;
    __syncthreads();

    // Compute the histogram for each thread. Not great for bank conflicts, but
    // at least it's easy.
    @meta for(int i = 0; i < vt; ++i)
      ++shared.hist16[nt * x[i] + tid];
    __syncthreads();

    // Cooperatively scan the entire histogram. Each thread loads 9 words,
    // which corresponds to 18 histogram slots. The odd grain size avoids
    // smem bank conflicts on all architectures.
    uint sum = 0;
    uint counters[num_slots];
    @meta for(int i = 0; i < num_slots; ++i) {
      counters[i] = shared.hist32[num_slots * tid + i];
      sum += counters[i];
    }
    __syncthreads();

    // Scan the reductions.
    uint carry_in = scan_t().scan(sum, shared.scan).scan;
    carry_in += (carry_in>> 16) | (carry_in<< 16);

    // Write the scanned histogram back to shared memory.
    @meta for(int i = 0; i < num_slots; ++i) {
      // Add .low to .high
      carry_in += counters[i]<< 16;
      carry_in += counters[i] + (counters[i]>> 16);
      shared.hist32[num_slots * tid + i] = carry_in;
    }
    __syncthreads();

    // Get the digit totals. This is a maximally-conflicted operation.
    uint digit_count = 0;
    if(tid < num_bins) {
      digit_count = shared.hist16[nt * tid + nt - 1];
      int left = subgroupShuffleUp(digit_count, 1);
      if(tid)
        digit_count -= left;
    }

    __syncthreads();
    return digit_count;
  }

  // Return the scatter indices for all keys plus the cta-wide scan for
  // each digit.
  template<int vt>
  result_t<vt> scatter(std::array<uint, vt> x, storage_t& shared) {
    int tid = glcomp_LocalInvocationID.x;

    // Cooperatively zero out the histogram smem.
    @meta for(int i = 0; i < num_slots; ++i)
      shared.hist32[nt * i + tid] = 0;
    __syncthreads();

    // Compute the histogram for each thread. Not great for bank conflicts, but
    // at least it's easy.
    @meta for(int i = 0; i < vt; ++i)
      ++shared.hist16[nt * x[i] + tid];
    __syncthreads();

    // Cooperatively scan the entire histogram. Each thread loads 9 words,
    // which corresponds to 18 histogram slots. The odd grain size avoids
    // smem bank conflicts on all architectures.
    uint sum = 0;
    uint counters[num_slots];
    @meta for(int i = 0; i < num_slots; ++i) {
      counters[i] = shared.hist32[num_slots * tid + i];
      sum += counters[i];
    }
    __syncthreads();

    // Scan the reductions.
    uint carry_in = scan_t().scan(sum, shared.scan).scan;
    carry_in += (carry_in>> 16) | (carry_in<< 16);

    // Write the scanned histogram back to shared memory.
    @meta for(int i = 0; i < num_slots; ++i) {
      // Add .low to .high
      carry_in += counters[i]<< 16;
      shared.hist32[num_slots * tid + i] = carry_in;
      carry_in += counters[i] + (counters[i]>> 16);
    }
    __syncthreads();

    // Get the digit totals. This is a maximally-conflicted operation.
    uint digit_scan = tid < num_bins ? shared.hist16[nt * tid] : 0;
    __syncthreads();

    // Make a downsweep pass by counting the digits a second time.
    std::array<uint, vt> scatter;
    @meta for(int i = 0; i < vt; ++i)
      scatter[i] = shared.hist16[nt * x[i] + tid]++;
    __syncthreads();

    return { scatter, digit_scan };
  }
};

////////////////////////////////////////////////////////////////////////////////
// Use ballot instruction to rank digits. Currently this only works on 
// 32-lane subgroups.

template<int nt, int num_bits>
struct cta_radix_rank_ballot_t {
  enum { 
    num_bins = 1<< num_bits,
    warp_size = 32,
    num_warps = nt / warp_size,
    counters_per_thread = num_bins * warp_size / nt + 1,
    num_counters = nt * counters_per_thread,
  };

  @meta printf("counters_per_thread = %d\n", counters_per_thread);

  // Simpler to require as many threads as thehre are histogram bins.
  static_assert(nt >= num_bins);

  typedef cta_scan_t<nt, uint> scan_t;

  template<int vt>
  struct result_t {
    // All threads return scatter indices for each value.
    std::array<uint, vt> indices;

    // The first num_bins threads return the corresponding digit count.
    uint digit_scan;
  };

  union storage_t {
    uint32_t counters[num_counters];
    uint32_t hist32[num_warps][num_bins];
    typename scan_t::storage_t scan;
  };

  uint32_t get_matching_lanes(uint digit) {
    // Start with all lanes matching.
    uint32_t match = -1;
    @meta for(int i = 0; i < num_bits; ++i) {{
      const uint flag = 1<< i;
      uint mask = flag & digit;
      uint32_t b = gl_subgroupBallot(mask).x;

      // Clear lanes from the match if they have a different bit.
      if(!mask) b = ~b;
      match &= b;
    }}

    return match;
  }

  template<int vt>
  uint reduce(std::array<uint, vt> x, storage_t& shared) {
    // Cooperatively zero out the histogram smem.
    int tid = glcomp_LocalInvocationID.x;
    int lane = gl_SubgroupInvocationID;
    int warp = gl_SubgroupID;

    // Cooperatively zero out the shared memory.
    @meta for(int i = 0; i < counters_per_thread; ++i)
      shared.counters[nt * i + tid] = 0;
    __syncthreads();

    // Process each digit.
    @meta for(int i = 0; i < vt; ++i) {{
      // Get a bitfield of lanes with matching digits.
      uint32_t match = get_matching_lanes(x[i]);

      // Increment the histogram bin to indicate the digit count.
      // Only the lowest lane in the match mask does this.
      if(0 == (gl_SubgroupLtMask & match))
        shared.hist32[warp][x[i]] += bitCount(match);
    }}
    __syncthreads();

    // Do a digit-wise reduction across warps.
    int digit_count = 0;
    if(tid < num_bins) {
      @meta for(int warp = 0; warp < num_warps; ++warp)
        digit_count += shared.hist32[warp][tid];
    }
    __syncthreads();

    return digit_count;
  }

  template<int vt>
  result_t<vt> scatter(std::array<uint, vt> x, storage_t& shared) {
    // Cooperatively zero out the histogram smem.
    int tid = glcomp_LocalInvocationID.x;
    int lane = gl_SubgroupInvocationID;
    int warp = gl_SubgroupID;

    // Cooperatively zero out the shared memory.
    @meta for(int i = 0; i < counters_per_thread; ++i)
      shared.counters[nt * i + tid] = 0;
    __syncthreads();

    // Process each digit.
    uint matches[vt];
    @meta for(int i = 0; i < vt; ++i) {{
      // Get a bitfield of lanes with matching digits.
      matches[i] = get_matching_lanes(x[i]);

      // Increment the histogram bin to indicate the digit count.
      // Only the lowest lane in the match mask does this.
      if(0 == (gl_SubgroupLtMask & matches[i]))
        shared.hist32[warp][x[i]] += bitCount(matches[i]);
    }}
    __syncthreads();

    // NOTE HOW WARP DIGITS MUST BE INTERLEAVED?

    if(tid < num_bins) {
      // Reduce the digit counts over the warps and keep a copy of the 
      // counters.
      uint counters[num_warps];
      uint digit_count = 0;
      @meta for(int warp = 0; warp < num_warps; ++warp)
        digit_count += counters[warp] = shared.hist32[warp][tid];

      // Do a cooperative CTA scan.
      uint scan = scan_t().scan(digit_count, shared.scan).scan;

      // Add back into the warp counters.
      @meta for(int warp = 0; warp < num_warps; ++warp) { 
        shared.hist32[warp][tid] = scan;
        scan += counters[warp];
      }
    }
    __syncthreads();

    // Make a second pass and compute scatter indices.
    std::array<uint, vt> scatter;
    @meta for(int i = 0; i < vt; ++i) {{
      uint lower_mask = gl_SubgroupLtMask.x & matches[i];
      scatter[i] = shared.hist32[warp][x[i]] + bitCount(lower_mask);
      if(!lower_mask)
        shared.hist32[warp][x[i]] = scatter[i] + bitCount(matches[i]);
    }}
    __syncthreads();

    return { scatter, 0 };
  }
};

END_MGPU_NAMESPACE
