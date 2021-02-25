#include "cta_scan.hxx"

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


template<int nt, int num_bits = 4>
struct cta_radix_rank_t {
  enum { num_bins = 1<< num_bits, num_slots = num_bins / 2 + 1 };
  typedef cta_scan_t<nt, uint32_t> scan_t;

  template<int vt>
  struct result_t {
    // All threads return scatter indices for each value.
    std::array<uint32_t, vt> indices;

    // The first num_bins threads return the corresponding digit count.
    uint digit_count;
  };

  union storage_t {
    uint16_t hist16[nt * num_bins];
    uint32_t hist32[nt * num_slots];
    typename scan_t::storage_t scan;
  };

  // Return the scatter indices for all keys plus the cta-wide reduction for
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

    // Make a downsweep pass by counting the digits a second time.
    std::array<uint, vt> scatter;
    @meta for(int i = 0; i < vt; ++i)
      scatter[i] = shared.hist16[nt * x[i] + tid]++;
    __syncthreads();

    // Get the digit totals. This is a maximally-conflicted operation.
    uint digit_count = tid < num_bits ? shared.hist16[nt * tid + nt - 1] : 0;
    __syncthreads();

    return { scatter, digit_count };
  }
};

END_MGPU_NAMESPACE
