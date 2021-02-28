#pragma once
#include "../common/cta_radix.hxx"
#include "scan.hxx"

BEGIN_MGPU_NAMESPACE

namespace vk {

////////////////////////////////////////////////////////////////////////////////
// Scan counters for all bins. This avoids having to scatter into a single
// ordered array, which becomes inefficient as num_bins becomes large.

void radix_scan_8(void* aux_data, size_t& aux_size, 
  cmd_buffer_t& cmd_buffer, uint* counts, int num_frames) {

  enum { nt = 1024, num_bins = 256 };
  int num_ctas = div_up(num_frames, 32);
  if(!num_ctas) return;

  if(1 == num_ctas) {
    // Require no extra memory for single-CTA case.
    if(!aux_data) return;

    // Support up to 32 frames reduction in one CTA.
    launch<nt>(1, cmd_buffer, [=](int tid, int cta) mutable {
      typedef cta_scan_t<nt, uint> scan_t;
      __shared__ struct {
        // Use non-overlapping space for scan and to store counts.
        typename scan_t::storage_t scan;
        uint counts[nt];
      } shared;

      // The single-cta scan chops up 32 frames into four sections of 8
      // registers each. 
      int section = tid / 256;
      int lane = tid & 255;
      int frame0 = 8 * section;

      counts += num_bins * frame0 + lane;
      num_frames -= frame0;

      uint x[8];
      uint reduction = 0;
      @meta for(int i = 0; i < 8; ++i) {
        if(i < num_frames) {
          x[i] = counts[i * num_bins];
          reduction += x[i];
        }
      }
      shared.counts[tid] = reduction;
      __syncthreads();

      // Further reduce them across lanes.
      uint total = 0;
      if(0 == frame0) {
        total = reduction;
        @meta for(int i = 1; i < nt / num_bins; ++i)
          total += shared.counts[tid + i * num_bins];
      }
      __syncthreads();

      // Scan the first 256 array elements.
      uint scan = scan_t().scan(total, shared.scan).scan;

      if(tid < num_bins)
        counts[tid] = scan;

      // Apply the carry-in to all section reductions.
      if(0 == frame0) {
        shared.counts[tid] = scan;
        scan += reduction;

        @meta for(int i = 1; i < nt / num_bins; ++i) {
          reduction = shared.counts[tid + i * num_bins];
          shared.counts[tid + i * num_bins] = scan;
          scan += reduction;
        }
      }
      __syncthreads();

      // Scan and output the cached counts.
      scan = shared.counts[tid];
      @meta for(int i = 0; i < 8; ++i) {
        if(i < num_frames) {
          counts[i * num_bins] = scan;
          scan += x[i];
        }
      }
    });

  } else {
    if(!aux_data) {
      // Reserve one 256-item frame for each 32 input frames.
      aux_size += sizeof(uint) * num_bins * num_ctas;
      radix_scan_8(aux_data, aux_size, cmd_buffer, (uint*)nullptr, num_ctas);
      return;
    }

    // Write and read to a list of partials.
    uint* partials = advance_pointer<uint>(aux_data, num_bins * num_ctas);

    // Upsweep to reduce 32 frames into 1 frame.
    launch<nt>(num_ctas, cmd_buffer, [=](int tid, int cta) mutable {
      __shared__ uint shared_counts[nt];

      int frame0 = tid / 256 + 32 * cta;
      int lane = tid & 255;
      num_frames -= frame0;

      counts += num_bins * frame0 + lane;

      uint reduction = 0;
      @meta for(int i = 0; i < 32; i += 4) {
        if(i < num_frames)
          reduction += counts[i * num_bins];
      }
      shared_counts[tid] = reduction;
      __syncthreads();

      if(tid < num_bins) {
        @meta for(int i = 1; i < nt / num_bins; ++i)
          reduction += shared_counts[i * num_bins + tid];

        partials[num_bins * cta + tid] = reduction;
      }
    });

    // Recurse on the partials.
    radix_scan_8(aux_data, aux_size, cmd_buffer, partials, num_ctas);

    // Downsweep to scan 32 frames with carry-in from the partials.
    launch<nt>(num_ctas, cmd_buffer, [=](int tid, int cta) mutable {
      typedef cta_scan_t<nt, uint> scan_t;
      __shared__ struct {
        // Use non-overlapping space for scan and to store counts.
        uint counts[nt];
        typename scan_t::storage_t scan;
      } shared;
      
      // Spread the sections 8 frames across.
      int section = tid / 256;
      int lane = tid & 255;
      int frame0 = 8 * section;

      counts += (32 * cta + frame0) * num_bins + lane;
      num_frames -= 32 * cta + frame0;

      uint x[8];
      uint reduction = 0;
      @meta for(int i = 0; i < 8; ++i) {
        if(i < num_frames) {
          x[i] = counts[i * num_bins];
          reduction += x[i];
        }
      }
      shared.counts[tid] = reduction;
      __syncthreads();

      if(0 == frame0) {
        // Add the carry-in from the partials.
        uint scan = partials[num_bins * cta + tid];

        // Apply the carry-in to all section reductions.
        shared.counts[tid] = scan;
        scan += reduction;

        @meta for(int i = 1; i < nt / num_bins; ++i) {
          reduction = shared.counts[i * num_bins + tid];
          shared.counts[i * num_bins + tid] = scan;
          scan += reduction;
        }
      }
      __syncthreads();

      // Scan and output the cached counts.
      uint scan = shared.counts[tid];
      @meta for(int i = 0; i < 8; ++i) {
        if(i < num_frames) {
          counts[i * num_bins] = scan;
          scan += x[i];
        }
      }
    });
  }
}


////////////////////////////////////////////////////////////////////////////////
// Radix sort entry point that accepts 4 bit sort (shared memory histogram)
// or 8 bit sort (ballot with one histogram per warp). The ballot version uses
// much less memory, but the partial reduction scatter is devastating with
// 256 transactions, so a special radix scan is devised to handle both 
// implementations.

template<int nt = 128, int vt = 15, int num_bits = 4, 
  typename key_t>
void radix_sort(void* aux_data, size_t& aux_size, cmd_buffer_t& cmd_buffer, 
  key_t* data, int count) {

  enum { 
    nv = nt * vt, 
    num_bins = 1<< num_bits,
  };
  typedef unsigned_int_by_size_t<sizeof(key_t)> unsigned_type;

  static_assert(4 == num_bits || 8 == num_bits);
  
  // Require a 32-lane warp for ballot radix sort.
  // TODO: Write a 64-lane version.
  typedef cta_radix_rank_t<
    nt, 
    num_bits, 
    4 == num_bits ? radix_kind_shared : radix_kind_ballot
  > radix_t;
  const int subgroup_size = 8 == num_bits ? 32 : -1;

  if(!count) return;
  int num_ctas = div_up(count, nv);


  if(1 == num_ctas) {
    if(!aux_data) return;

    ////////////////////////////////////////////////////////////////////////////
    // Fully radix sort data within a CTA.

    launch<nt>(num_ctas, cmd_buffer, [=](int tid, int cta) {
      __shared__ union {
        typename radix_t::storage_t radix;
        unsigned_type keys[nv];
      } shared;

      int lane = gl_SubgroupInvocationID;
      int warp = gl_SubgroupID;
      int warp_size = gl_SubgroupSize;

      // Load the data into strided order.
      std::array<unsigned_type, vt> keys;
      @meta for(int i = 0; i < vt; ++i) {{
        int offset = nt * i + tid;
        if(offset < count) {
          // If the key is in range, load it and convert to radix bits.
          keys[i] = radix_permute_t<key_t>::to_radix_bits(data[offset]);

        } else {
          // Otherwise set all radix bits so this key is sorted to the end.
          keys[i] = -1;
        }
      }}

      // Move the keys into shared memory.
      reg_to_shared_strided<nt>(keys, tid, shared.keys);

      @meta for(int bit = 0; bit < 8 * sizeof(num_bits); bit += num_bits) {
        // Load the keys from shared memory.
        if constexpr(8 == num_bits)
          keys = shared_to_reg_warp<nt, vt>(shared.keys, lane, warp, warp_size);
        else
          keys = shared_to_reg_thread<nt, vt>(shared.keys, tid);

        // Extract the digits for each key.
        std::array digits { 
          (uint)bitfieldExtract(keys...[:], bit, num_bits) ...
        };

        // Compute the radix rank of each digit.
        auto result = radix_t().scatter(digits, shared.radix);

        // Scatter the keys into shared memory.
        shared.keys[result.indices...[:]] = keys...[:] ...;
        __syncthreads();
      }

      // Write from shared memory to device memory.
      @meta for(int i = 0; i < vt; ++i) {{
        int offset = nt * i + tid;
        if(offset < count) {
          unsigned_type u = shared.keys[offset];
          data[offset] = radix_permute_t<key_t>::from_radix_bits(u);
        }
      }}
    }); 

  } else {
    ////////////////////////////////////////////////////////////////////////////
    // Make multiple passes to sort the input.

    if(!aux_data) {
      // ping-pong keys buffer.
      aux_size += sizeof(key_t) * nv * num_ctas;

      // partials reduction buffer.
      aux_size += sizeof(uint) * num_bins * num_ctas;

      // scan auxiliary storage.
      if constexpr(8 == num_bits)
        radix_scan_8(aux_data, aux_size, cmd_buffer, (uint*)nullptr, num_ctas);
      else
        scan(nullptr, aux_size, cmd_buffer, (uint*)nullptr, num_bins * num_ctas);
      
      return;
    }

    // Allocate a second buffer to ping-pong.
    key_t* data2 = advance_pointer<key_t>(aux_data, nv * num_ctas);

    // Allocate space for each digit count.
    uint* partials = advance_pointer<uint>(aux_data, num_bins * num_ctas);

    for(int bit = 0; bit < 8 * sizeof(key_t); bit += num_bits) {

      //////////////////////////////////////////////////////////////////////////
      // Upsweep.

      launch<nt>(num_ctas, cmd_buffer, 
        [=](int tid, int cta) mutable {

        __shared__ union {
          typename radix_t::storage_t radix;
        } shared;

        int cur = nv * cta;
        data += cur;
        count -= cur;

        // Load the data into strided order.
        std::array<unsigned_type, vt> keys;
        @meta for(int i = 0; i < vt; ++i) {{
          int offset = nt * i + tid;
          if(offset < count) {
            // If the key is in range, load it and convert to radix bits.
            keys[i] = radix_permute_t<key_t>::to_radix_bits(data[offset]);

          } else {
            // Otherwise set all radix bits so this key is sorted to the end.
            keys[i] = -1;
          }
        }}

        // The upsweep doesn't care about the order of keys. Keep them in
        // strided order.

        // Extract the digits for each key.
        std::array digits { 
          (uint)bitfieldExtract(keys...[:], bit, num_bits) ...
        };

        // Compute the radix rank of each digit.
        uint digit_count = radix_t().reduce(digits, shared.radix);

        // Write the reductions to the counter.
        if(tid < num_bins) {
          int index = 0;
          if constexpr(8 == num_bits) {
            // Write densely to the output because we use the special radix
            // scan.
            index = num_bins * cta + tid;
          } else {
            // Interleave to the output because we use an ordinary scan.
            index = num_ctas * tid + cta;
          }

          partials[index] = digit_count;
        }
      });

      //////////////////////////////////////////////////////////////////////////
      // Scan.

      if constexpr(8 == num_bits) {
        radix_scan_8(aux_data, aux_size, cmd_buffer, partials, num_ctas);

      } else {
        scan(aux_data, aux_size, cmd_buffer, partials, num_bins * num_ctas);
      }

      //////////////////////////////////////////////////////////////////////////
      // Downsweep.

      launch<nt>(num_ctas, cmd_buffer, 
        [=](int tid, int cta) mutable {

        __shared__ union {
          typename radix_t::storage_t radix;
          unsigned_type keys[nv];
          ivec2 offsets[num_bins];
        } shared;

        int lane = gl_SubgroupInvocationID;
        int warp = gl_SubgroupID;
        int warp_size = gl_SubgroupSize;

        int cur = nv * cta;
        data += cur;
        count -= cur;

        // Load the data into strided order.
        std::array<unsigned_type, vt> keys;
        @meta for(int i = 0; i < vt; ++i) {{
          int offset = nt * i + tid;
          if(offset < count) {
            // If the key is in range, load it and convert to radix bits.
            keys[i] = radix_permute_t<key_t>::to_radix_bits(data[offset]);

          } else {
            // Otherwise set all radix bits so this key is sorted to the end.
            keys[i] = -1;
          }
        }}

        // Move the keys into shared memory.
        reg_to_shared_strided<nt>(keys, tid, shared.keys);

        // Load the keys from shared memory.
        if constexpr(8 == num_bits)
          keys = shared_to_reg_warp<nt, vt>(shared.keys, lane, warp, warp_size);
        else
          keys = shared_to_reg_thread<nt, vt>(shared.keys, tid);

        // Extract the digits for each key.
        std::array digits { 
          (uint)bitfieldExtract(keys...[:], bit, num_bits) ...
        };

        // Compute the radix rank of each digit.
        auto result = radix_t().scatter(digits, shared.radix);

        // Scatter the keys into shared memory.
        shared.keys[result.indices...[:]] = keys...[:] ...;
        __syncthreads();

        // Load keys in strided order.
        keys = shared_to_reg_strided<nt, vt>(shared.keys, tid);
      
        // Load the offset for each digit into global output.
        if(tid < num_bins) {
          int index = 0;
          if constexpr(8 == num_bits)
            index = num_bins * cta + tid;
          else
            index = num_ctas * tid + cta;

          shared.offsets[tid] = ivec2(
            result.digit_scan,    // local digit offset
            partials[index]       // global digit offset
          );
        }
        __syncthreads();

        // Write from shared memory to device memory.
        @meta for(int i = 0; i < vt; ++i) {{
          int offset = nt * i + tid;
          if(offset < count) {
            // Extract the digit.
            uint digit = bitfieldExtract(keys[i], bit, num_bits);

            // Look up the first occurrence of this digit within the CTA and
            // within the global output for this CTA. The difference is the
            // position at which we scatter to device memory.
            ivec2 offsets = shared.offsets[digit];
            offset += offsets.y - offsets.x;
            data2[offset] = radix_permute_t<key_t>::from_radix_bits(keys[i]);
          }
        }}
      });

      std::swap(data, data2);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////



} // namespace vk

END_MGPU_NAMESPACE
