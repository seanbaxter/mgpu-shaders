#pragma once
#include "../common/cta_radix.hxx"
#include "scan.hxx"

BEGIN_MGPU_NAMESPACE

namespace vk {

////////////////////////////////////////////////////////////////////////////////
// Multiple radix sort passes.

template<int nt, int vt, int num_bits = 4, typename key_t>
void radix_sort(cmd_buffer_t& cmd_buffer, memcache_t& cache, key_t* data,
  int count) {

  enum { nv = nt * vt, num_bins = 1<< num_bits };
  typedef unsigned_int_by_size_t<sizeof(key_t)> unsigned_type;
  
  if(!count) return;
  int num_ctas = div_up(count, nv);

  if(1 == num_ctas) {
    ////////////////////////////////////////////////////////////////////////////
    // Fully radix sort data within a CTA.

    launch<nt>(num_ctas, cmd_buffer, [=](int tid, int cta) {
      typedef cta_radix_rank_t<nt, num_bits> radix_t;
      __shared__ union {
        typename radix_t::storage_t radix;
        unsigned_type keys[nv];
      } shared;

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

      for(int bit = 0; bit < 8 * sizeof(key_t); bit += num_bits) {
        // Load the keys from shared memory.
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

    // Allocate space for each digit count.
    int* partials = data + 2 * count; //cache.allocate<int>(num_bins * num_ctas);

    // Allocate a second buffer to ping-pong.
    key_t* data2 = data + count;

    for(int bit = 0; bit < 32; bit += num_bits) {
      //////////////////////////////////////////////////////////////////////////
      // Upsweep.

      launch<nt>(num_ctas, cmd_buffer, [=](int tid, int cta) mutable {
        typedef cta_radix_rank_t<nt, num_bits> radix_t;
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
        if(tid < num_bins)
          partials[num_ctas * tid + cta] = digit_count;
      });

      //////////////////////////////////////////////////////////////////////////
      // Scan.

      scan(cmd_buffer, cache, partials, num_bins * num_ctas);

      //////////////////////////////////////////////////////////////////////////
      // Downsweep.

      launch<nt>(num_ctas, cmd_buffer, [=](int tid, int cta) mutable {
        typedef cta_radix_rank_t<nt, num_bits> radix_t;
        __shared__ union {
          typename radix_t::storage_t radix;
          unsigned_type keys[nv];
          ivec2 offsets[num_bins];
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

        // Move the keys into shared memory.
        reg_to_shared_strided<nt>(keys, tid, shared.keys);

        // Load the keys from shared memory.
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
          shared.offsets[tid] = ivec2(
            result.digit_scan,                // local digit offset
            partials[num_ctas * tid + cta]    // global digit offset
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

} // namespace vk

END_MGPU_NAMESPACE
