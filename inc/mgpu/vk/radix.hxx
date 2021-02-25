#pragma once
#include "../common/cta_radix.hxx"
#include "context.hxx"
#include "transform.hxx"

BEGIN_MGPU_NAMESPACE

namespace vk {

// Fully radix sort data within a CTA.
template<int nt, int vt, int num_bits = 4, typename key_t>
void radix_sort_cta(cmd_buffer_t& cmd_buffer, key_t* data, int count) {
  enum { nv = nt * vt };
  typedef unsigned_int_by_size_t<sizeof(key_t)> unsigned_type;

  if(!count) return;
  int num_ctas = div_up(count, nv);
  assert(1 == num_ctas);

  launch<nt>(num_ctas, cmd_buffer, [=](int tid, int cta) {
    typedef cta_radix_rank_t<nt, num_bits> radix_t;
    __shared__ union {
      typename radix_t::storage_t radix;
      unsigned_type keys[nv];
    } shared;

    // Load the data into thread order.
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
}

template<int nt, int vt, int num_bits = 4, typename key_t>
void radix_sort_pass(cmd_buffer_t& cmd_buffer, memcache_t& cache, 
  const key_t* data_in, key_t* data_out, int bit_offset, int count) {

  enum { nv = nt * vt };
  typedef unsigned_int_by_size_t<sizeof(key_t)> unsigned_type;
  
  if(!count) return;
  int num_ctas = div_up(count, nv);


}

} // namespace vk

END_MGPU_NAMESPACE
