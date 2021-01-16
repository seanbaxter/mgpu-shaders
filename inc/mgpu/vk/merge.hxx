#pragma once
#include "../common/kernel_merge.hxx"
#include "partition.hxx"  // TODO: PORT PARTITION

#include <cstdio>

BEGIN_MGPU_NAMESPACE

namespace vk {

// Key-value merge.
template<
  int nt = 128, int vt = 7, // Optional tuning parameters.
  typename a_keys_it, typename a_vals_it, 
  typename b_keys_it, typename b_vals_it,
  typename c_keys_it, typename c_vals_it, 
  typename comp_t
>
void merge(
  cmd_buffer_t& cmd_buffer, memcache_t& cache,
  a_keys_it a_keys, a_vals_it a_vals, int a_count, 
  b_keys_it b_keys, b_vals_it b_vals, int b_count,
  c_keys_it c_keys, c_vals_it c_vals, comp_t comp) {

  typedef typename std::iterator_traits<a_keys_it>::value_type type_t;
  typedef typename std::iterator_traits<a_vals_it>::value_type val_t;
  constexpr int nv = nt * vt;

  int num_partitions = num_merge_partitions(a_count + b_count, nv);
  int* partitions = cache.allocate<int>(num_partitions);

  merge_path_partitions<bounds_lower>(cmd_buffer, a_keys, a_count, 
    b_keys, b_count, partitions, nv, comp);

  int num_blocks = div_up(a_count + b_count, nv);
  launch<nt>(num_blocks, cmd_buffer, [=](int tid, int block) {
    if constexpr(std::is_same_v<val_t, empty_t>) {
      kernel_merge<nt, vt>(partitions, a_keys, a_vals, a_count, b_keys, 
        b_vals, b_count, c_keys, c_vals, comp);

    } else {
      kernel_merge<nt, vt>(partitions, a_keys, (empty_t*)nullptr, a_count,
        b_keys, (empty_t*)nullptr, b_count, c_keys, (empty_t*)nullptr, comp);
    }
  });
}

// Key-only merge.
template<int nt = 128, int vt = 7,
  typename a_keys_it, typename b_keys_it, typename c_keys_it,
  typename comp_t>
void merge(cmd_buffer_t& cmd_buffer, memcache_t& cache, a_keys_it a_keys, 
  int a_count, b_keys_it b_keys, int b_count, c_keys_it c_keys, comp_t comp) {

  merge<nt, vt>(cmd_buffer, cache, a_keys, (const empty_t*)nullptr, a_count, 
    b_keys, (const empty_t*)nullptr, b_count, c_keys, (empty_t*)nullptr, comp);
}

} // namespace vk

END_MGPU_NAMESPACE
