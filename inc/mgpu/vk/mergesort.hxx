#pragma once
#include "../common/kernel_mergesort.hxx"
#include "context.hxx"
#include "transform.hxx"

BEGIN_MGPU_NAMESPACE

namespace vk {

template<int nt = 128, int vt = 7, bool sort_indices = false, 
  typename key_t, typename val_t, typename comp_t = std::less<key_t> >
void mergesort_kv(
  void* aux_data, size_t& aux_size, cmd_buffer_t& cmd_buffer,
  key_t* keys, val_t* vals, int count, comp_t comp = comp_t()) {

  static_assert(!sort_indices || std::is_same_v<int, val_t>);
  constexpr bool has_values = !std::is_same_v<empty_t, val_t>;

  int num_ctas = div_up(count, nt * vt);
  int num_passes = find_log2(num_ctas, true);

  if(0 == num_passes) {
    if(!aux_data) return;

    // For a single CTA, sort in place and don't require any cache memory.
    launch<nt>(num_ctas, cmd_buffer, [=](int tid, int cta) {
      kernel_blocksort<sort_indices, nt, vt>(keys, vals, keys, vals, 
        count, comp);
    });

  } else {
    int num_partitions = num_ctas + 1;

    // Allocate temporary storage for the partitions and ping-pong buffers.
    if(!aux_data) {
      aux_size += sizeof(int) * num_partitions;
      aux_size += sizeof(key_t) * count;
      if(has_values)
        aux_size += sizeof(val_t) * count;
      return;
    }

    int* mp = advance_pointer<int>(aux_data, num_partitions);
    key_t* keys2 = advance_pointer<key_t>(aux_data, count);
    val_t* vals2 = advance_pointer<val_t>(aux_data, count);

    key_t* keys_blocksort = (1 & num_passes) ? keys2 : keys;
    val_t* vals_blocksort = (1 & num_passes) ? vals2 : vals;

    // Blocksort the input.
    launch<nt>(num_ctas, cmd_buffer, [=](int tid, int cta) {
      kernel_blocksort<sort_indices, nt, vt>(keys, vals, keys_blocksort,
        vals_blocksort, count, comp);
    });

    if(1 & num_passes) {
      std::swap(keys, keys2);
      std::swap(vals, vals2);
    }

    for(int pass = 0; pass < num_passes; ++pass) {
      int coop = 2<< pass;

      // Partition the partially-sorted inputs.
      transform(num_partitions, cmd_buffer, [=](int index) {
        int spacing = nt * vt;
        merge_range_t range = compute_mergesort_range(count, index, coop, 
          spacing);
        int diag = min(spacing * index, count) - range.a_begin;
        mp[index] = merge_path<bounds_lower>(keys + range.a_begin,
          range.a_count(), keys + range.b_begin, range.b_count(), diag, comp);
      });

      // Launch the merge pass.
      launch<nt>(num_ctas, cmd_buffer, [=](int tid, int cta) {
        kernel_mergesort_pass<nt, vt>(mp, keys, vals, keys2, vals2, count,
          coop, comp);
      });
      
      std::swap(keys, keys2);
      std::swap(vals, vals2);
    }
  }
}

template<int nt = 128, int vt = 7, typename key_t,
  typename comp_t = std::less<key_t> >
void mergesort_keys(void* aux_data, size_t& aux_size, cmd_buffer_t& cmd_buffer,
  key_t* keys, int count, comp_t comp = comp_t()) {

  mergesort_kv<nt, vt, false>(aux_data, aux_size, cmd_buffer, keys, 
    (empty_t*)nullptr, count, comp);
}

template<int nt = 128, int vt = 7, typename key_t,
  typename comp_t = std::less<key_t> >
void mergesort_indices(void* aux_data, size_t& aux_size, 
  cmd_buffer_t& cmd_buffer, key_t* keys, int* indices, int count, 
  comp_t comp = comp_t()) {

  mergesort_kv<nt, vt, true>(aux_data, aux_size, cmd_buffer, keys, indices,
    count, comp);
}

} // namespace vk

END_MGPU_NAMESPACE
