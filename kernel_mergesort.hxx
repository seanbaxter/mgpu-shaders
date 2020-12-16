#pragma once
#include "cta_mergesort.hxx"

BEGIN_MGPU_NAMESPACE

// Sort full tiles in place.
template<
  int nt, int vt,
  typename keys_in_it, typename vals_in_it,
  typename keys_out_it, typename vals_out_it,
  typename comp_t
>
void kernel_blocksort(
  keys_in_it keys_in, vals_in_it vals_in,
  keys_out_it keys_out, vals_out_it vals_out, 
  comp_t comp, int count) {

  typedef typename std::iterator_traits<keys_in_it>::value_type key_t;
  typedef typename std::iterator_traits<vals_in_it>::value_type val_t;
  enum { has_values = !std::is_same<val_t, empty_t>::value };

  typedef cta_sort_t<nt, vt, key_t, val_t> sort_t;
  sort_t sort;

  [[spirv::shared]] typename sort_t::storage_t shared;

  const int nv = nt * vt;
  int tid = threadIdx.x;
  int cta = blockIdx.x;
  range_t tile = get_tile(cta, nv, count);

  // Load the keys and values.
  kv_array_t<key_t, val_t, vt> unsorted;
  unsorted.keys = mem_to_reg_thread<nt, vt>(keys_in + tile.begin, tid, 
    tile.count(), shared.keys);
  if(has_values)
    unsorted.vals = mem_to_reg_thread<nt, vt>(vals_in + tile.begin, tid,
      tile.count(), shared.vals);

  // Blocksort.
  kv_array_t<key_t, val_t, vt> sorted = sort_t().block_sort(unsorted,
    tid, tile.count(), comp, shared.sort);

  // Store the keys and values.
  reg_to_mem_thread<nt, vt>(sorted.keys, tid, tile.count(), 
    keys_out + tile.begin, shared.keys);
  if(has_values)
    reg_to_mem_thread<nt, vt>(sorted.vals, tid, tile.count(), 
      vals_out + tile.begin, shared.vals);
}

template<int nt, int vt, typename params_t, int ubo>
[[using spirv: comp, local_size(nt)]]
void kernel_blocksort() {
  params_t params = shader_uniform<ubo, params_t>;

  kernel_blocksort<nt, vt>(
    params.keys_in,
    params.vals_in,
    params.keys_out,
    params.vals_out,
    params.comp,
    params.count
  );
}

////////////////////////////////////////////////////////////////////////////////

template<
  int nt, int vt,
  typename mp_it,
  typename keys_in_it, typename vals_in_it,
  typename keys_out_it, typename vals_out_it,
  typename comp_t
>
void kernel_mergesort_pass(mp_it mp_data,
  keys_in_it keys_in, vals_in_it vals_in,
  keys_out_it keys_out, vals_out_it vals_out, 
  comp_t comp, int count, int coop) {

  typedef typename std::iterator_traits<keys_in_it>::value_type key_t;
  typedef typename std::iterator_traits<vals_in_it>::value_type val_t;
  enum { has_values = !std::is_same<val_t, empty_t>::value };

  const int nv = nt * vt;
  int tid = threadIdx.x;
  int cta = blockIdx.x;

  [[spirv::shared]] key_t keys[nv + 1];
  [[spirv::shared]] int   indices[nv];

  range_t tile = get_tile(cta, nv, count);

  // Load the range for this CTA and merge the values into register.
  merge_range_t range = compute_mergesort_range(count, cta, coop, nv, 
    mp_data[cta + 0], mp_data[cta + 1]);

  merge_pair_t<key_t, vt> merge = cta_merge_from_mem<bounds_lower, nt, vt>(
    keys_in, keys_in, range, tid, comp, keys);

  // Store merged values back out.
  reg_to_mem_thread<nt>(merge.keys, tid, tile.count(), 
    keys_out + tile.begin, keys);

  if(has_values) {
    // Transpose the indices from thread order to strided order.
    std::array<int, vt> indices = reg_thread_to_strided<nt>(merge.indices,
      tid, indices);

    // Gather the input values and merge into the output values.
    transfer_two_streams_strided<nt>(vals_in + range.a_begin, 
      range.a_count(), vals_in + range.b_begin, range.b_count(),
      indices, tid, vals_out + tile.begin);
  }
}

template<int nt, int vt, typename params_t, int mp, int ubo>
[[using spirv: comp, local_size(nt)]]
void kernel_mergesort_pass() {
  params_t params = shader_uniform<ubo, params_t>;

  kernel_mergesort_pass<nt, vt>(
    readonly_iterator_t<int, mp>(),
    params.keys_in,
    params.vals_in,
    params.keys_out,
    params.vals_out,
    params.comp,
    params.count,
    params.coop
  );
}

template<
  typename keys_in_it,
  typename vals_in_it,
  typename keys_out_it,
  typename vals_out_it,
  typename comp_t
> struct mergesort_params_t {
  keys_in_it keys_in;
  vals_in_it vals_in;
  keys_out_it keys_out;

  int count;
  int coop; // 2<< pass.
};




END_MGPU_NAMESPACE
