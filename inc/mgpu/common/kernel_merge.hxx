#pragma once
#include "cta_merge.hxx"

BEGIN_MGPU_NAMESPACE

////////////////////////////////////////////////////////////////////////////////
// Generic merge code for a compute kernel.

template<
  int nt, int vt, 
  typename mp_it,
  typename a_keys_it, typename a_vals_it,
  typename b_keys_it, typename b_vals_it,
  typename c_keys_it, typename c_vals_it,
  typename comp_t
>
void kernel_merge(
  mp_it mp_data,
  a_keys_it a_keys, a_vals_it a_vals, int a_count,
  b_keys_it b_keys, b_vals_it b_vals, int b_count,
  c_keys_it c_keys, c_vals_it c_vals, comp_t comp) {

  typedef typename std::iterator_traits<a_keys_it>::value_type key_t;
  typedef typename std::iterator_traits<a_vals_it>::value_type val_t;

  const int nv = nt * vt;
  int tid = threadIdx.x;
  int cta = blockIdx.x;
 
  __shared__ ALIAS_UNION {
    key_t keys[nv + 1];
    int indices[nv];
  } shared;

  // Load the range for this CTA and merge the values into register.
  int mp0 = mp_data[cta + 0];
  int mp1 = mp_data[cta + 1];
  merge_range_t range = compute_merge_range(a_count, b_count, cta, nv, 
    mp0, mp1);

  merge_pair_t<key_t, vt> merge = cta_merge_from_mem<bounds_lower, nt, vt>(
    a_keys, b_keys, range, tid, comp, shared.keys);

  int dest_offset = nv * cta;
  reg_to_mem_thread<nt>(merge.keys, tid, range.total(), c_keys + dest_offset,
    shared.keys);

  if constexpr(!std::is_same_v<empty_t, val_t>) {
    // Transpose the indices from thread order to strided order.
    std::array<int, vt> indices = reg_thread_to_strided<nt>(merge.indices, tid, 
      shared.indices);
  
    // Gather the input values and merge into the output values.
    transfer_two_streams_strided<nt>(a_vals + range.a_begin, range.a_count(),
      b_vals + range.b_begin, range.b_count(), indices, tid, 
      c_vals + dest_offset);
  }
}

END_MGPU_NAMESPACE
