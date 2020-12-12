#pragma once
#include "cta_merge.hxx"
#include "transform.hxx"
#include "kernel_partition.hxx"

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

  typedef typename std::iterator_traits<a_keys_it>::value_type type_t;
  typedef typename std::iterator_traits<a_vals_it>::value_type val_t;

  const int nv = nt * vt;
  int tid = glcomp_LocalInvocationID.x;
  int cta = glcomp_WorkGroupID.x;
 
  struct [[spirv::alias]] shared_t {
    type_t keys[nv + 1];
    int indices[nv];
  };
  [[spirv::shared]] shared_t shared;

  // Load the range for this CTA and merge the values into register.
  int mp0 = mp_data[cta + 0];
  int mp1 = mp_data[cta + 1];
  merge_range_t range = compute_merge_range(a_count, b_count, cta, nv, 
    mp0, mp1);

  merge_pair_t<type_t, vt> merge = cta_merge_from_mem<bounds_lower, nt, vt>(
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

template<int nt, int vt, typename params_t, int mp, int ubo>
[[using spirv: comp, local_size(nt)]]
void kernel_merge() {
  params_t params = shader_uniform<ubo, params_t>;

  kernel_merge<nt, vt>(
    readonly_iterator_t<int, mp>(),
    
    params.a_keys,
    params.a_vals,
    params.a_count,

    params.b_keys,
    params.b_vals,
    params.b_count,

    params.c_keys,
    params.c_vals,

    params.comp
  );
}

template<
  typename a_keys_it,
  typename a_values_it,
  typename b_keys_it,
  typename b_values_it,
  typename c_keys_it,
  typename c_values_it,
  typename comp_t>
struct merge_params_t {
  a_keys_it a_keys;
  b_keys_it b_keys;
  c_keys_it c_keys;

  int a_count;
  int b_count;
  int spacing;           // NV * VT

  // Put the potentially empty objects together to take up less space.
  a_values_it a_vals;
  b_values_it b_vals;
  c_values_it c_vals;
  comp_t comp;
};

template<int nt, int vt, typename params_t, int mp, int ubo = 0>
void launch_merge(int count) {
  // First launch the partition kernel.
  launch_partition<bounds_lower, params_t, mp, ubo>(count, nt * vt);

  // Launch the CTA merge kernel.
  int num_ctas = div_up(count, nt * vt);
  // gl_dispatch_kernel<kernel_merge<nt, vt, params_t, mp, ubo> >(num_ctas);
}


END_MGPU_NAMESPACE
