#pragma once
#include "cta_merge.hxx"

BEGIN_MGPU_NAMESPACE

[[using spirv: buffer, readonly, binding(0)]]
int mp_data_in[];

template<typename type_t>
[[using spirv: buffer, readonly, binding(1)]]
type_t keysA_in[];

template<typename type_t>
[[using spirv: buffer, writeonly, binding(2)]]
type_t keysB_in[];

template<typename type_t>
[[using spirv: buffer, writeonly, binding(3)]]
type_t keysC_out[];

template<typename type_t>
[[using spirv: buffer, readonly, binding(4)]]
type_t valuesA_in[];

template<typename type_t>
[[using spirv: buffer, readonly, binding(5)]]
type_t valuesB_in[];

template<typename type_t>
[[using spirv: buffer, writeonly, binding(6)]]
type_t valuesC_out[];

template<typename comp_t>
struct merge_params_t {
  int a_count, b_count;
  comp_t comp;
};

template<typename comp_t>
[[using spirv: uniform, binding(0)]]
merge_params_t<comp_t> merge_params;

template<
  int nt, int vt, 
  typename a_keys_it, typename a_vals_it,
  typename b_keys_it, typename b_vals_it,
  typename c_keys_it, typename c_vals_it,
  typename comp_t
>
void kernel_merge(
  a_keys_it a_keys, a_vals_it a_vals, int a_count,
  b_keys_it b_keys, b_vals_it b_vals, int b_count,
  c_keys_it c_keys, c_vals_it c_vals, comp_t comp) {

  typedef typename std::iterator_traits<a_keys_it>::value_type type_t;
  typedef typename std::iterator_traits<a_vals_it>::value_type val_t;

  const int nv = nt * vt;
  int tid = glcomp_LocalInvocationID.x;
  int cta = glcomp_WorkGroupID.x;

  merge_params_t<comp_t> params = merge_params<comp_t>;

  // TODO: Replace with [[spirv::alias]]
  struct shared_t {
    type_t keys[nv + 1];
    int indices[nv];
  };
  [[spirv::alias]] shared_t shared;

  // Load the range for this CTA and merge the values into register.
  int mp0 = mp_data_in[cta + 0];
  int mp1 = mp_data_in[cta + 1];

  merge_range_t range = compute_merge_range(a_count, b_count, cta, nv, 
    mp0, mp1);

  merge_pair_t<type_t, vt> merge = cta_merge_from_mem<bounds_lower, nt, vt>(
    a_keys, b_keys, range, tid, params.comp, shared.keys);

  int dest_offset = nv * cta;
  reg_to_mem_thread<nt>(merge.keys, tid, range.total(), c_keys + dest_offset,
    shared.keys);
  
  if constexpr(!std::is_same_v<std::nullptr_t, val_t>) {

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
