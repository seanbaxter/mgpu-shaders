#pragma once
#include "loadstore.hxx"
#include "kernel_merge.hxx"

BEGIN_MGPU_NAMESPACE

template<bounds_t bounds, typename a_it, typename b_it, typename comp_t>
void kernel_partition(int* mp_data, a_it a, int a_count, b_it b, int b_count, 
  int spacing, comp_t comp) {

  int num_partitions = (int)div_up(a_count + b_count, spacing) + 1;
  int index = glcomp_GlobalInvocationID.x;
  if(index < num_partitions) {
    int diag = min(spacing * index, a_count + b_count);
    mp_data[index] = merge_path<bounds>(a, a_count, b, b_count, diag, comp); 
  }
}

template<bounds_t bounds, typename type_t, typename comp_t = std::less<type_t>>
[[using spirv: comp, local_size(128)]]
void kernel_partition() {
  kernel_partition<bounds>(
    mp_data_out, 
    keysA_in<type_t>,
    merge_params<comp_t>.a_count,
    keysB_in<type_t>,
    merge_params<comp_t>.b_count,
    merge_params<comp_t>.spacing,
    merge_params<comp_t>.comp
  );
}

template<bounds_t bounds, typename type_t, typename comp_t = std::less<type_t>>
void merge_path_partitions(GLuint mp_data, int a_count, int b_count, 
  int spacing) {

  int num_partitions = (int)div_up(a_count + b_count, spacing) + 1;
  int num_ctas = div_up(num_partitions, 128);

  gl_dispatch_kernel<kernel_partition<bounds, type_t, comp_t>>(num_ctas);
}

END_MGPU_NAMESPACE
