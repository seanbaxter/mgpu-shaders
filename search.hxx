#pragma once
#include "loadstore.hxx"
#include "kernel_merge.hxx"

BEGIN_MGPU_NAMESPACE

template<bounds_t bounds, typename mp_it, typename a_it, typename b_it, 
  typename comp_t>
void kernel_partition(mp_it mp_data, a_it a, int a_count, b_it b, int b_count, 
  int spacing, comp_t comp) {

  int num_partitions = (int)div_up(a_count + b_count, spacing) + 1;
  int index = glcomp_GlobalInvocationID.x;
  if(index < num_partitions) {
    int diag = min(spacing * index, a_count + b_count);
    mp_data[index] = merge_path<bounds>(a, a_count, b, b_count, diag, comp); 
  }
}

template<bounds_t bounds, typename params_t, int ubo>
[[using spirv: comp, local_size(128)]]
void kernel_partition() {
  // Load the kernel parameters from the uniform buffer at binding=ubo.
  params_t params = shader_uniform<ubo, params_t>;

  // Launch the kernel using kernel parameters.
  kernel_partition<bounds>(
    params.mp_data,
    params.a,
    params.a_count,
    params.b,
    params.b_count,
    params.spacing,
    params.comp
  );
}

template<
  typename a_it, 
  typename b_it, 
  typename mp_it, 
  typename comp_t = std::less<decltype(std::declval<a_it>()[0])>
>
struct merge_path_partitions_t {
  a_it a;
  b_it b;
  mp_it mp_data;
  comp_t comp;

  int a_count;
  int b_count;
  int spacing;

  template<bounds_t bounds, int ubo = 0>
  void launch() {
    int num_partitions = (int)div_up(a_count + b_count, spacing) + 1;
    int num_ctas = div_up(num_partitions, 128);

    gl_dispatch_kernel<kernel_partition<bounds, merge_path_partitions_t, ubo> >(
      num_ctas
    );
  }
};

END_MGPU_NAMESPACE
