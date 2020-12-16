#pragma once
#include "loadstore.hxx"
#include "kernel_merge.hxx"

BEGIN_MGPU_NAMESPACE

// Basic partitioning used for merge and load balancing search.
// Mergesort and segmented sort have their own partitioning kernels.
template<bounds_t bounds, typename mp_it, typename a_it, typename b_it, 
  typename comp_t>
void kernel_partition(mp_it mp_data, a_it a, int a_count, b_it b, int b_count, 
  int spacing, comp_t comp) {

  int num_partitions = (int)div_up(a_count + b_count, spacing) + 1;
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index < num_partitions) {
    int diag = min(spacing * index, a_count + b_count);
    mp_data[index] = merge_path<bounds>(a, a_count, b, b_count, diag, comp); 
  }
}

template<bounds_t bounds, typename params_t, int mp, int ubo>
[[using spirv: comp, local_size(128)]]
void kernel_partition() {
  // Load the kernel parameters from the uniform buffer at binding=ubo.
  params_t params = shader_uniform<ubo, params_t>;

  // Launch the kernel using kernel parameters.
  kernel_partition<bounds>(
    writeonly_iterator_t<int, mp>(),
    params.a_keys,
    params.a_count,
    params.b_keys,
    params.b_count,
    params.spacing,
    params.comp
  );
}

int num_merge_partitions(int count, int spacing) {
  return div_up(count, spacing) + 1;
}

template<bounds_t bounds, typename params_t, int mp, int ubo = 0>
void launch_partition(int count, int spacing) {
  int num_ctas = div_up(num_merge_partitions(count, spacing), 128);
  gl_dispatch_kernel<kernel_partition<bounds, params_t, mp, ubo> >(num_ctas);
}

END_MGPU_NAMESPACE
