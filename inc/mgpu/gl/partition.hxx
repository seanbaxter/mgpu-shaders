#pragma once
#include "../common/cta_merge.hxx"
#include "transform.hxx"

BEGIN_MGPU_NAMESPACE

template<bounds_t bounds, typename params_t, int mp, int ubo>
[[using spirv: comp, local_size(128)]]
void kernel_partition() {
  // Load the kernel parameters from the uniform buffer at binding=ubo.
  params_t params = shader_uniform<ubo, params_t>;
  int a_count = params.a_count;
  int b_count = params.b_count;
  int spacing = params.spacing;

  int num_partitions = num_merge_partitions(a_count + b_count, spacing);
  int index = threadIdx.x + blockDim.x * blockIdx.x;

  if(index < num_partitions) {
    int diag = min(spacing * index, a_count + b_count);

    writeonly_iterator_t<int, mp> mp_data;
    mp_data[index] = merge_path<bounds>(params.a_keys, a_count, params.b_keys,
      b_count, diag, params.comp);
  }
}

template<bounds_t bounds, typename params_t, int mp, int ubo = 0>
void launch_partition(int count, int spacing) {
  int num_ctas = div_up(num_merge_partitions(count, spacing), 128);
  gl_dispatch_kernel<kernel_partition<bounds, params_t, mp, ubo> >(num_ctas);
}

END_MGPU_NAMESPACE
