#pragma once
#include "../common/kernel_partition.hxx"
#include "transform.hxx"

BEGIN_MGPU_NAMESPACE

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
