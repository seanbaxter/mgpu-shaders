#pragma once
#include "../common/cta_merge.hxx"
#include "transform.hxx"

BEGIN_MGPU_NAMESPACE

namespace vk {

template<bounds_t bounds, typename a_keys_it, typename b_keys_it,
  typename comp_t>
void merge_path_partitions(cmd_buffer_t& cmd_buffer, a_keys_it a, 
  int a_count, b_keys_it b, int b_count, int* mp_data, int spacing, 
  comp_t comp) {

  int num_partitions = num_merge_partitions(a_count + b_count, spacing);
  transform(num_partitions, cmd_buffer, [=](int index) {
    int diag = min(spacing * index, a_count + b_count);
    mp_data[index] = merge_path<bounds>(a, a_count, b, b_count, diag, comp);
  });
}

} // namespace vk

END_MGPU_NAMESPACE
