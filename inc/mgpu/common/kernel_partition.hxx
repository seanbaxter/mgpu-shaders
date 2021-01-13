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

END_MGPU_NAMESPACE
