#pragma once
#include "launch.hxx"

BEGIN_MGPU_NAMESPACE

namespace vk {

// Launch a grid and pass (tid, gid)

template<int nt, typename func_t>
[[using spirv: comp, local_size(nt), push]]
void launch_cs(func_t func) {
  func(threadIdx.x, blockIdx.x);
}

template<int nt, typename func_t>
static void launch(int num_blocks, cmd_buffer_t& cmd_buffer, func_t func) {
  launch_cs<nt><<<num_blocks, cmd_buffer>>>(func);
}

template<int nt = 256, typename func_t>
[[using spirv: comp, local_size(nt), push]]
void transform_cs(int count, func_t func) {
  int gid = glcomp_GlobalInvocationID.x;

  if(gid >= count)
    return;

  func(gid);
}

template<int nt = 256, typename func_t>
static void transform(int count, cmd_buffer_t& cmd_buffer, func_t func) {
  int num_blocks = div_up(count, nt);
  transform_cs<nt><<<num_blocks, cmd_buffer>>>(count, func);
}

} // namespace vk

END_MGPU_NAMESPACE