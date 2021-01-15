#pragma once
#include "context.hxx"

BEGIN_MGPU_NAMESPACE

namespace vk {

// Chevron launch on a SPIR-V compute shader performs ADL lookup to find
// this symbol, and overload resolution to select this overload.
template<auto F, typename... params_t>
void spirv_compute_dispatch(int num_blocks, cmd_buffer_t& cmd_buffer, 
  params_t... params) {

  static_assert((... && std::is_trivially_copyable_v<params_t>));
  tuple_t<params_t...> storage { params... };

  cmd_buffer.context.dispatch_compute(
    cmd_buffer,
    @spirv(F),
    cmd_buffer.context.create_module(__spirv_data, __spirv_size),
    num_blocks,
    sizeof(storage),
    &storage
  );
}

} // namespace vk

END_MGPU_NAMESPACE