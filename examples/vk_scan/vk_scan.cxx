#pragma spirv GL_EXT_shared_memory_block

#include <mgpu/vk/transform.hxx>
#include <mgpu/vk/scan.hxx>
#include <mgpu/common/cta_scan.hxx>
#include <cstdio>
#include <tuple>
#include <array>

using namespace mgpu;
using namespace mgpu::vk;

int main() {
  context_t context;

  // Allocate test data storage.
  int count = 10000;
  int* host = context.alloc_cpu<int>(count);
  int* gpu  = context.alloc_gpu<int>(count);

  // Generate test data.
  for(int i = 0; i < count; ++i)
    host[i] = i;

  // Create a command buffer.
  cmd_buffer_t cmd_buffer(context);
  cmd_buffer.begin();

  // Upload test data to GPU memory.
  cmd_buffer.memcpy(gpu, host, sizeof(int) * count);
  cmd_buffer.host_barrier();

  // Execute the scan.
  void* aux_data = nullptr;
  size_t aux_size = 0;
  vk::scan(aux_data, aux_size, cmd_buffer, gpu, count);
  printf("aux size = %ld\n", aux_size);
  aux_data = context.alloc_gpu(aux_size);

  vk::scan(aux_data, aux_size, cmd_buffer, gpu, count);

  // Retrieve the results.
  cmd_buffer.memcpy(host, gpu, sizeof(int) * count);
  cmd_buffer.host_barrier();

  // End and submit the command buffer.
  cmd_buffer.end();
  context.submit(cmd_buffer);

  // And wait for it to be done.
  vkQueueWaitIdle(context.queue);

  for(int i = 0; i < count; ++i)
    printf("%3d: %2d\n", i, host[i]);

  context.free(aux_data);
  context.free(host);
  context.free(gpu);

  return 0;
}

