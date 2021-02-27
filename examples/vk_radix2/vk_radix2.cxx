#pragma spirv GL_EXT_shared_memory_block

#include "mgpu/vk/radix.hxx"
#include <cstdio>
#include <ctime>

using namespace mgpu;
using namespace mgpu::vk;

int main() {
  context_t context;

  // Allocate test data storage.
  enum { nt = 256, num_bits = 8, num_bins = 1<< num_bits, vt = 16, nv = nt * vt };

  typedef uint type_t;
  int count = nt * vt;
  type_t* host = context.alloc_cpu<type_t>(count);
  type_t* gpu  = context.alloc_gpu<type_t>(count);

  // Generate test data.
  for(int i = 0; i < count; ++i) {
    host[i] = rand();
  }

  // Create a command buffer.
  cmd_buffer_t cmd_buffer(context);
  cmd_buffer.begin();

  // Upload test data to GPU memory.
  cmd_buffer.memcpy(gpu, host, sizeof(type_t) * count);
  cmd_buffer.host_barrier();

  void* aux_data = nullptr;
  size_t aux_size = 0;
  radix_sort<nt, vt, num_bits>(aux_data, aux_size, cmd_buffer, gpu, count);
  aux_data = context.alloc_gpu(aux_size);

  radix_sort<nt, vt, num_bits>(aux_data, aux_size, cmd_buffer, gpu, count);

  // Retrieve the results.
  cmd_buffer.memcpy(host, gpu, sizeof(type_t) * count);
  cmd_buffer.host_barrier();

  // End and submit the command buffer.
  cmd_buffer.end();
  context.submit(cmd_buffer);

  // And wait for it to be done.
  vkQueueWaitIdle(context.queue);

  int target[num_bins];
  target...[:] = -1 ...;
  for(int i = 0; i < count; ++i) {
    printf("%3d: 0x%08x\n", i, host[i]);
    uint a = host[std::max(0, i - 1)];
    uint b = host[i];
    if(a > b) {
      printf("Error at %d: %d vs %d\n", i - 1, a, b);
      exit(1);
    }
  }

  // context.free(aux_data);
  context.free(host);
  context.free(gpu);

}

