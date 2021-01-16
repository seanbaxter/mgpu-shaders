#include <mgpu/vk/mergesort.hxx>
#include <cstdio>

using namespace mgpu::vk;

int main() {
  context_t context;

  int count = 10000;
  float* host = context.alloc_cpu<float>(count);
  float* gpu = context.alloc_gpu<float>(count);

  // Generate test data into the staging buffers.
  for(int i = 0; i < count; ++i)
    host[i] = rand() % 100000;

  // Create a command buffer.
  cmd_buffer_t cmd_buffer(context);
  cmd_buffer.begin();

  // Upload to GPU memory.
  cmd_buffer.memcpy(gpu, host, sizeof(float) * count);
  cmd_buffer.host_barrier();

  // Execute the parallel mergesort.
  mergesort_cache_t cache(context);
  mergesort_keys(cmd_buffer, cache, gpu, count);

  // Retrieve the results.
  cmd_buffer.memcpy(host, gpu, sizeof(float) * count);
  cmd_buffer.host_barrier();

  // End and submite the command buffer.
  cmd_buffer.end();
  context.submit(cmd_buffer);

  vkQueueWaitIdle(context.queue);

  // Print our results.
  for(int i = 0; i < count; ++i)
    printf("%5d: %f\n", i, host[i]);

  context.free(host);
  context.free(gpu);
}