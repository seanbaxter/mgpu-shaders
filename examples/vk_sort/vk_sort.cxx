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
  void* aux_data = nullptr;
  size_t aux_size = 0;
  mergesort_keys(aux_data, aux_size, cmd_buffer, gpu, count);
  aux_data = context.alloc_gpu(aux_size);

  mergesort_keys(aux_data, aux_size, cmd_buffer, gpu, count);

  // Retrieve the results.
  cmd_buffer.memcpy(host, gpu, sizeof(float) * count);
  cmd_buffer.host_barrier();

  // End and submit the command buffer.
  cmd_buffer.end();
  context.submit(cmd_buffer);

  vkQueueWaitIdle(context.queue);

  // Print our results.
  for(int i = 0; i < count; ++i)
    printf("%5d: %f\n", i, host[i]);

  context.free(aux_data);
  context.free(host);
  context.free(gpu);
}