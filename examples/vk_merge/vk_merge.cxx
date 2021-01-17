#include <mgpu/vk/merge.hxx>
#include <cstdio>

using namespace mgpu::vk;

int main() {
  context_t context;

  int a_count = 10000;
  int b_count = 10000;
  int count = a_count + b_count;

  float* a_host = context.alloc_cpu<float>(a_count);
  float* b_host = context.alloc_cpu<float>(b_count);
  float* c_host = context.alloc_cpu<float>(a_count + b_count);

  // Generate test data into the staging buffers.
  for(int i = 0; i < a_count; ++i)
    a_host[i] = rand() % 100000;
  for(int i = 0; i < b_count; ++i)
    b_host[i] = rand() % 100000;

  // Sort both input sequences.
  std::sort(a_host, a_host + a_count);
  std::sort(b_host, b_host + b_count);

  float* a_gpu = context.alloc_gpu<float>(a_count);
  float* b_gpu = context.alloc_gpu<float>(b_count);
  float* c_gpu = context.alloc_gpu<float>(count);

  // Create a command buffer.
  cmd_buffer_t cmd_buffer(context);
  cmd_buffer.begin();

  // Upload to GPU memory.
  cmd_buffer.memcpy(a_gpu, a_host, sizeof(float) * a_count);
  cmd_buffer.memcpy(b_gpu, b_host, sizeof(float) * b_count);
  cmd_buffer.host_barrier();

  // Execute the parallel merge.
  memcache_t memcache(context);
  merge(cmd_buffer, memcache, a_gpu, a_count, b_gpu, b_count, c_gpu, 
    std::less<float>());

  // Retrieve the results.
  cmd_buffer.memcpy(c_host, c_gpu, sizeof(float) * count);
  cmd_buffer.host_barrier();

  // End and submite the command buffer.
  cmd_buffer.end();
  context.submit(cmd_buffer);

  vkQueueWaitIdle(context.queue);

  // Print our results.
  for(int i = 0; i < count; ++i)
    printf("%5d: %f\n", i, c_host[i]);

  context.free(a_host);
  context.free(b_host);
  context.free(c_host);
  context.free(a_gpu);
  context.free(b_gpu);
  context.free(c_gpu);  
}