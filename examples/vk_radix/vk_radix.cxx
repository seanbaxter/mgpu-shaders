#pragma spirv GL_EXT_shared_memory_block

#include "mgpu/vk/radix.hxx"
#include <cstdio>
#include <ctime>

using namespace mgpu;
using namespace mgpu::vk;

// Return a throughput in elements/second.
template<typename type_t, int nt, int vt>
double benchmark_radix(context_t& context, int count, int num_iterations,
  const type_t* init) {

  // Allocate test data storage.
  type_t* gpu = context.alloc_gpu<type_t>(count);

  // Create a command buffer.
  cmd_buffer_t cmd_buffer(context);
  cmd_buffer.begin();

  // Copy in host data.
  cmd_buffer.memcpy(gpu, init, sizeof(type_t) * count);

  // Query the radix sort for auxiliary data size and allocate.
  void* aux_data = nullptr;
  size_t aux_size = 0;
  radix_sort<nt, vt, 4>(aux_data, aux_size, cmd_buffer, gpu, count);

  // Do one run to let the shader compiler work.
  aux_data = context.alloc_gpu(aux_size);
  radix_sort<nt, vt, 4>(aux_data, aux_size, cmd_buffer, gpu, count);
  
  // End and submit the command buffer.
  cmd_buffer.end();
  context.submit(cmd_buffer);
  cmd_buffer.reset();

  // And wait for it to be done.
  vkQueueWaitIdle(context.queue);

  timespec start;
  clock_gettime(CLOCK_REALTIME, &start);
  
  cmd_buffer.begin();
  for(int i = 0; i < num_iterations; ++i)
    radix_sort<nt, vt, 8>(aux_data, aux_size, cmd_buffer, gpu, count);

  cmd_buffer.end();
  context.submit(cmd_buffer);
  vkQueueWaitIdle(context.queue);

  timespec end;
  clock_gettime(CLOCK_REALTIME, &end);

  double elapsed = (end.tv_sec - start.tv_sec) + 
    (end.tv_nsec - start.tv_nsec) * 1.0e-9;
  context.free(aux_data);
  context.free(gpu);

  return elapsed;
}

int main() {
  context_t context;

  typedef uint type_t;
  int max_count = 50'000'000;

  type_t* host = context.alloc_cpu<type_t>(max_count);
  for(int i = 0; i < max_count; ++i)
    host[i] = rand() + 2 * rand(); // fill all 32 bits.

  type_t* gpu = context.alloc_gpu<type_t>(max_count);

  // Create a command buffer.
  cmd_buffer_t cmd_buffer(context);
  cmd_buffer.begin();

  // Copy in host data.
  cmd_buffer.memcpy(gpu, host, sizeof(type_t) * max_count);

  cmd_buffer.end();
  context.submit(cmd_buffer);

  // Allocate auxiliary storage.
  void* aux_data;
  size_t aux_size = 0;
  radix_sort<128, 4, 4>(aux_data, aux_size, cmd_buffer, gpu, max_count);
  aux_data = context.alloc_gpu(aux_size);

  vkQueueWaitIdle(context.queue);

  int sizes[] { 1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 25, 30, 35, 40, 45, 50 };

  enum { nt = 256, vt = 16 };
  for(int size : sizes) {
    // Sort 5 billion keys at least.
    int count = 1'000'000 * size;
    int num_iterations = (int)ceil(5.0e9 / count);

    cmd_buffer.reset();
    cmd_buffer.begin();

    timespec start;
    clock_gettime(CLOCK_REALTIME, &start);
    
    cmd_buffer.begin();
    for(int i = 0; i < num_iterations; ++i)
      radix_sort<nt, vt, 8>(aux_data, aux_size, cmd_buffer, gpu, count);
    cmd_buffer.end();

    context.submit(cmd_buffer);
    vkQueueWaitIdle(context.queue);

    timespec end;
    clock_gettime(CLOCK_REALTIME, &end);

    double elapsed = (end.tv_sec - start.tv_sec) + 
      (end.tv_nsec - start.tv_nsec) * 1.0e-9;

    double rate = (double)count * num_iterations / elapsed / 1.0e6;

    printf("%9d: %20.5f  time=%f, iterations=%d\n", count, rate, elapsed, 
      num_iterations);
  }
   
  context.free(aux_data);
  context.free(gpu);
  context.free(host);
}

