#define ALIASED_SMEM

#include "mgpu/vk/radix.hxx"
#include <cstdio>
#include <ctime>

using namespace mgpu;
using namespace mgpu::vk;

int main() {
  context_t context;

  // Allocate test data storage.
  enum { nt = 256, num_bits = 8, num_bins = 1<< num_bits, vt = 1, nv = nt * vt };

  typedef uint type_t;
  int count = nv * 32 * 32 * 32;
  int num_ctas = div_up(count, nv);
  std::vector<uint> ref(count);

  type_t* host = context.alloc_cpu<type_t>(count);
  type_t* gpu  = context.alloc_gpu<type_t>(count);

  // Generate test data.
  for(int i = 0; i < count; ++i) {
    ref[i] = host[i] = rand();
  }

  //for(int i = 0; i < count; i += nv) {
  //  std::sort(ref.begin() + i, ref.begin() + std::min(count, i + nv));
  //}
  std::sort(ref.begin(), ref.end());

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
  radix_sort<nt, vt, num_bits>(aux_data, aux_size, cmd_buffer, gpu, count);
  radix_sort<nt, vt, num_bits>(aux_data, aux_size, cmd_buffer, gpu, count);
  radix_sort<nt, vt, num_bits>(aux_data, aux_size, cmd_buffer, gpu, count);
  radix_sort<nt, vt, num_bits>(aux_data, aux_size, cmd_buffer, gpu, count);
  radix_sort<nt, vt, num_bits>(aux_data, aux_size, cmd_buffer, gpu, count);

  // Retrieve the results.
  cmd_buffer.memcpy(host, gpu, sizeof(type_t) * count);
  cmd_buffer.host_barrier();

  // End and submit the command buffer.
  cmd_buffer.end();
  context.submit(cmd_buffer);

  // And wait for it to be done.
  vkQueueWaitIdle(context.queue);

  for(int i = 0; i < count; ++i) {
  //  printf("%6d: %9d\n", i, host[i]); // - %3d - %3d\n", i, host[i], ref[i], scans[i / 32][i % 32]);
  //  printf("%3d: %5d \n", i, host[i]);

   if(host[i] != ref[i]) {
     printf("Error at %d: %d vs %d\n", i, host[i], ref[i]);
     exit(1);
   }
  }

  printf("MATCH\n");
  

  context.free(aux_data);
  context.free(host);
  context.free(gpu);

}

