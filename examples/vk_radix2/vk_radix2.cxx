#pragma spirv GL_EXT_shared_memory_block

#include "mgpu/vk/radix.hxx"
#include <cstdio>
#include <ctime>

using namespace mgpu;
using namespace mgpu::vk;

int main() {
  context_t context;

  // Allocate test data storage.
  enum { nt = 256, num_bits = 8, num_bins = 1<< num_bits, vt = 9, nv = nt * vt };

  typedef uint type_t;
  int count = nt * vt;
  type_t* host = context.alloc_cpu<type_t>(count);
  type_t* gpu  = context.alloc_gpu<type_t>(count);

  int hist[num_bins] { };

  // Generate test data.
  for(int i = 0; i < count; ++i) {
    host[i] = rand() % num_bins;
    hist[host[i]]++;
  }

  // Create a command buffer.
  cmd_buffer_t cmd_buffer(context);
  cmd_buffer.begin();

  // Upload test data to GPU memory.
  cmd_buffer.memcpy(gpu, host, sizeof(type_t) * count);
  cmd_buffer.host_barrier();

  launch<nt>(1, cmd_buffer, [=](int tid, int cta) {
    typedef cta_radix_rank_ballot_t<nt, num_bits> radix_t;
    __shared__ union {
      typename radix_t::storage_t radix;
      type_t keys[nv];
    } shared;

    std::array<uint, vt> x = mem_to_reg_thread<nt, vt>(gpu, tid, count,
      shared.keys);

    auto result = radix_t().scatter(x, shared.radix);
    shared.keys[result.indices...[:]] = x...[:] ...;
    __syncthreads();

    mem_to_mem<nt, vt>(shared.keys, tid, count, gpu);
  });


  // Retrieve the results.
  cmd_buffer.memcpy(host, gpu, sizeof(type_t) * count);
  cmd_buffer.host_barrier();

  // End and submit the command buffer.
  cmd_buffer.end();
  context.submit(cmd_buffer);

  // And wait for it to be done.
  vkQueueWaitIdle(context.queue);

  int total = 0;
  for(int i = 0; i < count; ++i) {
    total += host[i];
    printf("%3d: %2d - %2d\n", i, host[i], hist[i]);
  }
  printf("total = %d\n", total);

  context.free(host);
  context.free(gpu);

}

