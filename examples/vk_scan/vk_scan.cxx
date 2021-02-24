#include <mgpu/vk/transform.hxx>
#include <mgpu/common/cta_scan.hxx>
#include <cstdio>
#include <tuple>
#include <array>

using namespace mgpu;
using namespace mgpu::vk;

int main() {
  context_t context;

  // Allocate test data storage.
  enum { nt = 128 };
  int count = nt;
  int num_blocks = div_up(count, nt);
  int* data = context.alloc_gpu<int>(count);

  // Create a command buffer.
  cmd_buffer_t cmd_buffer(context);
  cmd_buffer.begin();

  launch<nt>(num_blocks, cmd_buffer, [=](int tid, int cta) {
    // We can now put the scan's storage_t into a union!
    typedef cta_scan_t<nt, int> scan_t;
    [[spirv::shared]] scan_t::storage_t shared;

    // Just scan 1.
    auto result = scan_t().scan<scan_type_exc>(1, shared);

    data[tid] = result.scan;
  });

  // Copy the data to host memory.
  int* host = context.alloc_cpu<int>(count);

  cmd_buffer.host_barrier();
  context.memcpy(cmd_buffer, host, data, sizeof(int) * count);
  cmd_buffer.host_barrier();

  // End and submit the command buffer.
  cmd_buffer.end();
  context.submit(cmd_buffer);

  // And wait for it to be done.
  vkQueueWaitIdle(context.queue);

  for(int i = 0; i < count; ++i)
    printf("%3d: %2d\n", i, host[i]);

  context.free(data);
  context.free(host);

  return 0;
}

