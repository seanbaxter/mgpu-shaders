#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>
#include <time.h>

int main(void) {
  typedef uint type_t;
  int max_count = 50'000'000;

  thrust::host_vector<int> host(max_count);
  for(int i = 0; i < max_count; ++i)
    host[i] = rand() + 2 * rand();  // fill all 32 bits.

  // Copy in host data.
  thrust::device_vector<int> gpu = host;

  int sizes[] { 1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 25, 30, 35, 40, 45, 50 };

  for(int size : sizes) {
    // Sort 5 billion keys at least.
    int count = 1'000'000 * size;
    int num_iterations = (int)ceil(5.0e9 / count);

    cudaDeviceSynchronize();
    timespec start;
    clock_gettime(CLOCK_REALTIME, &start);
    
    for(int i = 0; i < num_iterations; ++i)
      thrust::sort(gpu.begin(), gpu.begin() + count);
    
    cudaDeviceSynchronize();

    timespec end;
    clock_gettime(CLOCK_REALTIME, &end);

    double elapsed = (end.tv_sec - start.tv_sec) + 
      (end.tv_nsec - start.tv_nsec) * 1.0e-9;

    double rate = (double)count * num_iterations / elapsed / 1.0e6;

    printf("%9d: %20.5f  time=%f, iterations=%d\n", count, rate, elapsed, 
      num_iterations);
  }

  return 0;
}