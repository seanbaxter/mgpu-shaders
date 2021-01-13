#include <mgpu/gl/merge.hxx>
#include <mgpu/gl/app.hxx>

#include <cstdio>
#include <algorithm>

using namespace mgpu;

template<int nt, int vt, typename type_t, typename comp_t = std::less<type_t> >
std::vector<type_t> gpu_merge(const std::vector<type_t>& a, 
  const std::vector<type_t>& b, comp_t comp = comp_t()) {

  // Upload keys to OpenGL SSBOs.
  gl_buffer_t<type_t[]> a_keys(a);
  gl_buffer_t<type_t[]> b_keys(b);

  // Allocate an SSBO for the result.
  gl_buffer_t<type_t[]> c_keys(a.size() + b.size());

  // Merge the keys!
  merge_pipeline_t<type_t> pipeline;
  pipeline.launch(a_keys, a.size(), b_keys, b.size(), c_keys, comp);

  // Return the result in host memory.
  return c_keys.get_data();
}

int main() {
  app_t app("merge demo");

  int a_count = 10000;
  int b_count = 10000;
  std::vector<float> a(a_count), b(b_count);
  a[:] = rand() % 100000...; std::sort(a.begin(), a.end());
  b[:] = rand() % 100000...; std::sort(b.begin(), b.end());

  std::vector<float> c = gpu_merge<128, 7>(a, b);
  printf("%d: %f\n", @range(), c[:])...;
}