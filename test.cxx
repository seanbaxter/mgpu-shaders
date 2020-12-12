#include "kernel_merge.hxx"
#include <vector>

using namespace mgpu;

template<typename type_t>
std::vector<type_t> gpu_merge(std::vector<type_t>& a, 
  std::vector<type_t>& b) {

  gl_buffer_t<type_t> a_keys(a);
  gl_buffer_t<type_t> b_keys(b);
  gl_buffer_t<type_t> c_keys(a.size() + b.size());

  merge_params_t<
    readonly_iterator_t<type_t, 0>,
    empty_iterator_t,

    readonly_iterator_t<type_t, 1>,
    empty_iterator_t,

    writeonly_iterator_t<type_t, 2>,
    empty_iterator_t,

    // Use default comparison.
    std::less<int>
  > params;

  params.a_count = a.size();
  params.b_count = b.size();

  const int nt = 128;
  const int vt = 7;
  params.spacing = nt * vt;

  launch_merge<nt, vt, decltype(params), 3>(a.size() + b.size());

  return c_keys.get_data();
}

int main() {
  std::vector<float> a(100), b(100);
  std::vector<float> c = gpu_merge(a, b);
}