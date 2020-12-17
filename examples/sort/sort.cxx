#include <mgpu/kernel_mergesort.hxx>

using namespace mgpu;
// key-index sort.
// sort keys in place and fill with gather indices.

template<typename type_t>
void gpu_sort(std::vector<type_t>& data) {
  gl_buffer_t<type_t[]> a(data);

  mergesort_pipeline_t<type_t, empty_t> pipeline;
  pipeline.sort_keys(a, data.size());

  a.get_data(data.data());
}

int main() {
  int count = 128;
  std::vector<float> data(count);
  for(int i = 0; i < count; ++i)
    data[i] = rand() % 100;

  gpu_sort(data);
}