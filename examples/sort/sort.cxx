#include <mgpu/kernel_mergesort.hxx>
#include <mgpu/app.hxx>

using namespace mgpu;
// key-index sort.
// sort keys in place and fill with gather indices.

template<int nt, int vt, typename type_t>
void gpu_sort(std::vector<type_t>& data) {
  gl_buffer_t<type_t[]> a(data);

  mergesort_pipeline_t<type_t, empty_t> pipeline;
  pipeline.template sort_keys<nt, vt>(a, data.size());

  a.get_data(data.data());
}

int main() {
  app_t app("sort demo");

  const int nt = 256;
  const int vt = 11;
  int count = nt * vt * 512;
  std::vector<float> data(count);
  for(int i = 0; i < count; ++i)
    data[i] = rand() % 10000;

  gpu_sort<nt, vt>(data);
  printf("%d: %f\n", @range(), data[:])...;

  bool is_sorted = (... && (data[:] <= data[1:]));
  printf("IS SORTED = %d\n", is_sorted);
}