#include <mgpu/kernel_mergesort.hxx>
#include <mgpu/app.hxx>

using namespace mgpu;
// key-index sort.
// sort keys in place and fill with gather indices.

template<int nt, int vt, typename type_t>
void gpu_sort(std::vector<type_t>& data) {
  gl_buffer_t<type_t[]> a(data);
  gl_buffer_t<int[]> b(data.size());

  mergesort_pipeline_t<type_t, int> pipeline;
  pipeline.template sort_keys_indices<nt, vt>(a, b, data.size());

  std::vector<int> indices = b.get_data();
  std::vector<float> gathered = [data[indices[:]]...];

  a.get_data(data.data());

  bool is_inverse = gathered == data;
  printf("is_inverse = %d\n", is_inverse);

}

int main() {
  app_t app("sort demo");

  const int nt = 128;
  const int vt = 7;
  int count = 10000;
  std::vector<float> data(count);
  for(int i = 0; i < count; ++i)
    data[i] = rand() % 10000;
  
  gpu_sort<nt, vt>(data);
  

  bool is_sorted = (... && (data[:] <= data[1:]));
  printf("IS SORTED = %d\n", is_sorted);
}