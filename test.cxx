#include "search.hxx"

using namespace mgpu;

struct search_params_t {
  readonly_iterator_t<float, 0> a;
  readonly_iterator_t<float, 1> b;
  writeonly_iterator_t<int, 2> mp_data;
  std::less<float> comp;

  int a_count;
  int b_count;
  int spacing;
};

int main() {
  merge_path_partitions_t<
    readonly_iterator_t<float, 0>,
    readonly_iterator_t<float, 1>,
    writeonly_iterator_t<int, 2>
  > params;

  params.a_count = 100;
  params.b_count = 200;
  params.spacing = 128 * 7;

  params.launch<bounds_lower>();
}