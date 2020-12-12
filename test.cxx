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
  search_params_t params;
  params.a_count = 100;
  params.b_count = 200;
  params.spacing = 128 * 7;

  // Specify the type of the search parameters and the UBO binding at which
  // to find them.
  mgpu::merge_path_partitions<mgpu::bounds_lower, search_params_t, 0>(
    params.a_count,
    params.b_count,
    params.spacing
  );
}