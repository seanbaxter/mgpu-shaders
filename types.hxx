#pragma once
#include "meta.hxx"
#include <array>
#include <functional>

BEGIN_MGPU_NAMESPACE

struct alignas(16) range_t {
  int begin, end;
  int size() const { return end - begin; }
  int count() const { return size(); }
  bool valid() const { return begin < end; }
};

inline range_t get_tile(int cta, int nv, int count) {
  return range_t { nv * cta, min(count, nv * (cta + 1)) };
}

struct alignas(16) merge_range_t {
  int a_begin, a_end, b_begin, b_end;

  int a_count() const { return a_end - a_begin; }
  int b_count() const { return b_end - b_begin; }
  int total()   const { return a_count() + b_count(); }

  range_t a_range() const { 
    return { a_begin, a_end };
  }
  range_t b_range() const {
    return { b_begin, b_end };
  }

  merge_range_t to_local() const {
    return { 0, a_count(), a_count(), total() };
  }
  
  // Partition from mp to the end.
  merge_range_t partition(int mp0, int diag) const {
    return { a_begin + mp0, a_end, b_begin + diag - mp0, b_end };
  }

  // Partition from mp0 to mp1.
  merge_range_t partition(int mp0, int diag0, int mp1, int diag1) const {
    return { 
      a_begin + mp0, 
      a_begin + mp1,
      b_begin + diag0 - mp0,
      b_begin + diag1 - mp1
    };
  }

  bool a_valid() const { 
    return a_begin < a_end; 
  }
  bool b_valid() const {
    return b_begin < b_end;
  }
};

template<typename type_t, int size>
struct merge_pair_t {
  std::array<type_t, size> keys;
  std::array<int, size> indices;
};

enum bounds_t { 
  bounds_lower,
  bounds_upper
};

END_MGPU_NAMESPACE
