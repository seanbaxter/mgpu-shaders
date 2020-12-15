#pragma once
#include "types.hxx"

BEGIN_MGPU_NAMESPACE

////////////////////////////////////////////////////////////////////////////////
// Odd-even transposition sorting network. Sorts keys and values in-place in
// register.
// http://en.wikipedia.org/wiki/Odd%E2%80%93even_sort

template<typename type_t, int vt, typename comp_t>
std::array<type_t, vt> odd_even_sort(std::array<type_t, vt> x, comp_t comp, 
  int flags = 0) { 

  @meta for(int I = 0; I < vt; ++I) {
    @meta for(int i = 1 & I; i < vt - 1; i += 2) {
      if((0 == ((2<< i) & flags)) && comp(x[i + 1], x[i]))
        std::swap(x[i], x[i + 1]);
    }
  }
  return x;
}

template<typename key_t, typename val_t, int vt, typename comp_t>
kv_array_t<key_t, val_t, vt> odd_even_sort(kv_array_t<key_t, val_t, vt> x, 
  comp_t comp, int flags = 0) { 

  @meta for(int I = 0; I < vt; ++I) {
    @meta for(int i = 1 & I; i < vt - 1; i += 2) {
      if((0 == ((2<< i) & flags)) && comp(x.keys[i + 1], x.keys[i])) {
        std::swap(x.keys[i], x.keys[i + 1]);
        std::swap(x.vals[i], x.vals[i + 1]);
      }
    }
  }
  return x;
}

END_MGPU_NAMESPACE
