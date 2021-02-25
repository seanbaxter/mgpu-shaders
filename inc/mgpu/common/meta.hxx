#pragma once
#include <cstdint>
#include <cassert>
#include <cstring>
#include <type_traits>
#include <functional>

#define BEGIN_MGPU_NAMESPACE namespace mgpu {
#define END_MGPU_NAMESPACE }

BEGIN_MGPU_NAMESPACE

struct empty_t { };

template<typename... types_t>
struct tuple_t {
  types_t @(int...) ...;
};

constexpr int div_up(int x, int y) {
  return (x + y - 1) / y;
}
constexpr int64_t div_up(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}
constexpr size_t div_up(size_t x, size_t y) {
  return (x + y - 1) / y;
}

template<typename type_t>
constexpr bool is_pow2(type_t x) {
  static_assert(std::is_integral_v<type_t>);
  return 0 == (x & (x - 1));
}

// Find log2(x) and optionally round up to the next integer logarithm.
inline int find_log2(int x, bool round_up = false) {
  int a = 31 - __builtin_clz(x);
  if(round_up) a += !is_pow2(x);
  return a;
} 

constexpr int s_log2(int x) {
  int i = 0;
  while(x) {
    x>>= 1;
    ++i;
  }
  return i;
}

template<int count, typename func_t>
void iterate(func_t f) {
  @meta for(int i = 0; i < count; ++i)
    f(i);
}

// Invoke unconditionally.
template<int nt, int vt, typename func_t>
void strided_iterate(func_t f, int tid) {
  @meta for(int i = 0; i < vt; ++i)
    f(i, nt * i + tid);
}

// Check range.
template<int nt, int vt, int vt0 = vt, typename func_t>
void strided_iterate(func_t f, int tid, int count) {
  // Unroll the first vt0 elements of each thread.
  if constexpr(vt0) {
    if(vt0 > 1 && count >= nt * vt0) {
      strided_iterate<nt, vt0>(f, tid);    // No checking
    
    } else {
      @meta for(int i = 0; i < vt0; ++i) {{
        int index = nt * i + tid;
        if(index < count) f(i, index);
      }}
    }
  }
  
  @meta for(int i = vt0; i < vt; ++i) {{
    int index = nt * i + tid;
    if(index < count) f(i, index);
  }}
}

template<int vt, typename func_t>
void thread_iterate(func_t f, int tid) {
  @meta for(int i = 0; i < vt; ++i)
    f(i, vt * tid + i);
}

template<int size> 
struct integer_by_size_t;

template<> struct integer_by_size_t<1> {
  typedef  int8_t signed_type;
  typedef uint8_t unsigned_type;
};
template<> struct integer_by_size_t<2> {
  typedef  int16_t signed_type;
  typedef uint16_t unsigned_type;
};
template<> struct integer_by_size_t<4> {
  typedef  int32_t signed_type;
  typedef uint32_t unsigned_type;
};
template<> struct integer_by_size_t<8> {
  typedef  int64_t signed_type;
  typedef uint64_t unsigned_type;
};

template<int size> 
using signed_int_by_size_t = typename integer_by_size_t<size>::signed_type;

template<int size> 
using unsigned_int_by_size_t = typename integer_by_size_t<size>::unsigned_type;

END_MGPU_NAMESPACE
