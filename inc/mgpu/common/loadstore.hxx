// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once
#include "types.hxx"

BEGIN_MGPU_NAMESPACE

////////////////////////////////////////////////////////////////////////////////
// reg<->shared

template<int nt, int vt, typename type_t, int shared_size>
void reg_to_shared_thread(std::array<type_t, vt> x, int tid,
  type_t (&shared)[shared_size], bool sync = true) {

  static_assert(shared_size >= nt * vt,
    "reg_to_shared_thread must have at least nt * vt storage");

  // thread_iterate<vt>([&](int i, int j) { 
  //   shared[j] = x[i]; 
  // }, tid);

  @meta for(int i = 0; i < vt; ++i)
    shared[vt * tid + i] = x[i];

  if(sync) __syncthreads();
}

template<int nt, int vt, typename type_t, int shared_size>
std::array<type_t, vt> shared_to_reg_thread(
  const type_t (&shared)[shared_size], int tid, bool sync = true) {

  static_assert(shared_size >= nt * vt,
    "reg_to_shared_thread must have at least nt * vt storage");

  std::array<type_t, vt> x;
  thread_iterate<vt>([&](int i, int j) { 
    x[i] = shared[j];
  }, tid);
  if(sync) __syncthreads();
  return x;
}

////////////////////////////////////////////////////////////////////////////////

template<int nt, int vt, typename type_t, int shared_size>
void reg_to_shared_strided(std::array<type_t, vt> x, int tid,
  type_t (&shared)[shared_size], bool sync = true) {

  static_assert(shared_size >= nt * vt,
    "reg_to_shared_strided must have at least nt * vt storage");

  // strided_iterate<nt, vt>([&](int i, int j) { shared[j] = x[i]; }, tid);

  @meta for(int i = 0; i < vt; ++i)
    shared[nt * i + tid] = x[i];

  if(sync) __syncthreads();
}

template<int nt, int vt, typename type_t, int shared_size>
std::array<type_t, vt> shared_to_reg_strided(
  const type_t (&shared)[shared_size], int tid, bool sync = true) {

  static_assert(shared_size >= nt * vt,
    "shared_to_reg_strided must have at least nt * vt storage");

  std::array<type_t, vt> x;
  strided_iterate<nt, vt>([&](int i, int j) { x[i] = shared[j]; }, tid);
  if(sync) __syncthreads();
  return x;
}

////////////////////////////////////////////////////////////////////////////////

template<int nt, int vt, typename type_t, int shared_size>
std::array<type_t, vt> shared_to_reg_warp(const type_t (&shared)[shared_size],
  int lane, int warp, int warp_size, bool sync = true) {

  uint cur = vt * warp_size * warp + lane;

  std::array<type_t, vt> x;
  @meta for(int i = 0; i < vt; ++i)
    x[i] = shared[cur + i * warp_size];
  if(sync) __syncthreads();
  return x;
}

////////////////////////////////////////////////////////////////////////////////

template<int nt, int vt, typename type_t, int shared_size>
std::array<type_t, vt> shared_gather(const type_t(&data)[shared_size],
  std::array<int, vt> indices, bool sync = true) {

  static_assert(shared_size >= nt * vt,
    "shared_gather must have at least nt * vt storage");

  std::array<type_t, vt> x { data[indices...[:]]... };
  if(sync) __syncthreads();
  return x;
}

template<int nt, int vt, typename type_t, int shared_size>
std::array<type_t, vt> thread_to_strided(std::array<type_t, vt> x, 
  int tid, type_t (&shared)[shared_size]) {

  reg_to_shared_thread<nt, vt>(x, tid, shared);
  return shared_to_reg_strided<nt, vt>(shared, tid);
}



////////////////////////////////////////////////////////////////////////////////
// reg<->memory

template<int nt, int vt, int vt0 = vt, typename type_t, typename it_t>
void reg_to_mem_strided(std::array<type_t, vt> x, int tid, 
  int count, it_t mem) {

  // strided_iterate<nt, vt, vt0>([=](int i, int j) { 
  //   mem[j] = x[i]; 
  // }, tid, count);
  @meta for(int i = 0; i < vt; ++i) {{
    int k = nt * i + tid;
    if(k < count)
      mem[k] = x[i];
  }}
}

template<int nt, int vt, int vt0 = vt, typename it_t>
std::array<typename std::iterator_traits<it_t>::value_type, vt> 
mem_to_reg_strided(it_t mem, int tid, int count) {
  typedef typename std::iterator_traits<it_t>::value_type type_t;
  std::array<type_t, vt> x;
  
  // strided_iterate<nt, vt, vt0>([&](int i, int j) { 
  //  x[i] = mem[j]; 
  //  }, tid, count);

  @meta for(int i = 0; i < vt; ++i) {{
    int k = nt * i + tid;
    if(k < count)
      x[i] = mem[k];
  }}

  return x;
}

template<int nt, int vt, int vt0 = vt, typename type_t, typename it_t, 
  int shared_size>
void reg_to_mem_thread(std::array<type_t, vt> x, int tid,
  int count, it_t mem, type_t (&shared)[shared_size]) {

  reg_to_shared_thread<nt>(x, tid, shared);
  std::array<type_t, vt> y = shared_to_reg_strided<nt, vt>(shared, tid);
  reg_to_mem_strided<nt, vt, vt0>(y, tid, count, mem);
}

template<int nt, int vt, int vt0 = vt, typename type_t, typename it_t, 
  int shared_size>
std::array<type_t, vt> mem_to_reg_thread(it_t mem, int tid,
  int count, type_t (&shared)[shared_size]) {

  std::array<type_t, vt> x = mem_to_reg_strided<nt, vt, vt0>(mem, tid, count);
  reg_to_shared_strided<nt, vt>(x, tid, shared);
  std::array<type_t, vt> y = shared_to_reg_thread<nt, vt>(shared, tid);
  return y;
}

template<int nt, int vt, int vt0 = vt, typename input_it, typename output_it>
void mem_to_mem(input_it input, int tid, int count,
  output_it output) {
  typedef typename std::iterator_traits<input_it>::value_type type_t;
  type_t x[vt];

  strided_iterate<nt, vt, vt0>([&](int i, int j) {
    x[i] = input[j];
  }, tid, count);
  strided_iterate<nt, vt, vt0>([&](int i, int j) {
    output[j] = x[i];
  }, tid, count);
}

////////////////////////////////////////////////////////////////////////////////
// memory<->memory

template<int nt, int vt, int vt0 = vt, typename type_t, typename it_t>
void mem_to_shared(it_t mem, int tid, int count, type_t* shared, 
  bool sync = true) {

  std::array<type_t, vt> x = mem_to_reg_strided<nt, vt, vt0>(mem, tid, count);
  strided_iterate<nt, vt, vt0>([&](int i, int j) {
    shared[j] = x[i];
  }, tid, count);
  if(sync) __syncthreads();
}

template<int nt, int vt, typename type_t, typename it_t>
void shared_to_mem(const type_t* shared, int tid, int count,
  it_t mem, bool sync = true) {

  strided_iterate<nt, vt>([&](int i, int j) { 
    mem[j] = shared[j]; 
  }, tid, count);
  if(sync) __syncthreads();
}

////////////////////////////////////////////////////////////////////////////////
// reg<->reg

template<int nt, int vt, typename type_t, int shared_size>
std::array<type_t, vt> reg_thread_to_strided(std::array<type_t, vt> x,
  int tid, type_t (&shared)[shared_size]) {

  reg_to_shared_thread<nt>(x, tid, shared);
  return shared_to_reg_strided<nt, vt>(shared, tid);
}

template<int nt, int vt, typename type_t, int shared_size>
std::array<type_t, vt> reg_strided_to_thread(std::array<type_t, vt> x,
  int tid, type_t (&shared)[shared_size]) {

  reg_to_shared_strided<nt>(x, tid, shared);
  return shared_to_reg_thread<nt, vt>(shared, tid);
}

END_MGPU_NAMESPACE
