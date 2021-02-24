#pragma once
#include "meta.hxx"

BEGIN_MGPU_NAMESPACE

template<typename type_t>
type_t subgroupShuffle(type_t x, uint id) {
  if constexpr(std::is_array_v<type_t> ||
    requires { typename std::tuple_size<type_t>::type; }) {

    // Shuffle elemnents of arrays and tuples.
    x...[:] = subgroupShuffle(x...[:], id)...;

  } else if constexpr(std::is_class_v<type_t>) {
    // Shuffle all public base classes and data members of class objects.
    x...[:] = subgroupShuffle(x.@base_values(), id)...;
    x...[:] = subgroupShuffle(x.@member_values(), id)...;

  } else {
    // Plain shuffle scalars.
    x = gl_subgroupShuffle(x, id);
  }

  return x;
}

template<typename type_t>
type_t subgroupShuffleDown(type_t x, uint delta) {
  if constexpr(std::is_array_v<type_t> ||
    requires { typename std::tuple_size<type_t>::type; }) {
    x...[:] = subgroupShuffleDown(x...[:], delta)...;

  } else if constexpr(std::is_class_v<type_t>) {
    x...[:] = subgroupShuffleDown(x.@base_values(), delta)...;
    x...[:] = subgroupShuffleDown(x.@member_values(), delta)...;

  } else {
    x = gl_subgroupShuffleDown(x, delta);
  }
  return x;
}

template<typename type_t>
type_t subgroupShuffleUp(type_t x, uint delta) {
  if constexpr(std::is_array_v<type_t> ||
    requires { typename std::tuple_size<type_t>::type; }) {
    x...[:] = subgroupShuffleUp(x...[:], delta)...;

  } else if constexpr(std::is_class_v<type_t>) {
    x...[:] = subgroupShuffleUp(x.@base_values(), delta)...;
    x...[:] = subgroupShuffleUp(x.@member_values(), delta)...;

  } else {
    x = gl_subgroupShuffleUp(x, delta);
  }
  return x;
}

END_MGPU_NAMESPACE
