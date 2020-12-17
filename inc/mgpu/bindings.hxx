#pragma once
#include "meta.hxx"
#include <vector>
#include <optional>

BEGIN_MGPU_NAMESPACE

template<auto index, typename type_t = @enum_type(index)>
[[using spirv: uniform, binding((int)index)]]
type_t shader_uniform;

template<auto index, typename type_t = @enum_type(index)>
[[using spirv: buffer, readonly, binding(index)]]
type_t shader_readonly;

template<auto index, typename type_t = @enum_type(index)>
[[using spirv: buffer, writeonly, binding(index)]]
type_t shader_writeonly;

template<auto index, typename type_t = @enum_type(index)>
[[using spirv: buffer, binding(index)]]
type_t shader_buffer;

////////////////////////////////////////////////////////////////////////////////

// Provide an a common iterator type.
template<typename accessor_t, typename type_t = decltype(accessor_t::access(0))>
struct iterator_t : std::iterator_traits<const std::remove_reference_t<type_t>*> {

  iterator_t() = default;
  explicit iterator_t(int offset) : offset(offset) { }

  iterator_t(const iterator_t&) = default;
  iterator_t& operator=(const iterator_t&) = default;

  iterator_t operator+(int diff) const noexcept {
    return iterator_t(offset + diff);
  }
  iterator_t& operator+=(int diff) noexcept {
    offset += diff;
    return *this;
  }
  friend iterator_t operator+(int diff, iterator_t rhs) noexcept {
    return iterator_t(diff + rhs.offset);
  }

  iterator_t operator-(int diff) const noexcept {
    return iterator_t(offset - diff);
  }
  iterator_t& operator-=(int diff) noexcept {
    offset -= diff;
    return *this;
  }

  int operator-(iterator_t rhs) const noexcept {
    return offset - rhs.offset;
  }

  decltype(auto) operator*() const noexcept {
    return accessor_t::access(offset);
  }

  decltype(auto) operator[](int index) const noexcept {
    return accessor_t::access(offset + index);
  }

  int offset = 0;
};

template<typename type_t, int binding>
struct readonly_access_t {
  static type_t access(int index) noexcept {
    return shader_readonly<binding, type_t[]>[index];
  }
};

template<typename type_t, int binding>
using readonly_iterator_t = iterator_t<readonly_access_t<type_t, binding> >;

template<typename type_t, int binding>
struct writeonly_access_t {
  static type_t& access(int index) noexcept {
    return shader_writeonly<binding, type_t[]>[index];
  }
};
template<typename type_t, int binding>
using writeonly_iterator_t = iterator_t<writeonly_access_t<type_t, binding> >;

template<typename type_t, int binding>
struct buffer_access_t {
  static type_t& access(int index) noexcept {
    return shader_buffer<binding, type_t[]>[index];
  }
};
template<typename type_t, int binding>
using buffer_iterator_t = iterator_t<buffer_access_t<type_t, binding> >;

struct empty_iterator_t : std::iterator_traits<const empty_t*> {
  // Don't provide additional interface. The caller should check the 
  // iterator_traits prior to subscripting.
};

////////////////////////////////////////////////////////////////////////////////

END_MGPU_NAMESPACE

