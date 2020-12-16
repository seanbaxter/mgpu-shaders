#pragma once
#include "meta.hxx"
#include <vector>

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

template<typename T, bool is_array = std::is_array_v<T> >
struct gl_buffer_t {
  typedef std::remove_extent_t<T> type_t;

  gl_buffer_t() : buffer(0), count(0) { }

  gl_buffer_t(int count, const type_t* data = nullptr) noexcept : count(count) {
    glCreateBuffers(1, &buffer);
    glNamedBufferStorage(buffer, sizeof(type_t) * count, data, 
      GL_DYNAMIC_STORAGE_BIT);
  }
  gl_buffer_t(const std::vector<type_t>& data) noexcept :
    gl_buffer_t(data.size(), data.data()) { }

  ~gl_buffer_t() {
    if(buffer)
      glDeleteBuffers(1, &buffer);
  }

  gl_buffer_t(const gl_buffer_t&) = delete;
  gl_buffer_t& operator=(const gl_buffer_t) = delete;

  operator GLuint() noexcept { return buffer; }

  void set_data(const type_t* data) noexcept {
    if(count) {
      assert(buffer);
      glNamedBufferSubData(buffer, 0, sizeof(type_t) * count, data);
    }
  }
  void get_data(type_t* data) noexcept {
    if(count) {
      assert(buffer);
      glGetNamedBufferSubData(buffer, 0, sizeof(type_t) * count, data);
    }
  }

  void bind_ubo(GLuint index) {
    glBindBufferBase(GL_UNIFORM_BUFFER, index, buffer);
  }
  void bind_ssbo(GLuint index) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, index, buffer);
  }

  std::vector<type_t> get_data() {
    std::vector<type_t> vec(count);
    get_data(vec.data());
    return vec;
  }

  void resize(int count2) {
    gl_buffer_t buffer2(count2);
    std::swap(buffer, buffer2.buffer);
    std::swap(count, buffer2.count);
  }

  GLuint buffer;
  int count;
};

template<typename type_t>
struct gl_buffer_t<type_t, false> {
  gl_buffer_t(const type_t* data = nullptr) noexcept {
    glCreateBuffers(1, &buffer);
    glNamedBufferStorage(buffer, sizeof(type_t), data, 
      GL_DYNAMIC_STORAGE_BIT);
  }

  ~gl_buffer_t() {
    glDeleteBuffers(1, &buffer);
  }

  gl_buffer_t(const gl_buffer_t&) = delete;
  gl_buffer_t& operator=(const gl_buffer_t) = delete;

  operator GLuint() noexcept { return buffer; }

  void set_data(const type_t& data) noexcept {
    assert(buffer);
    glNamedBufferSubData(buffer, 0, sizeof(type_t), &data);
  }
  void get_data(type_t* data) noexcept {
    assert(buffer);
    glGetNamedBufferSubData(buffer, 0, sizeof(type_t), data);
  }
  type_t get_data() noexcept {
    type_t x;
    get_data(&x);
    return x;
  }

  void bind_ubo(GLuint index) {
    glBindBufferBase(GL_UNIFORM_BUFFER, index, buffer);
  }
  void bind_ssbo(GLuint index) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, index, buffer);
  }

  GLuint buffer;
};


END_MGPU_NAMESPACE

