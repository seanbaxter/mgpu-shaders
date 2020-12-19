#pragma once
#include "bindings.hxx"

#define GL_GLEXT_PROTOTYPES
#include <GL/gl3w.h>

BEGIN_MGPU_NAMESPACE

template<
  typename T, 
  bool is_array = std::is_array_v<T>, 
  bool is_const = std::is_const_v<T> 
>
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

  void swap(gl_buffer_t& rhs) noexcept {
    std::swap(buffer, rhs.buffer);
    std::swap(count, rhs.count);
  }

  void set_data(const type_t* data) noexcept {
    if(count) {
      assert(buffer);
      glNamedBufferSubData(buffer, 0, sizeof(type_t) * count, data);
    }
  }
  void set_data(const std::vector<type_t>& data) {
    resize(data.size());
    set_data(data.data());
  }
  void set_data_range(const type_t* data, int first, int count) {
    assert(first + count <= this->count);
    if(count) {
      assert(buffer);
      glNamedBufferSubData(buffer, sizeof(type_t) * first, 
        count * sizeof(type_t), data);
    }
  }

  void get_data(type_t* data) noexcept {
    if(count) {
      assert(buffer);
      glGetNamedBufferSubData(buffer, 0, sizeof(type_t) * count, data);
    }
  }

  void clear_bytes() {
    if(count && buffer) {
      char zero = 0;
      glClearNamedBufferData(buffer, GL_R8I, GL_RED_INTEGER, 
        GL_UNSIGNED_BYTE, &zero);
    }
  }

  void bind_ubo(GLuint index) {
    glBindBufferBase(GL_UNIFORM_BUFFER, index, buffer);
  }
  void bind_ssbo(GLuint index) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, index, buffer);
  }

  template<int bind>
  buffer_iterator_t<type_t, bind> bind_ssbo() {
    bind_ssbo(bind);
    return { };
  }

  std::vector<type_t> get_data() {
    std::vector<type_t> vec(count);
    get_data(vec.data());
    return vec;
  }

  void resize(int count2, bool preserve = false) {
    if(count != count2) {
      gl_buffer_t buffer2(count2);

      if(preserve && count && count2) {
        // Copy the old data into the new buffer.
        glCopyNamedBufferSubData(buffer, buffer2, 0, 0, 
          std::min(count, count2) * sizeof(type_t));
      }

      std::swap(buffer, buffer2.buffer);
      std::swap(count, buffer2.count); 
    }
  }

  GLuint buffer;
  int count;
};

template<typename type_t>
struct gl_buffer_t<type_t, false, false> {
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

// A const non-array type keeps a copy of the object on the CPU.
template<typename T>
struct gl_buffer_t<T, false, true> {
  typedef std::remove_const_t<T> type_t;

  gl_buffer_t() : buffer(0), invalid(true) {
    glCreateBuffers(1, &buffer);
    glNamedBufferStorage(buffer, sizeof(type_t), nullptr, 
      GL_DYNAMIC_STORAGE_BIT);
  }

  gl_buffer_t(const type_t& x) : data(x) {
    glCreateBuffers(1, &buffer);
    glNamedBufferStorage(buffer, sizeof(type_t), &data, 
      GL_DYNAMIC_STORAGE_BIT);
    invalid = false;
  }

  ~gl_buffer_t() {
    glDeleteBuffers(1, &buffer);
  }
  
  void set_data(const type_t& x) noexcept {
    if(!data || memcmp(&x, &data, sizeof(type_t))) {
      data.emplace(x);
      invalid = true;
    }
  }

  void update() {
    if(invalid) {
      assert(data);
      glNamedBufferSubData(buffer, 0, sizeof(type_t), &*data);
      invalid = false;
    }
  }

  void bind_ubo(GLuint index) {
    update();
    glBindBufferBase(GL_UNIFORM_BUFFER, index, buffer);
  }

  void bind_ubo_range(GLuint index, size_t offset, size_t size) {
    update();
    glBindBufferRange(GL_UNIFORM_BUFFER, index, buffer, offset, size);
  }

  GLuint buffer;
  bool invalid;
  std::optional<type_t> data;
};

END_MGPU_NAMESPACE
