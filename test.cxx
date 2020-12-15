#include "kernel_merge.hxx"
#include "kernel_mergesort.hxx"

#include <cstdio>

#define GL_GLEXT_PROTOTYPES
#include <GL/gl3w.h>
#include <GLFW/glfw3.h>

using namespace mgpu;

template<typename type_t>
std::vector<int> gpu_merge_partition(const std::vector<type_t>& a, 
  const std::vector<type_t>& b) {

  merge_params_t<
    readonly_iterator_t<type_t, 0>,
    empty_iterator_t,

    readonly_iterator_t<type_t, 1>,
    empty_iterator_t,

    writeonly_iterator_t<type_t, 2>,
    empty_iterator_t,

    // Use default comparison.
    std::less<int>
  > params;

  const int nt = 128;
  const int vt = 7;
  params.spacing = nt * vt;
  params.a_count = a.size();
  params.b_count = b.size();

  int count = params.a_count + params.b_count;
  int num_partitions = num_merge_partitions(count, params.spacing);

  gl_buffer_t<type_t>           a_keys(a);
  gl_buffer_t<type_t>           b_keys(b);
  gl_buffer_t<int>              mp(num_partitions);
  gl_buffer_t<type_t>           c_keys(count);
  gl_buffer_t<decltype(params)> ubo(1, &params);
  
  // Select the SSBOs.
  a_keys.bind_ssbo(0);
  b_keys.bind_ssbo(1);
  c_keys.bind_ssbo(2);
  mp    .bind_ssbo(3);
  ubo   .bind_ubo(0);

  // Select the parameters.
  launch_partition<bounds_lower, decltype(params), 3, 0>(
    params.a_count + params.b_count, params.spacing);

  return mp.get_data();
}

template<int nt, int vt, typename type_t>
std::vector<type_t> gpu_merge(const std::vector<type_t>& a, 
  const std::vector<type_t>& b) {

  merge_params_t<
    readonly_iterator_t<type_t, 0>,
    empty_iterator_t,

    readonly_iterator_t<type_t, 1>,
    empty_iterator_t,

    writeonly_iterator_t<type_t, 2>,
    empty_iterator_t,

    // Use default comparison.
    std::less<int>
  > params;

  params.spacing = nt * vt;
  params.a_count = a.size();
  params.b_count = b.size();

  int count = params.a_count + params.b_count;
  int num_partitions = num_merge_partitions(count, params.spacing);

  gl_buffer_t<type_t>           a_keys(a);
  gl_buffer_t<type_t>           b_keys(b);
  gl_buffer_t<type_t>           c_keys(count);
  gl_buffer_t<int>              mp(num_partitions);
  gl_buffer_t<decltype(params)> ubo(1, &params);
  
  // Select the SSBOs.
  a_keys.bind_ssbo(0);
  b_keys.bind_ssbo(1);
  c_keys.bind_ssbo(2);
  mp    .bind_ssbo(3);
  ubo   .bind_ubo(0);

  // Select the parameters.
  launch_merge<nt, vt, decltype(params), 3, 0>(count);

  return c_keys.get_data();
}

struct app_t {
  app_t();

protected:
  virtual void debug_callback(GLenum source, GLenum type, GLuint id, 
    GLenum severity, GLsizei length, const GLchar* message);

  GLFWwindow* window = nullptr;

private:
  static void _debug_callback(GLenum source, GLenum type, GLuint id, 
    GLenum severity, GLsizei length, const GLchar* message, 
    const void* user_param);

};

app_t::app_t() {
  glfwWindowHint(GLFW_DOUBLEBUFFER, 1);
  glfwWindowHint(GLFW_DEPTH_BITS, 24);
  glfwWindowHint(GLFW_STENCIL_BITS, 8);
  glfwWindowHint(GLFW_SAMPLES, 4); // HQ 4x multisample.
  glfwWindowHint(GLFW_DECORATED, GLFW_TRUE);

  // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);

  int width = 320;
  int height = 200;
  window = glfwCreateWindow(width, height, "sort test", nullptr, nullptr);
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  glEnable(GL_DEBUG_OUTPUT);
  glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
  glDebugMessageCallback(_debug_callback, this);

  glfwGetWindowSize(window, &width, &height);
  glViewport(0, 0, width, height);
}

void app_t::debug_callback(GLenum source, GLenum type, GLuint id, 
  GLenum severity, GLsizei length, const GLchar* message) { 

  if(GL_DEBUG_SEVERITY_HIGH == severity ||
    GL_DEBUG_SEVERITY_MEDIUM == severity)
    printf("OpenGL: %s\n", message);

  if(GL_DEBUG_SEVERITY_HIGH == severity)
    exit(1);
}
void app_t::_debug_callback(GLenum source, GLenum type, GLuint id, 
  GLenum severity, GLsizei length, const GLchar* message, 
  const void* user_param) {

  app_t* app = (app_t*)user_param;
  app->debug_callback(source, type, id, severity, length, message);
}

int main() {
  glfwInit();
  gl3wInit();
  app_t app;

  int a_count = 10000;
  int b_count = 10000;
  std::vector<float> a(a_count), b(b_count);
  
  for(int i = 0; i < a_count; ++i)
    a[i] = 4 * i + a_count / 3;
  for(int i = 0; i < b_count; ++i)
    b[i] = 5 * i + 1;

  std::vector<float> c = gpu_merge<128, 7>(a, b);

  for(int i = 0; i < c.size(); ++i)
    printf("%d: %f\n", i, c[i]);
}