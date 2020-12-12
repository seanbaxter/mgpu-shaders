#include "kernel_merge.hxx"
#include <cstdio>

#define GL_GLEXT_PROTOTYPES
#include <GL/gl3w.h>
#include <GLFW/glfw3.h>

using namespace mgpu;

template<typename type_t>
std::vector<type_t> gpu_merge(std::vector<type_t>& a, 
  std::vector<type_t>& b) {

  gl_buffer_t<type_t> a_keys(a);
  gl_buffer_t<type_t> b_keys(b);
  gl_buffer_t<type_t> c_keys(a.size() + b.size());

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

  params.a_count = a.size();
  params.b_count = b.size();

  const int nt = 128;
  const int vt = 7;
  params.spacing = nt * vt;

  launch_merge<nt, vt, decltype(params), 3>(a.size() + b.size());

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

  std::vector<float> a(100), b(100);
  for(int i = 0; i < 100; ++i)
    a[i] = 2 * i, b[i] = 2 * i + 1;

  std::vector<float> c = gpu_merge(a, b);

  printf("%f\n", c[:])...;
}