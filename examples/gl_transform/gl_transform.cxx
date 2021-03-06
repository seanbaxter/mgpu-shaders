#include <mgpu/gl/transform.hxx>
#include <mgpu/gl/app.hxx>
#include <cmath>
#include <cstdio>

using namespace mgpu::gl;

int main() {
  // Initialize OpenGL and create an offscreen window.
  app_t app("lambda test");

  // Allocate storage for 10000 floats.
  int count = 10000;
  gl_buffer_t<float[]> data(count);

  // Bind to binding=0. Return a buffer_iterator_t that samples 
  // shader_buffer<0, float[]>.
  auto p = data.bind_ssbo<0>();

  // Launch a compute shader from a lambda.
  gl_transform([=](int index) {
    p[index] = sqrt((float)index);
  }, count);

  std::vector<float> data2 = data.get_data();
  printf("%5d: %f\n", @range(), data2[:])...;
}