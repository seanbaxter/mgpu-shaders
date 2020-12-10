#pragma once
#include <gl3w/GL/gl3w.h>

struct shader_t {
  GLuint shader;
};

template<auto kernel>
void gl_dispatch_kernel(int x, int y = 1, int z = 1) {
  static GLuint program = 0;
  if(!program) {
    GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
    glShaderBinary(1, &cs, GL_SHADER_BINARY_FORMAT_SPIR_V_ARB,
      __spirv_data, __spirv_size);
    glSpecializeShader(cs, @spirv(kernel), 0, nullptr, nullptr);

    program = glCreateProgram();
    glAttachShader(program, cs);
    glLinkProgram(program);
  }

  glUseProgram(program);
  glDispatchCompute(x, y, z);
}

