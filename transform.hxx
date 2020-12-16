#pragma once
#include <gl3w/GL/gl3w.h>
#include "bindings.hxx"

BEGIN_MGPU_NAMESPACE

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

  if(x && y && z) {
    glUseProgram(program);
    glDispatchCompute(x, y, z); 
  }
}

template<int ubo, int nt, typename data_t>
[[using spirv: comp, local_size(nt)]]
void kernel_transform() {
  data_t data = shader_uniform<ubo, data_t>;

  int gid = threadIdx.x + nt * blockIdx.x;
  if(gid < data.count)
    data.func(gid);
}

template<int ubo = 0, int nt = 128, typename func_t>
void gl_transform(func_t func, int count) {
  static_assert(std::is_copy_constructible_v<func_t>);
  
  struct data_t {
    func_t func;
    int count;
  };

  // Keep a cache for the UBO. Only calls glNamedBufferSubData if 
  // its contents are different from the last bind operation.
  static gl_buffer_t<const data_t> buffer;
  buffer.set_data({ func, count });
  buffer.bind_ubo(ubo);

  int num_ctas = div_up(count, nt);
  gl_dispatch_kernel<kernel_transform<ubo, nt, data_t> >(num_ctas);
}

END_MGPU_NAMESPACE