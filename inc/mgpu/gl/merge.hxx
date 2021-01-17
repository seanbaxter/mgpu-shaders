#pragma once
#include "../common/kernel_merge.hxx"
#include "../common/bindings.hxx"
#include "partition.hxx"

BEGIN_MGPU_NAMESPACE

namespace gl {

template<int nt, int vt, typename params_t, int mp, int ubo>
[[using spirv: comp, local_size(nt)]]
void kernel_merge() {
  params_t params = shader_uniform<ubo, params_t>;

  kernel_merge<nt, vt>(
    readonly_iterator_t<int, mp>(),
    
    params.a_keys,
    params.a_vals,
    params.a_count,

    params.b_keys,
    params.b_vals,
    params.b_count,

    params.c_keys,
    params.c_vals,

    params.comp
  );
}

template<
  typename a_keys_it,
  typename a_values_it,
  typename b_keys_it,
  typename b_values_it,
  typename c_keys_it,
  typename c_values_it,
  typename comp_t>
struct merge_params_t {
  a_keys_it a_keys;
  b_keys_it b_keys;
  c_keys_it c_keys;

  int spacing;           // NV * VT
  int a_count;
  int b_count;

  // Put the potentially empty objects together to take up less space.
  a_values_it a_vals;
  b_values_it b_vals;
  c_values_it c_vals;
  comp_t comp;
};

template<int nt, int vt, typename params_t, int mp, int ubo = 0>
void launch_merge(int count) {
  // First launch the partition kernel.
  launch_partition<bounds_lower, params_t, mp, ubo>(count, nt * vt);

  // Launch the CTA merge kernel.
  int num_ctas = div_up(count, nt * vt);
  gl_dispatch_kernel<kernel_merge<nt, vt, params_t, mp, ubo> >(num_ctas);
}

// merge_pipeline_t is a convenient entry point for using the merge 
// kernel. It loads data from SSBOs and writes to an SSBO. Storage for 
// the parameters UBO and merge paths SSBO is handled automatically.
template<typename key_t, typename val_t = empty_t, 
  typename comp_t = std::less<key_t> >
struct merge_pipeline_t {
  void reserve(int count, int spacing) {
    int num_partitions = num_merge_partitions(count, spacing);
    if(num_partitions > partitions_ssbo.count)
      partitions_ssbo.resize(num_partitions);
  }

  template<int nt = 128, int vt = 7>
  void launch(GLuint a_keys, int a_count, GLuint b_keys, int b_count, 
    GLuint c_keys, comp_t comp = comp_t()) {

    static_assert(std::is_same_v<empty_t, val_t>);

    // Bind the merge path SSBO.
    reserve(a_count + b_count, nt * vt);
    partitions_ssbo.bind_ssbo(3);

    params_t params { };
    params.spacing = nt * vt;
    params.a_count = a_count;
    params.b_count = b_count;
    params.comp = comp;

    // Upload and bind the UBO.
    params_ubo.set_data(params);
    params_ubo.bind_ubo(0);

    // Bind the data.
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, a_keys);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, b_keys);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, c_keys);

    launch_merge<nt, vt, params_t, 3, 0>(a_count + b_count);
  }

  template<int nt = 256, int vt = 7>
  void launch(GLuint a_keys, GLuint a_vals, int a_count, GLuint b_keys,
    GLuint b_vals, int b_count, GLuint c_keys, GLuint c_vals, 
    comp_t comp = comp_t()) {

    static_assert(!std::is_same_v<empty_t, val_t>);

    // Bind the merge path SSBO.
    reserve(a_count + b_count, nt * vt);
    partitions_ssbo.bind_ssbo(3);

    params_t params { };
    params.spacing = nt * vt;
    params.a_count = a_count;
    params.b_count = b_count;
    params.comp = comp;

    // Upload and bind the UBO.
    params_ubo.set_data(params);
    params_ubo.bind_ubo(0);

    // Bind the data.
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, a_keys);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, b_keys);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, c_keys);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, a_vals);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, b_vals);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, c_vals);

    launch_merge<nt, vt, params_t, 3, 0>(a_count + b_count);
  }

  typedef merge_params_t<
    // A
    readonly_iterator_t<key_t, 0>,
    readonly_iterator_t<val_t, 4>,

    // B
    readonly_iterator_t<key_t, 1>,
    readonly_iterator_t<val_t, 5>,

    // C
    writeonly_iterator_t<key_t, 2>,
    writeonly_iterator_t<val_t, 6>,

    comp_t
  > params_t;

  gl_buffer_t<int[]>    partitions_ssbo;
  gl_buffer_t<const params_t> params_ubo;
};

} // namespace gl

END_MGPU_NAMESPACE