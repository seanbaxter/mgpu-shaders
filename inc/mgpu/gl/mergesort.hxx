#pragma once
#include "../common/kernel_mergesort.hxx"
#include "transform.hxx"

BEGIN_MGPU_NAMESPACE

namespace gl {

template<bool sort_indices, int nt, int vt, typename params_t, int ubo>
[[using spirv: comp, local_size(nt)]]
void kernel_blocksort() {
  params_t params = shader_uniform<ubo, params_t>;

  kernel_blocksort<sort_indices, nt, vt>(
    params.keys_block,
    params.vals_block,
    params.keys_out,
    params.vals_out,
    params.count,
    params.comp
  );

  // Zero out the pass identifiers at the end of the mp data. The partition
  // and mergesort pass kernels use these terms to know which pass they're
  // working on.
  //if(params.num_partitions + threadIdx.x + blockIdx.x)
    params.mp_data[params.num_partitions] = 0;
}

template<typename mp_it, typename keys_it, typename comp_t>
void kernel_mergesort_partition(mp_it mp_data, keys_it keys, int count, 
  int num_partitions, int spacing, int coop, comp_t comp) {

  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index < num_partitions) {
    merge_range_t range = compute_mergesort_range(count, index, coop, spacing);
    int diag = min(spacing * index, count) - range.a_begin;
    mp_data[index] = merge_path<bounds_lower>(keys + range.a_begin, 
      range.a_count(), keys + range.b_begin, range.b_count(), diag, comp);
  }
}

template<typename params_t, int ubo = 0>
[[using spirv: comp, local_size(128)]]
void kernel_mergesort_partition() {
  params_t params = shader_uniform<ubo, params_t>;

  // Load the pass.
  int pass = params.mp_data[params.num_partitions];

  // The first thread should increment the pass.
  int first_thread = !threadIdx.x && !blockIdx.x;
  if(first_thread)
    params.mp_data[params.num_partitions + 1] = pass;

  int coop = 2<< pass;
  kernel_mergesort_partition(
    params.mp_data,
    params.keys_in,
    params.count,
    params.num_partitions,
    params.spacing,
    coop,
    params.comp
  );
}

template<int nt, int vt, typename params_t, int ubo>
[[using spirv: comp, local_size(nt)]]
void kernel_mergesort_pass() {
  params_t params = shader_uniform<ubo, params_t>;

  // Load the pass.
  int pass = params.mp_data[params.num_partitions + 1];

  // The first thread should increment the pass.
  int first_thread = !threadIdx.x && !blockIdx.x;
  if(first_thread)
    params.mp_data[params.num_partitions] = pass + 1;

  int coop = 2<< pass;
  kernel_mergesort_pass<nt, vt>(
    params.mp_data,
    params.keys_in,
    params.vals_in,
    params.keys_out,
    params.vals_out,
    params.count,
    coop,
    params.comp
  );
}

template<
  int mp,
  typename keys_block_it,
  typename vals_block_it,
  typename keys_in_it,
  typename vals_in_it,
  typename keys_out_it,
  typename vals_out_it,
  typename comp_t
> struct mergesort_params_t {
  buffer_iterator_t<int, mp> mp_data;

  // Inputs to the blocksort.
  keys_block_it keys_block;
  vals_block_it vals_block;

  // Inputs to the partition and merge passes.
  keys_in_it keys_in;
  vals_in_it vals_in;
 
  // Outputs for blocksort and merge passes.
  keys_out_it keys_out;
  vals_out_it vals_out;

  int count;
  int num_partitions;
  int spacing;
  comp_t comp;
};

template<typename key_t, typename val_t = empty_t, 
  typename comp_t = std::less<key_t>>
struct mergesort_pipeline_t {
  enum { has_values = !std::is_same_v<val_t, empty_t> };

  struct info_t {
    int num_passes;
    int num_ctas;
    int num_partitions;
    int num_partition_ctas;
  };

  info_t reserve(int count, int nv) {
    int num_ctas = div_up(count, nv);
    int num_passes = find_log2(num_ctas, true);
    int num_partitions = num_ctas > 1 ? num_ctas + 1 : 0;

    if(num_passes) {
      // Reserve two extra slots for the pass.
      partitions_ssbo.resize(num_partitions + 2);
      keys_ssbo.resize(count);
      if(has_values)
        vals_ssbo.resize(count);
    }

    int num_partition_ctas = div_up(num_partitions, 128);
    return { num_passes, num_ctas, num_partitions, num_partition_ctas };
  }

  template<int nt = 128, int vt = 7>
  void sort_keys(GLuint keys, int count, comp_t comp = comp_t()) {
    static_assert(!has_values);
    const int nv = nt * vt;

    if(!count) return;

    params_t params { };
    info_t info = reserve(count, nv);

    params.count = count;
    params.spacing = nv;
    params.num_partitions = info.num_partitions;
    params.comp = comp;

    // Ping pong with this buffer.
    GLuint keys2 = keys_ssbo.buffer;
    
    // Upload the UBO.
    params_ubo.set_data(params);
    params_ubo.bind_ubo(0);

    // Bind the partitions buffer.
    if(info.num_passes)
      partitions_ssbo.bind_ssbo(2);

    // Execute the block sort.
    if(info.num_passes % 2) {
      // Read the input and write to the aux buffer.
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, keys);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, keys2);
      std::swap(keys, keys2);

    } else {
      // Read the input and write to the input.
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, keys);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, keys);
    }

    // Launch the blocksort kernel.
    gl_dispatch_kernel<kernel_blocksort<false, nt, vt, params_t, 0> >(
      info.num_ctas
    );
    
    // Execute the merge passes.
    for(int pass = 0; pass < info.num_passes; ++pass) {
      // Bind the inputs and outputs.
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, keys);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, keys2);

      // Launch the partitions kernel.
      gl_dispatch_kernel<kernel_mergesort_partition<params_t> >(
        info.num_partition_ctas);

      // Launch the mergesort pass kernel.
      gl_dispatch_kernel<kernel_mergesort_pass<nt, vt, params_t, 0> >(
        info.num_ctas
      );

      // Swap the source and destintation buffers.
      std::swap(keys, keys2);
    }
  }

  template<int nt = 128, int vt = 7>
  void sort_keys_indices(GLuint keys, GLuint vals, int count, 
    comp_t comp = comp_t()) {
    
    sort_keys_values<nt, vt, true>(keys, vals, count, comp);
  }

  template<int nt = 256, int vt = 7, bool sort_indices = false>
  void sort_keys_values(GLuint keys, GLuint vals, int count,
    comp_t comp = comp_t()) {

    static_assert(has_values);
    const int nv = nt * vt;

    if(!count) return;

    params_t params { };
    info_t info = reserve(count, nv);

    params.count = count;
    params.spacing = nv;
    params.num_partitions = info.num_partitions;
    params.comp = comp;

    // Ping pong with this buffer.
    GLuint keys2 = keys_ssbo.buffer;
    GLuint vals2 = vals_ssbo.buffer;
    
    // Upload the UBO.
    params_ubo.set_data(params);
    params_ubo.bind_ubo(0);

    // Bind the partitions buffer.
    if(info.num_passes)
      partitions_ssbo.bind_ssbo(2);

    // Execute the block sort.
    if(info.num_passes % 2) {
      // Read the input and write to the aux buffer.
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, keys);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, keys2);

      if constexpr(!sort_indices)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, vals);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, vals2);

      std::swap(keys, keys2);
      std::swap(vals, vals2);

    } else {
      // Read the input and write to the input.
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, keys);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, keys);

      if constexpr(!sort_indices)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, vals);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, vals);
    }

    // Launch the blocksort kernel.
    gl_dispatch_kernel<kernel_blocksort<sort_indices, nt, vt, params_t, 0> >(
      info.num_ctas
    );

    // Execute the merge passes.
    for(int pass = 0; pass < info.num_passes; ++pass) {
      // Bind the inputs and outputs.
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, keys);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, keys2);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, vals);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, vals2);

      // Launch the partitions kernel.
      gl_dispatch_kernel<kernel_mergesort_partition<params_t> >(
        info.num_partition_ctas);

      // Launch the mergesort pass kernel.
      gl_dispatch_kernel<kernel_mergesort_pass<nt, vt, params_t, 0> >(
        info.num_ctas
      );

      // Swap the source and destintation buffers.
      std::swap(keys, keys2);
      std::swap(vals, vals2);
    }
  }

  typedef mergesort_params_t<
    2,    // 2 is reserved for partitions.
    buffer_iterator_t<key_t, 0>,
    buffer_iterator_t<val_t, 3>,
    readonly_iterator_t<key_t, 0>,
    readonly_iterator_t<val_t, 3>,
    buffer_iterator_t<key_t, 1>,
    buffer_iterator_t<val_t, 4>,
    comp_t
  > params_t;

  // Keep a parameters UBO. The value is cached so glNamedBufferSubData is only
  // called when something changes.
  gl_buffer_t<const params_t> params_ubo;

  // Keep storage for keys and values to ping-pong between passes.
  gl_buffer_t<key_t[]> keys_ssbo;
  gl_buffer_t<val_t[]> vals_ssbo;
  gl_buffer_t<int[]>   partitions_ssbo;
};

} // namespace gl

END_MGPU_NAMESPACE
