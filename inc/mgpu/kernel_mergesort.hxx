#pragma once
#include "cta_mergesort.hxx"
#include "transform.hxx"
#include "bindings.hxx"
#include "buffer.hxx"

BEGIN_MGPU_NAMESPACE

////////////////////////////////////////////////////////////////////////////////
// Sort full tiles in place.

template<
  bool sort_indices, 
  int nt, int vt,
  typename keys_in_it, typename vals_in_it,
  typename keys_out_it, typename vals_out_it,
  typename comp_t
>
void kernel_blocksort(
  keys_in_it keys_in, vals_in_it vals_in,
  keys_out_it keys_out, vals_out_it vals_out, 
  int count, comp_t comp) {
  
  typedef typename std::iterator_traits<keys_in_it>::value_type key_t;
  typedef typename std::iterator_traits<vals_in_it>::value_type val_t;
  enum { has_values = !std::is_same<val_t, empty_t>::value };

  typedef cta_sort_t<nt, vt, key_t, val_t> sort_t;
  sort_t sort;

  [[spirv::shared]] typename sort_t::storage_t shared;

  const int nv = nt * vt;
  int tid = threadIdx.x;
  int cta = blockIdx.x;
  range_t tile = get_tile(cta, nv, count);

  // Load the keys and values.
  kv_array_t<key_t, val_t, vt> unsorted;
  unsorted.keys = mem_to_reg_thread<nt, vt>(keys_in + tile.begin, tid, 
    tile.count(), shared.keys);

  if constexpr(sort_indices) {
    // If we're sorting key/index pairs, sythesize the data without sampling
    // the counting_iterator, which would perform a trip through shared 
    // memory.
    int index = nv * cta + vt * tid;
    @meta for(int i = 0; i < vt; ++i)
      unsorted.vals[i] = index + i;

  } else if constexpr(has_values) {
    unsorted.vals = mem_to_reg_thread<nt, vt>(vals_in + tile.begin, tid,
      tile.count(), shared.vals);
  }

  // Blocksort.
  kv_array_t<key_t, val_t, vt> sorted = sort_t().block_sort(unsorted,
    tid, tile.count(), comp, shared);

  // Store the keys and values.
  reg_to_mem_thread<nt, vt>(sorted.keys, tid, tile.count(), 
    keys_out + tile.begin, shared.keys);

  if constexpr(has_values)
    reg_to_mem_thread<nt, vt>(sorted.vals, tid, tile.count(), 
      vals_out + tile.begin, shared.vals);
}

template<bool sort_indices, int nt, int vt, typename params_t, int ubo>
[[using spirv: comp, local_size(nt)]]
void kernel_blocksort() {
  params_t params = shader_uniform<ubo, params_t>;

  kernel_blocksort<sort_indices, nt, vt>(
    params.keys_in,
    params.vals_in,
    params.keys_out,
    params.vals_out,
    params.count,
    params.comp
  );
}

template<
  typename keys_in_it, typename vals_in_it,
  typename keys_out_it, typename vals_out_it,
  typename comp_t
>
struct blocksort_params_t {
  keys_in_it keys_in;
  vals_in_it vals_in;
  keys_out_it keys_out;
  vals_out_it vals_out;

  int count;
  comp_t comp;
};

////////////////////////////////////////////////////////////////////////////////

template<typename mp_it, typename keys_it, typename comp_t>
void kernel_mergesort_partition(mp_it mp_data, keys_it keys, int count, 
  int spacing, int coop, comp_t comp) {

  int num_partitions = (int)div_up(count, spacing) + 1;
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index < num_partitions) {
    merge_range_t range = compute_mergesort_range(count, index, coop, spacing);
    int diag = min(spacing * index, count) - range.a_begin;
    mp_data[index] = merge_path<bounds_lower>(keys + range.a_begin, 
      range.a_count(), keys + range.b_begin, range.b_count(), diag, comp);
  }
}

template<typename params_t, int mp, int ubo = 0>
[[using spirv: comp, local_size(128)]]
void kernel_mergesort_partition() {
  params_t params = shader_uniform<ubo, params_t>;
  kernel_mergesort_partition(
    writeonly_iterator_t<int, mp>(),
    params.keys_in,
    params.count,
    params.spacing,
    params.coop,
    params.comp
  );
}

////////////////////////////////////////////////////////////////////////////////
// Join two fully sorted sequences into one sequence.

template<
  int nt, int vt,
  typename mp_it,
  typename keys_in_it, typename vals_in_it,
  typename keys_out_it, typename vals_out_it,
  typename comp_t
>
void kernel_mergesort_pass(mp_it mp_data,
  keys_in_it keys_in, vals_in_it vals_in,
  keys_out_it keys_out, vals_out_it vals_out, 
  comp_t comp, int count, int coop) {

  typedef typename std::iterator_traits<keys_in_it>::value_type key_t;
  typedef typename std::iterator_traits<vals_in_it>::value_type val_t;
  enum { has_values = !std::is_same<val_t, empty_t>::value };

  const int nv = nt * vt;
  int tid = threadIdx.x;
  int cta = blockIdx.x;

  struct shared_t {
    key_t keys[nv + 1];
    int indices[nv];
  };
  [[spirv::shared]] shared_t shared; 

  range_t tile = get_tile(cta, nv, count);

  // Load the range for this CTA and merge the values into register.
  merge_range_t range = compute_mergesort_range(count, cta, coop, nv, 
    mp_data[cta + 0], mp_data[cta + 1]);

//  merge_pair_t<key_t, vt> merge = cta_merge_from_mem<bounds_lower, nt, vt>(
//    keys_in, keys_in, range, tid, comp, shared.keys);
//
//  // Store merged values back out.
//  reg_to_mem_thread<nt>(merge.keys, tid, tile.count(), 
//    keys_out + tile.begin, shared.keys);
//
//  if(has_values) {
//    // Transpose the indices from thread order to strided order.
//    std::array<int, vt> indices = reg_thread_to_strided<nt>(merge.indices,
//      tid, shared.indices);
//
//    // Gather the input values and merge into the output values.
//    transfer_two_streams_strided<nt>(vals_in + range.a_begin, 
//      range.a_count(), vals_in + range.b_begin, range.b_count(),
//      indices, tid, vals_out + tile.begin);
//  }
}

template<int nt, int vt, typename params_t, int mp, int ubo>
[[using spirv: comp, local_size(nt)]]
void kernel_mergesort_pass() {
  params_t params = shader_uniform<ubo, params_t>;

  kernel_mergesort_pass<nt, vt>(
    readonly_iterator_t<int, mp>(),
    params.keys_in,
    params.vals_in,
    params.keys_out,
    params.vals_out,
    params.comp,
    params.count,
    params.coop
  );
}

template<
  typename keys_in_it,
  typename vals_in_it,
  typename keys_out_it,
  typename vals_out_it,
  typename comp_t
> struct mergesort_pass_params_t {
  keys_in_it keys_in;
  vals_in_it vals_in;
  keys_out_it keys_out;
  vals_out_it vals_out;

  int spacing;  // For partitioning
  int count;
  int coop;     // 2<< pass.
  comp_t comp;
};

////////////////////////////////////////////////////////////////////////////////

template<
  typename keys_in_it,   typename vals_in_it,
  typename keys_pass_it, typename vals_pass_it,
  typename keys_out_it,  typename vals_out_it,
  typename comp_t,       int max_passes = 16
>
struct mergesort_params_t {
  // Reserve bind slot 2 for merge path array.

  // One structure for the block sort. This will be in-place if the number of 
  // passes is even.
  typedef blocksort_params_t<
    keys_in_it,
    vals_in_it,
    keys_out_it,
    vals_out_it,
    comp_t
  > blocksort_t;

  // One structure for each pass.
  typedef mergesort_pass_params_t<
    keys_pass_it, 
    vals_pass_it,
    keys_out_it,
    vals_out_it,
    comp_t
  > pass_t;

  blocksort_t blocksort;
  pass_t passes[max_passes];

  int configure(int count, int spacing, comp_t comp) {
    int num_ctas = div_up(count, spacing);
    int num_passes = s_log2(num_ctas);

    blocksort.count = count;
    blocksort.comp = comp;

    for(int i = 0; i < num_passes; ++i) {
      passes[i].spacing = spacing;
      passes[i].count = count;
      passes[i].coop = 2<< i;
      passes[i].comp = comp;
    }

    return num_passes;
  }
};

template<typename key_t, typename val_t = empty_t, 
  typename comp_t = std::less<key_t>, int max_passes = 16>
struct mergesort_pipeline_t {
  enum { has_values = !std::is_same_v<val_t, empty_t> };

  struct ctas_t {
    int num_ctas;
    int num_partition_ctas;
  };

  ctas_t reserve(int num_passes, int count, int nv) {
    int num_ctas = div_up(count, nv);
    int num_partitions = num_ctas + 1;

    if(num_passes > 1) {
      partitions_ssbo.resize(num_partitions);
      keys_ssbo.resize(count);
      if(has_values)
        vals_ssbo.resize(count);
    }

    return { num_ctas, num_partitions };
  }

  template<int nt = 256, int vt = 7>
  void sort_keys(GLuint keys, int count, comp_t comp = comp_t()) {
    static_assert(!has_values);
    const int nv = nt * vt;

    if(!count) return;

    params_t params { };
    int num_passes = params.configure(count, nv, comp);
    num_passes = 0;
    ctas_t ctas = reserve(num_passes, count, nv);

    // Ping pong with this buffer.
    GLuint keys2 = keys_ssbo.buffer;
    
    // Upload the UBO.
    params_ubo.set_data(params);

    // Execute the block sort.
    {
      params_ubo.bind_ubo_range(
        0, 
        offsetof(params_t, blocksort), 
        sizeof(blocksort_t)
      );

      if(num_passes % 2) {
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
      gl_dispatch_kernel<kernel_blocksort<false, nt, vt, blocksort_t, 0> >(
        ctas.num_ctas
      );
    }
/*
    // Bind the partitions buffer.
    partitions_ssbo.bind_ssbo(2);

    // Execute the merge passes.
    for(int pass = 0; pass < num_passes; ++pass) {
      // Update the UBO view into the parameters.
      params_ubo.bind_ubo_range(
        0, 
        offsetof(params_t, passes) + pass * sizeof(pass_t),
        sizeof(pass_t)
      );

      // Bind the inputs and outputs.
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, keys);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, keys2);

      // Launch the partitions kernel.
     gl_dispatch_kernel<kernel_mergesort_partition<pass_t, 2> >(
       ctas.num_partition_ctas);

   //  // Launch the mergesort pass kernel.
   //  gl_dispatch_kernel<kernel_mergesort_pass<nt, vt, pass_t, 2, 0> >(
   //    ctas.num_ctas
   //  );

      std::swap(keys, keys2);
    }
    */
  }

  template<int nt = 256, int vt = 7>
  void sort_key_indices(GLuint keys, GLuint indices, int count, 
    comp_t comp = comp_t()) {
    static_assert(std::is_same_v<int, val_t>);
    if(!count) return;

  }

  template<int nt = 256, int vt = 7>
  void sort_key_values(GLuint keys, GLuint values, int count,
    comp_t comp = comp_t()) {
    static_assert(has_values);
    if(!count) return;
  }

  typedef mergesort_params_t<
    readonly_iterator_t<key_t, 0>,
    readonly_iterator_t<val_t, 3>,
    readonly_iterator_t<key_t, 0>,
    readonly_iterator_t<val_t, 3>,
    writeonly_iterator_t<key_t, 1>,
    writeonly_iterator_t<val_t, 4>,
    comp_t,
    max_passes
  > params_t;

  typedef typename params_t::blocksort_t blocksort_t;
  typedef typename params_t::pass_t pass_t;

  // Keep a parameters UBO. The value is cached so glNamedBufferSubData is only
  // called when something changes.
  gl_buffer_t<const params_t> params_ubo;

  // Keep storage for keys and values to ping-pong between passes.
  gl_buffer_t<key_t[]> keys_ssbo;
  gl_buffer_t<val_t[]> vals_ssbo;
  gl_buffer_t<int[]>   partitions_ssbo;
};


END_MGPU_NAMESPACE
