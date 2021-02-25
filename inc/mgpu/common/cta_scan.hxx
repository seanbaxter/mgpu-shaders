#pragma once
#include "loadstore.hxx"
#include "subgroup.hxx"

BEGIN_MGPU_NAMESPACE

////////////////////////////////////////////////////////////////////////////////

template<int nt, typename type_t>
struct cta_reduce_t {
  struct storage_t {
    type_t warps[nt / 8];
  };

  // Reduce the values across a cta. Only thread 0 returns a value. If all
  // threads want the value, store to shared memory and broadcast.
  template<typename op_t = std::plus<type_t> >
  type_t reduce(type_t x, storage_t& shared, op_t op = op_t()) {
    int warp_size = gl_SubgroupSize;
    int num_warps = gl_NumSubgroups;
    int lane = gl_SubgroupInvocationID;
    int warp = gl_SubgroupID;

    // Reduce within a warp.
    for(int offset = 1; offset < warp_size; offset<<= 1) {
      type_t y = subgroupShuffleDown(x, offset);
      if(lane + offset < warp_size)
        x = op(x, y);
    }

    // The first lane in each warp writes its reduction.
    if(!lane)
      shared.warps[warp] = x;
    __syncthreads();

    // Scan the reductions. This assumes we can do it in one shot.
    if(lane < num_warps) {
      x = shared.warps[lane];
      for(int offset = 1; offset < num_warps; offset<<= 1) {
        type_t y = subgroupShuffleDown(x, offset);
        if(lane + offset < num_warps)
          x = op(x, y);
      }
    }
    __syncthreads();

    return x;
  }

  template<int vt, typename op_t = std::plus<type_t> >
  type_t reduce(std::array<type_t, vt> x, storage_t& shared, op_t op = op_t()) {
    // Reduce within a thread.
    @meta for(int i = 1; i < vt; ++i)
      x[0] = op(x[0], x[i]);

    // Reduce across threads.
    return reduce(x[0], shared, op);
  }
};

////////////////////////////////////////////////////////////////////////////////

enum scan_type_t {
  scan_type_exc,
  scan_type_inc,
};

template<typename type_t, int vt = 0, bool is_array = (vt > 0)>
struct scan_result_t {
  type_t scan;
  type_t reduction;
};

template<typename type_t, int vt>
struct scan_result_t<type_t, vt, true> {
  std::array<type_t, vt> scan;
  type_t reduction;
};

template<int nt, typename type_t>
struct cta_scan_t {
  struct storage_t {
    int warps[nt / 8];
  };

  // Scalar scan.
  template<
    scan_type_t scan_type = scan_type_exc, 
    typename op_t = std::plus<type_t>
  >
  scan_result_t<type_t> scan(type_t x, storage_t& shared, 
    type_t init = type_t(), op_t op = op_t()) {

    int warp_size = gl_SubgroupSize;
    int num_warps = gl_NumSubgroups;
    int lane = gl_SubgroupInvocationID;
    int warp = gl_SubgroupID;

    // Use subgroupShuffleUp to prefix sum over a warp.
    for(int offset = 1; offset < warp_size; offset<<= 1) {
      type_t y = subgroupShuffleUp(x, offset);
      if(offset <= lane)
        x = op(y, x);
    }

    // The last lane in each warp writes its reduction.
    if(warp_size - 1 == lane)
      shared.warps[warp] = x;
    __syncthreads();

    // Scan the reductions. This assumes we can do it in one shot.
    if(lane < num_warps) {
      type_t x = shared.warps[lane];
      for(int offset = 1; offset < num_warps; offset<<= 1) {
        type_t y = subgroupShuffleUp(x, offset);
        if(offset <= lane)
          x = op(y, x);
      }
      shared.warps[lane] = x;
    }
    __syncthreads();

    if constexpr(scan_type_exc == scan_type) {
      // For exclusive scan, get the value of the warp scan to the left.
      type_t left = subgroupShuffleUp(x, 1);
      x = lane ? left : init;
    }

    if(warp)
      x = op(shared.warps[warp - 1], x);
    type_t reduction = shared.warps[num_warps - 1];
    __syncthreads();

    return { x, reduction };
  }

  // CTA vectorized scan. Accepts multiple values per thread and adds in
  // optional global carry-in.
  template<
    scan_type_t scan_type = scan_type_exc,
    int vt, 
    typename op_t = std::plus<type_t>
  > 
  scan_result_t<type_t, vt> scan(std::array<type_t, vt> x, storage_t& shared,
    type_t carry_in = type_t(), type_t init = type_t(), op_t op = op_t()) {

    int tid = glcomp_LocalInvocationID.x;

    // Start with an inclusive scan of the in-range elements.
    @meta for(int i = 1; i < vt; ++i)
      x[i] = op(x[i - 1], x[i]);

    // Scan the thread-local reductions for a carry-in for each thread.
    scan_result_t<type_t> result = scan(x[vt - 1], shared, init, op);

    // Perform the scan downsweep and add both global carry-in and the thread
    // carry-in to the values.
    result.reduction = op(carry_in, result.reduction);
    result.scan = op(carry_in, result.scan);

    if constexpr(scan_type_exc == scan_type) {
      @meta for(int i = vt - 1; i > 0; --i)
        x[i] = op(result.scan, x[i - 1]);
      x[0] = result.scan;

    } else {
      // Add the carry-in.
      @meta for(int i = 0; i < vt; ++i)
        x[i] = op(result.scan, x[i]);
    }

    return { x, result.reduction };
  }
};

END_MGPU_NAMESPACE
