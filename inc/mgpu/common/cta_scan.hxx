#pragma once
#include "loadstore.hxx"
#include "subgroup.hxx"

BEGIN_MGPU_NAMESPACE

enum scan_type_t {
  scan_type_exc,
  scan_type_inc,
};

template<int nt, typename type_t>
struct cta_scan_t {
  struct storage_t {
    int warps[nt / 8];
  };

  struct result_t {
    type_t scan;
    type_t reduction;
  };

  template<
    scan_type_t scan_type = scan_type_exc, 
    typename op_t = std::plus<type_t>
  >
  result_t scan(type_t x, storage_t& shared, type_t init = type_t(), 
    op_t op = op_t()) {

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
};

END_MGPU_NAMESPACE
