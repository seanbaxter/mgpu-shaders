#pragma once
#include "loadstore.hxx"

BEGIN_MGPU_NAMESPACE

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
struct scan_result_t {
  std::array<type_t, vt> scan;
  type_t reduction;
};

template<int nt, typename type_t>
struct cta_scan_t {

  struct storage_t {
    type_t data[2 * nt];
  };

  template<typename comp_t>
  type_t warp_scan(int tid, type_t x, comp_t comp, storage_t& storage) {
    if constexpr(is_plus_type_v<comp_t>) {
      x = gl_subgroupInclusiveAdd(x);

    } else {

    }
    return x;
  }

  template<
    scan_type_t scan_type = scan_type_exc, 
    typename op_t = std::plus<type_t>
  >
  scan_result_t<type_t> scan(int tid, type_t x, storage_t& storage,
    int count = nt, op_t op = op_t(), type_t init = type_t()) const {

    int lane = gl_SubgroupInvocationID;
    int warp = gl_SubgroupID;

    // Treat the warp size as the min of gl_SubgroupSize and nt.
    int warp_size = gl_SubgroupSize & (nt - 1);

    type_t s = warp_scan(lane, )
    if(lane == warp_size - 1)

  }
};

gl_SubgroupInvocationID = lane
gl_SubgroupID = warp
gl_SubroupSize = warp size
gl_NumSubgroups = num warps

gl_NumSubgroups is the number of subgroups within the local workgroup.
gl_SubgroupID is the ID of the subgroup within the local workgroup, an integer in the range [0..gl_NumSubgroups).

[[spirv::builtin]] uint  gl_SubgroupSize;
[[spirv::builtin]] uint  gl_SubgroupInvocationID;

END_MGPU_NAMESPACE


template<typename type_t, typename op_t> 
MGPU_DEVICE type_t shfl_up_op(type_t x, int offset, op_t op, 
  int width = warp_size) {

  type_t y = shfl_up(x, offset, width);
  int lane = (width - 1) & threadIdx.x;
  if(lane >= offset) x = op(x, y);
  return x;
}

template<typename type_t, typename op_t> 
MGPU_DEVICE type_t shfl_down_op(type_t x, int offset, op_t op, 
  int width = warp_size) {

  type_t y = shfl_down(x, offset, width);
  int lane = (width - 1) & threadIdx.x;
  if(lane < width - offset) x = op(x, y);
  return x;
}
