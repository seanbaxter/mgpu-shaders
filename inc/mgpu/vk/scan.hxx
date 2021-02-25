#pragma once
#include "../common/cta_scan.hxx"
#include "context.hxx"
#include "transform.hxx"

BEGIN_MGPU_NAMESPACE

namespace vk {

template<int nt = 128, int vt = 7, typename type_t,
  typename op_t = std::plus<type_t> >
void scan(cmd_buffer_t& cmd_buffer, memcache_t& cache,
  type_t* data, int count, type_t init = type_t(), op_t op = op_t()) {

  enum { nv = nt * vt };
  int num_ctas = div_up(count, nv);

  if(num_ctas <= 8) {
    // The small input pass. Perform the scan with a single CTA.
    launch<nt>(1, cmd_buffer, [=](int tid, int cta) {

      // TODO: Edit compiler to flatten out top-level unions.
      typedef cta_scan_t<nt, int> scan_t;
      __shared__ union {
        typename scan_t::storage_t scan;
        type_t values[nv];
      } shared;

      type_t carry_in = type_t();
      for(int cur = 0; cur < count; cur += nv) {
        // Load this tile's data.
        std::array<type_t, vt> x = mem_to_reg_thread<nt, vt>(data + cur, tid,
          count - cur, shared.values);

        // Scan the inputs.
        auto result = scan_t().scan(x, shared.scan, carry_in, init, op);

        // Store the scanned values back to global memory.
        reg_to_mem_thread<nt, vt>(result.scan, tid, count - cur, data + cur, 
          shared.values);

        carry_in = result.reduction;
      }
    });

  } else {
    // The recursive kernel.
    int num_passes = find_log2(num_ctas, true);

    // Allocate space for one reduction per tile.
    type_t* partials = cache.allocate<type_t>(num_ctas);

    // The upsweep reduces each tile into partials.
    launch<nt>(num_ctas, cmd_buffer, [=](int tid, int cta) {
      typedef cta_reduce_t<nt, type_t> reduce_t;

      __shared__ union {
        typename reduce_t::storage_t reduce;
        type_t values[nv];
      } shared;

      int cur = nv * cta;

      // Load this tile's data.
      std::array<type_t, vt> x = mem_to_reg_thread<nt, vt>(data + cur, tid,
        count - cur, shared.values);

      type_t reduce = reduce_t().reduce(x, shared.reduce);

      // Write to the partials.
      if(!tid)
        partials[cta] = reduce;
    });

    // Recursively scan the partials.
    scan<nt, vt>(cmd_buffer, cache, partials, num_ctas, init, op);

    // The downsweep performs a scan with carry-in.
    launch<nt>(num_ctas, cmd_buffer, [=](int tid, int cta) {
      typedef cta_scan_t<nt, type_t> scan_t;

      __shared__ union {
        typename scan_t::storage_t scan;
        type_t values[nv];
      } shared;

      int cur = nv * cta;

      // Load this tile's data.
      std::array<type_t, vt> x = mem_to_reg_thread<nt, vt>(data + cur, tid,
        count - cur, shared.values);

      // Load the carry-in.
      type_t carry_in = partials[cta];

      // Scan the inputs.
      auto result = scan_t().scan(x, shared.scan, carry_in, init, op);

      // Store the scanned values back to global memory.
      reg_to_mem_thread<nt, vt>(result.scan, tid, count - cur, data + cur, 
        shared.values);
    });
  }
}

} // namespace vk

END_MGPU_NAMESPACE
