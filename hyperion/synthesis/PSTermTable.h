/*
 * Copyright 2020 Associated Universities, Inc. Washington DC, USA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef HYPERION_SYNTHESIS_PS_TERM_TABLE_H_
#define HYPERION_SYNTHESIS_PS_TERM_TABLE_H_

#include <hyperion/synthesis/CFTable.h>

#include <array>
#include <cmath>

namespace hyperion {
namespace synthesis {

class HYPERION_EXPORT PSTermTable
  : public CFTable<CF_PS_SCALE> {
public:

  static const constexpr unsigned d_ps = 0;

  /**
   * PSTermTable constructor.
   *
   * @param ctx Legion Context
   * @param rt Legion Runtime pointer
   * @param cf_size dimensions of CF in image domain (not extended)
   * @param ps_scales PS scale axis values
   *
   * Commonly, \a ps_scales will have only a single element.
   */
  PSTermTable(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::array<Legion::coord_t, 2>& cf_size,
    const std::vector<typename cf_table_axis<CF_PS_SCALE>::type>& ps_scales);

  /**
   * compute the PS term convolution function values (in the image domain)
   *
   * @param ctx Legion Context
   * @param rt Legion Runtime pointer
   * @param partition table partition (optional)
   */
  void
  compute_cfs(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const ColumnSpacePartition& partition = ColumnSpacePartition()) const;

  /**
   * task name for compute_cfs_task
   */
  static const constexpr char* compute_cfs_task_name =
    "PSTermTable::compute_cfs_task";

  /**
   * taskID for compute_cfs_task
   */
  static Legion::TaskID compute_cfs_task_id;

  struct ComputeCFsTaskArgs {
    Table::Desc desc;
    array<cf_fp_t, 2> pixel_offset;
  };

  template <size_t N>
  static KOKKOS_INLINE_FUNCTION cf_fp_t
  sph(const array<cf_fp_t, N>& ary, cf_fp_t nu_lo, cf_fp_t nu_hi) {
    static_assert(N > 0);
    const cf_fp_t dn2 = nu_lo * nu_lo - nu_hi * nu_hi;
    cf_fp_t result = ary[N - 1];
    for (unsigned k = N - 1; k > 0; --k)
      result = dn2 * result + ary[k - 1];
    return result;
  }

  static KOKKOS_INLINE_FUNCTION cf_fp_t
  spheroidal(cf_fp_t nu) {
    cf_fp_t result;
    if (nu <= 0) {
      result = 1.0;
    } else if (nu < 0.75) {
      const array<cf_fp_t, 5>
        p{8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1};
      const array<cf_fp_t, 3>
        q{1.0000000e0, 8.212018e-1, 2.078043e-1};
      result = sph(p, nu, 0.75) / sph(q, nu, 0.75);
    } else if (nu < 1.0) {
      const array<cf_fp_t, 5>
        p{4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2};
      const array<cf_fp_t, 3>
        q{1.0000000e0, 9.599102e-1, 2.918724e-1};
      result = sph(p, nu, 1.0) / sph(q, nu, 1.0);
    } else {
      result = 0.0;
    }
    return result;
  }

#ifdef HYPERION_USE_KOKKOS
  template <typename execution_space>
  static void
  compute_cfs_task(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions,
    Legion::Context ctx,
    Legion::Runtime *rt) {

    const ComputeCFsTaskArgs& args =
      *static_cast<ComputeCFsTaskArgs*>(task->args);

    auto ptcr =
      PhysicalTable::create(
        rt,
        args.desc,
        task->regions.begin(),
        task->regions.end(),
        regions.begin(),
        regions.end())
      .value();
# if HAVE_CXX17
    auto& [pt, rit, pit] = ptcr;
# else // !HAVE_CXX17
    auto& pt = std::get<0>(ptcr);
    auto& rit = std::get<1>(ptcr);
    auto& pit = std::get<2>(ptcr);
# endif // HAVE_CXX17
    assert(rit == task->regions.end());
    assert(pit == regions.end());

    auto tbl = CFPhysicalTable<CF_PS_SCALE>(pt);

    auto ps_scales =
      tbl.ps_scale<Legion::AffineAccessor>().view<execution_space, READ_ONLY>();

    auto value_col = tbl.value<Legion::AffineAccessor>();
    auto value_rect = value_col.rect();
    auto values = value_col.view<execution_space, WRITE_ONLY>();
    auto weights =
      tbl.weight<Legion::AffineAccessor>().view<execution_space, WRITE_ONLY>();

    auto& pixel_offset = args.pixel_offset;
    Kokkos::MDRangePolicy<Kokkos::Rank<3>, execution_space> range(
      rt->get_executing_processor(ctx).kokkos_work_space(),
      rect_lo(value_rect),
      rect_hi(value_rect));
    Kokkos::parallel_for(
      "ComputePSTerm",
      range,
      KOKKOS_LAMBDA(
        Legion::coord_t i_ps,
        Legion::coord_t i_x,
        Legion::coord_t i_y) {
        const cf_fp_t x = static_cast<cf_fp_t>(i_x) + pixel_offset[0];
        const cf_fp_t y = static_cast<cf_fp_t>(i_y) + pixel_offset[1];
        const cf_fp_t rs = std::sqrt(x * x + y * y) * ps_scales(i_ps);
        if (rs <= (cf_fp_t)1.0) {
          const cf_fp_t v = spheroidal(rs) * ((cf_fp_t)1.0 - rs * rs);
          values(i_ps, i_x, i_y) = v;
          weights(i_ps, i_x, i_y) = v * v;
        } else {
          values(i_ps, i_x, i_y) = (cf_fp_t)0.0;
          weights(i_ps, i_x, i_y) = std::numeric_limits<cf_fp_t>::quiet_NaN();
        }
      });
  }
#else // !HYPERION_USE_KOKKOS
  static void
  compute_cfs_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* rt);
#endif // HYPERION_USE_KOKKOS

  static void
  preregister_tasks();
};

} // end namespace synthesis
} // end namespace hyperion

#endif // HYPERION_SYNTHESIS_PS_TERM_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
