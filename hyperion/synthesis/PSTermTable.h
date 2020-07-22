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

  PSTermTable(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::array<Legion::coord_t, 2>& cf_size,
    const std::vector<typename cf_table_axis<CF_PS_SCALE>::type>& ps_scales);

  void
  compute_cfs(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const ColumnSpacePartition& partition = ColumnSpacePartition()) const;

  static const constexpr char* compute_cfs_task_name =
    "PSTermTable::compute_cfs_task";

  static Legion::TaskID compute_cfs_task_id;

  template <size_t N>
  static KOKKOS_INLINE_FUNCTION double
  sph(const array<double, N>& ary, double nu_lo, double nu_hi) {
    static_assert(N > 0);
    const double dn2 = nu_lo * nu_lo - nu_hi * nu_hi;
    double result = ary[N - 1];
    for (unsigned k = N - 1; k > 0; --k)
      result = dn2 * result + ary[k - 1];
    return result;
  }

  static KOKKOS_INLINE_FUNCTION double
  spheroidal(double nu) {
    double result;
    if (nu <= 0) {
      result = 1.0;
    } else if (nu < 0.75) {
      const array<double, 5>
        p{8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1};
      const array<double, 3>
        q{1.0000000e0, 8.212018e-1, 2.078043e-1};
      result = sph(p, nu, 0.75) / sph(q, nu, 0.75);
    } else if (nu < 1.0) {
      const array<double, 5>
        p{4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2};
      const array<double, 3>
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

    const Table::Desc& desc = *static_cast<Table::Desc*>(task->args);

    auto ptcr =
      PhysicalTable::create(
        rt,
        desc,
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
    typedef decltype(value_col)::value_t::value_type fp_t;

    Kokkos::MDRangePolicy<Kokkos::Rank<3>, execution_space> range(
      rt->get_executing_processor(ctx).kokkos_work_space(),
      rect_lo(value_rect),
      rect_hi(value_rect));
    Kokkos::parallel_for(
      "ComputePSTerm",
      range,
      KOKKOS_LAMBDA (Legion::coord_t ps, Legion::coord_t x, Legion::coord_t y) {
        const fp_t yp =
          std::sqrt((static_cast<fp_t>(x) * x) + (static_cast<fp_t>(y) * y))
          * ps_scales(ps);
        const fp_t v =
          std::max(
            static_cast<fp_t>(spheroidal(yp)) * ((fp_t)1.0 - yp * yp),
            (fp_t)0.0);
        values(ps, x, y) = v;
        weights(ps, x, y) = v * v;
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
