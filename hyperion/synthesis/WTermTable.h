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
#ifndef HYPERION_SYNTHESIS_W_TERM_TABLE_H_
#define HYPERION_SYNTHESIS_W_TERM_TABLE_H_

#include <hyperion/synthesis/CFTable.h>

#include <array>
#include <cmath>

namespace hyperion {
namespace synthesis {

class HYPERION_EXPORT WTermTable
  : public CFTable<CF_W> {
public:

  WTermTable(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::array<Legion::coord_t, 2>& cf_bounds_lo,
    const std::array<Legion::coord_t, 2>& cf_bounds_hi,
    const std::array<double, 2>& cell_size,
    const std::vector<typename cf_table_axis<CF_W>::type>& w_values);

  WTermTable(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const Legion::coord_t& cf_x_radius,
    const Legion::coord_t& cf_y_radius,
    const std::array<double, 2>& cell_size,
    const std::vector<typename cf_table_axis<CF_W>::type>& w_values);

  void
  compute_cfs(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const ColumnSpacePartition& partition = ColumnSpacePartition()) const;

  std::array<double, 2> m_cell_size;

  static const constexpr char* compute_cfs_task_name =
    "WTermTable::compute_cfs_task";

  static Legion::TaskID compute_cfs_task_id;

  static const constexpr double twopi = 2 * 3.141592653589793;

  struct ComputeCFSTaskArgs {
    Table::Desc desc;
    decltype(m_cell_size) cell_size; 
  };

#ifdef HYPERION_USE_KOKKOS
  template <typename execution_space>
  static void
  compute_cfs_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* rt) {

  const ComputeCFSTaskArgs& args =
    *static_cast<const ComputeCFSTaskArgs*>(task->args);
  const decltype(m_cell_size)& cell_size = args.cell_size;

  auto ptcr =
    PhysicalTable::create(
      rt,
      args.desc,
      task->regions.begin(),
      task->regions.end(),
      regions.begin(),
      regions.end())
    .value();
#if HAVE_CXX17
  auto& [pt, rit, pit] = ptcr;
#else // !HAVE_CXX17
  auto& pt = std::get<0>(ptcr);
  auto& rit = std::get<1>(ptcr);
  auto& pit = std::get<2>(ptcr);
#endif // HAVE_CXX17
  assert(rit == task->regions.end());
  assert(pit == regions.end());

  auto tbl = CFPhysicalTable<CF_W>(pt);

  auto w_value_acc = tbl.w<Legion::AffineAccessor>().accessor<READ_ONLY>();
  typedef typename cf_table_axis<CF_W>::type fp_t;
  Kokkos::View<const fp_t *, execution_space> w_values =
    w_value_acc.accessor;

  auto value_col = tbl.value<Legion::AffineAccessor>();
  auto value_rect = value_col.rect();
  auto value_acc = value_col.accessor<WRITE_ONLY>();
  Kokkos::View<typename CFTableBase::cf_value_t ***, execution_space> values =
    value_acc.accessor;

  Kokkos::MDRangePolicy<Kokkos::Rank<3>, execution_space> range(
    rt->get_executing_processor(ctx).kokkos_work_space(),
    rect_lo(value_rect),
    rect_hi(value_rect));
  Kokkos::parallel_for(
    "ComputeWTerm",
    range,
    KOKKOS_LAMBDA (Legion::coord_t w, Legion::coord_t x, Legion::coord_t y) {
      const fp_t l = cell_size[0] * x;
      const fp_t m = cell_size[1] * y;
      const fp_t r2 = l * l + m * m;
      const fp_t phase =
        ((r2 <= (fp_t)1.0)
         ? (((fp_t)twopi * w_values(w))
            * (std::sqrt((fp_t)1.0 - r2) - (fp_t)1.0))
         : (fp_t)0.0);
      values(w, x, y) = std::polar((fp_t)1.0, phase);
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

#endif // HYPERION_SYNTHESIS_W_TERM_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
