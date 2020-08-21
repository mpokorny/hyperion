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
#include <hyperion/synthesis/GridCoordinateTable.h>

#include <array>
#include <cmath>

namespace hyperion {
namespace synthesis {

class HYPERION_EXPORT WTermTable
  : public CFTable<CF_W> {
public:

  static const constexpr unsigned d_w = 0;

  /**
   * WTermTable constructor
   *
   * @param ctx Legion Context
   * @param rt Legion Runtime pointer
   * @param grid_size size of CF grid in either dimension
   * @param w_values W axis values
   */
  WTermTable(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const size_t& grid_size,
    const std::vector<typename cf_table_axis<CF_W>::type>& w_values);

  /**
   * compute the W term convolution function values, given a cell size for
   * function evaluation (in image domain)
   *
   * @param ctx Legion Context
   * @param rt Legion Runtime pointer
   * @param gc grid coordinate system
   * @param partition table partition (optional)
   *
   * The coordinate system is used in the CF evaluation directly, on the
   * assumption that its coordinate values are (l, m) values
   */
  void
  compute_cfs(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const GridCoordinateTable& gc,
    const ColumnSpacePartition& partition = ColumnSpacePartition()) const;

  /**
   * task name for compute_cfs_task
   */
  static const constexpr char* compute_cfs_task_name =
    "WTermTable::compute_cfs_task";

  /**
   * task id for compute_cfs_task
   */
  static Legion::TaskID compute_cfs_task_id;

  static const constexpr double twopi = 2 * 3.141592653589793;

  struct ComputeCFsTaskArgs {
    Table::Desc w;
    Table::Desc gc;
  };

  template <typename execution_space>
  static void
  compute_cfs_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* rt) {

  const ComputeCFsTaskArgs& args =
    *static_cast<const ComputeCFsTaskArgs*>(task->args);
  std::vector<Table::Desc> tdesc{args.w, args.gc};

  auto ptcrs =
    PhysicalTable::create_many(
      rt,
      tdesc,
      task->regions.begin(),
      task->regions.end(),
      regions.begin(),
      regions.end())
    .value();
#if HAVE_CXX17
  auto& [pts, rit, pit] = ptcrs;
#else // !HAVE_CXX17
  auto& pts = std::get<0>(ptcrs);
  auto& rit = std::get<1>(ptcrs);
  auto& pit = std::get<2>(ptcrs);
#endif // HAVE_CXX17
  assert(rit == task->regions.end());
  assert(pit == regions.end());

  auto w_tbl = CFPhysicalTable<CF_W>(pts[0]);
  auto gc_tbl = CFPhysicalTable<CF_PARALLACTIC_ANGLE>(pts[1]);

  auto w_values =
    w_tbl.w<Legion::AffineAccessor>().view<execution_space, LEGION_READ_ONLY>();

  auto value_col = w_tbl.value<Legion::AffineAccessor>();
  auto value_rect = value_col.rect();
  auto values = value_col.view<execution_space, LEGION_WRITE_ONLY>();
  auto weights =
    w_tbl
    .weight<Legion::AffineAccessor>()
    .view<execution_space, LEGION_WRITE_ONLY>();

  auto cs_x_col =
    GridCoordinateTable::CoordColumn<Legion::AffineAccessor>(
      *gc_tbl.column(GridCoordinateTable::COORD_X_NAME).value());
  auto i_pa = cs_x_col.rect().lo[GridCoordinateTable::d_pa];
  auto cs_x =
    Kokkos::subview(
      cs_x_col.view<execution_space, LEGION_READ_ONLY>(),
      i_pa,
      Kokkos::ALL,
      Kokkos::ALL);
  auto cs_y =
    Kokkos::subview(
      GridCoordinateTable::CoordColumn<Legion::AffineAccessor>(
        *gc_tbl.column(GridCoordinateTable::COORD_Y_NAME).value())
      .view<execution_space, LEGION_READ_ONLY>(),
      i_pa,
      Kokkos::ALL,
      Kokkos::ALL);

  Kokkos::MDRangePolicy<Kokkos::Rank<3>, execution_space> range(
    rt->get_executing_processor(ctx).kokkos_work_space(),
    rect_lo(value_rect),
    rect_hi(value_rect));
  Kokkos::parallel_for(
    "ComputeWTerm",
    range,
    KOKKOS_LAMBDA (
      Legion::coord_t i_w,
      Legion::coord_t i_x,
      Legion::coord_t i_y) {

      const cf_fp_t l = cs_x(i_x, i_y);
      const cf_fp_t m = cs_y(i_x, i_y);
      const cf_fp_t r2 = l * l + m * m;
      if (r2 <= (cf_fp_t)1.0) {
        const cf_fp_t phase =
          (cf_fp_t)twopi * w_values(i_w)
          * (std::sqrt((cf_fp_t)1.0 - r2) - (cf_fp_t)1.0);
        values(i_w, i_x, i_y) = std::polar((cf_fp_t)1.0, phase);
        weights(i_w, i_x, i_y) = (cf_fp_t)1.0;
      } else {
        values(i_w, i_x, i_y) = (cf_fp_t)0.0;
        weights(i_w, i_x, i_y) = std::numeric_limits<cf_fp_t>::quiet_NaN();
      }
    });
  }

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
