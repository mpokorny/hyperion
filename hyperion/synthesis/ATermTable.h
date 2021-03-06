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
#ifndef HYPERION_SYNTHESIS_A_TERM_TABLE_H_
#define HYPERION_SYNTHESIS_A_TERM_TABLE_H_

#include <hyperion/synthesis/CFTable.h>

#include <array>
#include <atomic>
#include <map>
#include <memory>

#define HYPERION_A_TERM_TABLE_AXES                                      \
  CF_BASELINE_CLASS, CF_PARALLACTIC_ANGLE, CF_FREQUENCY, CF_STOKES_OUT, CF_STOKES_IN

#include <hyperion/synthesis/ATermZernikeModel.h>
#include <hyperion/synthesis/ATermIlluminationFunction.h>
#include <hyperion/synthesis/GridCoordinateTable.h>
#include <hyperion/synthesis/FFT.h>

#include <fftw3.h>
#ifdef HYPERION_USE_CUDA
# include <cufft.h>
#endif

namespace hyperion {
namespace synthesis {

class HYPERION_EXPORT ATermTable
  : public CFTable<HYPERION_A_TERM_TABLE_AXES> {
public:

  ATermTable(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const size_t& grid_size,
    const std::vector<typename cf_table_axis<CF_BASELINE_CLASS>::type>&
      baseline_classes,
    const std::vector<typename cf_table_axis<CF_PARALLACTIC_ANGLE>::type>&
      parallactic_angles,
    const std::vector<typename cf_table_axis<CF_FREQUENCY>::type>&
      frequencies,
    const std::vector<typename cf_table_axis<CF_STOKES_OUT>::type>&
      stokes_out_values,
    const std::vector<typename cf_table_axis<CF_STOKES_IN>::type>&
      stokes_in_values);

  static const constexpr unsigned d_blc = 0;
  static const constexpr unsigned d_pa = d_blc + 1;
  static const constexpr unsigned d_frq = d_pa + 1;
  static const constexpr unsigned d_sto_out = d_frq + 1;
  static const constexpr unsigned d_sto_in = d_sto_out + 1;

  /**
   * compute the ATerm convolution functions
   *
   * @param ctx Legion Context
   * @param rt Legion Runtime
   * @param gc grid coordinate system
   * @param zernike_coefficients unordered vector of Zernike
   *                             expansion coefficients
   *
   * The grid coordinate system should normally be based on a
   * casacore::LinearCoordinate of rank 2, with radius equal to 1.0. Note that
   * the gc value is not a const referenence, as this method creates additional
   * columns in gc.
   */
  void
  compute_cfs(
    Legion::Context ctx,
    Legion::Runtime* rt,
    GridCoordinateTable& gc,
    const std::vector<ZCoeff>& zernike_coefficients,
    const ColumnSpacePartition& partition = ColumnSpacePartition()) const;

  static const constexpr char* compute_cfs_task_name =
    "ATermTable::compute_cfs_task";

  static Legion::TaskID compute_cfs_task_id;

  struct ComputeCFsTaskArgs {
    Table::Desc aif;
    Table::Desc aterm;
  };

  template <typename execution_space>
  static void
  compute_cfs_task(
    const Legion::Task*task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* rt) {

    const ComputeCFsTaskArgs& args =
      *static_cast<const ComputeCFsTaskArgs*>(task->args);

    std::vector<Table::Desc> descs{args.aif, args.aterm};
    auto pts =
      PhysicalTable::create_all_unsafe(rt, descs, task->regions, regions);

    auto kokkos_work_space =
      rt->get_executing_processor(ctx).kokkos_work_space();

    CFPhysicalTable<HYPERION_A_TERM_ILLUMINATION_FUNCTION_AXES> jones(pts[0]);
    CFPhysicalTable<HYPERION_A_TERM_TABLE_AXES> aterm(pts[1]);

    // Stokes CF functions.  We expect to have all of the Stokes index column
    // accessible
    auto jones_stokes_col = jones.template stokes<Legion::AffineAccessor>();
    auto jones_stokes =
      jones_stokes_col.template view<execution_space, LEGION_READ_ONLY>();
    auto jones_stokes_rect = jones_stokes_col.rect();
    // a map from stokes_t value to Stokes index in jones
    Kokkos::View<
      Legion::coord_t*,
      execution_space,
      Kokkos::MemoryTraits<Kokkos::RandomAccess>>
      stokes_indexes("stokes_indexes", num_stokes_t::value);
    Kokkos::parallel_for(
      Kokkos::RangePolicy<execution_space>(
        kokkos_work_space,
        0,
        num_stokes_t::value),
      KOKKOS_LAMBDA(const int sto) {
        stokes_indexes(sto) = -1;
      });
    Kokkos::parallel_for(
      Kokkos::RangePolicy<execution_space>(
        kokkos_work_space,
        0,
        jones_stokes_rect.hi[0] - jones_stokes_rect.lo[0] + 1),
      KOKKOS_LAMBDA(const int i) {
        stokes_indexes(static_cast<int>(jones_stokes(i)) - 1) = i;
      });
    auto jones_value_col = jones.template value<Legion::AffineAccessor>();
    auto jones_values =
      jones_value_col.template view<execution_space, LEGION_READ_ONLY>();
    auto jones_weight_col = jones.template weight<Legion::AffineAccessor>();
    auto jones_weights =
      jones_weight_col.template view<execution_space, LEGION_READ_ONLY>();

    // Final CF functions
    auto aterm_stokes_out_col =
      aterm.template stokes_out<Legion::AffineAccessor>();
    auto aterm_stokes_out_values =
      aterm_stokes_out_col.template view<execution_space, LEGION_READ_ONLY>();
    auto aterm_stokes_in_col =
      aterm.template stokes_in<Legion::AffineAccessor>();
    auto aterm_stokes_in_values =
      aterm_stokes_out_col.template view<execution_space, LEGION_READ_ONLY>();
    auto aterm_value_col = aterm.template value<Legion::AffineAccessor>();
    auto aterm_values =
      aterm_value_col.template view<execution_space, LEGION_WRITE_ONLY>();
    auto aterm_weight_col = aterm.template weight<Legion::AffineAccessor>();
    auto aterm_weights =
      aterm_weight_col.template view<execution_space, LEGION_WRITE_ONLY>();
    auto aterm_rect = aterm_value_col.rect();

    // we use hierarchical parallelism here where the thread teams range over
    // the outer dimensions of aterm_rect
    Legion::Rect<index_rank> truncated_aterm_rect;
    for (size_t i = 0; i < index_rank; ++i) {
      truncated_aterm_rect.lo[i] = aterm_rect.lo[i];
      truncated_aterm_rect.hi[i] = aterm_rect.hi[i];
    }
    // thread range of X
    auto x_size = aterm_rect.hi[d_x] - aterm_rect.lo[d_x] + 1;
    // vector range of Y
    auto y_size = aterm_rect.hi[d_y] - aterm_rect.lo[d_y] + 1;

    unsigned dd_blc = d_blc;
    unsigned dd_pa = d_pa;
    unsigned dd_frq = d_frq;
    unsigned dd_sto_out = d_sto_out;
    unsigned dd_sto_in = d_sto_in;

    typedef typename Kokkos::TeamPolicy<execution_space>::member_type
      member_type;
    Kokkos::parallel_for(
      Kokkos::TeamPolicy<execution_space>(
        kokkos_work_space,
        linearized_index_range(truncated_aterm_rect),
        Kokkos::AUTO,
        y_size),
      KOKKOS_LAMBDA(const member_type& team_member) {
        auto pt =
          multidimensional_index_l(
            static_cast<Legion::coord_t>(team_member.league_rank()),
            truncated_aterm_rect);
        auto& blc_l = pt[dd_blc];
        auto& pa_l = pt[dd_pa];
        auto& frq_l = pt[dd_frq];
        auto& sto_out_l = pt[dd_sto_out];
        auto& sto_in_l = pt[dd_sto_in];
        auto ats =
          Kokkos::subview(
            aterm_values,
            blc_l,
            pa_l,
            frq_l,
            sto_out_l,
            sto_in_l,
            Kokkos::ALL,
            Kokkos::ALL);
        Legion::coord_t sto_left_l =
          stokes_indexes(
            static_cast<int>(aterm_stokes_out_values(sto_out_l)) - 1);
        auto left =
          Kokkos::subview(
            jones_values,
            blc_l,
            pa_l,
            frq_l,
            sto_left_l,
            Kokkos::ALL,
            Kokkos::ALL);
        Legion::coord_t sto_right_l =
          stokes_indexes(
            static_cast<int>(aterm_stokes_in_values(sto_in_l)) - 1);
        auto right =
          Kokkos::subview(
            jones_values,
            blc_l,
            pa_l,
            frq_l,
            sto_right_l,
            Kokkos::ALL,
            Kokkos::ALL);
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team_member, x_size),
          [=](const auto x0) {
            // TODO: measure performance impact of using subviews
            auto ats_x = Kokkos::subview(ats, x0, Kokkos::ALL);
            auto left_x = Kokkos::subview(left, x0, Kokkos::ALL);
            auto right_x = Kokkos::subview(right, x0, Kokkos::ALL);
            Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(team_member, y_size),
              [=](const auto y0) {
                ats_x(y0) = left_x(y0) * Kokkos::conj(right_x(y0));
              });
          });
      });
  }

  static void
  preregister_tasks();
};

} // end namespace synthesis
} // end namespace hyperion

#endif // HYPERION_SYNTHESIS_A_TERM_TABLE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
