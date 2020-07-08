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

#define HYPERION_A_TERM_TABLE_AXES                                      \
  CF_BASELINE_CLASS, CF_PARALLACTIC_ANGLE, CF_FREQUENCY, CF_STOKES_OUT, CF_STOKES_IN

#include <hyperion/synthesis/ATermAux0.h>
#include <hyperion/synthesis/ATermAux1.h>

namespace hyperion {
namespace synthesis {

class HYPERION_EXPORT ATermTable
  : public CFTable<HYPERION_A_TERM_TABLE_AXES> {
public:

  ATermTable(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::array<Legion::coord_t, 2>& cf_bounds_lo,
    const std::array<Legion::coord_t, 2>& cf_bounds_hi,
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

  ATermTable(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const Legion::coord_t& cf_x_radius,
    const Legion::coord_t& cf_y_radius,
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

  void
  compute_cfs(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<ZCoeff>& zernike_coefficients,
    const ColumnSpacePartition& partition = ColumnSpacePartition()) const;

  static const constexpr char* compute_cfs_task_name =
    "ATermTable::compute_cfs_task";

  static Legion::TaskID compute_cfs_task_id;

  struct ComputeCFsTaskArgs {
    Table::Desc aux1;
    Table::Desc aterm;
  };

#ifdef HYPERION_USE_KOKKOS
  template <typename execution_space>
  static void
  compute_cfs_task(
    const Legion::Task*task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* rt) {

    const ComputeCFsTaskArgs& args =
      *static_cast<const ComputeCFsTaskArgs*>(task->args);

    std::vector<Table::Desc> descs{args.aux1, args.aterm};
    auto ptcrs =
      PhysicalTable::create_many(
        rt,
        descs,
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

    auto kokkos_work_space =
      rt->get_executing_processor(ctx).kokkos_work_space();

    // Stokes CF functions
    auto aux1_tbl = CFPhysicalTable<HYPERION_A_TERM_AUX1_AXES>(pts[0]);
    // we expect to have all of the Stokes index column accessible
    auto aux1_stokes_col = aux1_tbl.stokes<Legion::AffineAccessor>();
    auto aux1_stokes = aux1_stokes_col.view<execution_space, READ_ONLY>();
    auto aux1_stokes_rect = aux1_stokes_col.rect();
    // a map from stokes_t value to Stokes index in aux1
    Kokkos::View<
      Legion::coord_t[num_stokes_t::value],
      execution_space,
      Kokkos::MemoryTraits<Kokkos::RandomAccess>>
      stokes_indexes("stokes_indexes");
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
        aux1_stokes_rect.lo,
        aux1_stokes_rect.hi + 1),
      KOKKOS_LAMBDA(const int i) {
        stokes_indexes(static_cast<int>(aux1_stokes(i)) - 1) = i;
      });
    auto aux1_value_col = aux1_tbl.value<Legion::AffineAccessor>();
    auto aux1_values = aux1_value_col.view<execution_space, READ_ONLY>();
    auto aux1_weight_col = aux1_tbl.weight<Legion::AffineAccessor>();
    auto aux1_weights = aux1_weight_col.view<execution_space, READ_ONLY>();

    // Final CF functions
    auto aterm_tbl = CFPhysicalTable<HYPERION_A_TERM_TABLE_AXES>(pts[1]);
    auto aterm_stokes_out_col = aterm_tbl.stokes_out<Legion::AffineAccessor>();
    auto aterm_stokes_out_values =
      aterm_stokes_out_col.view<execution_space, READ_ONLY>();
    auto aterm_stokes_in_col = aterm_tbl.stokes_in<Legion::AffineAccessor>();
    auto aterm_stokes_in_values =
      aterm_stokes_out_col.view<execution_space, READ_ONLY>();
    auto aterm_value_col = aterm_tbl.value<Legion::AffineAccessor>();
    auto aterm_values = aterm_value_col.view<execution_space, WRITE_ONLY>();
    auto aterm_weight_col = aterm_tbl.weight<Legion::AffineAccessor>();
    auto aterm_weights = aterm_weight_col.view<execution_space, WRITE_ONLY>();
    auto aterm_rect = aterm_value_col.rect();

    // we use hierarchical parallelism here where the thread teams range over
    // the outer dimensions of aterm_rect
    Legion::Rect<5> truncated_aterm_rect;
    for (size_t i = 0; i < 5; ++i) {
      truncated_aterm_rect.lo[i] = aterm_rect.lo[i];
      truncated_aterm_rect.hi[i] = aterm_rect.hi[i];
    }
    // thread range of X
    auto x_size = aterm_rect.hi[5] - aterm_rect.lo[5] + 1;
    // vector range of Y
    auto y_size = aterm_rect.hi[6] - aterm_rect.lo[6] + 1;

    typedef typename Kokkos::TeamPolicy<execution_space>::member_type
      member_type;
    Kokkos::parallel_for(
      Kokkos::TeamPolicy<execution_space>(
        kokkos_work_space,
        linearized_range_size(truncated_aterm_rect),
        Kokkos::AUTO(),
        y_size),
      KOKKOS_LAMBDA(const member_type& team_member) {
        auto pt =
          delinearized_in_range(
            static_cast<Legion::coord_t>(team_member.league_rank()),
            truncated_aterm_rect);
        auto& blc = pt[0];
        auto& pa = pt[1];
        auto& frq = pt[2];
        auto& sto_out = pt[3];
        auto& sto_in = pt[4];
        auto ats =
          Kokkos::subview(
            aterm_values,
            blc,
            pa,
            frq,
            sto_out,
            sto_in,
            Kokkos::ALL,
            Kokkos::ALL);
        Legion::coord_t sto_left =
          stokes_indexes(
            static_cast<int>(aterm_stokes_out_values(sto_out)) - 1);
        auto left =
          Kokkos::subview(
            aux1_values,
            blc,
            pa,
            frq,
            sto_left,
            Kokkos::ALL,
            Kokkos::ALL);
        Legion::coord_t sto_right =
          stokes_indexes(
            static_cast<int>(aterm_stokes_in_values(sto_in)) - 1);
        auto right =
          Kokkos::subview(
            aux1_values,
            blc,
            pa,
            frq,
            sto_right,
            Kokkos::ALL,
            Kokkos::ALL);
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team_member, x_size),
          [=](const auto x0) {
            const auto x = x0 + aterm_rect.lo[5];
            auto ats_x = Kokkos::subview(ats, x, Kokkos::ALL);
            auto left_x = Kokkos::subview(left, x, Kokkos::ALL);
            auto right_x = Kokkos::subview(right, x, Kokkos::ALL);
            Kokkos::parallel_for(
              Kokkos::TeamVectorRange(team_member, y_size),
              [=](const auto y0) {
                const auto y = y0 + aterm_rect.lo[6];
                ats_x(y) = left_x(y) * Kokkos::conj(right_x(y));
              });
          });
      });
#if 0
    // Kokkos MDRangePolicy has max rank of 6, so we put the inner loops into
    // the kernel
    Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<5>, execution_space>(
        rt->get_executing_processor(ctx).kokkos_work_space(),
        rect_lo(truncated_aterm_rect),
        rect_hi(truncated_aterm_rect)),
      KOKKOS_LAMBDA(
        const Legion::coord_t& blc,
        const Legion::coord_t& pa,
        const Legion::coord_t& frq,
        const Legion::coord_t& sto_out,
        const Legion::coord_t& sto_in) {
        Legion::coord_t sto_left =
          stokes_indexes(
            static_cast<int>(aterm_stokes_out_values(sto_out)) - 1);
        Legion::coord_t sto_right =
          stokes_indexes(
            static_cast<int>(aterm_stokes_in_values(sto_in)) - 1);
        for (Legion::coord_t x = aterm_rect.lo[5]; x <= aterm_rect.hi[5]; ++x)
          for (Legion::coord_t y = aterm_rect.lo[6]; x <= aterm_rect.lo[6]; ++y)
            aterm_values(blc, pa, frq, sto_out, sto_in, x, y) =
              aux1_values(blc, pa, frq, sto_left, x, y) *
              Kokkos::conj(aux1_values(blc, pa, frq, sto_right, x, y));
      });
#endif
  }
#endif // HYPERION_USE_KOKKOS

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
