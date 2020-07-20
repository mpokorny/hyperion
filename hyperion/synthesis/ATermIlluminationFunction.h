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
#ifndef HYPERION_SYNTHESIS_A_TERM_ILLUMINATION_FUNCTION_H_
#define HYPERION_SYNTHESIS_A_TERM_ILLUMINATION_FUNCTION_H_

#include <hyperion/synthesis/CFTable.h>
#include <cmath>

#define HYPERION_A_TERM_ILLUMINATION_FUNCTION_AXES                  \
  CF_BASELINE_CLASS, CF_PARALLACTIC_ANGLE, CF_FREQUENCY, CF_STOKES

#include <hyperion/synthesis/ATermZernikeModel.h>
#include <fftw3.h>

namespace hyperion {
namespace synthesis {

/**
 * Helper table for ATermTable. For aperture illumination function values on a
 * grid, with dependence on baseline class, parallactic angle, frequency, and
 * Stokes value.
 */
class HYPERION_EXPORT ATermIlluminationFunction
  : public CFTable<HYPERION_A_TERM_ILLUMINATION_FUNCTION_AXES> {
public:

  ATermIlluminationFunction(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const Legion::Rect<2>& cf_bounds,
    unsigned zernike_order,
    const std::vector<typename cf_table_axis<CF_BASELINE_CLASS>::type>&
      baseline_classes,
    const std::vector<typename cf_table_axis<CF_PARALLACTIC_ANGLE>::type>&
      parallactic_angles,
     const std::vector<typename cf_table_axis<CF_FREQUENCY>::type>&
      frequencies,
    const std::vector<typename cf_table_axis<CF_STOKES>::type>&
      stokes_values);

  static const constexpr unsigned d_blc =
    cf_indexing::index_of<
      CF_BASELINE_CLASS,
      HYPERION_A_TERM_ILLUMINATION_FUNCTION_AXES>
    ::type::value;
  static const constexpr unsigned d_pa =
    cf_indexing::index_of<
      CF_PARALLACTIC_ANGLE,
      HYPERION_A_TERM_ILLUMINATION_FUNCTION_AXES>
    ::type::value;
  static const constexpr unsigned d_frq =
    cf_indexing::index_of<
      CF_FREQUENCY,
      HYPERION_A_TERM_ILLUMINATION_FUNCTION_AXES>
    ::type::value;
  static const constexpr unsigned d_sto =
    cf_indexing::index_of<
      CF_STOKES,
      HYPERION_A_TERM_ILLUMINATION_FUNCTION_AXES>
    ::type::value;

  static const constexpr unsigned d_x = index_rank;
  static const constexpr unsigned d_y = d_x + 1;
  static const constexpr unsigned d_power = d_y + 1;
  static const constexpr unsigned ept_rank = d_power + 1;

  /**
   * column of function evaluation point values on the grid
   *
   * This exists to provide for the evaluation of rotated functions
   */
  static const constexpr Legion::FieldID EPT_X_FID = 88;
  static const constexpr Legion::FieldID EPT_Y_FID = 89;
  static const constexpr char* EPT_X_NAME = "EPT_X";
  static const constexpr char* EPT_Y_NAME = "EPT_Y";
  typedef CFTableBase::cf_fp_t ept_t;
  template <Legion::PrivilegeMode MODE, bool CHECK_BOUNDS=HYPERION_CHECK_BOUNDS>
  using ept_accessor_t =
    Legion::FieldAccessor<
      MODE,
      ept_t,
      ept_rank,
      Legion::coord_t,
      Legion::AffineAccessor<ept_t, ept_rank, Legion::coord_t>,
      CHECK_BOUNDS>;
  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  using EPtColumn =
    PhysicalColumnTD<
    ValueType<ept_t>::DataType,
    index_rank,
    ept_rank,
    A,
    COORD_T>;

  /**
   * compute the values of the function evaluation points column
   */
  void
  compute_epts(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const ColumnSpacePartition& partition = ColumnSpacePartition()) const;

  /**
   * compute the values of the aperture illumination function column
   **/
  void
  compute_jones(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const ATermZernikeModel& zmodel,
    const ColumnSpacePartition& partition = ColumnSpacePartition(),
    unsigned fftw_flags = FFTW_MEASURE,
    double fftw_timelimit = 5.0) const;

//protected:

  static const constexpr char* compute_epts_task_name =
    "ATermIlluminationFunction::compute_epts_task";

  static Legion::TaskID compute_epts_task_id;

#ifdef HYPERION_USE_KOKKOS
  template <typename execution_space>
  static void
  compute_epts_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* rt) {

    const Table::Desc& tdesc = *static_cast<const Table::Desc*>(task->args);

    auto ptcr =
      PhysicalTable::create(
        rt,
        tdesc,
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

    auto aif = CFPhysicalTable<HYPERION_A_TERM_ILLUMINATION_FUNCTION_AXES>(pt);

    // parallactic angle column
    auto parallactic_angle_col =
      aif.parallactic_angle<Legion::AffineAccessor>();
    auto parallactic_angles =
      parallactic_angle_col.view<execution_space, READ_ONLY>();
    typedef decltype(parallactic_angle_col)::value_t pa_t;

    // polynomial function evaluation points columns
    auto xpt_col =
      EPtColumn<Legion::AffineAccessor>(*aif.column(EPT_X_NAME).value());
    auto xpts = xpt_col.view<execution_space, WRITE_DISCARD>();
    auto ypt_col =
      EPtColumn<Legion::AffineAccessor>(*aif.column(EPT_Y_NAME).value());
    auto ypts = ypt_col.view<execution_space, WRITE_DISCARD>();

    Legion::Rect<ept_rank> ept_rect = xpt_col.rect();
    Legion::Rect<ept_rank> full_ept_rect = xpt_col.values().value();

    auto kokkos_work_space =
      rt->get_executing_processor(ctx).kokkos_work_space();

    // compute grid values for zero parallactic angle
    //
    // we support (potentially different) block partitions of X and Y axes,
    // but only allow squares for full XY plane
    assert(full_ept_rect.hi[d_x] - full_ept_rect.lo[d_x]
           == full_ept_rect.hi[d_y] - full_ept_rect.lo[d_y]);
    // the XY origin must lie in the middle of full_ept_rect
    assert((full_ept_rect.hi[d_x] - full_ept_rect.lo[d_x] + 1) / 2
           + full_ept_rect.lo[d_x] == 0);
    assert((full_ept_rect.hi[d_y] - full_ept_rect.lo[d_y] + 1) / 2
           + full_ept_rect.lo[d_y] == 0);
    Kokkos::View<ept_t*, execution_space>
      g_x(
        Kokkos::ViewAllocateWithoutInitializing("xvals"),
        ept_rect.hi[d_x] - ept_rect.lo[d_x] + 1);
    Kokkos::View<ept_t*, execution_space>
      g_y(
        Kokkos::ViewAllocateWithoutInitializing("yvals"),
        ept_rect.hi[d_y] - ept_rect.lo[d_y] + 1);
    const ept_t step =
      (ept_t)2.0 / (full_ept_rect.hi[d_x] - full_ept_rect.lo[d_x] + 1);
    const ept_t offset_x =
      (ept_t)-1.0 + step / (ept_t)2.0
      + (ept_rect.lo[d_x] - full_ept_rect.lo[d_x]) * step;
    const ept_t offset_y =
      (ept_t)-1.0 + step / (ept_t)2.0
      + (ept_rect.lo[2] - full_ept_rect.lo[2]) * step;
    Kokkos::parallel_for(
      "init_xvals",
      Kokkos::RangePolicy<execution_space>(kokkos_work_space, 0, g_x.extent(0)),
      KOKKOS_LAMBDA(const int i) {
        g_x(i) = i * step + offset_x;
      });
    Kokkos::parallel_for(
      "init_yvals",
      Kokkos::RangePolicy<execution_space>(kokkos_work_space, 0, g_y.extent(0)),
      KOKKOS_LAMBDA(const int i) {
        g_y(i) = i * step + offset_y;
      });

    // compute polynomial evaluation points with function rotation
    Kokkos::parallel_for(
      "compute_polynomial_evaluation_points",
      Kokkos::MDRangePolicy<Kokkos::Rank<ept_rank - 1>, execution_space>(
        kokkos_work_space,
        {ept_rect.lo[d_blc], ept_rect.lo[d_pa], ept_rect.lo[d_frq],
         ept_rect.lo[d_sto], ept_rect.lo[d_x], ept_rect.lo[d_y]},
        {ept_rect.hi[d_blc] + 1, ept_rect.hi[d_pa] + 1, ept_rect.hi[d_frq] + 1,
         ept_rect.hi[d_sto] + 1, ept_rect.hi[d_x] + 1, ept_rect.hi[d_y] + 1}),

      KOKKOS_LAMBDA(
        const Legion::coord_t& blc,
        const Legion::coord_t& pa,
        const Legion::coord_t& frq,
        const Legion::coord_t& sto,
        const Legion::coord_t& x,
        const Legion::coord_t& y) {

        // apply parallactic angle rotation
        const auto neg_parallactic_angle = -parallactic_angles(pa);
        const auto cs = std::cos(neg_parallactic_angle);
        const auto sn = std::sin(neg_parallactic_angle);
        const auto x0 = x - ept_rect.lo[d_x];
        const auto y0 = y - ept_rect.lo[d_y];
        const auto rx = cs * g_x(x0) - sn * g_y(y0);
        const auto ry = sn * g_x(x0) + cs * g_y(y0);
        // Outside of the unit disk, the function should evaluate to zero,
        // which can be achieved by setting the X and Y vectors to zero.
        ept_t ept0 = ((rx * rx + ry * ry <= 1.0) ? 1.0 : 0.0);
        xpts(blc, pa, frq, sto, x, y, 0) = ept0;
        ypts(blc, pa, frq, sto, x, y, 0) = ept0;
        xpts(blc, pa, frq, sto, x, y, 1) = rx * ept0;
        ypts(blc, pa, frq, sto, x, y, 1) = ry * ept0;
      });
  }
#else // !HYPERION_USE_KOKKOS
  static void
  compute_epts_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* rt);
#endif // HYPERION_USE_KOKKOS

  static const constexpr char* compute_aifs_task_name =
    "ATermIlluminationFunction::compute_aifs_task";

  static Legion::TaskID compute_aifs_task_id;

  struct ComputeAIFsTaskArgs {
    Table::Desc zmodel;
    Table::Desc aif;
  };

#ifdef HYPERION_USE_KOKKOS
  template <typename execution_space>
  static void
  compute_aifs_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* rt) {

    const ComputeAIFsTaskArgs& args =
      *static_cast<ComputeAIFsTaskArgs*>(task->args);
    std::vector<Table::Desc> descs{args.zmodel, args.aif};

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

    auto zmodel = CFPhysicalTable<HYPERION_A_TERM_ZERNIKE_MODEL_AXES>(pts[0]);
    auto aif =
      CFPhysicalTable<HYPERION_A_TERM_ILLUMINATION_FUNCTION_AXES>(pts[1]);

    // polynomial function coefficients column
    auto pc_col =
      ATermZernikeModel::PCColumn<Legion::AffineAccessor>(
        *zmodel.column(ATermZernikeModel::PC_NAME).value());
    auto pcs = pc_col.view<execution_space, READ_ONLY>();

    // polynomial function evaluation points columns
    auto xpt_col =
      EPtColumn<Legion::AffineAccessor>(*aif.column(EPT_X_NAME).value());
    auto xpts = xpt_col.view<execution_space, READ_ONLY>();
    auto ypt_col =
      EPtColumn<Legion::AffineAccessor>(*aif.column(EPT_Y_NAME).value());
    auto ypts = ypt_col.view<execution_space, READ_ONLY>();

    // polynomial function values column
    auto value_col = aif.value<Legion::AffineAccessor>();
    auto value_rect = value_col.rect();
    auto values = value_col.view<execution_space, WRITE_DISCARD>();

    // CUDA compilation fails without the following redundant definitions. Note
    // that similar usage in compute_epts_task works. TODO: remove these
    unsigned dd_blc = d_blc;
    unsigned dd_pa = d_pa;
    unsigned dd_frq = d_frq;
    unsigned dd_sto = d_sto;
    unsigned dd_x = d_x;
    unsigned dd_y = d_y;
    auto kokkos_work_space =
      rt->get_executing_processor(ctx).kokkos_work_space();
    typedef typename Kokkos::TeamPolicy<execution_space>::member_type
      member_type;
    typedef Kokkos::View<
      ATermZernikeModel::pc_t*,
      typename execution_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>> shared_pc_1d;
    Kokkos::parallel_for(
      Kokkos::TeamPolicy<execution_space>(
        kokkos_work_space,
        linearized_index_range(value_rect),
        Kokkos::AUTO())
      .set_scratch_size(
        0,
        Kokkos::PerTeam(
          (zernike_max_order::value + 1) * sizeof(ATermZernikeModel::pc_t))),
      KOKKOS_LAMBDA(const member_type& team_member) {
        auto pt =
          multidimensional_index(
            static_cast<Legion::coord_t>(team_member.league_rank()),
            value_rect);
        auto& blc = pt[dd_blc];
        auto& pa = pt[dd_pa];
        auto& frq = pt[dd_frq];
        auto& sto = pt[dd_sto];
        auto& x = pt[dd_x];
        auto& y = pt[dd_y];
        auto xpt = Kokkos::subview(xpts, blc, pa, frq, sto, x, y, Kokkos::ALL);
        auto ypt = Kokkos::subview(ypts, blc, pa, frq, sto, x, y, Kokkos::ALL);
        auto pc = Kokkos::subview(pcs, blc, frq, sto, Kokkos::ALL, Kokkos::ALL);
        auto tmp = shared_pc_1d(team_member.team_scratch(0), pc.extent(0));
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team_member, pc.extent(0)),
          [=](const int& i) {
            tmp(i) = (ATermZernikeModel::pc_t)0.0;
            for (int j = pc.extent(1) - 1; j > 0; --j)
              tmp(i) = (tmp(i) + pc(i, j)) * ypt(1);
            tmp(i) = (tmp(i) + pc(i, 0)) * ypt(0);
          });
        team_member.team_barrier();
        auto& v = values(blc, pa, frq, sto, x, y);
        v = (ATermZernikeModel::pc_t)0.0;
        for (int i = pc.extent(0) - 1; i > 0; --i)
          v = (v + tmp(i)) * xpt(1);
        v = (v + tmp(0)) * xpt(0);
      });
  }
#else // !HYPERION_USE_KOKKOS
  static void
  compute_aifs_task(
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

#endif // HYPERION_SYNTHESIS_A_TERM_ILLUMINATION_FUNCTION_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
