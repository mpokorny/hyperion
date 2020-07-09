
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
#ifndef HYPERION_SYNTHESIS_A_TERM_AUX1_H_
#define HYPERION_SYNTHESIS_A_TERM_AUX1_H_

#include <hyperion/synthesis/CFTable.h>
#include <cmath>

#define HYPERION_A_TERM_AUX1_AXES                                   \
  CF_BASELINE_CLASS, CF_PARALLACTIC_ANGLE, CF_FREQUENCY, CF_STOKES

#include <hyperion/synthesis/ATermTable.h>
#include <hyperion/synthesis/ATermZernikeModel.h>

namespace hyperion {
namespace synthesis {

class ATermTable;
class ATermZernikeModel;

class HYPERION_EXPORT ATermAux1
  : public CFTable<HYPERION_A_TERM_AUX1_AXES> {
public:

  ATermAux1(
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
    cf_indexing::index_of<CF_BASELINE_CLASS, HYPERION_A_TERM_AUX1_AXES>
    ::type::value;
  static const constexpr unsigned d_pa =
    cf_indexing::index_of<CF_PARALLACTIC_ANGLE, HYPERION_A_TERM_AUX1_AXES>
    ::type::value;
  static const constexpr unsigned d_frq =
    cf_indexing::index_of<CF_FREQUENCY, HYPERION_A_TERM_AUX1_AXES>
    ::type::value;
  static const constexpr unsigned d_sto =
    cf_indexing::index_of<CF_STOKES, HYPERION_A_TERM_AUX1_AXES>
    ::type::value;

  static const constexpr unsigned d_x = index_rank;
  static const constexpr unsigned d_y = d_x + 1;
  static const constexpr unsigned d_power = d_y + 1;
  static const constexpr unsigned ept_rank = d_power + 1;

  static const constexpr Legion::FieldID EPT_X_FID = 88;
  static const constexpr Legion::FieldID EPT_Y_FID = 89;
  static const constexpr char* EPT_X_NAME = "EPT_X";
  static const constexpr char* EPT_Y_NAME = "EPT_Y";
  typedef double ept_t;
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

  struct ComputeEPtsTaskArgs {
    Table::Desc aterm;
    Table::Desc zmodel;
    Table::Desc aux1;
  };

  void
  compute_epts(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const ATermTable& aterm_table,
    const ATermZernikeModel& zmodel,
    const ColumnSpacePartition& partition = ColumnSpacePartition()) const;

  void
  compute_cfs(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const ColumnSpacePartition& partition = ColumnSpacePartition()) const;

//protected:

  static const constexpr char* compute_epts_task_name =
    "ATermAux1::compute_epts_task";

  static Legion::TaskID compute_epts_task_id;

#ifdef HYPERION_USE_KOKKOS
  template <typename execution_space>
  static void
  compute_epts_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* rt) {

    const ComputeEPtsTaskArgs& args =
      *static_cast<const ComputeEPtsTaskArgs*>(task->args);

    std::vector<Table::Desc> descs{args.aterm, args.zmodel, args.aux1};
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

    // parallactic angle column
    auto aterm_tbl = CFPhysicalTable<HYPERION_A_TERM_TABLE_AXES>(pts[0]);
    auto parallactic_angle_col =
      aterm_tbl.parallactic_angle<Legion::AffineAccessor>();
    auto parallactic_angles =
      parallactic_angle_col.view<execution_space, READ_ONLY>();
    typedef decltype(parallactic_angle_col)::value_t pa_t;

    // polynomial function coefficients column
    auto zmodel =
      CFPhysicalTable<HYPERION_A_TERM_ZERNIKE_MODEL_AXES>(pts[1]);
    auto pc_col =
      ATermZernikeModel::PCColumn<Legion::AffineAccessor>(
        *zmodel.column(ATermZernikeModel::PC_NAME).value());
    auto pcs = pc_col.view<execution_space, READ_ONLY>();
    typedef ATermZernikeModel::pc_t pc_t;

    // polynomial function evaluation points columns
    auto aux1_tbl = CFPhysicalTable<HYPERION_A_TERM_AUX1_AXES>(pts[2]);
    auto xpt_col =
      EPtColumn<Legion::AffineAccessor>(*aux1_tbl.column(EPT_X_NAME).value());
    auto xpts = xpt_col.view<execution_space, WRITE_DISCARD>();
    auto ypt_col =
      EPtColumn<Legion::AffineAccessor>(*aux1_tbl.column(EPT_Y_NAME).value());
    auto ypts = ypt_col.view<execution_space, WRITE_DISCARD>();

    Legion::Rect<ept_rank> ept_rect = xpt_col.rect();
    Legion::Rect<ept_rank> full_ept_rect = xpt_col.values().value();
    auto zernike_order = xpts.extent(d_power) - 1;
    
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

        // TODO: measure performance impact of using subviews
        auto powx = Kokkos::subview(xpts, blc, pa, frq, sto, x, y, Kokkos::ALL);
        auto powy = Kokkos::subview(ypts, blc, pa, frq, sto, x, y, Kokkos::ALL);

        // apply parallactic angle rotation
        auto neg_parallactic_angle = -parallactic_angles(pa);
        double cs = std::cos(neg_parallactic_angle);
        double sn = std::sin(neg_parallactic_angle);
        const auto x0 = x - ept_rect.lo[d_x];
        const auto y0 = y - ept_rect.lo[d_y];
        double rx = cs * g_x(x0) - sn * g_y(y0);
        double ry = sn * g_x(x0) + cs * g_y(y0);
        // Fill powx, powy with powers of rx, ry
        //
        // Outside of the unit disk, the function should evaluate to zero,
        // which can be achieved by setting the X and Y vectors to zero.
        ept_t ept0 = ((rx * rx + ry * ry <= 1.0) ? 1.0 : 0.0);
        powx(0) = ept0;
        powy(0) = ept0;
        for (unsigned d = 1; d <= zernike_order; ++d) {
          powx(d) = rx * powx(d - 1);
          powy(d) = ry * powy(d - 1);
        }
      }); 
  }
#else
  static void
  compute_epts_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* rt)
#endif

  static void
  preregister_tasks();

};

} // end namespace synthesis
} // end namespace hyperion

#endif // HYPERION_SYNTHESIS_A_TERM_AUX1_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
