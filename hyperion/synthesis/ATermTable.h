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
#define HYPERION_SYNTHESIS_N_TERM_TABLE_H_

#include <hyperion/synthesis/CFTable.h>
#include <hyperion/synthesis/Zernike.h>

#include <array>
#include <cmath>

#ifdef HYPERION_USE_KOKKOS_KERNELS
# include <KokkosBlas3_gemm.hpp>
#endif

#define HYPERION_A_TERM_TABLE_AXES                                  \
  CF_BASELINE_CLASS, CF_PARALLACTIC_ANGLE, CF_STOKES, CF_FREQUENCY

namespace hyperion {
namespace synthesis {

class HYPERION_EXPORT ATermTable
  : public CFTable<HYPERION_A_TERM_TABLE_AXES> {
public:

  static const constexpr Legion::FieldID ZC_FID = 84;
  static const constexpr char* ZC_NAME = "ZERNIKE_COEFFICIENTS";
#ifdef HYPERION_USE_KOKKOS
  typedef Kokkos::complex<double> zc_t;
#else
  typedef std::complex<double> zc_t;
#endif

  // acceptable fractional error in frequency value for "exact" zernike
  // expansion coefficient matching (i.e, stop searching for a closer frequency
  // value when one has been found within this tolerance)
  static const constexpr float zc_exact_frequency_tolerance = 0.01;

  struct ZCoeff {
    typename cf_table_axis<CF_BASELINE_CLASS>::type baseline_class;
    typename cf_table_axis<CF_FREQUENCY>::type frequency;
    typename cf_table_axis<CF_STOKES>::type stokes;
    int m;
    unsigned n;
    zc_t coefficient;
  };

  static const constexpr Legion::FieldID PC_FID = 84;
  static const constexpr char* PC_NAME = "POLYNOMIAL_COEFFICIENTS";
  typedef zc_t pc_t;

  ATermTable(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::array<Legion::coord_t, 2>& cf_bounds_lo,
    const std::array<Legion::coord_t, 2>& cf_bounds_hi,
    unsigned zernike_order,
    const std::vector<ZCoeff>& zernike_coefficients,
    const std::vector<typename cf_table_axis<CF_BASELINE_CLASS>::type>&
    baseline_classes,
    const std::vector<typename cf_table_axis<CF_PARALLACTIC_ANGLE>::type>&
    parallactic_angles,
    const std::vector<typename cf_table_axis<CF_STOKES>::type>&
    stokes_values,
    const std::vector<typename cf_table_axis<CF_FREQUENCY>::type>&
    frequencies);

  ATermTable(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const Legion::coord_t& cf_x_radius,
    const Legion::coord_t& cf_y_radius,
    unsigned zernike_order,
    const std::vector<ZCoeff>& zernike_coefficients,
    const std::vector<typename cf_table_axis<CF_BASELINE_CLASS>::type>&
    baseline_classes,
    const std::vector<typename cf_table_axis<CF_PARALLACTIC_ANGLE>::type>&
    parallactic_angles,
    const std::vector<typename cf_table_axis<CF_STOKES>::type>&
    stokes_values,
    const std::vector<typename cf_table_axis<CF_FREQUENCY>::type>&
    frequencies);

  void
  compute_cfs(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const ColumnSpacePartition& partition = ColumnSpacePartition()) const;

protected:

  void
  create_zc_region(
    Legion::Context ctx,
    Legion::Runtime* rt,
    unsigned zernike_order,
    const std::vector<ZCoeff>& zernike_coefficients,
    const std::vector<typename cf_table_axis<CF_BASELINE_CLASS>::type>&
    baseline_classes,
    const std::vector<typename cf_table_axis<CF_STOKES>::type>&
    stokes_values,
    const std::vector<typename cf_table_axis<CF_FREQUENCY>::type>&
    frequencies);

  void
  create_pc_region(
    Legion::Context ctx,
    Legion::Runtime* rt,
    unsigned zernike_order,
    unsigned num_baseline_classes,
    unsigned num_stokes_values,
    unsigned num_frequencies);

public:

  unsigned m_zernike_order;

  Legion::LogicalRegion m_zc_region;

  Legion::LogicalRegion m_pc_region;

  static const constexpr char* compute_cfs_task_name =
    "ATermTable::compute_cfs_task";

  static Legion::TaskID compute_cfs_task_id;

#ifdef HYPERION_USE_KOKKOS
  template <typename execution_space>
  static void
  compute_cfs_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* rt) {

    const Table::Desc& tdesc = *static_cast<const Table::Desc*>(task->args);

    // polynomial coefficients region (the array Z, as described elsewhere),
    // computed for each value of (baseline_class, stokes_value, frequency)
    const Legion::FieldAccessor<
      READ_ONLY,
      pc_t,
      5,
      coord_t,
      Legion::AffineAccessor<pc_t, 5, coord_t>> pc_acc(regions[0], PC_FID);
    Kokkos::View<const pc_t*****, execution_space> pcoeffs = pc_acc.accessor;
    Legion::Rect<5> pcoeffs_rect =
      rt->get_index_space_domain(task->regions[0].region.get_index_space());
    auto zernike_order = pcoeffs.extent(4) - 1;

    // ATermTable physical instance
    auto ptcr =
      PhysicalTable::create(
        rt,
        tdesc,
        task->regions.begin() + 1,
        task->regions.end(),
        regions.begin() + 1,
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

    auto tbl = CFPhysicalTable<HYPERION_A_TERM_TABLE_AXES>(pt);

    // baseline class column
    auto baseline_class_col = tbl.baseline_class<Legion::AffineAccessor>();
    auto baseline_class_rect = baseline_class_col.rect();
    auto baseline_class_extent =
      baseline_class_rect.hi[0] - baseline_class_rect.lo[0] + 1;
    auto baseline_classes =
      baseline_class_col.view<execution_space, READ_ONLY>();
    typedef decltype(baseline_class_col)::value_t bl_t;

    // frequency column
    auto frequency_col = tbl.frequency<Legion::AffineAccessor>();
    auto frequency_rect = frequency_col.rect();
    auto frequency_extent =
      frequency_rect.hi[0] - frequency_rect.lo[0] + 1;
    auto frequencies = frequency_col.view<execution_space, READ_ONLY>();
    typedef decltype(frequency_col)::value_t frq_t;

    // parallactic angle column
    auto parallactic_angle_col = tbl.parallactic_angle<Legion::AffineAccessor>();
    auto parallactic_angle_rect = parallactic_angle_col.rect();
    auto parallactic_angle_extent =
      parallactic_angle_rect.hi[0] - parallactic_angle_rect.lo[0] + 1;
    auto parallactic_angles =
      parallactic_angle_col.view<execution_space, READ_ONLY>();
    typedef decltype(parallactic_angle_col)::value_t pa_t;

    // stokes column
    auto stokes_value_col = tbl.stokes<Legion::AffineAccessor>();
    auto stokes_value_rect = stokes_value_col.rect();
    auto stokes_value_extent =
      stokes_value_rect.hi[0] - stokes_value_rect.lo[0] + 1;
    auto stokes_values = stokes_value_col.view<execution_space, READ_ONLY>();
    typedef decltype(stokes_value_col)::value_t sto_t;

    // A-term cf values column
    auto cf_value_col = tbl.value<Legion::AffineAccessor>();
    auto cf_value_rect = cf_value_col.rect();
    auto cf_values = cf_value_col.view<execution_space, WRITE_DISCARD>();

    // initialize cf_values values to 0; not sure that it's necessary, it depends
    // on whether GEMM accesses the array even when beta is 0
    {
      Kokkos::MDRangePolicy<Kokkos::Rank<6>, execution_space> range(
        rt->get_executing_processor(ctx).kokkos_work_space(),
        rect_lo(cf_value_rect),
        rect_hi(cf_value_rect));
      Kokkos::parallel_for(
        range,
        KOKKOS_LAMBDA(
          Legion::coord_t blc,
          Legion::coord_t pa,
          Legion::coord_t sto,
          Legion::coord_t frq,
          Legion::coord_t x,
          Legion::coord_t y) {
          cf_values(blc, pa, sto, frq, x, y) = 0;
        });
    }

    // the matrices X and Y (for each parallactic angle); these need to have
    // complex values in order to use GEMM for the matrix multiplication
    // implementation
    Kokkos::View<pc_t***, execution_space> xp(
      Kokkos::ViewAllocateWithoutInitializing("xp"),
      parallactic_angle_extent,
      zernike_order + 1,
      cf_value_rect.hi[4] - cf_value_rect.lo[4] + 1);
    Kokkos::View<pc_t***, execution_space> yp(
      Kokkos::ViewAllocateWithoutInitializing("yp"),
      parallactic_angle_extent,
      zernike_order + 1,
      cf_value_rect.hi[5] - cf_value_rect.lo[5] + 1);
    {
      assert(xp.extent(2) == yp.extent(2)); // squares only
      Kokkos::View<double*, execution_space>
        g(Kokkos::ViewAllocateWithoutInitializing("gridvals"), xp.extent(2));
      const double step = 2.0 / xp.extent(2);
      const double offset = -1.0 + step / 2.0;
      Kokkos::parallel_for(
        "init_grid",
        Kokkos::RangePolicy<execution_space>(0, xp.extent(2)),
        KOKKOS_LAMBDA(const int i) {
          g(i) = i * step + offset;
        });
    
      Kokkos::parallel_for(
        "compute_grid_powers",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>, execution_space>(
          rt->get_executing_processor(ctx).kokkos_work_space(),
          {0, 0, 0},
          {parallactic_angle_extent, (long)xp.extent(0), (long)yp.extent(0)}),

        KOKKOS_LAMBDA(
          const Legion::coord_t& pa0,
          const Legion::coord_t& x0,
          const Legion::coord_t& y0) {

          auto powx = Kokkos::subview(xp, pa0, Kokkos::ALL, x0);
          auto powy = Kokkos::subview(yp, pa0, Kokkos::ALL, y0);

          // apply parallactic angle rotation
          auto neg_parallactic_angle =
            -parallactic_angles(pa0 + parallactic_angle_rect.lo[0]);
          double cs = std::cos(neg_parallactic_angle);
          double sn = std::sin(neg_parallactic_angle);
          double rx = cs * g(x0) - sn * g(y0);
          double ry = sn * g(x0) + cs * g(y0);
          // Outside of the unit disk, the function should evaluate to zero,
          // which can be achieved by setting the X and Y vectors to zero.
          // Recall that xp and yp were created without value initialization.
          powx(0) = powy(0) = ((rx * rx + ry * ry <= 1.0) ? 1.0 : 0.0);
          for (unsigned d = 1; d <= zernike_order; ++d) {
            powx(d) = rx * powx(d - 1).real();
            powy(d) = ry * powy(d - 1).real();
          }
        });
    }

    // do X^T Z Y products
    {
      // do multiplication as Q = Z Y, result = X^T Q; need a scratch array for
      // Q
      Kokkos::View<pc_t******, execution_space> zy(
        "gemm_scratch",
        baseline_class_extent,
        parallactic_angle_extent,
        stokes_value_extent,
        frequency_extent,
        yp.extent(1),
        yp.extent(2));
      for (Legion::coord_t blc0 = 0; blc0 < baseline_class_extent; ++blc0)
        for (Legion::coord_t pa0 = 0; pa0 < parallactic_angle_extent; ++pa0)
          for (Legion::coord_t sto0 = 0; sto0 < stokes_value_extent; ++sto0)
            for (Legion::coord_t frq0 = 0; frq0 < frequency_extent; ++frq0) {
              auto Z =
                Kokkos::subview(
                  pcoeffs,
                  blc0 + pcoeffs_rect.lo[0],
                  sto0 + pcoeffs_rect.lo[1],
                  frq0 + pcoeffs_rect.lo[2],
                  Kokkos::ALL,
                  Kokkos::ALL);
              auto X = Kokkos::subview(xp, pa0, Kokkos::ALL, Kokkos::ALL);
              auto Y = Kokkos::subview(yp, pa0, Kokkos::ALL, Kokkos::ALL);
              auto Q =
                Kokkos::subview(
                  zy,
                  blc0,
                  pa0,
                  sto0,
                  frq0,
                  Kokkos::ALL,
                  Kokkos::ALL);
              auto CF =
                Kokkos::subview(
                  cf_values,
                  blc0 + cf_value_rect.lo[0],
                  pa0 + cf_value_rect.lo[1],
                  sto0 + cf_value_rect.lo[2],
                  frq0 + cf_value_rect.lo[3],
                  Kokkos::ALL,
                  Kokkos::ALL);
#ifdef HYPERION_USE_KOKKOS_KERNELS
              KokkosBlas::gemm("N", "N", 1.0, Z, Y, 0.0, Q);
              KokkosBlas::gemm("T", "N", 1.0, X, Q, 0.0, CF);
#else // !HYPERION_USE_KOKKOS_KERNELS
              typedef typename Kokkos::TeamPolicy<execution_space>::member_type
                member_type;
              // Q = Z Y
              Kokkos::parallel_for(
                Kokkos::TeamPolicy<execution_space>(Z.extent(0), Kokkos::AUTO()),
                KOKKOS_LAMBDA(const member_type& team_member) {
                  Legion::coord_t i = team_member.league_rank();
                  Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team_member, Y.extent(1)),
                    [=](Legion::coord_t j) {
                      Q(i, j) = 0.0;
                      Kokkos::parallel_reduce(
                        Kokkos::TeamVectorRange(team_member, Y.extent(0)),
                        [=](Legion::coord_t k, pc_t& sum) {
                          sum += Z(i, k) * Y(k, j);
                        },
                        Q(i, j));
                    });
                });
              // CF = X^T Q
              Kokkos::parallel_for(
                Kokkos::TeamPolicy<execution_space>(X.extent(1), Kokkos::AUTO()),
                KOKKOS_LAMBDA(const member_type& team_member) {
                  Legion::coord_t i = team_member.league_rank();
                  Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team_member, Q.extent(1)),
                    [=](Legion::coord_t j) {
                      CF(i, j) = 0.0;
                      Kokkos::parallel_reduce(
                        Kokkos::TeamVectorRange(team_member, X.extent(0)),
                        [=](Legion::coord_t k, pc_t& sum) {
                          sum += X(k, i) * Q(k, j);                          
                        },
                        CF(i, j));
                    });
                });
#endif // HYPERION_USE_KOKKOS_KERNELS
            };
    }
  }
#else // !HYPERION_USE_KOKKOS
  static void
  compute_cfs_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* rt);
#endif // HYPERION_USE_KOKKOS


  // Let the sequence {z(0,0), z(-1,1), z(1,1), ..., z(N,N-2), z(N,N)} represent
  // a linear combination of Zernike polynomials Z_(m, n) with coefficients z(m,
  // n), up to Zernike order N. This sequence represents the expansion of a
  // function F in the Zernike basis. To evaluate F on a grid, we will express
  // this sequence as a polynomial function of two variables, x and y. The idea
  // is to create a matrix Z, where Z(i, j) is the coefficient of x^i y^j term,
  // and then to evaluate the function by forming the vector-matrix-vector
  // product X^T Z Y, where X is the vector for a given value x with the
  // elements x^0, x^1, x^2, ... x^N (similar definition for Y). This product
  // expression can be generalized into a triple matrix product form to get the
  // polynomial function value at multiple values of (x, y) in a single
  // expression.
  static const constexpr char* compute_pcs_task_name =
    "ATermTable::compute_pcs_task";

  static Legion::TaskID compute_pcs_task_id;

#ifdef HYPERION_USE_KOKKOS
  template <typename execution_space>
  static void
  compute_pcs_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* rt) {

    // Zernike coefficients
    const Legion::FieldAccessor<
      READ_ONLY,
      zc_t,
      4,
      coord_t,
      Legion::AffineAccessor<zc_t, 4, coord_t>> zc_acc(regions[0], ZC_FID);
    Kokkos::View<const zc_t****, execution_space> zcs_array = zc_acc.accessor;
    Legion::Rect<4> rect =
      rt->get_index_space_domain(task->regions[0].region.get_index_space());

    // polynomial expansion coefficients
    const Legion::FieldAccessor<
      WRITE_ONLY,
      pc_t,
      5,
      coord_t,
      Legion::AffineAccessor<pc_t, 5, coord_t>> pc_acc(regions[1], PC_FID);
    Kokkos::View<const pc_t*****, execution_space> pcs_array = pc_acc.accessor;

    Kokkos::parallel_for(
      "compute_pcs",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>, execution_space>(
        rt->get_executing_processor(ctx).kokkos_work_space(),
        {rect.lo[0], rect.lo[1], rect.lo[2]},
        {rect.hi[0] + 1, rect.hi[1] + 1, rect.hi[2] + 1}),
      KOKKOS_LAMBDA(
        const Legion::coord_t& blc,
        const Legion::coord_t& pa,
        const Legion::coord_t& sto) {

        auto zcs = Kokkos::subview(zcs_array, blc, pa, sto, Kokkos::ALL);
        auto pcs =
          Kokkos::subview(pcs_array, blc, pa, sto, Kokkos::ALL, Kokkos::ALL);
        switch (zcs.extent(3) - 1) {
#define ZEXP(N)                                       \
          case N:                                     \
            zernike_basis<zc_t, N>::expand(zcs, pcs); \
            break
          ZEXP(0);
          ZEXP(1);
          ZEXP(2);
          ZEXP(3);
          ZEXP(4);
          ZEXP(5);
          ZEXP(6);
          ZEXP(7);
          ZEXP(8);
          ZEXP(9);
          ZEXP(10);
#undef ZEXP
        default:
          assert(false);
          break;
        }
      });    
  }
#else // !HYPERION_USE_KOKKOS
  static void
  compute_pcs_task(
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
