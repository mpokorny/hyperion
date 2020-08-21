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
#ifndef HYPERION_SYNTHESIS_A_TERM_ZERNIKE_MODEL_H_
#define HYPERION_SYNTHESIS_A_TERM_ZERNIKE_MODEL_H_

#include <hyperion/synthesis/CFTable.h>
#include <hyperion/synthesis/Zernike.h>

#include <cmath>

#define HYPERION_A_TERM_ZERNIKE_MODEL_AXES      \
  CF_BASELINE_CLASS, CF_FREQUENCY, CF_STOKES

namespace hyperion {
namespace synthesis {

typedef complex<CFTableBase::cf_fp_t> zc_t;/**< Zernike coefficient type */

/**
 * self-described Zernike expansion coefficient value
 */
struct HYPERION_EXPORT ZCoeff {
  /** baseline class identifier (label) */
  typename cf_table_axis<CF_BASELINE_CLASS>::type baseline_class;
  /** frequency (label) */
  typename cf_table_axis<CF_FREQUENCY>::type frequency;
  /** Stokes component (label) */
  stokes_t stokes;
  /** Zernike coefficient M value (label) */
  int m;
  /** Zernike coefficient N value (label) */
  unsigned n;
  /** value of the Zernike expansion coefficient with the given labels */
  zc_t coefficient;
};

/**
 * Helper table for use by ATermTable. For Zernike model parameters and
 * model polynomial, without dependence on parallactic angle.
 *
 * Although this class is a sub-class of CFTable, that has been done only for
 * convenience: the value and weight columns are unused
 */
class HYPERION_EXPORT ATermZernikeModel
  : public CFTable<HYPERION_A_TERM_ZERNIKE_MODEL_AXES> {
public:

  /**
   * ATermZernikeModel constructor
   *
   * @param ctx Legion Context
   * @param rt Legion Runtime pointer
   * @param zernike_coefficients vector of Zernike expansion coefficients
   *                             (values may be in any order)
   * @param baseline_classes baseline class axis values
   * @param frequencies frequency axis values
   * @param stokes_values Stokes axis values
   */
  ATermZernikeModel(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<ZCoeff>& zernike_coefficients,
    const std::vector<typename cf_table_axis<CF_BASELINE_CLASS>::type>&
      baseline_classes,
     const std::vector<typename cf_table_axis<CF_FREQUENCY>::type>&
      frequencies,
    const std::vector<typename cf_table_axis<CF_STOKES>::type>&
      stokes_values);

  /** baseline class axis dimension index */
  static const constexpr unsigned d_blc =
    cf_indexing::index_of<CF_BASELINE_CLASS, HYPERION_A_TERM_ZERNIKE_MODEL_AXES>
    ::type::value;
  /** frequency axis dimension index */
  static const constexpr unsigned d_frq =
    cf_indexing::index_of<CF_FREQUENCY, HYPERION_A_TERM_ZERNIKE_MODEL_AXES>
    ::type::value;
  /** Stokes axis dimension index */
  static const constexpr unsigned d_sto =
    cf_indexing::index_of<CF_STOKES, HYPERION_A_TERM_ZERNIKE_MODEL_AXES>
    ::type::value;

  // column for Zernike expansion coeffiecients
  //
  // The size of this axis is determined upon construction, given a vector of
  // ZCoeff values. The ordering of values on this axis conforms to the OSA/ANSI
  // standard order
  static const constexpr unsigned d_zc = index_rank;
  static const constexpr unsigned zc_rank = d_zc + 1;
  static const constexpr Legion::FieldID ZC_FID = 84;
  static const constexpr char* ZC_NAME = "ZERNIKE_COEFFICIENTS";
  template <Legion::PrivilegeMode MODE, bool CHECK_BOUNDS=HYPERION_CHECK_BOUNDS>
  using zc_accessor_t =
    const Legion::FieldAccessor<
      MODE,
      zc_t,
      zc_rank,
      Legion::coord_t,
      Legion::AffineAccessor<zc_t, zc_rank, Legion::coord_t>,
      CHECK_BOUNDS>;
  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  using ZCColumn =
    PhysicalColumnTD<
      ValueType<zc_t>::DataType,
      index_rank,
      zc_rank,
      A,
      COORD_T>;

  // acceptable fractional error in frequency value for "exact" zernike
  // expansion coefficient matching (i.e, stop searching for a closer frequency
  // value when one has been found within this tolerance)
  static const constexpr typename cf_table_axis<CF_FREQUENCY>::type
    zc_exact_frequency_tolerance = 0.01;

  // column for coefficients of polynomial function representation of Zernike
  // expansion
  //
  // Two dimensions, typically for powers of x and y. The polynomial function
  // coefficient of the x^p y^q term is at index [p, q] (within the subspace for
  // a given baseline class, frequency and Stokes parameter value).
  static const constexpr unsigned d_pc0 = index_rank;
  static const constexpr unsigned d_pc1 = d_pc0 + 1;
  static const constexpr unsigned pc_rank = d_pc1 + 1;
  static const constexpr Legion::FieldID PC_FID = 86;
  static const constexpr char* PC_NAME = "POLYNOMIAL_COEFFICIENTS";
  typedef zc_t pc_t;
  template <Legion::PrivilegeMode MODE, bool CHECK_BOUNDS=HYPERION_CHECK_BOUNDS>
  using pc_accessor_t =
    const Legion::FieldAccessor<
      MODE,
      pc_t,
      pc_rank,
      Legion::coord_t,
      Legion::AffineAccessor<pc_t, pc_rank, Legion::coord_t>,
      CHECK_BOUNDS>;
  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  using PCColumn =
    PhysicalColumnTD<
      ValueType<pc_t>::DataType,
      index_rank,
      pc_rank,
      A,
      COORD_T>;

  /**
   * compute the coefficients of the polynomial function representation of the
   * Zernike expansion
   */
  void
  compute_pcs(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const ColumnSpacePartition& partition = ColumnSpacePartition()) const;

protected:

  /**
   * Initialize the Zernike coefficients column from unsorted vector of baseline
   * class-, frequency- and Stokes-dependent coefficient values. This is used by
   * the class constructor, and should not be used otherwise.
   */
  void
  init_zc_region(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<ZCoeff>& zernike_coefficients,
    const std::vector<typename cf_table_axis<CF_BASELINE_CLASS>::type>&
      baseline_classes,
    const std::vector<typename cf_table_axis<CF_FREQUENCY>::type>&
      frequencies,
    const std::vector<typename cf_table_axis<CF_STOKES>::type>&
      stokes_values) const;

public:
  // Let the sequence {z(0,0), z(-1,1), z(1,1), ..., z(N,N-2), z(N,N)} represent
  // a linear combination of Zernike polynomials Z_(m, n) with coefficients z(m,
  // n), up to Zernike order N. This sequence represents the expansion of a
  // function F in the Zernike basis. To evaluate F on a grid, we will express
  // this sequence as a polynomial function of two variables, x and y. The idea
  // is to create a matrix Z, where Z(i, j) is the coefficient of x^i y^j term,
  // and then to evaluate the function by forming the vector-matrix-vector
  // product X^T Z Y, where X is the vector for a given value x with the
  // elements x^0, x^1, x^2, ... x^N (similar definition for Y).
  static const constexpr char* compute_pcs_task_name =
    "ATermZernikeModel::compute_pcs_task";

  static Legion::TaskID compute_pcs_task_id;

  // N.B: this implementation does not work in Kokkos::Cuda, primarily because
  // of the static array definitions in Zernike.h
  template <typename execution_space>
  static void
  compute_pcs_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* rt) {

    const Table::Desc& tdesc = *static_cast<Table::Desc*>(task->args);

    // ATermZernikeModel physical instance
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
    auto& [tbl, rit, pit] = ptcr;
#else // !HAVE_CXX17
    auto& tbl = std::get<0>(ptcr);
    auto& rit = std::get<1>(ptcr);
    auto& pit = std::get<2>(ptcr);
#endif // HAVE_CXX17
    assert(rit == task->regions.end());
    assert(pit == regions.end());

    // Zernike coefficients
    auto zc_col =
      ZCColumn<Legion::AffineAccessor>(*tbl.column(ZC_NAME).value());
    auto zc_rect = zc_col.rect();
    auto zcs = zc_col.view<execution_space, READ_ONLY>();

    // polynomial function coefficients
    auto pc_col =
      PCColumn<Legion::AffineAccessor>(*tbl.column(PC_NAME).value());
    auto pcs = pc_col.view<execution_space, WRITE_ONLY>();

    Kokkos::parallel_for(
      "compute_pcs",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>, execution_space>(
        rt->get_executing_processor(ctx).kokkos_work_space(),
        {zc_rect.lo[d_blc], zc_rect.lo[d_frq], zc_rect.lo[d_sto]},
        {zc_rect.hi[d_blc] + 1, zc_rect.hi[d_frq] + 1, zc_rect.hi[d_sto] + 1}),
      [=]( // no KOKKOS_LAMBDA here, to avoid __device__ modifier
        const Legion::coord_t& blc,
        const Legion::coord_t& frq,
        const Legion::coord_t& sto) {

        auto zcs0 = Kokkos::subview(zcs, blc, frq, sto, Kokkos::ALL);
        auto pcs0 =
          Kokkos::subview(pcs, blc, frq, sto, Kokkos::ALL, Kokkos::ALL);
        switch (pcs.extent(3) - 1) {
#define ZEXP(N)                                       \
          case N:                                       \
            zernike_basis<zc_t, N>::expand(pcs0, zcs0); \
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

  static void
  preregister_tasks();

};

} // end namespace synthesis
} // end namespace hyperion

#endif // HYPERION_SYNTHESIS_A_TERM_ZERNIKE_MODEL_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
