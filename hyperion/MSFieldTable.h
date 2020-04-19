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
#ifndef HYPERION_MS_FIELD_TABLE_H_
#define HYPERION_MS_FIELD_TABLE_H_

#include <hyperion/hyperion.h>
#include <hyperion/PhysicalTable.h>
#include <hyperion/PhysicalColumn.h>
#include <hyperion/MSTableColumns.h>

#pragma GCC visibility push(default)
# ifdef HYPERION_USE_CASACORE
#  include <casacore/measures/Measures/MDirection.h>
#  include <casacore/measures/Measures/MCDirection.h>
#  include <casacore/measures/Measures/MEpoch.h>
#  include <casacore/measures/Measures/MCEpoch.h>
# endif // HYPERION_USE_CASACORE

# include <any>
# include <unordered_map>
# include <vector>
#pragma GCC visibility pop

namespace hyperion {

class HYPERION_API MSFieldTable
  : public PhysicalTable {
public:

  typedef MSTableColumns<MS_FIELD> C;

  MSFieldTable(const PhysicalTable& pt)
    : PhysicalTable(pt) {
    assert(pt.axes_uid() == Axes<typename MSTable<MS_FIELD>::Axes>::uid);
    assert(pt.index_axes() == std::vector{static_cast<int>(FIELD_ROW)});
  }

  static const constexpr unsigned row_rank = 1;

  //
  // NAME
  //
  static const constexpr unsigned name_rank =
    row_rank + C::element_ranks[C::col_t::MS_FIELD_COL_NAME];

  bool
  has_name() const {
    return m_columns.count(HYPERION_COLUMN_NAME(FIELD, NAME)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_STRING, row_rank, name_rank, A, COORD_T>
  name() const {
    return decltype(name())(*m_columns.at(HYPERION_COLUMN_NAME(FIELD, NAME)));
  }

  //
  // CODE
  //
  static const constexpr unsigned code_rank =
    row_rank + C::element_ranks[C::col_t::MS_FIELD_COL_CODE];

  bool
  has_code() const {
    return m_columns.count(HYPERION_COLUMN_NAME(FIELD, CODE)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_SHORT, row_rank, code_rank, A, COORD_T>
  code() const {
    return decltype(code())(*m_columns.at(HYPERION_COLUMN_NAME(FIELD, CODE)));
  }

  //
  // TIME
  //
  static const constexpr unsigned time_rank =
    row_rank + C::element_ranks[C::col_t::MS_FIELD_COL_TIME];

  bool
  has_time() const {
    return m_columns.count(HYPERION_COLUMN_NAME(FIELD, TIME)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_DOUBLE, row_rank, time_rank, A, COORD_T>
  time() const {
    return decltype(time())(*m_columns.at(HYPERION_COLUMN_NAME(FIELD, TIME)));
  }

  // TODO: timeQuant()?

#ifdef HYPERION_USE_CASACORE
  bool
  has_time_meas() const {
    return
      has_time() && m_columns.at(HYPERION_COLUMN_NAME(FIELD, TIME))->mr_drs();
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTMD<
    HYPERION_TYPE_DOUBLE,
    MClass::M_EPOCH,
    row_rank,
    time_rank,
    1,
    A,
    COORD_T>
  time_meas() const {
    return
      decltype(time_meas())(*m_columns.at(HYPERION_COLUMN_NAME(FIELD, TIME)));
  }
#endif // HYPERION_USE_CASACORE

  //
  // NUM_POLY
  //
  static const constexpr unsigned num_poly_rank =
    row_rank + C::element_ranks[C::col_t::MS_FIELD_COL_NUM_POLY];

  bool
  has_num_poly() const {
    return m_columns.count(HYPERION_COLUMN_NAME(FIELD, NUM_POLY)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, num_poly_rank, A, COORD_T>
  num_poly() const {
    return
      decltype(num_poly())(
        *m_columns.at(HYPERION_COLUMN_NAME(FIELD, NUM_POLY)));
  }

  //
  // DELAY_DIR
  //
  static const constexpr unsigned delay_dir_rank =
    row_rank + C::element_ranks[MS_FIELD_COL_DELAY_DIR];

  bool
  has_delay_dir() const {
    return m_columns.count(HYPERION_COLUMN_NAME(FIELD, DELAY_DIR)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_DOUBLE, row_rank, delay_dir_rank, A, COORD_T>
  delay_dir() const {
    return
      decltype(delay_dir())(
        *m_columns.at(HYPERION_COLUMN_NAME(FIELD, DELAY_DIR)));
  }

#ifdef HYPERION_USE_CASACORE
  bool
  has_delay_dir_meas() const {
    return has_delay_dir() &&
      m_columns.at(HYPERION_COLUMN_NAME(FIELD, DELAY_DIR))->mr_drs();
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTMD<
    HYPERION_TYPE_DOUBLE,
    MClass::M_DIRECTION,
    row_rank,
    delay_dir_rank,
    2,
    A,
    COORD_T>
  delay_dir_meas() const {
    return
      decltype(delay_dir_meas())(
        *m_columns.at(HYPERION_COLUMN_NAME(FIELD, DELAY_DIR)));
  }

#ifdef SAVE_ME_FOR_REFERENCE
  template <typename T>
  class DelayDirWriterMixin
    : public T {
  public:
    using T::T;

    void
    write(
      const Legion::Point<row_rank, Legion::coord_t>& pt,
      std::vector<casacore::MDirection>& val) {

      static_assert(row_rank == 1);
      static_assert(delay_dir_rank == 3);

      auto cvt = T::m_cm.convert_at(pt);
      // until either the num_poly index space is writable or there's some
      // convention to interpret a difference in polynomial order, the following
      // precondition is intended to avoid unexpected results
      auto np = T::num_poly[pt];
      assert(val.size() == static_cast<unsigned>(np) + 1);

      for (int i = 0; i < np + 1; ++i) {
        auto d = cvt(val[i]);
        auto vs = d.getAngle(T::m_units).getValue();
        T::m_delay_dir[Legion::Point<delay_dir_rank>(pt[0], i, 0)] = vs[0];
        T::m_delay_dir[Legion::Point<delay_dir_rank>(pt[0], i, 1)] = vs[1];
      }
    }
  };

  template <typename T>
  class DelayDirReaderMixin
    : public T {
  public:
    using T::T;

    casacore::MDirection
    read(
      const Legion::Point<row_rank, Legion::coord_t>& pt,
      double time=0.0) const {

      static_assert(row_rank == 1);
      static_assert(delay_dir_rank == 3);

      auto mr = T::m_cm.meas_ref_at(pt);
      const DataType<HYPERION_TYPE_DOUBLE>::ValueType* ds =
        T::m_delay_dir.ptr(Legion::Point<delay_dir_rank>(pt[0], 0, 0));

      if (time == 0.0)
        return to_mdirection(ds, mr);

      // TODO: support ephemerides as in casacore::MSFieldColumns
      std::vector<casacore::MDirection> dir_poly;
      auto np = T::m_num_poly[pt];
      dir_poly.reserve(np + 1);
      for (int i = 0; i < np + 1; ++i) {
        dir_poly.push_back(to_mdirection(ds, mr));
        ds += 2;
      }
      return interpolate_dir_meas(dir_poly, time, T::m_time[pt]);
    }

  private:

    casacore::MDirection
    to_mdirection(
      const DataType<HYPERION_TYPE_DOUBLE>::ValueType* ds,
      const casacore::MeasRef<casacore::MDirection>& mr) const {
      return
        casacore::MDirection(
          casacore::Quantity(ds[0], T::m_units),
          casacore::Quantity(ds[1], T::m_units),
          mr);
    };
  };
#endif // SAVE_ME_FOR_REFERENCE
#endif // HYPERION_USE_CASACORE

  //
  // PHASE_DIR
  //
  static const constexpr unsigned phase_dir_rank =
    row_rank + C::element_ranks[C::col_t::MS_FIELD_COL_PHASE_DIR];

  bool
  has_phase_dir() const {
    return m_columns.count(HYPERION_COLUMN_NAME(FIELD, PHASE_DIR)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_DOUBLE, row_rank, phase_dir_rank, A, COORD_T>
  phase_dir() const {
    return
      decltype(phase_dir())(
        *m_columns.at(HYPERION_COLUMN_NAME(FIELD, PHASE_DIR)));
  }

#ifdef HYPERION_USE_CASACORE
  bool
  has_phase_dir_meas() const {
    return has_phase_dir() &&
      m_columns.at(HYPERION_COLUMN_NAME(FIELD, PHASE_DIR))->mr_drs();
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTMD<
    HYPERION_TYPE_DOUBLE,
    MClass::M_DIRECTION,
    row_rank,
    phase_dir_rank,
    2,
    A,
    COORD_T>
  phase_dir_meas() const {
    return
      decltype(phase_dir_meas())(
        *m_columns.at(HYPERION_COLUMN_NAME(FIELD, PHASE_DIR)));
  }
#endif // HYPERION_USE_CASACORE

  //
  // REFERENCE_DIR
  //
  static const constexpr unsigned reference_dir_rank =
    row_rank + C::element_ranks[C::col_t::MS_FIELD_COL_REFERENCE_DIR];

  bool
  has_reference_dir() const {
    return m_columns.count(HYPERION_COLUMN_NAME(FIELD, REFERENCE_DIR)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    HYPERION_TYPE_DOUBLE,
    row_rank,
    reference_dir_rank,
    A,
    COORD_T>
  reference_dir() const {
    return
      decltype(reference_dir())(
        *m_columns.at(HYPERION_COLUMN_NAME(FIELD, REFERENCE_DIR)));
  }

#ifdef HYPERION_USE_CASACORE
  bool
  has_reference_dir_meas() const {
    return has_reference_dir() &&
      m_columns.at(HYPERION_COLUMN_NAME(FIELD, REFERENCE_DIR))->mr_drs();
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTMD<
    HYPERION_TYPE_DOUBLE,
    MClass::M_DIRECTION,
    row_rank,
    reference_dir_rank,
    2,
    A,
    COORD_T>
  reference_dir_meas() const {
    return
      decltype(reference_dir_meas())(
        *m_columns.at(HYPERION_COLUMN_NAME(FIELD, REFERENCE_DIR)));
  }
#endif // HYPERION_USE_CASACORE

  //
  // SOURCE_ID
  //
  static const constexpr unsigned source_id_rank =
    row_rank + C::element_ranks[C::col_t::MS_FIELD_COL_SOURCE_ID];

  bool
  has_source_id() const {
    return m_columns.count(HYPERION_COLUMN_NAME(FIELD, SOURCE_ID)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, source_id_rank, A, COORD_T>
  source_id() const {
    return
      decltype(source_id())(
        *m_columns.at(HYPERION_COLUMN_NAME(FIELD, SOURCE_ID)));
  }

  //
  // EPHEMERIS_ID
  //
  static const constexpr unsigned ephemeris_id_rank =
    row_rank + C::element_ranks[C::col_t::MS_FIELD_COL_EPHEMERIS_ID];

  bool
  has_ephemermis_id() const {
    return m_columns.count(HYPERION_COLUMN_NAME(FIELD, EPHEMERIS_ID)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, ephemeris_id_rank, A, COORD_T>
  ephemeris_id() const {
    return
      decltype(ephemeris_id())(
        *m_columns.at(HYPERION_COLUMN_NAME(FIELD, EPHEMERIS_ID)));
  }

  //
  // FLAG_ROW
  //
  static const constexpr unsigned flag_row_rank =
    row_rank + C::element_ranks[C::col_t::MS_FIELD_COL_FLAG_ROW];

  bool
  has_flag_row() const {
    return m_columns.count(HYPERION_COLUMN_NAME(FIELD, FLAG_ROW)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_BOOL, row_rank, flag_row_rank, A, COORD_T>
  flag_row() const {
    return
      decltype(flag_row())(
        *m_columns.at(HYPERION_COLUMN_NAME(FIELD, FLAG_ROW)));
  }

#ifdef HYPERION_USE_CASACORE
private:

  static casacore::MDirection
  interpolate_dir_meas(
    const std::vector<casacore::MDirection>& dir_poly,
    double interTime,
    double timeOrigin);
#endif // HYPERION_USE_CASACORE
};

} // end namespace hyperion

#endif // HYPERION_FIELD_TABLE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
