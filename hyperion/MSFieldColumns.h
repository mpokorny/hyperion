/*
 * Copyright 2019 Associated Universities, Inc. Washington DC, USA.
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
#ifndef HYPERION_MS_FIELD_COLUMNS_H_
#define HYPERION_MS_FIELD_COLUMNS_H_

#include <hyperion/hyperion.h>
#include <hyperion/Column.h>
#include <hyperion/MSTableColumns.h>

#pragma GCC visibility push(default)
# include <casacore/measures/Measures/MDirection.h>
# include <casacore/measures/Measures/MCDirection.h>
# include <casacore/measures/Measures/MEpoch.h>
# include <casacore/measures/Measures/MCEpoch.h>
# include <memory>
# include <optional>
# include <unordered_map>
# include <vector>
#pragma GCC visibility pop

namespace hyperion {

class HYPERION_API MSFieldColumns {
public:

  typedef MSTableColumns<MS_FIELD> C;

  MSFieldColumns(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const Legion::RegionRequirement& rows_requirement,
    const std::unordered_map<std::string, std::vector<Legion::PhysicalRegion>>&
    regions);

private:

  template <TypeTag T, int N, legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using FieldAccessor =
    Legion::FieldAccessor<
      MODE,
      typename DataType<T>::ValueType,
      N,
      Legion::coord_t,
      Legion::AffineAccessor<
        typename DataType<T>::ValueType,
        N,
        Legion::coord_t>,
      CHECK_BOUNDS>;

public:

  static const constexpr unsigned row_rank = 1;

  Legion::DomainT<row_rank>
  rows(Legion::Runtime* rt) const {
    return
      rt->get_index_space_domain(
        m_rows_requirement.region.get_index_space());
  }

  //
  // NAME
  //
  static const constexpr unsigned name_rank =
    row_rank + C::element_ranks[C::col_t::MS_FIELD_COL_NAME];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using NameAccessor =
    FieldAccessor<HYPERION_TYPE_STRING, name_rank, MODE, CHECK_BOUNDS>;

  bool
  has_name() const {
    return m_regions.count(C::col_t::MS_FIELD_COL_NAME) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  NameAccessor<MODE, CHECK_BOUNDS>
  name() const {
    return
      NameAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_FIELD_COL_NAME),
        Column::VALUE_FID);
  }

  //
  // CODE
  //
  static const constexpr unsigned code_rank =
    row_rank + C::element_ranks[C::col_t::MS_FIELD_COL_CODE];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using CodeAccessor =
    FieldAccessor<HYPERION_TYPE_SHORT, code_rank, MODE, CHECK_BOUNDS>;

  bool
  has_code() const {
    return m_regions.count(C::col_t::MS_FIELD_COL_CODE) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  CodeAccessor<MODE, CHECK_BOUNDS>
  code() const {
    return
      CodeAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_FIELD_COL_CODE),
        Column::VALUE_FID);
  }

  //
  // TIME
  //
  static const constexpr unsigned time_rank =
    row_rank + C::element_ranks[C::col_t::MS_FIELD_COL_TIME];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using TimeAccessor =
    FieldAccessor<HYPERION_TYPE_DOUBLE, time_rank, MODE, CHECK_BOUNDS>;

  bool
  has_time() const {
    return m_regions.count(C::col_t::MS_FIELD_COL_TIME) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  TimeAccessor<MODE, CHECK_BOUNDS>
  time() const {
    return
      TimeAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_FIELD_COL_TIME),
        Column::VALUE_FID);
  }

  // TODO: timeQuant()?

#ifdef HYPERION_USE_CASACORE
  template <typename T>
  class TimeMeasWriterMixin
    : public T {
  public:
    using T::T;

    void
    write(
      const Legion::Point<time_rank, Legion::coord_t>& pt,
      const casacore::MEpoch& val) {

      auto t = T::m_convert(val);
      T::m_time[pt] = t.get(T::m_units).getValue();
    }
  };

  template <typename T>
  class TimeMeasReaderMixin
    : public T {
  public:
    using T::T;

    casacore::MEpoch
    read(const Legion::Point<time_rank, Legion::coord_t>& pt) const {
      const DataType<HYPERION_TYPE_DOUBLE>::ValueType& t = T::m_time[pt];
      return casacore::MEpoch(casacore::Quantity(t, T::m_units), *T::m_mr);
    }
  };

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  class TimeMeasAccessorBase {
  public:
    TimeMeasAccessorBase(
      const Legion::PhysicalRegion& region,
      const std::shared_ptr<casacore::MeasRef<casacore::MEpoch>>& mr)
      : m_time(region, Column::VALUE_FID)
      , m_mr(mr)
      , m_units(C::col_t::MS_FIELD_COL_TIME) {
      m_convert.setOut(*m_mr);
    }

  protected:

    TimeAccessor<MODE, CHECK_BOUNDS> m_time;

    std::shared_ptr<casacore::MeasRef<casacore::MEpoch>> m_mr;

    const char *m_units;

    casacore::MEpoch::Convert m_convert;
  };

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  class TimeMeasAccessor
    : public TimeMeasWriterMixin<TimeMeasAccessorBase<MODE, CHECK_BOUNDS>> {
    // this implementation supports MODE=WRITE_ONLY and MODE=WRITE_DISCARD
    typedef TimeMeasWriterMixin<TimeMeasAccessorBase<MODE, CHECK_BOUNDS>> T;
  public:
    using T::T;
  };

  template <bool CHECK_BOUNDS>
  class TimeMeasAccessor<READ_ONLY, CHECK_BOUNDS>
    : public TimeMeasReaderMixin<
        TimeMeasAccessorBase<READ_ONLY, CHECK_BOUNDS>> {
    typedef TimeMeasReaderMixin<
      TimeMeasAccessorBase<READ_ONLY, CHECK_BOUNDS>> T;
  public:
    using T::T;
  };

  template <bool CHECK_BOUNDS>
  class TimeMeasAccessor<READ_WRITE, CHECK_BOUNDS>
    : public TimeMeasReaderMixin<
        TimeMeasWriterMixin<
          TimeMeasAccessorBase<READ_WRITE, CHECK_BOUNDS>>> {
    typedef TimeMeasReaderMixin<
      TimeMeasWriterMixin<
        TimeMeasAccessorBase<READ_WRITE, CHECK_BOUNDS>>> T;
  public:
    using T::T;
  };

  bool
  has_timeMeas() const {
    return has_time() && m_time_mr;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  TimeMeasAccessor<MODE, CHECK_BOUNDS>
  timeMeas() const {
    return
      TimeMeasAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_FIELD_COL_TIME),
        m_time_mr);
  }
#endif // HYPERION_USE_CASACORE

  //
  // NUM_POLY
  //
  static const constexpr unsigned numPoly_rank =
    row_rank + C::element_ranks[C::col_t::MS_FIELD_COL_NUM_POLY];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using NumPolyAccessor =
    FieldAccessor<HYPERION_TYPE_INT, numPoly_rank, MODE, CHECK_BOUNDS>;

  bool
  has_numPoly() const {
    return m_regions.count(C::col_t::MS_FIELD_COL_NUM_POLY) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  NumPolyAccessor<MODE, CHECK_BOUNDS>
  numPoly() const {
    return
      NumPolyAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_FIELD_COL_NUM_POLY),
        Column::VALUE_FID);
  }

  //
  // DELAY_DIR
  //
  static const constexpr unsigned delayDir_rank =
    row_rank + C::element_ranks[MS_FIELD_COL_DELAY_DIR];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using DelayDirAccessor =
    FieldAccessor<HYPERION_TYPE_DOUBLE, delayDir_rank, MODE, CHECK_BOUNDS>;

  bool
  has_delayDir() const {
    return m_regions.count(C::col_t::MS_FIELD_COL_DELAY_DIR) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  DelayDirAccessor<MODE, CHECK_BOUNDS>
  delayDir() const {
    return
      DelayDirAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_FIELD_COL_DELAY_DIR),
        Column::VALUE_FID);
  }

#ifdef HYPERION_USE_CASACORE
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
      static_assert(delayDir_rank == 3);

      // until either the num_poly index space is writable or there's some
      // convention to interpret a difference in polynomial order, the following
      // precondition is intended to avoid unexpected results
      auto np = T::num_poly[pt];
      assert(val.size() == static_cast<unsigned>(np) + 1);

      for (int i = 0; i < np + 1; ++i) {
        auto d = T::m_convert(val[i]);
        auto vs = d.getAngle(T::m_units).getValue();
        T::m_delay_dir[Legion::Point<delayDir_rank>(pt[0], i, 0)] = vs[0];
        T::m_delay_dir[Legion::Point<delayDir_rank>(pt[0], i, 1)] = vs[1];
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
      static_assert(delayDir_rank == 3);

      const DataType<HYPERION_TYPE_DOUBLE>::ValueType* ds =
        T::m_delay_dir.ptr(Legion::Point<delayDir_rank>(pt[0], 0, 0));

      if (time == 0.0)
        return to_mdirection(ds);

      // TODO: support ephemerides as in casacore::MSFieldColumns
      std::vector<casacore::MDirection> dir_poly;
      auto np = T::m_num_poly[pt];
      dir_poly.reserve(np + 1);
      for (int i = 0; i < np + 1; ++i) {
        dir_poly.push_back(to_mdirection(ds));
        ds += 2;
      }
      return interpolateDirMeas(dir_poly, time, T::m_time[pt]);
    }

  private:

    casacore::MDirection
    to_mdirection(const DataType<HYPERION_TYPE_DOUBLE>::ValueType* ds) const {
      return
        casacore::MDirection(
          casacore::Quantity(ds[0], T::m_units),
          casacore::Quantity(ds[1], T::m_units),
          *T::m_mr);
    };
  };

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  class DelayDirMeasAccessorBase {
  public:
    DelayDirMeasAccessorBase(
      const char* units,
      const Legion::PhysicalRegion& delay_dir_region,
      const Legion::PhysicalRegion& num_poly_region,
      const Legion::PhysicalRegion& time_region,
      const std::shared_ptr<casacore::MeasRef<casacore::MDirection>>& mr)
      : m_units(units)
      , m_delay_dir(delay_dir_region, Column::VALUE_FID)
      , m_num_poly(num_poly_region, Column::VALUE_FID)
      , m_time(time_region, Column::VALUE_FID) {
      m_convert.setOut(*mr);
    }

  private:

    const char* m_units;

    DelayDirAccessor<MODE, CHECK_BOUNDS> m_delay_dir;

    NumPolyAccessor<READ_ONLY, CHECK_BOUNDS> m_num_poly;

    TimeAccessor<READ_ONLY, CHECK_BOUNDS> m_time;

    casacore::MDirection::Convert m_convert;

  };

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  class DelayDirMeasAccessor
    : public DelayDirWriterMixin<
        DelayDirMeasAccessorBase<MODE, CHECK_BOUNDS>> {
    // this implementation supports MODE=WRITE_ONLY and MODE=WRITE_DISCARD
    typedef DelayDirWriterMixin<
      DelayDirMeasAccessorBase<MODE, CHECK_BOUNDS>> T;
  public:
    using T::T;
  };

  template <bool CHECK_BOUNDS>
  class DelayDirMeasAccessor<READ_ONLY, CHECK_BOUNDS>
    : public DelayDirReaderMixin<
        DelayDirMeasAccessorBase<READ_ONLY, CHECK_BOUNDS>> {
    typedef DelayDirReaderMixin<
      DelayDirMeasAccessorBase<READ_ONLY, CHECK_BOUNDS>> T;
  public:
    using T::T;
  };

  template <bool CHECK_BOUNDS>
  class DelayDirMeasAccessor<READ_WRITE, CHECK_BOUNDS>
    : public DelayDirReaderMixin<
        DelayDirWriterMixin<
          DelayDirMeasAccessorBase<READ_WRITE, CHECK_BOUNDS>>> {
    typedef DelayDirReaderMixin<
      DelayDirWriterMixin<
        DelayDirMeasAccessorBase<READ_WRITE, CHECK_BOUNDS>>> T;
  public:
    using T::T;
  };

  bool
  has_delayDirMeas() const {
    return has_delayDir() && m_delay_dir_mr
      && has_numPoly() && has_time();
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  DelayDirMeasAccessor<MODE, CHECK_BOUNDS>
  delayDirMeas() const {
    return
      DelayDirMeasAccessor<MODE, CHECK_BOUNDS>(
        C::units.at(MS_FIELD_COL_DELAY_DIR),
        m_regions.at(C::col_t::MS_FIELD_COL_DELAY_DIR),
        m_regions.at(C::col_t::MS_FIELD_COL_NUM_POLY),
        m_regions.at(C::col_t::MS_FIELD_COL_TIME),
        m_delay_dir_mr);
  }
#endif // HYPERION_USE_CASACORE

  //
  // PHASE_DIR
  //
  static const constexpr unsigned phaseDir_rank =
    row_rank + C::element_ranks[C::col_t::MS_FIELD_COL_PHASE_DIR];

  static_assert(phaseDir_rank == delayDir_rank);

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using PhaseDirAccessor = DelayDirAccessor<MODE, CHECK_BOUNDS>;

  bool
  has_phaseDir() const {
    return m_regions.count(C::col_t::MS_FIELD_COL_PHASE_DIR) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  PhaseDirAccessor<MODE, CHECK_BOUNDS>
  phaseDir() const {
    return
      PhaseDirAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_FIELD_COL_PHASE_DIR),
        Column::VALUE_FID);
  }

#ifdef HYPERION_USE_CASACORE
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using PhaseDirMeasAccessor = DelayDirMeasAccessor<MODE, CHECK_BOUNDS>;

  bool
  has_phaseDirMeas() const {
    return has_phaseDir() && m_phase_dir_mr
      && has_numPoly() && has_time();
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  PhaseDirMeasAccessor<MODE, CHECK_BOUNDS>
  phaseDirMeas() const {
    return
      PhaseDirMeasAccessor<MODE, CHECK_BOUNDS>(
        C::units.at(MS_FIELD_COL_PHASE_DIR),
        m_regions.at(C::col_t::MS_FIELD_COL_PHASE_DIR),
        m_regions.at(C::col_t::MS_FIELD_COL_NUM_POLY),
        m_regions.at(C::col_t::MS_FIELD_COL_TIME),
        m_phase_dir_mr);
  }
#endif // HYPERION_USE_CASACORE

  //
  // REFERENCE_DIR
  //
  static const constexpr unsigned referenceDir_rank =
    row_rank + C::element_ranks[C::col_t::MS_FIELD_COL_REFERENCE_DIR];

  static_assert(referenceDir_rank == delayDir_rank);

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using ReferenceDirAccessor = DelayDirAccessor<MODE, CHECK_BOUNDS>;

  bool
  has_referenceDir() const {
    return m_regions.count(C::col_t::MS_FIELD_COL_REFERENCE_DIR) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  ReferenceDirAccessor<MODE, CHECK_BOUNDS>
  referenceDir() const {
    return
      ReferenceDirAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_FIELD_COL_REFERENCE_DIR),
        Column::VALUE_FID);
  }

#ifdef HYPERION_USE_CASACORE
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using ReferenceDirMeasAccessor = DelayDirMeasAccessor<MODE, CHECK_BOUNDS>;

  bool
  has_referenceDirMeas() const {
    return has_referenceDir() && m_reference_dir_mr
      && has_numPoly() && has_time();
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  ReferenceDirMeasAccessor<MODE, CHECK_BOUNDS>
  referenceDirMeas() const {
    return
      ReferenceDirMeasAccessor<MODE, CHECK_BOUNDS>(
        C::units.at(MS_FIELD_COL_REFERENCE_DIR),
        m_regions.at(C::col_t::MS_FIELD_COL_REFERENCE_DIR),
        m_regions.at(C::col_t::MS_FIELD_COL_NUM_POLY),
        m_regions.at(C::col_t::MS_FIELD_COL_TIME),
        m_reference_dir_mr);
  }
#endif // HYPERION_USE_CASACORE

  //
  // SOURCE_ID
  //
  static const constexpr unsigned sourceId_rank =
    row_rank + C::element_ranks[C::col_t::MS_FIELD_COL_SOURCE_ID];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using SourceIdAccessor =
    FieldAccessor<HYPERION_TYPE_INT, sourceId_rank, MODE, CHECK_BOUNDS>;

  bool
  has_sourceId() const {
    return m_regions.count(C::col_t::MS_FIELD_COL_SOURCE_ID) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  SourceIdAccessor<MODE, CHECK_BOUNDS>
  sourceId() const {
    return SourceIdAccessor<MODE, CHECK_BOUNDS>(
      m_regions.at(C::col_t::MS_FIELD_COL_SOURCE_ID),
      Column::VALUE_FID);
  }

  //
  // EPHEMERIS_ID
  //
  static const constexpr unsigned ephemerisId_rank =
    row_rank + C::element_ranks[C::col_t::MS_FIELD_COL_EPHEMERIS_ID];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using EphemerisIdAccessor =
    FieldAccessor<HYPERION_TYPE_INT, ephemerisId_rank, MODE, CHECK_BOUNDS>;

  bool
  has_ephemermis_id() const {
    return m_regions.count(C::col_t::MS_FIELD_COL_EPHEMERIS_ID) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  EphemerisIdAccessor<MODE, CHECK_BOUNDS>
  ephemerisId() const {
    return EphemerisIdAccessor<MODE, CHECK_BOUNDS>(
      m_regions.at(C::col_t::MS_FIELD_COL_EPHEMERIS_ID),
      Column::VALUE_FID);
  }

  //
  // FLAG_ROW
  //
  static const constexpr unsigned flagRow_rank =
    row_rank + C::element_ranks[C::col_t::MS_FIELD_COL_FLAG_ROW];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using FlagRowAccessor =
    FieldAccessor<HYPERION_TYPE_BOOL, flagRow_rank, MODE, CHECK_BOUNDS>;

  bool
  has_flagRow() const {
    return m_regions.count(C::col_t::MS_FIELD_COL_FLAG_ROW) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  FlagRowAccessor<MODE, CHECK_BOUNDS>
  flagRow() const {
    return
      FlagRowAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_FIELD_COL_FLAG_ROW),
        Column::VALUE_FID);
  }

private:

  Legion::RegionRequirement m_rows_requirement;

  std::unordered_map<C::col_t, Legion::PhysicalRegion> m_regions;

#ifdef HYPERION_USE_CASACORE
  std::shared_ptr<casacore::MeasRef<casacore::MEpoch>> m_time_mr;
  std::shared_ptr<casacore::MeasRef<casacore::MDirection>> m_delay_dir_mr;
  std::shared_ptr<casacore::MeasRef<casacore::MDirection>> m_phase_dir_mr;
  std::shared_ptr<casacore::MeasRef<casacore::MDirection>> m_reference_dir_mr;
#endif

  static casacore::MDirection
  interpolateDirMeas(
    const std::vector<casacore::MDirection>& dir_poly,
    double interTime,
    double timeOrigin);

};

} // end namespace hyperion

#endif // HYPERION_FIELD_COLUMNS_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
