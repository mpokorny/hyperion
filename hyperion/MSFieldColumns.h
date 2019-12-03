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

#pragma GCC visibility push(default)
# include <casacore/measures/Measures/MDirection.h>
# include <casacore/measures/Measures/MCDirection.h>
# include <casacore/measures/Measures/MEpoch.h>
# include <casacore/measures/Measures/MCEpoch.h>
# include <memory>
# include <optional>
# include <vector>
#pragma GCC visibility pop

namespace hyperion {

class HYPERION_API MSFieldColumns {
public:

  MSFieldColumns(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const Legion::RegionRequirement& rows_requirement,
    const std::optional<Legion::PhysicalRegion>& name_region,
    const std::optional<Legion::PhysicalRegion>& code_region,
    const std::optional<Legion::PhysicalRegion>& time_region,
#ifdef HYPERION_USE_CASACORE
    const std::vector<Legion::PhysicalRegion>&
      time_epoch_regions,
#endif
    const std::optional<Legion::PhysicalRegion>& num_poly_region,
    const std::optional<Legion::PhysicalRegion>& delay_dir_region,
#ifdef HYPERION_USE_CASACORE
    const std::vector<Legion::PhysicalRegion>&
      delay_dir_direction_regions,
#endif
    const std::optional<Legion::PhysicalRegion>& phase_dir_region,
#ifdef HYPERION_USE_CASACORE
    const std::vector<Legion::PhysicalRegion>&
      phase_dir_direction_regions,
#endif
    const std::optional<Legion::PhysicalRegion>& reference_dir_region,
#ifdef HYPERION_USE_CASACORE
    const std::vector<Legion::PhysicalRegion>&
      reference_dir_direction_regions,
#endif
    const std::optional<Legion::PhysicalRegion>& source_id_region,
    const std::optional<Legion::PhysicalRegion>& ephemeris_id_region,
    const std::optional<Legion::PhysicalRegion>& flag_row_region);

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

  Legion::DomainT<1>
  rows(Legion::Runtime* rt) const {
    return
      rt->get_index_space_domain(
        m_rows_requirement.region.get_index_space());
  }

  //
  // NAME
  //
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using NameAccessor =
    FieldAccessor<HYPERION_TYPE_STRING, 1, MODE, CHECK_BOUNDS>;

  bool
  has_name() const {
    return m_name_region.has_value();
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  NameAccessor<MODE, CHECK_BOUNDS>
  name() const {
    return
      NameAccessor<MODE, CHECK_BOUNDS>(
        m_name_region.value(),
        Column::VALUE_FID);
  }

  //
  // CODE
  //
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using CodeAccessor =
    FieldAccessor<HYPERION_TYPE_SHORT, 1, MODE, CHECK_BOUNDS>;

  bool
  has_code() const {
    return m_code_region.has_value();
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  CodeAccessor<MODE, CHECK_BOUNDS>
  code() const {
    return
      CodeAccessor<MODE, CHECK_BOUNDS>(
        m_code_region.value(),
        Column::VALUE_FID);
  }

  //
  // TIME
  //
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using TimeAccessor =
    FieldAccessor<HYPERION_TYPE_DOUBLE, 1, MODE, CHECK_BOUNDS>;

  bool
  has_time() const {
    return m_time_region.has_value();
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  TimeAccessor<MODE, CHECK_BOUNDS>
  time() const {
    return
      TimeAccessor<MODE, CHECK_BOUNDS>(
        m_time_region.value(),
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
      const Legion::Point<1, Legion::coord_t>& pt,
      const casacore::MEpoch& val) {

      auto t = T::m_convert(val);
      T::m_time[pt] = t.get(MSFieldColumns::time_units).getValue();
    }
  };

  template <typename T>
  class TimeMeasReaderMixin
    : public T {
  public:
    using T::T;

    casacore::MEpoch
    read(const Legion::Point<1,Legion::coord_t>& pt) const {
      const DataType<HYPERION_TYPE_DOUBLE>::ValueType& t = T::m_time[pt];
      return
        casacore::MEpoch(
          casacore::Quantity(t, MSFieldColumns::time_units),
          *T::m_ref);
    }
  };

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  class TimeMeasAccessorBase {
  public:
    TimeMeasAccessorBase(
      const Legion::PhysicalRegion& region,
      const std::shared_ptr<casacore::MeasRef<casacore::MEpoch>>& ref)
      : m_time(region, Column::VALUE_FID)
      , m_ref(ref) {
      m_convert.setOut(*m_ref);
    }

  protected:

    TimeAccessor<MODE, CHECK_BOUNDS> m_time;

    std::shared_ptr<casacore::MeasRef<casacore::MEpoch>> m_ref;

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
    return has_time() && m_time_ref;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  TimeMeasAccessor<MODE, CHECK_BOUNDS>
  timeMeas() const {
    return
      TimeMeasAccessor<MODE, CHECK_BOUNDS>(m_time_region.value(), m_time_ref);
  }
#endif // HYPERION_USE_CASACORE

  //
  // NUM_POLY
  //
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using NumPolyAccessor =
    FieldAccessor<HYPERION_TYPE_INT, 1, MODE, CHECK_BOUNDS>;

  bool
  has_numPoly() const {
    return m_num_poly_region.has_value();
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  NumPolyAccessor<MODE, CHECK_BOUNDS>
  numPoly() const {
    return
      NumPolyAccessor<MODE, CHECK_BOUNDS>(
        m_num_poly_region.value(),
        Column::VALUE_FID);
  }

  //
  // DELAY_DIR
  //
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using DelayDirAccessor =
    FieldAccessor<HYPERION_TYPE_DOUBLE, 3, MODE, CHECK_BOUNDS>;

  bool
  has_delayDir() const {
    return m_delay_dir_region.has_value();
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  DelayDirAccessor<MODE, CHECK_BOUNDS>
  delayDir() const {
    return
      DelayDirAccessor<MODE, CHECK_BOUNDS>(
        m_delay_dir_region.value(),
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
      const Legion::Point<1, Legion::coord_t>& pt,
      std::vector<casacore::MDirection>& val) {

      // until either the num_poly index space is writable or there's some
      // convention to interpret a difference in polynomial order, the following
      // precondition is intended to avoid unexpected results
      auto np = T::num_poly[pt];
      assert(val.size() == static_cast<unsigned>(np) + 1);

      for (int i = 0; i < np + 1; ++i) {
        auto d = T::m_convert(val[i]);
        auto vs = d.getAngle(T::m_units).getValue();
        T::m_delay_dir[Legion::Point<3>(pt[0], i, 0)] = vs[0];
        T::m_delay_dir[Legion::Point<3>(pt[0], i, 1)] = vs[1];
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
      const Legion::Point<1, Legion::coord_t>& pt,
      double time=0.0) const {

      const DataType<HYPERION_TYPE_DOUBLE>::ValueType* ds =
        T::m_delay_dir.ptr(Legion::Point<3>(pt[0], 0, 0));

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
          *T::m_ref);
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
      const std::shared_ptr<casacore::MeasRef<casacore::MDirection>>& ref)
      : m_units(units)
      , m_delay_dir(delay_dir_region, Column::VALUE_FID)
      , m_num_poly(num_poly_region, Column::VALUE_FID)
      , m_time(time_region, Column::VALUE_FID) {
      m_convert.setOut(*ref);
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
    return has_delayDir() && m_delay_dir_ref
      && has_numPoly() && has_time();
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  DelayDirMeasAccessor<MODE, CHECK_BOUNDS>
  delayDirMeas() const {
    return
      DelayDirMeasAccessor<MODE, CHECK_BOUNDS>(
        delay_dir_units,
        m_delay_dir_region.value(),
        m_num_poly_region.value(),
        m_time_region.value(),
        m_delay_dir_ref);
  }
#endif // HYPERION_USE_CASACORE

  //
  // PHASE_DIR
  //
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using PhaseDirAccessor = DelayDirAccessor<MODE, CHECK_BOUNDS>;

  bool
  has_phaseDir() const {
    return m_phase_dir_region.has_value();
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  PhaseDirAccessor<MODE, CHECK_BOUNDS>
  phaseDir() const {
    return
      PhaseDirAccessor<MODE, CHECK_BOUNDS>(
        m_phase_dir_region.value(),
        Column::VALUE_FID);
  }

#ifdef HYPERION_USE_CASACORE
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using PhaseDirMeasAccessor = DelayDirMeasAccessor<MODE, CHECK_BOUNDS>;

  bool
  has_phaseDirMeas() const {
    return has_phaseDir() && m_phase_dir_ref
      && has_numPoly() && has_time();
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  PhaseDirMeasAccessor<MODE, CHECK_BOUNDS>
  phaseDirMeas() const {
    return
      PhaseDirMeasAccessor<MODE, CHECK_BOUNDS>(
        phase_dir_units,
        m_phase_dir_region.value(),
        m_num_poly_region.value(),
        m_time_region.value(),
        m_phase_dir_ref);
  }
#endif // HYPERION_USE_CASACORE

  //
  // REFERENCE_DIR
  //
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using ReferenceDirAccessor = DelayDirAccessor<MODE, CHECK_BOUNDS>;

  bool
  has_referenceDir() const {
    return m_reference_dir_region.has_value();
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  ReferenceDirAccessor<MODE, CHECK_BOUNDS>
  referenceDir() const {
    return
      ReferenceDirAccessor<MODE, CHECK_BOUNDS>(
        m_reference_dir_region.value(),
        Column::VALUE_FID);
  }

#ifdef HYPERION_USE_CASACORE
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using ReferenceDirMeasAccessor = DelayDirMeasAccessor<MODE, CHECK_BOUNDS>;

  bool
  has_referenceDirMeas() const {
    return has_referenceDir() && m_reference_dir_ref
      && has_numPoly() && has_time();
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  ReferenceDirMeasAccessor<MODE, CHECK_BOUNDS>
  referenceDirMeas() const {
    return
      ReferenceDirMeasAccessor<MODE, CHECK_BOUNDS>(
        reference_dir_units,
        m_reference_dir_region.value(),
        m_num_poly_region.value(),
        m_time_region.value(),
        m_reference_dir_ref);
  }
#endif // HYPERION_USE_CASACORE

  //
  // SOURCE_ID
  //
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using SourceIdAccessor =
    FieldAccessor<HYPERION_TYPE_INT, 1, MODE, CHECK_BOUNDS>;

  bool
  has_sourceId() const {
    return m_source_id_region.has_value();
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  SourceIdAccessor<MODE, CHECK_BOUNDS>
  sourceId() const {
    return SourceIdAccessor<MODE, CHECK_BOUNDS>(
      m_source_id_region.value(),
      Column::VALUE_FID);
  }

  //
  // EPHEMERIS_ID
  //
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using EphemerisIdAccessor =
    FieldAccessor<HYPERION_TYPE_INT, 1, MODE, CHECK_BOUNDS>;

  bool
  has_ephemermis_id() const {
    return m_ephemeris_id_region.has_value();
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  EphemerisIdAccessor<MODE, CHECK_BOUNDS>
  ephemerisId() const {
    return EphemerisIdAccessor<MODE, CHECK_BOUNDS>(
      m_ephemeris_id_region.value(),
      Column::VALUE_FID);
  }

  //
  // FLAG_ROW
  //
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using FlagRowAccessor =
    FieldAccessor<HYPERION_TYPE_BOOL, 1, MODE, CHECK_BOUNDS>;

  bool
  has_flagRow() const {
    return m_flag_row_region.has_value();
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  FlagRowAccessor<MODE, CHECK_BOUNDS>
  flagRow() const {
    return
      FlagRowAccessor<MODE, CHECK_BOUNDS>(
        m_flag_row_region.value(),
        Column::VALUE_FID);
  }

  static const constexpr char* time_units = "s";

  static const constexpr char* delay_dir_units = "rad";

  static const constexpr char* phase_dir_units = "rad";

  static const constexpr char* reference_dir_units = "rad";

private:

  Legion::RegionRequirement m_rows_requirement;

  std::optional<Legion::PhysicalRegion> m_name_region;
  std::optional<Legion::PhysicalRegion> m_code_region;
  std::optional<Legion::PhysicalRegion> m_time_region;
  std::optional<Legion::PhysicalRegion> m_num_poly_region;
  std::optional<Legion::PhysicalRegion> m_delay_dir_region;
  std::optional<Legion::PhysicalRegion> m_phase_dir_region;
  std::optional<Legion::PhysicalRegion> m_reference_dir_region;
  std::optional<Legion::PhysicalRegion> m_source_id_region;
  std::optional<Legion::PhysicalRegion> m_ephemeris_id_region;
  std::optional<Legion::PhysicalRegion> m_flag_row_region;

#ifdef HYPERION_USE_CASACORE
  std::shared_ptr<casacore::MeasRef<casacore::MEpoch>> m_time_ref;
  std::shared_ptr<casacore::MeasRef<casacore::MDirection>>
  m_delay_dir_ref;
  std::shared_ptr<casacore::MeasRef<casacore::MDirection>>
  m_phase_dir_ref;
  std::shared_ptr<casacore::MeasRef<casacore::MDirection>>
  m_reference_dir_ref;
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
