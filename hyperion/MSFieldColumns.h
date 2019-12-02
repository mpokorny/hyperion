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

  template <legion_privilege_mode_t MODE=READ_ONLY, bool CHECK_BOUNDS=false>
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

  template <legion_privilege_mode_t MODE=READ_ONLY, bool CHECK_BOUNDS=false>
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

  template <legion_privilege_mode_t MODE=READ_ONLY, bool CHECK_BOUNDS=false>
  TimeAccessor<MODE, CHECK_BOUNDS>
  time() const {
    return
      TimeAccessor<MODE, CHECK_BOUNDS>(
        m_time_region.value(),
        Column::VALUE_FID);
  }

  // TODO: timeQuant()?

#ifdef HYPERION_USE_CASACORE
  bool
  has_timeMeas() const {
    return has_time() && m_time_epoch;
  }

  casacore::MEpoch
  timeMeas(const DataType<HYPERION_TYPE_DOUBLE>::ValueType& t) const {
    return
      casacore::MEpoch(casacore::Quantity(t, time_units), *m_time_epoch);
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

  template <legion_privilege_mode_t MODE=READ_ONLY, bool CHECK_BOUNDS=false>
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

  template <legion_privilege_mode_t MODE=READ_ONLY, bool CHECK_BOUNDS=false>
  DelayDirAccessor<MODE, CHECK_BOUNDS>
  delayDir() const {
    return
      DelayDirAccessor<MODE, CHECK_BOUNDS>(
        m_delay_dir_region.value(),
        Column::VALUE_FID);
  }

#ifdef HYPERION_USE_CASACORE
  bool
  has_delayDirMeas() const {
    return has_delayDir() && m_delay_dir_direction;
  }

  casacore::MDirection
  delayDirMeas(const DataType<HYPERION_TYPE_DOUBLE>::ValueType* dd) const {
    return
      casacore::MDirection(
        casacore::Quantity(dd[0], delay_dir_units),
        casacore::Quantity(dd[1], delay_dir_units),
        *m_delay_dir_direction);
  }

  // Values dd, np, and t must be from the same row.
  //
  // TODO: Some alternatives are possible, but the ones I can think of require
  // accessor arguments (with many privilege parameter variants), or creating a
  // new logical region, neither of which I particularly like.
  casacore::MDirection
  delayDirMeas(
    const DataType<HYPERION_TYPE_DOUBLE>::ValueType* dd,
    const DataType<HYPERION_TYPE_INT>::ValueType& np,
    const DataType<HYPERION_TYPE_DOUBLE>::ValueType& t,
    double interTime) const;
#endif // HYPERION_USE_CASACORE

  //
  // PHASE_DIR
  //
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using PhaseDirAccessor =
    FieldAccessor<HYPERION_TYPE_DOUBLE, 3, MODE, CHECK_BOUNDS>;

  bool
  has_phaseDir() const {
    return m_phase_dir_region.has_value();
  }

  template <legion_privilege_mode_t MODE=READ_ONLY, bool CHECK_BOUNDS=false>
  PhaseDirAccessor<MODE, CHECK_BOUNDS>
  phaseDir() const {
    return
      PhaseDirAccessor<MODE, CHECK_BOUNDS>(
        m_phase_dir_region.value(),
        Column::VALUE_FID);
  }

#ifdef HYPERION_USE_CASACORE
  bool
  has_phaseDirMeas() const {
    return has_phaseDir() && m_phase_dir_direction;
  }

  casacore::MDirection
  phaseDirMeas(const DataType<HYPERION_TYPE_DOUBLE>::ValueType* dd) const {
    return
      casacore::MDirection(
        casacore::Quantity(dd[0], phase_dir_units),
        casacore::Quantity(dd[1], phase_dir_units),
        *m_phase_dir_direction);
  }

  // Values dd, np, and t must be from the same row.
  casacore::MDirection
  phaseDirMeas(
    const DataType<HYPERION_TYPE_DOUBLE>::ValueType* dd,
    const DataType<HYPERION_TYPE_INT>::ValueType& np,
    const DataType<HYPERION_TYPE_DOUBLE>::ValueType& t,
    double interTime) const;
#endif // HYPERION_USE_CASACORE

  //
  // REFERENCE_DIR
  //
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using ReferenceDirAccessor =
    FieldAccessor<HYPERION_TYPE_DOUBLE, 3, MODE, CHECK_BOUNDS>;

  bool
  has_referenceDir() const {
    return m_reference_dir_region.has_value();
  }

  template <legion_privilege_mode_t MODE=READ_ONLY, bool CHECK_BOUNDS=false>
  ReferenceDirAccessor<MODE, CHECK_BOUNDS>
  referenceDir() const {
    return
      ReferenceDirAccessor<MODE, CHECK_BOUNDS>(
        m_reference_dir_region.value(),
        Column::VALUE_FID);
  }

#ifdef HYPERION_USE_CASACORE
  bool
  has_referenceDirMeas() const {
    return has_referenceDir() && m_reference_dir_direction;
  }

  casacore::MDirection
  referenceDirMeas(const DataType<HYPERION_TYPE_DOUBLE>::ValueType* dd) const {
    return
      casacore::MDirection(
        casacore::Quantity(dd[0], reference_dir_units),
        casacore::Quantity(dd[1], reference_dir_units),
        *m_reference_dir_direction);
  }

  // Values dd, np, and t must be from the same row.
  casacore::MDirection
  referenceDirMeas(
    const DataType<HYPERION_TYPE_DOUBLE>::ValueType* dd,
    const DataType<HYPERION_TYPE_INT>::ValueType& np,
    const DataType<HYPERION_TYPE_DOUBLE>::ValueType& t,
    double interTime) const;
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

  template <legion_privilege_mode_t MODE=READ_ONLY, bool CHECK_BOUNDS=false>
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

  template <legion_privilege_mode_t MODE=READ_ONLY, bool CHECK_BOUNDS=false>
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

  template <legion_privilege_mode_t MODE=READ_ONLY, bool CHECK_BOUNDS=false>
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
  std::shared_ptr<casacore::MeasRef<casacore::MEpoch>> m_time_epoch;
#endif
#ifdef HYPERION_USE_CASACORE
  std::shared_ptr<casacore::MeasRef<casacore::MDirection>>
  m_delay_dir_direction;
#endif
#ifdef HYPERION_USE_CASACORE
  std::shared_ptr<casacore::MeasRef<casacore::MDirection>>
  m_phase_dir_direction;
#endif
#ifdef HYPERION_USE_CASACORE
  std::shared_ptr<casacore::MeasRef<casacore::MDirection>>
  m_reference_dir_direction;
#endif
};

} // end namespace hyperion

#endif // HYPERION_FIELD_COLUMNS_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
