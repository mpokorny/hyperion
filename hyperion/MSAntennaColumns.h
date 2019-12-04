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
#ifndef HYPERION_MS_ANTENNA_COLUMNS_H_
#define HYPERION_MS_ANTENNA_COLUMNS_H_

#include <hyperion/hyperion.h>
#include <hyperion/Column.h>

#pragma GCC visibility push(default)
# include <casacore/measures/Measures/MPosition.h>
# include <casacore/measures/Measures/MCPosition.h>
# include <memory>
# include <optional>
# include <unordered_map>
# include <vector>
#pragma GCC visibility pop

namespace hyperion {

class HYPERION_API MSAntennaColumns {
public:

  MSAntennaColumns(
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
  // STATION
  //
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using StationAccessor =
    FieldAccessor<HYPERION_TYPE_STRING, 1, MODE, CHECK_BOUNDS>;

  bool
  has_station() const {
    return m_station_region.has_value();
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  StationAccessor<MODE, CHECK_BOUNDS>
  station() const {
    return
      StationAccessor<MODE, CHECK_BOUNDS>(
        m_station_region.value(),
        Column::VALUE_FID);
  }

  //
  // TYPE
  //
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using TypeAccessor =
    FieldAccessor<HYPERION_TYPE_STRING, 1, MODE, CHECK_BOUNDS>;

  bool
  has_type() const {
    return m_type_region.has_value();
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  TypeAccessor<MODE, CHECK_BOUNDS>
  type() const {
    return
      TypeAccessor<MODE, CHECK_BOUNDS>(
        m_type_region.value(),
        Column::VALUE_FID);
  }

  //
  // MOUNT
  //
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using MountAccessor =
    FieldAccessor<HYPERION_TYPE_STRING, 1, MODE, CHECK_BOUNDS>;

  bool
  has_mount() const {
    return m_mount_region.has_value();
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  MountAccessor<MODE, CHECK_BOUNDS>
  mount() const {
    return
      MountAccessor<MODE, CHECK_BOUNDS>(
        m_mount_region.value(),
        Column::VALUE_FID);
  }

  //
  // POSITION
  //
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using PositionAccessor =
    FieldAccessor<HYPERION_TYPE_DOUBLE, 2, MODE, CHECK_BOUNDS>;

  bool
  has_position() const {
    return m_position_region.has_value();
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  PositionAccessor<MODE, CHECK_BOUNDS>
  position() const {
    return
      PositionAccessor<MODE, CHECK_BOUNDS>(
        m_position_region.value(),
        Column::VALUE_FID);
  }

#ifdef HYPERION_USE_CASACORE

  template <typename T>
  class PositionMeasWriterMixin
    : public T {
  public:
    using T::T;

    void
    write(
      const Legion::Point<1, Legion::coord_t>& pt,
      const casacore::MPosition& val) {

      auto p = T::m_convert(val);
      auto vs = p.get(*T::m_units).getValue();
      T::m_position[Legion::Point<2>(pt[0], 0)] = vs[0];
      T::m_position[Legion::Point<2>(pt[0], 1)] = vs[1];
      T::m_position[Legion::Point<2>(pt[0], 2)] = vs[2];
    }
  };

  template <typename T>
  class PositionMeasReaderMixin
    : public T {
  public:
    using T::T;

    casacore::MPosition
    read(const Legion::Point<1,Legion::coord_t>& pt) const {

      const DataType<HYPERION_TYPE_DOUBLE>::ValueType* mp =
        T::m_position.ptr(Legion::Point<2>(pt[0], 0));
      return
        casacore::MPosition(
          casacore::Quantity(mp[0], *T::m_units),
          casacore::Quantity(mp[1], *T::m_units),
          casacore::Quantity(mp[2], *T::m_units),
          *T::m_mr);
    }
  };

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  class PositionMeasAccessorBase {
  public:
    PositionMeasAccessorBase(
      const char* units,
      const Legion::PhysicalRegion& region,
      const std::shared_ptr<casacore::MeasRef<casacore::MPosition>>& mr)
      : m_units(units)
      , m_position(region, Column::VALUE_FID)
      , m_mr(mr) {
      m_convert.setOut(*m_mr);
    }

  protected:

    const char* m_units;

    PositionAccessor<MODE, CHECK_BOUNDS> m_position;

    std::shared_ptr<casacore::MeasRef<casacore::MPosition>> m_mr;

    casacore::MPosition::Convert m_convert;
  };

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  class PositionMeasAccessor:
    public PositionMeasWriterMixin<
      PositionMeasAccessorBase<MODE, CHECK_BOUNDS>> {
    typedef PositionMeasWriterMixin<
      PositionMeasAccessorBase<MODE, CHECK_BOUNDS>> T;
  public:
    using T::T;
  };

  template <bool CHECK_BOUNDS>
  class PositionMeasAccessor<READ_ONLY, CHECK_BOUNDS>
    : public PositionMeasReaderMixin<
        PositionMeasAccessorBase<READ_ONLY, CHECK_BOUNDS>> {
    typedef PositionMeasReaderMixin<
      PositionMeasAccessorBase<READ_ONLY, CHECK_BOUNDS>> T;
  public:
    using T::T;
  };

  template <bool CHECK_BOUNDS>
  class PositionMeasAccessor<READ_WRITE, CHECK_BOUNDS>
    : public PositionMeasReaderMixin<
        PositionMeasWriterMixin<
          PositionMeasAccessorBase<READ_WRITE, CHECK_BOUNDS>>> {
    typedef PositionMeasReaderMixin<
      PositionMeasWriterMixin<
        PositionMeasAccessorBase<READ_WRITE, CHECK_BOUNDS>>> T;
  public:
    using T::T;
  };

  bool
  has_positionMeas() const {
    return has_position() && m_position_mr;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  PositionMeasAccessor<MODE, CHECK_BOUNDS>
  positionMeas() const {
    return
      PositionMeasAccessor<MODE, CHECK_BOUNDS>(
        position_units,
        m_position_region.value(),
        m_position_mr);
  }
#endif // HYPERION_USE_CASACORE

  //
  // OFFSET
  //
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using OffsetAccessor = PositionAccessor<MODE, CHECK_BOUNDS>;

  bool
  has_offset() const {
    return m_offset_region.has_value();
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  OffsetAccessor<MODE, CHECK_BOUNDS>
  offset() const {
    return
      OffsetAccessor<MODE, CHECK_BOUNDS>(
        m_offset_region.value(),
        Column::VALUE_FID);
  }

#ifdef HYPERION_USE_CASACORE
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using OffsetMeasAccessor = PositionMeasAccessor<MODE, CHECK_BOUNDS>;

  bool
  has_offsetMeas() const {
    return has_offset() && m_offset_mr;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  OffsetMeasAccessor<MODE, CHECK_BOUNDS>
  offsetMeas() const {
    return
      OffsetMeasAccessor<MODE, CHECK_BOUNDS>(
        offset_units,
        m_offset_region.value(),
        m_offset_mr);
  }
#endif // HYPERION_USE_CASACORE

  //
  // DISH_DIAMETER
  //
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using DishDiameterAccessor =
    FieldAccessor<HYPERION_TYPE_DOUBLE, 1, MODE, CHECK_BOUNDS>;

  bool
  has_dishDiameter() const {
    return m_dish_diameter_region.has_value();
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  DishDiameterAccessor<MODE, CHECK_BOUNDS>
  dishDiameter() const {
    return
      DishDiameterAccessor<MODE, CHECK_BOUNDS>(
        m_dish_diameter_region.value(),
        Column::VALUE_FID);
  }

  //
  // ORBIT_ID
  //
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using OrbitIdAccessor =
    FieldAccessor<HYPERION_TYPE_INT, 1, MODE, CHECK_BOUNDS>;

  bool
  has_orbitId() const {
    return m_orbit_id_region.has_value();
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  OrbitIdAccessor<MODE, CHECK_BOUNDS>
  orbitId() const {
    return
      OrbitIdAccessor<MODE, CHECK_BOUNDS>(
        m_orbit_id_region.value(),
        Column::VALUE_FID);
  }

  //
  // MEAN_ORBIT
  //
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using MeanOrbitAccessor =
    FieldAccessor<HYPERION_TYPE_DOUBLE, 2, MODE, CHECK_BOUNDS>;

  bool
  has_meanOrbit() const {
    return m_mean_orbit_region.has_value();
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  MeanOrbitAccessor<MODE, CHECK_BOUNDS>
  meanOrbit() const {
    return
      MeanOrbitAccessor<MODE, CHECK_BOUNDS>(
        m_mean_orbit_region.value(),
        Column::VALUE_FID);
  }

  //
  // PHASED_ARRAY_ID
  //
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using PhasedArrayIdAccessor =
    FieldAccessor<HYPERION_TYPE_INT, 1, MODE, CHECK_BOUNDS>;

  bool
  has_phasedArrayId() const {
    return m_phased_array_id_region.has_value();
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  PhasedArrayIdAccessor<MODE, CHECK_BOUNDS>
  phasedArrayId() const {
    return
      PhasedArrayIdAccessor<MODE, CHECK_BOUNDS>(
        m_phased_array_id_region.value(),
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

  static const constexpr char* position_units = "m";

  static const constexpr char* offset_units = "m";

  static const constexpr char* dish_diameter_units = "m";

private:

  Legion::RegionRequirement m_rows_requirement;

  std::optional<Legion::PhysicalRegion> m_name_region;
  std::optional<Legion::PhysicalRegion> m_station_region;
  std::optional<Legion::PhysicalRegion> m_type_region;
  std::optional<Legion::PhysicalRegion> m_mount_region;
  std::optional<Legion::PhysicalRegion> m_position_region;
  std::optional<Legion::PhysicalRegion> m_offset_region;
  std::optional<Legion::PhysicalRegion> m_dish_diameter_region;
  std::optional<Legion::PhysicalRegion> m_orbit_id_region;
  std::optional<Legion::PhysicalRegion> m_mean_orbit_region;
  std::optional<Legion::PhysicalRegion> m_phased_array_id_region;
  std::optional<Legion::PhysicalRegion> m_flag_row_region;

#ifdef HYPERION_USE_CASACORE
  std::shared_ptr<casacore::MeasRef<casacore::MPosition>> m_position_mr;
  std::shared_ptr<casacore::MeasRef<casacore::MPosition>> m_offset_mr;
#endif
};

} // end namespace hyperion

#endif // HYPERION_ANTENNA_COLUMNS_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
