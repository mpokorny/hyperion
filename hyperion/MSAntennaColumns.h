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
#include <hyperion/MSTableColumns.h>

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

  typedef MSTableColumns<MS_ANTENNA> C;

  MSAntennaColumns(
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

  static const unsigned row_rank = 1;

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
    row_rank + C::element_ranks[C::col_t::MS_ANTENNA_COL_NAME];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using NameAccessor =
    FieldAccessor<HYPERION_TYPE_STRING, name_rank, MODE, CHECK_BOUNDS>;

  bool
  has_name() const {
    return m_regions.count(C::col_t::MS_ANTENNA_COL_NAME) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  NameAccessor<MODE, CHECK_BOUNDS>
  name() const {
    return
      NameAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_ANTENNA_COL_NAME),
        Column::VALUE_FID);
  }

  //
  // STATION
  //
  static const constexpr unsigned station_rank =
    row_rank + C::element_ranks[C::col_t::MS_ANTENNA_COL_STATION];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using StationAccessor =
    FieldAccessor<HYPERION_TYPE_STRING, station_rank, MODE, CHECK_BOUNDS>;

  bool
  has_station() const {
    return m_regions.count(C::col_t::MS_ANTENNA_COL_STATION) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  StationAccessor<MODE, CHECK_BOUNDS>
  station() const {
    return
      StationAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_ANTENNA_COL_STATION),
        Column::VALUE_FID);
  }

  //
  // TYPE
  //
  static const constexpr unsigned type_rank =
    row_rank + C::element_ranks[C::col_t::MS_ANTENNA_COL_TYPE];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using TypeAccessor =
    FieldAccessor<HYPERION_TYPE_STRING, type_rank, MODE, CHECK_BOUNDS>;

  bool
  has_type() const {
    return m_regions.count(C::col_t::MS_ANTENNA_COL_TYPE) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  TypeAccessor<MODE, CHECK_BOUNDS>
  type() const {
    return
      TypeAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_ANTENNA_COL_TYPE),
        Column::VALUE_FID);
  }

  //
  // MOUNT
  //
  static const constexpr unsigned mount_rank =
    row_rank + C::element_ranks[C::col_t::MS_ANTENNA_COL_MOUNT];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using MountAccessor =
    FieldAccessor<HYPERION_TYPE_STRING, mount_rank, MODE, CHECK_BOUNDS>;

  bool
  has_mount() const {
    return m_regions.count(C::col_t::MS_ANTENNA_COL_MOUNT) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  MountAccessor<MODE, CHECK_BOUNDS>
  mount() const {
    return
      MountAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_ANTENNA_COL_MOUNT),
        Column::VALUE_FID);
  }

  //
  // POSITION
  //
  static const constexpr unsigned position_rank =
    row_rank + C::element_ranks[C::col_t::MS_ANTENNA_COL_POSITION];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using PositionAccessor =
    FieldAccessor<HYPERION_TYPE_DOUBLE, position_rank, MODE, CHECK_BOUNDS>;

  bool
  has_position() const {
    return m_regions.count(C::col_t::MS_ANTENNA_COL_POSITION) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  PositionAccessor<MODE, CHECK_BOUNDS>
  position() const {
    return
      PositionAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_ANTENNA_COL_POSITION),
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
      const Legion::Point<row_rank, Legion::coord_t>& pt,
      const casacore::MPosition& val) {

      static_assert(row_rank == 1);
      static_assert(position_rank == 2);

      auto p = T::m_convert(val);
      auto vs = p.get(*T::m_units).getValue();
      T::m_position[Legion::Point<position_rank>(pt[0], 0)] = vs[0];
      T::m_position[Legion::Point<position_rank>(pt[0], 1)] = vs[1];
      T::m_position[Legion::Point<position_rank>(pt[0], 2)] = vs[2];
    }
  };

  template <typename T>
  class PositionMeasReaderMixin
    : public T {
  public:
    using T::T;

    casacore::MPosition
    read(const Legion::Point<row_rank,Legion::coord_t>& pt) const {

      static_assert(row_rank == 1);
      static_assert(position_rank == 2);

      const DataType<HYPERION_TYPE_DOUBLE>::ValueType* mp =
        T::m_position.ptr(Legion::Point<position_rank>(pt[0], 0));
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
        C::units.at(C::col_t::MS_ANTENNA_COL_POSITION),
        m_regions.at(C::col_t::MS_ANTENNA_COL_POSITION),
        m_position_mr);
  }
#endif // HYPERION_USE_CASACORE

  //
  // OFFSET
  //
  static const constexpr unsigned offset_rank =
    row_rank + C::element_ranks[C::col_t::MS_ANTENNA_COL_OFFSET];

  static_assert(offset_rank == position_rank);

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using OffsetAccessor = PositionAccessor<MODE, CHECK_BOUNDS>;

  bool
  has_offset() const {
    return m_regions.count(C::col_t::MS_ANTENNA_COL_OFFSET) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  OffsetAccessor<MODE, CHECK_BOUNDS>
  offset() const {
    return
      OffsetAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_ANTENNA_COL_OFFSET),
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
        C::units.at(C::col_t::MS_ANTENNA_COL_OFFSET),
        m_regions.at(C::col_t::MS_ANTENNA_COL_OFFSET),
        m_offset_mr);
  }
#endif // HYPERION_USE_CASACORE

  //
  // DISH_DIAMETER
  //
  static const constexpr unsigned dishDiameter_rank =
    row_rank + C::element_ranks[C::col_t::MS_ANTENNA_COL_DISH_DIAMETER];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using DishDiameterAccessor =
    FieldAccessor<HYPERION_TYPE_DOUBLE, dishDiameter_rank, MODE, CHECK_BOUNDS>;

  bool
  has_dishDiameter() const {
    return m_regions.count(C::col_t::MS_ANTENNA_COL_DISH_DIAMETER) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  DishDiameterAccessor<MODE, CHECK_BOUNDS>
  dishDiameter() const {
    return
      DishDiameterAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_ANTENNA_COL_DISH_DIAMETER),
        Column::VALUE_FID);
  }

  //
  // ORBIT_ID
  //
  static const constexpr unsigned orbitId_rank =
    row_rank + C::element_ranks[C::col_t::MS_ANTENNA_COL_ORBIT_ID];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using OrbitIdAccessor =
    FieldAccessor<HYPERION_TYPE_INT, orbitId_rank, MODE, CHECK_BOUNDS>;

  bool
  has_orbitId() const {
    return m_regions.count(C::col_t::MS_ANTENNA_COL_ORBIT_ID) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  OrbitIdAccessor<MODE, CHECK_BOUNDS>
  orbitId() const {
    return
      OrbitIdAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_ANTENNA_COL_ORBIT_ID),
        Column::VALUE_FID);
  }

  //
  // MEAN_ORBIT
  //
  static const constexpr unsigned meanOrbit_rank =
    row_rank + C::element_ranks[C::col_t::MS_ANTENNA_COL_MEAN_ORBIT];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using MeanOrbitAccessor =
    FieldAccessor<HYPERION_TYPE_DOUBLE, meanOrbit_rank, MODE, CHECK_BOUNDS>;

  bool
  has_meanOrbit() const {
    return m_regions.count(C::col_t::MS_ANTENNA_COL_MEAN_ORBIT) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  MeanOrbitAccessor<MODE, CHECK_BOUNDS>
  meanOrbit() const {
    return
      MeanOrbitAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_ANTENNA_COL_MEAN_ORBIT),
        Column::VALUE_FID);
  }

  //
  // PHASED_ARRAY_ID
  //
  static const constexpr unsigned phasedArrayId_rank =
    row_rank + C::element_ranks[C::col_t::MS_ANTENNA_COL_PHASED_ARRAY_ID];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using PhasedArrayIdAccessor =
    FieldAccessor<HYPERION_TYPE_INT, phasedArrayId_rank, MODE, CHECK_BOUNDS>;

  bool
  has_phasedArrayId() const {
    return m_regions.count(C::col_t::MS_ANTENNA_COL_PHASED_ARRAY_ID) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  PhasedArrayIdAccessor<MODE, CHECK_BOUNDS>
  phasedArrayId() const {
    return
      PhasedArrayIdAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_ANTENNA_COL_PHASED_ARRAY_ID),
        Column::VALUE_FID);
  }

  //
  // FLAG_ROW
  //
  static const constexpr unsigned flagRow_rank =
    row_rank + C::element_ranks[C::col_t::MS_ANTENNA_COL_FLAG_ROW];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using FlagRowAccessor =
    FieldAccessor<HYPERION_TYPE_BOOL, flagRow_rank, MODE, CHECK_BOUNDS>;

  bool
  has_flagRow() const {
    return m_regions.count(C::col_t::MS_ANTENNA_COL_FLAG_ROW) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  FlagRowAccessor<MODE, CHECK_BOUNDS>
  flagRow() const {
    return
      FlagRowAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_ANTENNA_COL_FLAG_ROW),
        Column::VALUE_FID);
  }

private:

  Legion::RegionRequirement m_rows_requirement;

  std::unordered_map<C::col_t, Legion::PhysicalRegion> m_regions;

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
