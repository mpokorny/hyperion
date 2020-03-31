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
#include <hyperion/PhysicalTable.h>
#include <hyperion/PhysicalColumn.h>
#include <hyperion/MSTableColumns.h>

#pragma GCC visibility push(default)
# include <casacore/measures/Measures/MPosition.h>
# include <casacore/measures/Measures/MCPosition.h>

# include <unordered_map>
#pragma GCC visibility pop

namespace hyperion {

class HYPERION_API MSAntennaTable
  : public PhysicalTable {
public:

  typedef MSTableColumns<MS_ANTENNA> C;

  MSAntennaTable(const PhysicalTable& pt)
    : PhysicalTable(pt) {}

  static const constexpr unsigned row_rank = 1;

  //
  // NAME
  //
  static const constexpr unsigned name_rank =
    row_rank + C::element_ranks[C::col_t::MS_ANTENNA_COL_NAME];

  bool
  has_name() const {
    return m_columns.count(HYPERION_COLUMN_NAME(ANTENNA, NAME)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_STRING, row_rank, name_rank, A, COORD_T>
  name() const {
    return decltype(name())(*m_columns.at(HYPERION_COLUMN_NAME(ANTENNA, NAME)));
  }

  //
  // STATION
  //
  static const constexpr unsigned station_rank =
    row_rank + C::element_ranks[C::col_t::MS_ANTENNA_COL_STATION];

  bool
  has_station() const {
    return m_columns.count(HYPERION_COLUMN_NAME(ANTENNA, STATION)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_STRING, row_rank, station_rank, A, COORD_T>
  station() const {
    return
      decltype(station())(
        *m_columns.at(HYPERION_COLUMN_NAME(ANTENNA, STATION)));
  }

  //
  // TYPE
  //
  static const constexpr unsigned type_rank =
    row_rank + C::element_ranks[C::col_t::MS_ANTENNA_COL_TYPE];

  bool
  has_type() const {
    return m_columns.count(HYPERION_COLUMN_NAME(ANTENNA, TYPE)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_STRING, row_rank, type_rank, A, COORD_T>
  type() const {
    return decltype(type())(*m_columns.at(HYPERION_COLUMN_NAME(ANTENNA, TYPE)));
  }

  //
  // MOUNT
  //
  static const constexpr unsigned mount_rank =
    row_rank + C::element_ranks[C::col_t::MS_ANTENNA_COL_MOUNT];

  bool
  has_mount() const {
    return m_columns.count(HYPERION_COLUMN_NAME(ANTENNA, MOUNT)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_STRING, row_rank, mount_rank, A, COORD_T>
  mount() const {
    return
      decltype(mount())(*m_columns.at(HYPERION_COLUMN_NAME(ANTENNA, MOUNT)));
  }

  //
  // POSITION
  //
  static const constexpr unsigned position_rank =
    row_rank + C::element_ranks[C::col_t::MS_ANTENNA_COL_POSITION];

  bool
  has_position() const {
    return m_columns.count(HYPERION_COLUMN_NAME(ANTENNA, POSITION)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_DOUBLE, row_rank, position_rank, A, COORD_T>
  position() const {
    return
      decltype(position())(
        *m_columns.at(HYPERION_COLUMN_NAME(ANTENNA, POSITION)));
  }

#ifdef HYPERION_USE_CASACORE
  bool
  has_position_meas() const {
    return has_position()
      && m_columns.at(HYPERION_COLUMN_NAME(ANTENNA, POSITION))->mr_drs();
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTMD<
    HYPERION_TYPE_DOUBLE,
    MClass::M_POSITION,
    row_rank,
    position_rank,
    3,
    A,
    COORD_T>
  position_meas() const {
    return
      decltype(position_meas())(
        *m_columns.at(HYPERION_COLUMN_NAME(ANTENNA, POSITION)));
  }
#endif // HYPERION_USE_CASACORE

  //
  // OFFSET
  //
  static const constexpr unsigned offset_rank =
    row_rank + C::element_ranks[C::col_t::MS_ANTENNA_COL_OFFSET];

  bool
  has_offset() const {
    return m_columns.count(HYPERION_COLUMN_NAME(ANTENNA, OFFSET)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_DOUBLE, row_rank, offset_rank, A, COORD_T>
  offset() const {
    return
      decltype(offset())(*m_columns.at(HYPERION_COLUMN_NAME(ANTENNA, OFFSET)));
  }

#ifdef HYPERION_USE_CASACORE
  bool
  has_offset_meas() const {
    return has_offset()
      && m_columns.at(HYPERION_COLUMN_NAME(ANTENNA, OFFSET))->mr_drs();
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTMD<
    HYPERION_TYPE_DOUBLE,
    MClass::M_POSITION,
    row_rank,
    offset_rank,
    3,
    A,
    COORD_T>
  offset_meas() const {
    return
      decltype(offset_meas())(
        *m_columns.at(HYPERION_COLUMN_NAME(ANTENNA, OFFSET)));
  }
#endif // HYPERION_USE_CASACORE

  //
  // DISH_DIAMETER
  //
  static const constexpr unsigned dish_diameter_rank =
    row_rank + C::element_ranks[C::col_t::MS_ANTENNA_COL_DISH_DIAMETER];

  bool
  has_dish_diameter() const {
    return
      m_columns.count(HYPERION_COLUMN_NAME(ANTENNA, DISH_DIAMETER)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    HYPERION_TYPE_DOUBLE,
    row_rank,
    dish_diameter_rank,
    A,
    COORD_T>
  dish_diameter() const {
    return
      decltype(dish_diameter())(
        *m_columns.at(HYPERION_COLUMN_NAME(ANTENNA, DISH_DIAMETER)));
  }

  //
  // ORBIT_ID
  //
  static const constexpr unsigned orbit_id_rank =
    row_rank + C::element_ranks[C::col_t::MS_ANTENNA_COL_ORBIT_ID];

  bool
  has_orbit_id() const {
    return m_columns.count(HYPERION_COLUMN_NAME(ANTENNA, ORBIT_ID)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, orbit_id_rank, A, COORD_T>
  orbit_id() const {
    return
      decltype(orbit_id())(
        *m_columns.at(HYPERION_COLUMN_NAME(ANTENNA, ORBIT_ID)));
  }

  //
  // MEAN_ORBIT
  //
  static const constexpr unsigned mean_orbit_rank =
    row_rank + C::element_ranks[C::col_t::MS_ANTENNA_COL_MEAN_ORBIT];

  bool
  has_mean_orbit() const {
    return
      m_columns.count(HYPERION_COLUMN_NAME(ANTENNA, MEAN_ORBIT)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_DOUBLE, row_rank, mean_orbit_rank, A, COORD_T>
  mean_orbit() const {
    return
      decltype(mean_orbit())(
        *m_columns.at(HYPERION_COLUMN_NAME(ANTENNA, MEAN_ORBIT)));
  }

  //
  // PHASED_ARRAY_ID
  //
  static const constexpr unsigned phased_array_id_rank =
    row_rank + C::element_ranks[C::col_t::MS_ANTENNA_COL_PHASED_ARRAY_ID];

  bool
  has_phased_array_id() const {
    return
      m_columns.count(HYPERION_COLUMN_NAME(ANTENNA, PHASED_ARRAY_ID))
      > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    HYPERION_TYPE_INT,
    row_rank,
    phased_array_id_rank,
    A,
    COORD_T>
  phased_array_id() const {
    return
      decltype(phased_array_id())(
        *m_columns.at(HYPERION_COLUMN_NAME(ANTENNA, PHASED_ARRAY_ID)));
  }

  //
  // FLAG_ROW
  //
  static const constexpr unsigned flag_row_rank =
    row_rank + C::element_ranks[C::col_t::MS_ANTENNA_COL_FLAG_ROW];

  bool
  has_flag_row() const {
    return m_columns.count(HYPERION_COLUMN_NAME(ANTENNA, FLAG_ROW)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_BOOL, row_rank, flag_row_rank, A, COORD_T>
  flag_row() const {
    return
      decltype(flag_row())(
        *m_columns.at(HYPERION_COLUMN_NAME(ANTENNA, FLAG_ROW)));
  }
};

} // end namespace hyperion

#endif // HYPERION_ANTENNA_COLUMNS_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
