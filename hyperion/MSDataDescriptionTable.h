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
#ifndef HYPERION_MS_DATA_DESCRIPTION_TABLE_H_
#define HYPERION_MS_DATA_DESCRIPTION_TABLE_H_

#include <hyperion/hyperion.h>
#include <hyperion/PhysicalTable.h>
#include <hyperion/PhysicalColumn.h>
#include <hyperion/MSTableColumns.h>

#include <unordered_map>
#include <vector>

namespace hyperion {

class HYPERION_EXPORT MSDataDescriptionTable
  : public PhysicalTable {
public:

  typedef MSTableColumns<MS_DATA_DESCRIPTION> C;

  MSDataDescriptionTable(const PhysicalTable& pt)
    : PhysicalTable(pt) {
    assert(pt.axes_uid()
      && (pt.axes_uid().value()
          == Axes<typename MSTable<MS_DATA_DESCRIPTION>::Axes>::uid));
    assert(
      pt.index_axes()
      == std::vector<int>{static_cast<int>(DATA_DESCRIPTION_ROW)});
  }

  static const constexpr unsigned row_rank = 1;

  //
  // SPECTRAL_WINDOW_ID
  //
  static const constexpr unsigned spectral_window_id_rank =
    row_rank
    + C::element_ranks[C::col_t::MS_DATA_DESCRIPTION_COL_SPECTRAL_WINDOW_ID];

  bool
  has_spectral_window_id() const {
    return
      m_columns
      .count(HYPERION_COLUMN_NAME(DATA_DESCRIPTION, SPECTRAL_WINDOW_ID)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    HYPERION_TYPE_INT,
    row_rank,
    spectral_window_id_rank,
    A,
    COORD_T>
  spectral_window_id() const {
    return
      decltype(spectral_window_id<A, COORD_T>())(
        *m_columns
        .at(HYPERION_COLUMN_NAME(DATA_DESCRIPTION, SPECTRAL_WINDOW_ID)));
  }

  //
  // POLARIZATION_ID
  //
  static const constexpr unsigned polarization_id_rank =
    row_rank
    + C::element_ranks[C::col_t::MS_DATA_DESCRIPTION_COL_POLARIZATION_ID];

  bool
  has_polarization_id() const {
    return
      m_columns
      .count(HYPERION_COLUMN_NAME(DATA_DESCRIPTION, POLARIZATION_ID)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    HYPERION_TYPE_INT,
    row_rank,
    polarization_id_rank,
    A,
    COORD_T>
  polarization_id() const {
    return
      decltype(polarization_id<A, COORD_T>())(
        *m_columns.at(HYPERION_COLUMN_NAME(DATA_DESCRIPTION, POLARIZATION_ID)));
  }

  //
  // LAG_ID
  //
  static const constexpr unsigned lag_id_rank =
    row_rank + C::element_ranks[C::col_t::MS_DATA_DESCRIPTION_COL_LAG_ID];
  bool
  has_lag_id() const {
    return m_columns.count(HYPERION_COLUMN_NAME(DATA_DESCRIPTION, LAG_ID)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, lag_id_rank, A, COORD_T>
  lag_id() const {
    return
      decltype(lag_id<A, COORD_T>())(
        *m_columns.at(HYPERION_COLUMN_NAME(DATA_DESCRIPTION, LAG_ID)));
  }

  //
  // FLAG_ROW
  //
  static const constexpr unsigned flag_row_rank =
    row_rank + C::element_ranks[C::col_t::MS_DATA_DESCRIPTION_COL_FLAG_ROW];

  bool
  has_flag_row() const {
    return
      m_columns.count(HYPERION_COLUMN_NAME(DATA_DESCRIPTION, FLAG_ROW)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_BOOL, row_rank, flag_row_rank, A, COORD_T>
  flag_row() const {
    return
      decltype(flag_row<A, COORD_T>())(
        *m_columns.at(HYPERION_COLUMN_NAME(DATA_DESCRIPTION, FLAG_ROW)));
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
