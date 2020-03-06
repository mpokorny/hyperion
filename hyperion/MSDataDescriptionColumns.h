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
#ifndef HYPERION_MS_DATA_DESCRIPTION_COLUMNS_H_
#define HYPERION_MS_DATA_DESCRIPTION_COLUMNS_H_

#include <hyperion/hyperion.h>
#include <hyperion/Column.h>
#include <hyperion/MSTableColumns.h>

#pragma GCC visibility push(default)
# include <unordered_map>
# include <vector>
#pragma GCC visibility pop

namespace hyperion {

class HYPERION_API MSDataDescriptionColumns
  : public MSTableColumnsBase {
public:

  typedef MSTableColumns<MS_DATA_DESCRIPTION> C;

  MSDataDescriptionColumns(
    Legion::Runtime* rt,
    const Legion::RegionRequirement& rows_requirement,
    const std::unordered_map<std::string, Regions>& regions);

  static const constexpr unsigned row_rank = 1;

  Legion::DomainT<row_rank>
  rows() const {
    return m_rows;
  }

  //
  // SPECTRAL_WINDOW_ID
  //
  static const constexpr unsigned spectral_window_id_rank =
    row_rank
    + C::element_ranks[C::col_t::MS_DATA_DESCRIPTION_COL_SPECTRAL_WINDOW_ID];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using SpectralWindowIdAccessor =
    FieldAccessor<
      HYPERION_TYPE_INT,
      spectral_window_id_rank,
      MODE,
      CHECK_BOUNDS>;

  bool
  has_spectral_window_id() const {
    return
      m_regions.count(C::col_t::MS_DATA_DESCRIPTION_COL_SPECTRAL_WINDOW_ID) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  SpectralWindowIdAccessor<MODE, CHECK_BOUNDS>
  spectral_window_id() const {
    return
      SpectralWindowIdAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_DATA_DESCRIPTION_COL_SPECTRAL_WINDOW_ID),
        C::fid(C::col_t::MS_DATA_DESCRIPTION_COL_SPECTRAL_WINDOW_ID));
  }

  //
  // POLARIZATION_ID
  //
  static const constexpr unsigned polarization_id_rank =
    row_rank
    + C::element_ranks[C::col_t::MS_DATA_DESCRIPTION_COL_POLARIZATION_ID];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using PolarizationIdAccessor =
    FieldAccessor<HYPERION_TYPE_INT, polarization_id_rank, MODE, CHECK_BOUNDS>;

  bool
  has_polarization_id() const {
    return
      m_regions.count(C::col_t::MS_DATA_DESCRIPTION_COL_POLARIZATION_ID) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  PolarizationIdAccessor<MODE, CHECK_BOUNDS>
  polarization_id() const {
    return
      PolarizationIdAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_DATA_DESCRIPTION_COL_POLARIZATION_ID),
        C::fid(C::col_t::MS_DATA_DESCRIPTION_COL_POLARIZATION_ID));
  }

  //
  // LAG_ID
  //
  static const constexpr unsigned lag_id_rank =
    row_rank + C::element_ranks[C::col_t::MS_DATA_DESCRIPTION_COL_LAG_ID];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using LagIdAccessor =
    FieldAccessor<HYPERION_TYPE_INT, lag_id_rank, MODE, CHECK_BOUNDS>;

  bool
  has_lag_id() const {
    return m_regions.count(C::col_t::MS_DATA_DESCRIPTION_COL_LAG_ID) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  LagIdAccessor<MODE, CHECK_BOUNDS>
  lag_id() const {
    return
      LagIdAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_DATA_DESCRIPTION_COL_LAG_ID),
        C::fid(C::col_t::MS_DATA_DESCRIPTION_COL_LAG_ID));
  }

  //
  // FLAG_ROW
  //
  static const constexpr unsigned flag_row_rank =
    row_rank + C::element_ranks[C::col_t::MS_DATA_DESCRIPTION_COL_FLAG_ROW];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using FlagRowAccessor =
    FieldAccessor<HYPERION_TYPE_BOOL, flag_row_rank, MODE, CHECK_BOUNDS>;

  bool
  has_flag_row() const {
    return m_regions.count(C::col_t::MS_DATA_DESCRIPTION_COL_FLAG_ROW) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  FlagRowAccessor<MODE, CHECK_BOUNDS>
  flag_row() const {
    return
      FlagRowAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_DATA_DESCRIPTION_COL_FLAG_ROW),
        C::fid(C::col_t::MS_DATA_DESCRIPTION_COL_FLAG_ROW));
  }

private:

  Legion::DomainT<row_rank> m_rows;

  std::unordered_map<C::col_t, Legion::PhysicalRegion> m_regions;
};

} // end namespace hyperion

#endif // HYPERION_ANTENNA_COLUMNS_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
