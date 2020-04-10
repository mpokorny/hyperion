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
#ifndef HYPERION_MS_MAIN_TABLE_H_
#define HYPERION_MS_MAIN_TABLE_H_

#include <hyperion/hyperion.h>
#include <hyperion/PhysicalTable.h>
#include <hyperion/PhysicalColumn.h>
#include <hyperion/MSTableColumns.h>

#pragma GCC visibility push(default)
# include <casacore/measures/Measures/MEpoch.h>
# include <casacore/measures/Measures/Muvw.h>

# include <unordered_map>
#pragma GCC visibility pop

namespace hyperion {

class MSMainTable
  : public PhysicalTable {
public:

  typedef MSTableColumns<MS_MAIN> C;

  MSMainTable(const PhysicalTable& pt)
    : PhysicalTable(pt) {}

  static const constexpr unsigned row_rank = 1;

  //
  // TIME
  //
  static const constexpr unsigned time_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_TIME];

  bool
  has_time() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, TIME)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_DOUBLE, row_rank, time_rank, A, COORD_T>
  time() const {
    return decltype(time())(*m_columns.at(HYPERION_COLUMN_NAME(MAIN, TIME)));
  }

#ifdef HYPERION_USE_CASACORE
  bool
  has_time_meas() const {
    return has_time()
      && m_columns.at(HYPERION_COLUMN_NAME(MAIN, TIME))->mr_drs();
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
      decltype(time_meas())(
        *m_columns.at(HYPERION_COLUMN_NAME(MAIN, TIME)));
  }
#endif // HYPERION_USE_CASACORE

  //
  // TIME_EXTRA_PREC
  //
  static const constexpr unsigned time_extra_prec_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_TIME_EXTRA_PREC];

  bool
  has_time_extra_prec() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, TIME_EXTRA_PREC)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    HYPERION_TYPE_DOUBLE,
    row_rank,
    time_extra_prec_rank,
    A,
    COORD_T>
  time_extra_prec() const {
    return
      decltype(time_extra_prec())(
        *m_columns.at(HYPERION_COLUMN_NAME(MAIN, TIME_EXTRA_PREC)));
  }

  //
  // ANTENNA1
  //
  static const constexpr unsigned antenna1_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_ANTENNA1];

  bool
  has_antenna1() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, ANTENNA1)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, antenna1_rank, A, COORD_T>
  antenna1() const {
    return
      decltype(antenna1())(*m_columns.at(HYPERION_COLUMN_NAME(MAIN, ANTENNA1)));
  }

  //
  // ANTENNA2
  //
  static const constexpr unsigned antenna2_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_ANTENNA2];

  bool
  has_antenna2() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, ANTENNA2)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, antenna2_rank, A, COORD_T>
  antenna2() const {
    return
      decltype(antenna2())(*m_columns.at(HYPERION_COLUMN_NAME(MAIN, ANTENNA2)));
  }

  //
  // ANTENNA3
  //
  static const constexpr unsigned antenna3_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_ANTENNA3];

  bool
  has_antenna3() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, ANTENNA3)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, antenna3_rank, A, COORD_T>
  antenna3() const {
    return
      decltype(antenna3())(*m_columns.at(HYPERION_COLUMN_NAME(MAIN, ANTENNA3)));
  }

  //
  // FEED1
  //
  static const constexpr unsigned feed1_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_FEED1];

  bool
  has_feed1() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, FEED1)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, feed1_rank, A, COORD_T>
  feed1() const {
    return
      decltype(feed1())(*m_columns.at(HYPERION_COLUMN_NAME(MAIN, FEED1)));
  }

  //
  // FEED2
  //
  static const constexpr unsigned feed2_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_FEED2];

  bool
  has_feed2() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, FEED2)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, feed2_rank, A, COORD_T>
  feed2() const {
    return
      decltype(feed2())(*m_columns.at(HYPERION_COLUMN_NAME(MAIN, FEED2)));
  }

  //
  // FEED3
  //
  static const constexpr unsigned feed3_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_FEED3];

  bool
  has_feed3() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, FEED3)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, feed3_rank, A, COORD_T>
  feed3() const {
    return
      decltype(feed3())(*m_columns.at(HYPERION_COLUMN_NAME(MAIN, FEED3)));
  }

  //
  // DATA_DESC_ID
  //
  static const constexpr unsigned data_desc_id_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_DATA_DESC_ID];

  bool
  has_data_desc_id() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, DATA_DESC_ID)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, data_desc_id_rank, A, COORD_T>
  data_desc_id() const {
    return
      decltype(data_desc_id())(
        *m_columns.at(HYPERION_COLUMN_NAME(MAIN, DATA_DESC_ID)));
  }

  //
  // PROCESSOR_ID
  //
  static const constexpr unsigned processor_id_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_PROCESSOR_ID];

  bool
  has_processor_id() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, PROCESSOR_ID)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, processor_id_rank, A, COORD_T>
  processor_id() const {
    return
      decltype(processor_id())(
        *m_columns.at(HYPERION_COLUMN_NAME(MAIN, PROCESSOR_ID)));
  }

  //
  // PHASE_ID
  //
  static const constexpr unsigned phase_id_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_PHASE_ID];

  bool
  has_phase_id() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, PHASE_ID)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, phase_id_rank, A, COORD_T>
  phase_id() const {
    return
      decltype(phase_id())(
        *m_columns.at(HYPERION_COLUMN_NAME(MAIN, PHASE_ID)));
  }

  //
  // FIELD_ID
  //
  static const constexpr unsigned field_id_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_FIELD_ID];

  bool
  has_field_id() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, FIELD_ID)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, field_id_rank, A, COORD_T>
  field_id() const {
    return
      decltype(field_id())(
        *m_columns.at(HYPERION_COLUMN_NAME(MAIN, FIELD_ID)));
  }

  //
  // INTERVAL
  //
  static const constexpr unsigned interval_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_INTERVAL];

  bool
  has_interval() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, INTERVAL)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_DOUBLE, row_rank, interval_rank, A, COORD_T>
  interval() const {
    return
      decltype(interval())(*m_columns.at(HYPERION_COLUMN_NAME(MAIN, INTERVAL)));
  }

  //
  // EXPOSURE
  //
  static const constexpr unsigned exposure_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_EXPOSURE];

  bool
  has_exposure() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, EXPOSURE)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_DOUBLE, row_rank, exposure_rank, A, COORD_T>
  exposure() const {
    return
      decltype(exposure())(*m_columns.at(HYPERION_COLUMN_NAME(MAIN, EXPOSURE)));
  }

  //
  // TIME_CENTROID
  //
  static const constexpr unsigned time_centroid_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_TIME_CENTROID];

  bool
  has_time_centroid() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, TIME_CENTROID)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    HYPERION_TYPE_DOUBLE,
    row_rank,
    time_centroid_rank,
    A,
    COORD_T>
  time_centroid() const {
    return
      decltype(time_centroid())(
        *m_columns.at(HYPERION_COLUMN_NAME(MAIN, TIME_CENTROID)));
  }

#ifdef HYPERION_USE_CASACORE
  bool
  has_time_centroid_meas() const {
    return has_time_centroid()
      && m_columns.at(HYPERION_COLUMN_NAME(MAIN, TIME_CENTROID))->mr_drs();
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTMD<
    HYPERION_TYPE_DOUBLE,
    MClass::M_EPOCH,
    row_rank,
    time_centroid_rank,
    1,
    A,
    COORD_T>
  time_centroid_meas() const {
    return
      decltype(time_centroid_meas())(
        *m_columns.at(HYPERION_COLUMN_NAME(MAIN, TIME_CENTROID)));
  }
#endif // HYPERION_USE_CASACORE

  //
  // PULSAR_BIN
  //
  static const constexpr unsigned pulsar_bin_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_PULSAR_BIN];

  bool
  has_pulsar_bin() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, PULSAR_BIN)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, pulsar_bin_rank, A, COORD_T>
  pulsar_bin() const {
    return
      decltype(pulsar_bin())(
        *m_columns.at(HYPERION_COLUMN_NAME(MAIN, PULSAR_BIN)));
  }

  //
  // PULSAR_GATE_ID
  //
  static const constexpr unsigned pulsar_gate_id_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_PULSAR_GATE_ID];

  bool
  has_pulsar_gate_id() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, PULSAR_GATE_ID)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, pulsar_gate_id_rank, A, COORD_T>
  pulsar_gate_id() const {
    return
      decltype(pulsar_gate_id())(
        *m_columns.at(HYPERION_COLUMN_NAME(MAIN, PULSAR_GATE_ID)));
  }

  //
  // SCAN_NUMBER
  //
  static const constexpr unsigned scan_number_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_SCAN_NUMBER];

  bool
  has_scan_number() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, SCAN_NUMBER)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, scan_number_rank, A, COORD_T>
  scan_number() const {
    return
      decltype(scan_number())(
        *m_columns.at(HYPERION_COLUMN_NAME(MAIN, SCAN_NUMBER)));
  }

  //
  // ARRAY_ID
  //
  static const constexpr unsigned array_id_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_ARRAY_ID];

  bool
  has_array_id() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, ARRAY_ID)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, array_id_rank, A, COORD_T>
  array_id() const {
    return
      decltype(array_id())(
        *m_columns.at(HYPERION_COLUMN_NAME(MAIN, ARRAY_ID)));
  }

  //
  // OBSERVATION_ID
  //
  static const constexpr unsigned observation_id_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_OBSERVATION_ID];

  bool
  has_observation_id() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, OBSERVATION_ID)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, observation_id_rank, A, COORD_T>
  observation_id() const {
    return
      decltype(observation_id())(
        *m_columns.at(HYPERION_COLUMN_NAME(MAIN, OBSERVATION_ID)));
  }

  //
  // STATE_ID
  //
  static const constexpr unsigned state_id_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_STATE_ID];

  bool
  has_state_id() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, STATE_ID)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, state_id_rank, A, COORD_T>
  state_id() const {
    return
      decltype(state_id())(
        *m_columns.at(HYPERION_COLUMN_NAME(MAIN, STATE_ID)));
  }

  //
  // BASELINE_REF
  //
  static const constexpr unsigned baseline_ref_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_BASELINE_REF];

  bool
  has_baseline_ref() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, BASELINE_REF)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_BOOL, row_rank, baseline_ref_rank, A, COORD_T>
  baseline_ref() const {
    return
      decltype(baseline_ref())(
        *m_columns.at(HYPERION_COLUMN_NAME(MAIN, BASELINE_REF)));
  }

  //
  // UVW
  //
  static const constexpr unsigned uvw_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_UVW];

  bool
  has_uvw() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, UVW)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_BOOL, row_rank, uvw_rank, A, COORD_T>
  uvw() const {
    return decltype(uvw())(*m_columns.at(HYPERION_COLUMN_NAME(MAIN, UVW)));
  }

#ifdef HYPERION_USE_CASACORE
  bool
  has_uvw_meas() const {
    return has_uvw() && m_columns.at(HYPERION_COLUMN_NAME(MAIN, UVW))->mr_drs();
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTMD<
    HYPERION_TYPE_DOUBLE,
    MClass::M_UVW,
    row_rank,
    uvw_rank,
    3,
    A,
    COORD_T>
  uvw_meas() const {
    return decltype(uvw_meas())(*m_columns.at(HYPERION_COLUMN_NAME(MAIN, UVW)));
  }
#endif // HYPERION_USE_CASACORE

  //
  // UVW2
  //
  static const constexpr unsigned uvw2_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_UVW2];

  bool
  has_uvw2() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, UVW2)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_BOOL, row_rank, uvw2_rank, A, COORD_T>
  uvw2() const {
    return decltype(uvw2())(*m_columns.at(HYPERION_COLUMN_NAME(MAIN, UVW2)));
  }

#ifdef HYPERION_USE_CASACORE
  bool
  has_uvw2_meas() const {
    return
      has_uvw2() && m_columns.at(HYPERION_COLUMN_NAME(MAIN, UVW2))->mr_drs();
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTMD<
    HYPERION_TYPE_DOUBLE,
    MClass::M_UVW,
    row_rank,
    uvw2_rank,
    3,
    A,
    COORD_T>
  uvw2_meas() const {
    return
      decltype(uvw2_meas())(*m_columns.at(HYPERION_COLUMN_NAME(MAIN, UVW2)));
  }
#endif // HYPERION_USE_CASACORE

  //
  // DATA
  //
  static const constexpr unsigned data_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_DATA];

  bool
  has_data() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, DATA)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_COMPLEX, row_rank, data_rank, A, COORD_T>
  data() const {
    return decltype(data())(*m_columns.at(HYPERION_COLUMN_NAME(MAIN, DATA)));
  }

  //
  // FLOAT_DATA
  //
  static const constexpr unsigned float_data_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_FLOAT_DATA];

  bool
  has_float_data() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, FLOAT_DATA)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_FLOAT, row_rank, float_data_rank, A, COORD_T>
  float_data() const {
    return
      decltype(float_data())(
        *m_columns.at(HYPERION_COLUMN_NAME(MAIN, FLOAT_DATA)));
  }

  //
  // VIDEO_POINT
  //
  static const constexpr unsigned video_point_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_VIDEO_POINT];

  bool
  has_video_point() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, VIDEO_POINT)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    HYPERION_TYPE_COMPLEX,
    row_rank,
    video_point_rank,
    A,
    COORD_T>
  video_point() const {
    return
      decltype(video_point())(
        *m_columns.at(HYPERION_COLUMN_NAME(MAIN, VIDEO_POINT)));
  }

  //
  // LAG_DATA
  //
  static const constexpr unsigned lag_data_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_LAG_DATA];

  bool
  has_lag_data() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, LAG_DATA)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_COMPLEX, row_rank, lag_data_rank, A, COORD_T>
  lag_data() const {
    return
      decltype(lag_data())(*m_columns.at(HYPERION_COLUMN_NAME(MAIN, LAG_DATA)));
  }

  //
  // SIGMA
  //
  static const constexpr unsigned sigma_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_SIGMA];

  bool
  has_sigma() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, SIGMA)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_FLOAT, row_rank, sigma_rank, A, COORD_T>
  sigma() const {
    return decltype(sigma())(*m_columns.at(HYPERION_COLUMN_NAME(MAIN, SIGMA)));
  }

  //
  // SIGMA_SPECTRUM
  //
  static const constexpr unsigned sigma_spectrum_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_SIGMA_SPECTRUM];

  bool
  has_sigma_spectrum() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, SIGMA_SPECTRUM)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    HYPERION_TYPE_FLOAT,
    row_rank,
    sigma_spectrum_rank,
    A,
    COORD_T>
  sigma_spectrum() const {
    return
      decltype(sigma_spectrum())(
        *m_columns.at(HYPERION_COLUMN_NAME(MAIN, SIGMA_SPECTRUM)));
  }

  //
  // WEIGHT
  //
  static const constexpr unsigned weight_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_WEIGHT];

  bool
  has_weight() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, WEIGHT)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_FLOAT, row_rank, weight_rank, A, COORD_T>
  weight() const {
    return decltype(weight())(*m_columns.at(HYPERION_COLUMN_NAME(MAIN, WEIGHT)));
  }

  //
  // WEIGHT_SPECTRUM
  //
  static const constexpr unsigned weight_spectrum_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_WEIGHT_SPECTRUM];

  bool
  has_weight_spectrum() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, WEIGHT_SPECTRUM)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    HYPERION_TYPE_FLOAT,
    row_rank,
    weight_spectrum_rank,
    A,
    COORD_T>
  weight_spectrum() const {
    return
      decltype(weight_spectrum())(
        *m_columns.at(HYPERION_COLUMN_NAME(MAIN, WEIGHT_SPECTRUM)));
  }

  //
  // FLAG
  //
  static const constexpr unsigned flag_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_FLAG];

  bool
  has_flag() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, FLAG)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_BOOL, row_rank, flag_rank, A, COORD_T>
  flag() const {
    return decltype(flag())(*m_columns.at(HYPERION_COLUMN_NAME(MAIN, FLAG)));
  }

  //
  // FLAG_CATEGORY
  //
  static const constexpr unsigned flag_category_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_FLAG_CATEGORY];

  bool
  has_flag_category() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, FLAG_CATEGORY)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_BOOL, row_rank, flag_category_rank, A, COORD_T>
  flag_category() const {
    return
      decltype(flag_category())(
        *m_columns.at(HYPERION_COLUMN_NAME(MAIN, FLAG_CATEGORY)));
  }

  //
  // FLAG_ROW
  //
  static const constexpr unsigned flag_row_rank =
    row_rank + C::element_ranks[C::col_t::MS_MAIN_COL_FLAG_ROW];

  bool
  has_flag_row() const {
    return m_columns.count(HYPERION_COLUMN_NAME(MAIN, FLAG_ROW)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_BOOL, row_rank, flag_row_rank, A, COORD_T>
  flag_row() const {
    return
      decltype(flag_row())(
        *m_columns.at(HYPERION_COLUMN_NAME(MAIN, FLAG_ROW)));
  }

};

} // end namespace hyperion

#endif // HYPERION_MAIN_TABLE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
