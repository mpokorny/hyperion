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
#ifndef HYPERION_MS_SP_WINDOW_TABLE_H_
#define HYPERION_MS_SP_WINDOW_TABLE_H_

#include <hyperion/hyperion.h>
#include <hyperion/PhysicalTable.h>
#include <hyperion/PhysicalColumn.h>
#include <hyperion/MSTableColumns.h>

#ifdef HYPERION_USE_CASACORE
# include <casacore/measures/Measures/MFrequency.h>
# include <casacore/measures/Measures/MCFrequency.h>
#endif // HYPERION_USE_CASACORE

#include <any>
#include <unordered_map>
#include <vector>

namespace hyperion {

class HYPERION_API MSSpWindowTable
  : public PhysicalTable {
public:

  typedef MSTableColumns<MS_SPECTRAL_WINDOW> C;

  MSSpWindowTable(const PhysicalTable& pt)
    : PhysicalTable(pt) {
    assert(
      pt.axes_uid()
      && (pt.axes_uid().value()
          == Axes<typename MSTable<MS_SPECTRAL_WINDOW>::Axes>::uid));
    assert(
      pt.index_axes()
      == std::vector<int>{static_cast<int>(SPECTRAL_WINDOW_ROW)});
  }

  static const constexpr unsigned row_rank = 1;

  //
  // NUM_CHAN
  //
  static const constexpr unsigned num_chan_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_NUM_CHAN];

  bool
  has_num_chan() const {
    return m_columns.count(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, NUM_CHAN)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, num_chan_rank, A, COORD_T>
  num_chan() const {
    return
      decltype(num_chan())(
        *m_columns.at(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, NUM_CHAN)));
  }

  //
  // NAME
  //
  static const constexpr unsigned name_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_NAME];

  bool
  has_name() const {
    return m_columns.count(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, NAME)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_STRING, row_rank, name_rank, A, COORD_T>
  name() const {
    return
      decltype(name())(
        *m_columns.at(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, NAME)));
  }

  //
  // REF_FREQUENCY
  //
  static const constexpr unsigned ref_frequency_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_REF_FREQUENCY];

  bool
  has_ref_frequency() const {
    return
      m_columns.count(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, REF_FREQUENCY)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    HYPERION_TYPE_DOUBLE,
    row_rank,
    ref_frequency_rank,
    A,
    COORD_T>
  ref_frequency() const {
    return
      decltype(ref_frequency())(
        *m_columns.at(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, REF_FREQUENCY)));
  }

#ifdef HYPERION_USE_CASACORE
  bool
  has_ref_frequency_meas() const {
    return has_ref_frequency() &&
      m_columns
      .at(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, REF_FREQUENCY))->mr_drs();
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTMD<
    HYPERION_TYPE_DOUBLE,
    MClass::M_FREQUENCY,
    row_rank,
    ref_frequency_rank,
    1,
    A,
    COORD_T>
  ref_frequency_meas() const {
    return
      decltype(ref_frequency_meas())(
        *m_columns
        .at(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, REF_FREQUENCY)));
  }
#endif // HYPERION_USE_CASACORE

  //
  // CHAN_FREQ
  //
  static const constexpr unsigned chan_freq_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_CHAN_FREQ];

  bool
  has_chan_freq() const {
    return
      m_columns.count(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, CHAN_FREQ)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_DOUBLE, row_rank, chan_freq_rank, A, COORD_T>
  chan_freq() const {
    return
      decltype(chan_freq())(
        *m_columns.at(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, CHAN_FREQ)));
  }

#ifdef HYPERION_USE_CASACORE
  bool
  has_chan_freq_meas() const {
    return has_chan_freq() &&
      m_columns.at(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, CHAN_FREQ))->mr_drs();
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTMD<
    HYPERION_TYPE_DOUBLE,
    MClass::M_FREQUENCY,
    row_rank,
    chan_freq_rank,
    1,
    A,
    COORD_T>
  chan_freq_meas() const {
    return
      decltype(chan_freq_meas())(
        *m_columns.at(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, CHAN_FREQ)));
  }
#endif

  //
  // CHAN_WIDTH
  //
  static const constexpr unsigned chan_width_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_CHAN_WIDTH];

  bool
  has_chan_width() const {
    return
      m_columns.count(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, CHAN_WIDTH)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_DOUBLE, row_rank, chan_width_rank, A, COORD_T>
  chan_width() const {
    return
      decltype(chan_width())(
        *m_columns.at(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, CHAN_WIDTH)));
  }

  //
  // EFFECTIVE_BW
  //
  static const constexpr unsigned effective_bw_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_EFFECTIVE_BW];

  bool
  has_effective_bw() const {
    return
      m_columns.count(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, EFFECTIVE_BW)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    HYPERION_TYPE_DOUBLE,
    row_rank,
    effective_bw_rank,
    A,
    COORD_T>
  effective_bw() const {
    return
      decltype(effective_bw())(
        *m_columns.at(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, EFFECTIVE_BW)));
  }

  //
  // RESOLUTION
  //
  static const constexpr unsigned resolution_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_RESOLUTION];

  bool
  has_resolution() const {
    return
      m_columns.count(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, RESOLUTION)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_DOUBLE, row_rank, resolution_rank, A, COORD_T>
  resolution() const {
    return
      decltype(resolution())(
        *m_columns.at(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, RESOLUTION)));
  }

  //
  // TOTAL_BANDWIDTH
  //
  static const constexpr unsigned total_bandwidth_rank =
    row_rank
    + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_TOTAL_BANDWIDTH];

  bool
  has_total_bandwidth() const {
    return
      m_columns
      .count(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, TOTAL_BANDWIDTH)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    HYPERION_TYPE_DOUBLE,
    row_rank,
    total_bandwidth_rank,
    A,
    COORD_T>
  total_bandwidth() const {
    return
      decltype(total_bandwidth())(
        *m_columns.at(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, TOTAL_BANDWIDTH)));
  }

  //
  // NET_SIDEBAND
  //
  static const constexpr unsigned net_sideband_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_NET_SIDEBAND];

  bool
  has_net_sideband() const {
    return
      m_columns.count(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, NET_SIDEBAND)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, net_sideband_rank, A, COORD_T>
  net_sideband() const {
    return
      decltype(net_sideband())(
        *m_columns.at(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, NET_SIDEBAND)));
  }

  //
  // BBC_NO
  //
  static const constexpr unsigned bbc_no_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_BBC_NO];

  bool
  has_bbc_no() const {
    return m_columns.count(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, BBC_NO)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, bbc_no_rank, A, COORD_T>
  bbc_no() const {
    return
      decltype(bbc_no())(
        *m_columns.at(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, BBC_NO)));
  }

  //
  // BBC_SIDEBAND
  //
  static const constexpr unsigned bbc_sideband_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_BBC_SIDEBAND];
  bool
  has_bbc_sideband() const {
    return
      m_columns.count(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, BBC_SIDEBAND)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, bbc_sideband_rank, A, COORD_T>
  bbc_sideband() const {
    return
      decltype(bbc_sideband())(
        *m_columns.at(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, BBC_SIDEBAND)));
  }

  //
  // IF_CONV_CHAIN
  //
  static const constexpr unsigned if_conv_chain_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_IF_CONV_CHAIN];

  bool
  has_if_conv_chain() const {
    return
      m_columns.count(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, IF_CONV_CHAIN)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, if_conv_chain_rank, A, COORD_T>
  if_conv_chain() const {
    return
      decltype(if_conv_chain())(
        *m_columns.at(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, IF_CONV_CHAIN)));
  }

  //
  // RECEIVER_ID
  //
  static const constexpr unsigned receiver_id_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_RECEIVER_ID];

  bool
  has_receiver_id() const {
    return
      m_columns.count(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, RECEIVER_ID)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, receiver_id_rank, A, COORD_T>
  receiver_id() const {
    return
      decltype(receiver_id())(
        *m_columns.at(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, RECEIVER_ID)));
  }

  //
  // FREQ_GROUP
  //
  static const constexpr unsigned freq_group_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_FREQ_GROUP];

  bool
  has_freq_group() const {
    return
      m_columns.count(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, FREQ_GROUP)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, freq_group_rank, A, COORD_T>
  freq_group() const {
    return
      decltype(freq_group())(
        *m_columns.at(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, FREQ_GROUP)));
  }

  //
  // FREQ_GROUP_NAME
  //
  static const constexpr unsigned freq_group_name_rank =
    row_rank
    + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_FREQ_GROUP_NAME];

  bool
  has_freq_group_name() const {
    return
      m_columns
      .count(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, FREQ_GROUP_NAME)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    HYPERION_TYPE_STRING,
    row_rank,
    freq_group_name_rank,
    A,
    COORD_T>
  freq_group_name() const {
    return
      decltype(freq_group_name())(
        *m_columns.at(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, FREQ_GROUP_NAME)));
  }

  //
  // DOPPLER_ID
  //
  static const constexpr unsigned doppler_id_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_DOPPLER_ID];

  bool
  has_doppler_id() const {
    return
      m_columns.count(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, DOPPLER_ID)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, doppler_id_rank, A, COORD_T>
  doppler_id() const {
    return
      decltype(doppler_id())(
        *m_columns.at(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, DOPPLER_ID)));
  }

  //
  // ASSOC_SPW_ID
  //
  static const constexpr unsigned assoc_spw_id_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_ASSOC_SPW_ID];

  bool
  has_assoc_spw_id() const {
    return
      m_columns.count(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, ASSOC_SPW_ID)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, assoc_spw_id_rank, A, COORD_T>
  assoc_spw_id() const {
    return
      decltype(assoc_spw_id())(
        *m_columns.at(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, ASSOC_SPW_ID)));
  }

  //
  // ASSOC_NATURE
  //
  static const constexpr unsigned assoc_nature_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_ASSOC_NATURE];

  bool
  has_assoc_nature() const {
    return
      m_columns.count(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, ASSOC_NATURE)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    HYPERION_TYPE_STRING,
    row_rank,
    assoc_nature_rank,
    A,
    COORD_T>
  assoc_nature() const {
    return
      decltype(assoc_nature())(
        *m_columns.at(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, ASSOC_NATURE)));
  }

  //
  // FLAG_ROW
  //
  static const constexpr unsigned flag_row_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_FLAG_ROW];

  bool
  has_flag_row() const {
    return m_columns.count(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, FLAG_ROW)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_BOOL, row_rank, flag_row_rank, A, COORD_T>
  flag_row() const {
    return
      decltype(flag_row())(
        *m_columns.at(HYPERION_COLUMN_NAME(SPECTRAL_WINDOW, FLAG_ROW)));
  }
};

} // end namespace hyperion

#endif // HYPERION_MS_SP_WINDOW_TABLE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
