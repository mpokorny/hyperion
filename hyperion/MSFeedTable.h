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
#ifndef HYPERION_MS_FEED_TABLE_H_
#define HYPERION_MS_FEED_TABLE_H_

#include <hyperion/hyperion.h>
#include <hyperion/PhysicalTable.h>
#include <hyperion/PhysicalColumn.h>
#include <hyperion/MSTableColumns.h>
#include <hyperion/MSTable.h>

#ifdef HYPERION_USE_CASACORE
# include <casacore/measures/Measures/MEpoch.h>
# include <casacore/measures/Measures/MCEpoch.h>
# include <casacore/measures/Measures/MPosition.h>
# include <casacore/measures/Measures/MCPosition.h>
# include <casacore/measures/Measures/MDirection.h>
# include <casacore/measures/Measures/MCDirection.h>
#endif // HYPERION_USE_CASACORE

#include <unordered_map>

namespace hyperion {

template <MSTable<MS_FEED>::Axes ...AXES>
class MSFeedTable
  : public PhysicalTable {
public:

  typedef table_indexing<MSTable<MS_FEED>::Axes> feed_indexing;

  typedef MSTableColumns<MS_FEED> C;

  MSFeedTable(const PhysicalTable& pt)
    : PhysicalTable(pt) {
    assert(pt.axes_uid()
           && (pt.axes_uid().value()
               == Axes<typename MSTable<MS_FEED>::Axes>::uid));
    assert(pt.index_axes() == std::vector<int>{static_cast<int>(AXES)...});
  }

  static const constexpr unsigned row_rank = sizeof...(AXES);

  //
  // ANTENNA_ID
  //
  static const constexpr unsigned antenna_id_rank =
    std::conditional<
      feed_indexing::includes<FEED_ANTENNA_ID, AXES...>,
      std::integral_constant<unsigned, 1>,
      std::integral_constant<unsigned, row_rank>>::type::value
    + C::element_ranks[C::col_t::MS_FEED_COL_ANTENNA_ID];

  bool
  has_antenna_id() const {
    return m_columns.count(HYPERION_COLUMN_NAME(FEED, ANTENNA_ID)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    HYPERION_TYPE_INT,
    antenna_id_rank - C::element_ranks[C::col_t::MS_FEED_COL_ANTENNA_ID],
    antenna_id_rank,
    A,
    COORD_T>
  antenna_id() const {
    return
      decltype(antenna_id<A, COORD_T>())(
        *m_columns.at(HYPERION_COLUMN_NAME(FEED, ANTENNA_ID)));
  }

  //
  // FEED_ID
  //
  static const constexpr unsigned feed_id_rank =
    std::conditional<
      feed_indexing::includes<FEED_FEED_ID, AXES...>,
      std::integral_constant<unsigned, 1>,
      std::integral_constant<unsigned, row_rank>>::type::value
    + C::element_ranks[C::col_t::MS_FEED_COL_FEED_ID];

  bool
  has_feed_id() const {
    return m_columns.count(HYPERION_COLUMN_NAME(FEED, FEED_ID)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    HYPERION_TYPE_INT,
    feed_id_rank - C::element_ranks[C::col_t::MS_FEED_COL_FEED_ID],
    feed_id_rank,
    A,
    COORD_T>
  feed_id() const {
    return
      decltype(feed_id<A, COORD_T>())(
        *m_columns.at(HYPERION_COLUMN_NAME(FEED, FEED_ID)));
  }

  //
  // SPECTRAL_WINDOW_ID
  //
  static const constexpr unsigned spectral_window_id_rank =
    std::conditional<
      feed_indexing::includes<FEED_SPECTRAL_WINDOW_ID, AXES...>,
      std::integral_constant<unsigned, 1>,
      std::integral_constant<unsigned, row_rank>>::type::value
    + C::element_ranks[C::col_t::MS_FEED_COL_SPECTRAL_WINDOW_ID];

  bool
  has_spectral_window_id() const {
    return m_columns.count(HYPERION_COLUMN_NAME(FEED, SPECTRAL_WINDOW_ID)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    HYPERION_TYPE_INT,
    spectral_window_id_rank
      - C::element_ranks[C::col_t::MS_FEED_COL_SPECTRAL_WINDOW_ID],
    spectral_window_id_rank,
    A,
    COORD_T>
  spectral_window_id() const {
    return
      decltype(spectral_window_id<A, COORD_T>())(
        *m_columns.at(HYPERION_COLUMN_NAME(FEED, SPECTRAL_WINDOW_ID)));
  }

  //
  // TIME
  //
  static const constexpr unsigned time_rank =
    std::conditional<
      feed_indexing::includes<FEED_TIME, AXES...>,
      std::integral_constant<unsigned, 1>,
      std::integral_constant<unsigned, row_rank>>::type::value
    + C::element_ranks[C::col_t::MS_FEED_COL_TIME];

  bool
  has_time() const {
    return m_columns.count(HYPERION_COLUMN_NAME(FEED, TIME)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    HYPERION_TYPE_DOUBLE,
    time_rank - C::element_ranks[C::col_t::MS_FEED_COL_TIME],
    time_rank,
    A,
    COORD_T>
  time() const {
    return decltype(time<A, COORD_T>())(
      *m_columns.at(HYPERION_COLUMN_NAME(FEED, TIME)));
  }

#ifdef HYPERION_USE_CASACORE
  bool
  has_time_meas() const {
    return has_time()
      && m_columns.at(HYPERION_COLUMN_NAME(FEED, TIME))->mr_drs();
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTMD<
    HYPERION_TYPE_DOUBLE,
    MClass::M_EPOCH,
    time_rank - C::element_ranks[C::col_t::MS_FEED_COL_TIME],
    time_rank,
    1,
    A,
    COORD_T>
  time_meas() const {
    return
      decltype(time_meas<A, COORD_T>())(
        *m_columns.at(HYPERION_COLUMN_NAME(FEED, TIME)));
  }
#endif // HYPERION_USE_CASACORE

  //
  // INTERVAL
  //
  static const constexpr unsigned interval_rank =
    std::conditional<
      feed_indexing::includes<FEED_INTERVAL, AXES...>,
      std::integral_constant<unsigned, 1>,
      std::integral_constant<unsigned, row_rank>>::type::value
    + C::element_ranks[C::col_t::MS_FEED_COL_INTERVAL];

  bool
  has_interval() const {
    return m_columns.count(HYPERION_COLUMN_NAME(FEED, INTERVAL)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    HYPERION_TYPE_DOUBLE,
    interval_rank - C::element_ranks[C::col_t::MS_FEED_COL_INTERVAL],
    interval_rank,
    A,
    COORD_T>
  interval() const {
    return
      decltype(interval<A, COORD_T>())(
        *m_columns.at(HYPERION_COLUMN_NAME(FEED, INTERVAL)));
  }

  //
  // NUM_RECEPTORS
  //
  static const constexpr unsigned num_receptors_rank =
    row_rank + C::element_ranks[C::col_t::MS_FEED_COL_NUM_RECEPTORS];

  bool
  has_num_receptors() const {
    return m_columns.count(HYPERION_COLUMN_NAME(FEED, NUM_RECEPTORS)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, num_receptors_rank, A, COORD_T>
  num_receptors() const {
    return
      decltype(num_receptors<A, COORD_T>())(
        *m_columns.at(HYPERION_COLUMN_NAME(FEED, NUM_RECEPTORS)));
  }

  //
  // BEAM_ID
  //
  static const constexpr unsigned beam_id_rank =
    row_rank + C::element_ranks[C::col_t::MS_FEED_COL_BEAM_ID];

  bool
  has_beam_id() const {
    return m_columns.count(HYPERION_COLUMN_NAME(FEED, BEAM_ID)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, beam_id_rank, A, COORD_T>
  beam_id() const {
    return
      decltype(beam_id<A, COORD_T>())(
        *m_columns.at(HYPERION_COLUMN_NAME(FEED, BEAM_ID)));
  }

  //
  // BEAM_OFFSET
  //
  static const constexpr unsigned beam_offset_rank =
    row_rank + C::element_ranks[C::col_t::MS_FEED_COL_BEAM_OFFSET];

  bool
  has_beam_offset() const {
    return m_columns.count(HYPERION_COLUMN_NAME(FEED, BEAM_OFFSET)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_DOUBLE, row_rank, beam_offset_rank, A, COORD_T>
  beam_offset() const {
    return decltype(beam_offset<A, COORD_T>())(
      *m_columns.at(HYPERION_COLUMN_NAME(FEED, BEAM_OFFSET)));
  }

#ifdef HYPERION_USE_CASACORE
  bool
  has_beam_offset_meas() const {
    return has_beam_offset()
      && m_columns.at(HYPERION_COLUMN_NAME(FEED, BEAM_OFFSET))->mr_drs();
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTMD<
    HYPERION_TYPE_DOUBLE,
    MClass::M_DIRECTION,
    row_rank,
    beam_offset_rank,
    2,
    A,
    COORD_T>
  beam_offset_meas() const {
    return
      decltype(beam_offset_meas<A, COORD_T>())(
        *m_columns.at(HYPERION_COLUMN_NAME(FEED, BEAM_OFFSET)));
  }
#endif // HYPERION_USE_CASACORE

  //
  // FOCUS_LENGTH
  //
  static const constexpr unsigned focus_length_rank =
    row_rank + C::element_ranks[C::col_t::MS_FEED_COL_FOCUS_LENGTH];

  bool
  has_focus_length() const {
    return m_columns.count(HYPERION_COLUMN_NAME(FEED, FOCUS_LENGTH)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    HYPERION_TYPE_DOUBLE,
    row_rank,
    focus_length_rank,
    A,
    COORD_T>
  focus_length() const {
    return
      decltype(focus_length<A, COORD_T>())(
        *m_columns.at(HYPERION_COLUMN_NAME(FEED, FOCUS_LENGTH)));
  }

  //
  // PHASED_FEED_ID
  //
  static const constexpr unsigned phased_feed_id_rank =
    row_rank + C::element_ranks[C::col_t::MS_FEED_COL_PHASED_FEED_ID];

  bool
  has_phased_feed_id() const {
    return m_columns.count(HYPERION_COLUMN_NAME(FEED, PHASED_FEED_ID)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_INT, row_rank, phased_feed_id_rank, A, COORD_T>
  phased_feed_id() const {
    return
      decltype(phased_feed_id<A, COORD_T>())(
        *m_columns.at(HYPERION_COLUMN_NAME(FEED, PHASED_FEED_ID)));
  }

  //
  // POLARIZATION_TYPE
  //
  static const constexpr unsigned polarization_type_rank =
    row_rank + C::element_ranks[C::col_t::MS_FEED_COL_POLARIZATION_TYPE];

  bool
  has_polarization_type() const {
    return m_columns.count(HYPERION_COLUMN_NAME(FEED, POLARIZATION_TYPE)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    HYPERION_TYPE_STRING,
    row_rank,
    polarization_type_rank,
    A,
    COORD_T>
  polarization_type() const {
    return
      decltype(polarization_type<A, COORD_T>())(
        *m_columns.at(HYPERION_COLUMN_NAME(FEED, POLARIZATION_TYPE)));
  }

  //
  // POL_RESPONSE
  //
  static const constexpr unsigned pol_response_rank =
    row_rank + C::element_ranks[C::col_t::MS_FEED_COL_POL_RESPONSE];

  bool
  has_pol_response() const {
    return m_columns.count(HYPERION_COLUMN_NAME(FEED, POL_RESPONSE)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    HYPERION_TYPE_COMPLEX,
    row_rank,
    pol_response_rank,
    A,
    COORD_T>
  pol_response() const {
    return
      decltype(pol_response<A, COORD_T>())(
        *m_columns.at(HYPERION_COLUMN_NAME(FEED, POL_RESPONSE)));
  }

  //
  // POSITION
  //
  static const constexpr unsigned position_rank =
    row_rank + C::element_ranks[C::col_t::MS_FEED_COL_POSITION];

  bool
  has_position() const {
    return m_columns.count(HYPERION_COLUMN_NAME(FEED, POSITION)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<HYPERION_TYPE_DOUBLE, row_rank, position_rank, A, COORD_T>
  position() const {
    return decltype(position<A, COORD_T>())(
      *m_columns.at(HYPERION_COLUMN_NAME(FEED, POSITION)));
  }

#ifdef HYPERION_USE_CASACORE
  bool
  has_position_meas() const {
    return has_position()
      && m_columns.at(HYPERION_COLUMN_NAME(FEED, POSITION))->mr_drs();
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
      decltype(position_meas<A, COORD_T>())(
        *m_columns.at(HYPERION_COLUMN_NAME(FEED, POSITION)));
  }
#endif // HYPERION_USE_CASACORE

  //
  // RECEPTOR_ANGLE
  //
  static const constexpr unsigned receptor_angle_rank =
    row_rank + C::element_ranks[C::col_t::MS_FEED_COL_RECEPTOR_ANGLE];

  bool
  has_receptor_angle() const {
    return m_columns.count(HYPERION_COLUMN_NAME(FEED, RECEPTOR_ANGLE)) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    HYPERION_TYPE_DOUBLE,
    row_rank,
    receptor_angle_rank,
    A,
    COORD_T>
  receptor_angle() const {
    return
      decltype(receptor_angle<A, COORD_T>())(
        *m_columns.at(HYPERION_COLUMN_NAME(FEED, RECEPTOR_ANGLE)));
  }

};

} // end namespace hyperion

#endif // HYPERION_FEED_TABLE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
