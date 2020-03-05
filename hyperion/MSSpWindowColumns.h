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
#ifndef HYPERION_MS_SP_WINDOW_COLUMNS_H_
#define HYPERION_MS_SP_WINDOW_COLUMNS_H_

#include <hyperion/hyperion.h>
#include <hyperion/Column.h>
#include <hyperion/MSTableColumns.h>

#pragma GCC visibility push(default)
# include <casacore/measures/Measures/MFrequency.h>
# include <casacore/measures/Measures/MCFrequency.h>

# include <any>
# include <memory>
# include <optional>
# include <unordered_map>
# include <vector>
#pragma GCC visibility pop

namespace hyperion {

class HYPERION_API MSSpWindowColumns
  : public MSTableColumnsBase {
public:

  typedef MSTableColumns<MS_SPECTRAL_WINDOW> C;

  MSSpWindowColumns(
    Legion::Runtime* rt,
    const Legion::RegionRequirement& rows_requirement,
    const std::unordered_map<std::string, Regions>& regions);

  static const constexpr unsigned row_rank = 1;

  Legion::DomainT<row_rank>
  rows() const {
    return m_rows;
  }

  //
  // NUM_CHAN
  //
  static const constexpr unsigned num_chan_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_NUM_CHAN];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using NumChanAccessor =
    FieldAccessor<HYPERION_TYPE_INT, num_chan_rank, MODE, CHECK_BOUNDS>;

  bool
  has_num_chan() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_NUM_CHAN) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  NumChanAccessor<MODE, CHECK_BOUNDS>
  num_chan() const {
    return
      NumChanAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_NUM_CHAN),
        C::fid(C::col_t::MS_SPECTRAL_WINDOW_COL_NUM_CHAN));
  }

  //
  // NAME
  //
  static const constexpr unsigned name_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_NAME];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using NameAccessor =
    FieldAccessor<HYPERION_TYPE_STRING, name_rank, MODE, CHECK_BOUNDS>;

  bool
  has_name() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_NAME) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  NameAccessor<MODE, CHECK_BOUNDS>
  name() const {
    return
      NameAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_NAME),
        C::fid(C::col_t::MS_SPECTRAL_WINDOW_COL_NAME));
  }

  //
  // REF_FREQUENCY
  //
  static const constexpr unsigned ref_frequency_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_REF_FREQUENCY];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using RefFrequencyAccessor =
    FieldAccessor<HYPERION_TYPE_DOUBLE, ref_frequency_rank, MODE, CHECK_BOUNDS>;

  bool
  has_ref_frequency() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_REF_FREQUENCY) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  RefFrequencyAccessor<MODE, CHECK_BOUNDS>
  ref_frequency() const {
    return
      RefFrequencyAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_REF_FREQUENCY),
        C::fid(C::col_t::MS_SPECTRAL_WINDOW_COL_REF_FREQUENCY));
  }

#ifdef HYPERION_USE_CASACORE
  template <typename T>
  class RefFrequencyMeasWriterMixin
    : public T {
  public:
    using T::T;

    void
    write(
      const Legion::Point<ref_frequency_rank, Legion::coord_t>& pt,
      const casacore::MFrequency& val) {

      auto cvt = T::m_cm.convert_at(pt);
      auto f = cvt(val);
      T::m_ref_frequency[pt] = f.get(T::m_units).getValue();
    }
  };

  template <typename T>
  class RefFrequencyMeasReaderMixin
    : public T {
  public:
    using T::T;

    casacore::MFrequency
    read(const Legion::Point<ref_frequency_rank, Legion::coord_t>& pt) const {

      const DataType<HYPERION_TYPE_DOUBLE>::ValueType& f =
        T::m_ref_frequency[pt];
      auto mr = T::m_cm.meas_ref_at(pt);
      return casacore::MFrequency(casacore::Quantity(f, T::m_units), mr);
    }
  };

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  class RefFrequencyMeasAccessorBase {
  public:
    RefFrequencyMeasAccessorBase(
      const Legion::PhysicalRegion& region,
      const mr_t<casacore::MFrequency>* mr)
      : m_ref_frequency(
        region,
        C::fid(C::col_t::MS_SPECTRAL_WINDOW_COL_REF_FREQUENCY))
      , m_units(C::units.at(C::col_t::MS_SPECTRAL_WINDOW_COL_REF_FREQUENCY))
      , m_cm(mr) {
    }

  protected:

    RefFrequencyAccessor<MODE, CHECK_BOUNDS> m_ref_frequency;

    const char* m_units;

    ColumnMeasure<
      casacore::MFrequency,
      row_rank,
      ref_frequency_rank,
      READ_ONLY,
      CHECK_BOUNDS> m_cm;
  };

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  class RefFrequencyMeasAccessor
    : public RefFrequencyMeasWriterMixin<
        RefFrequencyMeasAccessorBase<MODE, CHECK_BOUNDS>> {
    // this implementation supports MODE=WRITE_ONLY and MODE=WRITE_DISCARD
    typedef RefFrequencyMeasWriterMixin<
      RefFrequencyMeasAccessorBase<MODE, CHECK_BOUNDS>> T;
  public:
    using T::T;
  };

  template <bool CHECK_BOUNDS>
  class RefFrequencyMeasAccessor<READ_ONLY, CHECK_BOUNDS>
    : public RefFrequencyMeasReaderMixin<
        RefFrequencyMeasAccessorBase<READ_ONLY, CHECK_BOUNDS>> {
    typedef RefFrequencyMeasReaderMixin<
      RefFrequencyMeasAccessorBase<READ_ONLY, CHECK_BOUNDS>> T;
  public:
    using T::T;
  };

  template <bool CHECK_BOUNDS>
  class RefFrequencyMeasAccessor<READ_WRITE, CHECK_BOUNDS>
    : public RefFrequencyMeasReaderMixin<
        RefFrequencyMeasWriterMixin<
          RefFrequencyMeasAccessorBase<READ_WRITE, CHECK_BOUNDS>>> {
    typedef RefFrequencyMeasReaderMixin<
      RefFrequencyMeasWriterMixin<
        RefFrequencyMeasAccessorBase<READ_WRITE, CHECK_BOUNDS>>> T;
  public:
    using T::T;
  };

  bool
  has_ref_frequency_meas() const {
    return
      has_ref_frequency()
      && m_mrs.count(C::col_t::MS_SPECTRAL_WINDOW_COL_REF_FREQUENCY) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  RefFrequencyMeasAccessor<MODE, CHECK_BOUNDS>
  ref_frequency_meas() const {
    return
      RefFrequencyMeasAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_REF_FREQUENCY),
        std::any_cast<mr_t<casacore::MFrequency>>(
          &m_mrs.at(C::col_t::MS_SPECTRAL_WINDOW_COL_REF_FREQUENCY)));
  }
#endif // HYPERION_USE_CASACORE

  //
  // CHAN_FREQ
  //
  static const constexpr unsigned chan_freq_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_CHAN_FREQ];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using ChanFreqAccessor =
    FieldAccessor<HYPERION_TYPE_DOUBLE, chan_freq_rank, MODE, CHECK_BOUNDS>;

  bool
  has_chan_freq() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_CHAN_FREQ) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  ChanFreqAccessor<MODE, CHECK_BOUNDS>
  chan_freq() const {
    return
      ChanFreqAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_CHAN_FREQ),
        C::fid(C::col_t::MS_SPECTRAL_WINDOW_COL_CHAN_FREQ));
  }

#ifdef HYPERION_USE_CASACORE
  template <typename T>
  class ChanFreqMeasWriterMixin
    : public T {
  public:
    using T::T;

    void
    write(
      const Legion::Point<chan_freq_rank, Legion::coord_t>& pt,
      const casacore::MFrequency& val) {

      auto cvt = T::m_cm.convert_at(pt);
      auto f = cvt(val);
      T::m_chan_freq[pt] = f.get(T::m_units).getValue();
    }
  };

  template <typename T>
  class ChanFreqMeasReaderMixin
    : public T {
  public:
    using T::T;

    casacore::MFrequency
    read(const Legion::Point<chan_freq_rank, Legion::coord_t>& pt) const {
      const DataType<HYPERION_TYPE_DOUBLE>::ValueType& f =
        T::m_chan_freq[pt];

      auto mr = T::m_cm.meas_ref_at(pt);
      return casacore::MFrequency(casacore::Quantity(f, T::m_units), mr);
    }
  };

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  class ChanFreqMeasAccessorBase {
  public:
    ChanFreqMeasAccessorBase(
      const Legion::PhysicalRegion& chan_freq_region,
      const mr_t<casacore::MFrequency>* mr)
      : m_chan_freq(
        chan_freq_region,
        C::fid(C::col_t::MS_SPECTRAL_WINDOW_COL_CHAN_FREQ))
      , m_units(C::units.at(C::col_t::MS_SPECTRAL_WINDOW_COL_CHAN_FREQ))
      , m_cm(mr) {
    }

  protected:

    ChanFreqAccessor<MODE, CHECK_BOUNDS> m_chan_freq;

    const char* m_units;

    ColumnMeasure<
      casacore::MFrequency,
      row_rank,
      chan_freq_rank,
      READ_ONLY,
      CHECK_BOUNDS> m_cm;
  };

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  class ChanFreqMeasAccessor
    : public ChanFreqMeasWriterMixin<
        ChanFreqMeasAccessorBase<MODE, CHECK_BOUNDS>> {
    // this implementation supports MODE=WRITE_ONLY and MODE=WRITE_DISCARD
    typedef ChanFreqMeasWriterMixin<
      ChanFreqMeasAccessorBase<MODE, CHECK_BOUNDS>> T;
  public:
    using T::T;
  };

  template <bool CHECK_BOUNDS>
  class ChanFreqMeasAccessor<READ_ONLY, CHECK_BOUNDS>
    : public ChanFreqMeasReaderMixin<
        ChanFreqMeasAccessorBase<READ_ONLY, CHECK_BOUNDS>> {
    typedef ChanFreqMeasReaderMixin<
      ChanFreqMeasAccessorBase<READ_ONLY, CHECK_BOUNDS>> T;
  public:
    using T::T;
  };

  template <bool CHECK_BOUNDS>
  class ChanFreqMeasAccessor<READ_WRITE, CHECK_BOUNDS>
    : public ChanFreqMeasReaderMixin<
        ChanFreqMeasWriterMixin<
          ChanFreqMeasAccessorBase<READ_WRITE, CHECK_BOUNDS>>> {
    typedef ChanFreqMeasReaderMixin<
      ChanFreqMeasWriterMixin<
        ChanFreqMeasAccessorBase<READ_WRITE, CHECK_BOUNDS>>> T;
  public:
    using T::T;
  };

  bool
  has_chan_freq_meas() const {
    return
      has_chan_freq()
      && m_mrs.count(C::col_t::MS_SPECTRAL_WINDOW_COL_CHAN_FREQ) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  ChanFreqMeasAccessor<MODE, CHECK_BOUNDS>
  chan_freq_meas() const {
    return
      ChanFreqMeasAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_CHAN_FREQ),
        std::any_cast<mr_t<casacore::MFrequency>>(
          &m_mrs.at(C::col_t::MS_SPECTRAL_WINDOW_COL_CHAN_FREQ)));
  }
#endif

  //
  // CHAN_WIDTH
  //
  static const constexpr unsigned chan_width_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_CHAN_WIDTH];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using ChanWidthAccessor =
    FieldAccessor<HYPERION_TYPE_DOUBLE, chan_width_rank, MODE, CHECK_BOUNDS>;

  bool
  has_chan_width() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_CHAN_WIDTH) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  ChanWidthAccessor<MODE, CHECK_BOUNDS>
  chan_width() const {
    return
      ChanWidthAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_CHAN_WIDTH),
        C::fid(C::col_t::MS_SPECTRAL_WINDOW_COL_CHAN_WIDTH));
  }

  //
  // EFFECTIVE_BW
  //
  static const constexpr unsigned effective_bw_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_EFFECTIVE_BW];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using EffectiveBWAccessor =
    FieldAccessor<HYPERION_TYPE_DOUBLE, effective_bw_rank, MODE, CHECK_BOUNDS>;

  bool
  has_effective_bw() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_EFFECTIVE_BW) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  EffectiveBWAccessor<MODE, CHECK_BOUNDS>
  effective_bw() const {
    return
      EffectiveBWAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_EFFECTIVE_BW),
        C::fid(C::col_t::MS_SPECTRAL_WINDOW_COL_EFFECTIVE_BW));
  }

  //
  // RESOLUTION
  //
  static const constexpr unsigned resolution_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_RESOLUTION];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using ResolutionAccessor =
    FieldAccessor<HYPERION_TYPE_DOUBLE, resolution_rank, MODE, CHECK_BOUNDS>;

  bool
  has_resolution() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_RESOLUTION) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  ResolutionAccessor<MODE, CHECK_BOUNDS>
  resolution() const {
    return
      ResolutionAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_RESOLUTION),
        C::fid(C::col_t::MS_SPECTRAL_WINDOW_COL_RESOLUTION));
  }

  //
  // TOTAL_BANDWIDTH
  //
  static const constexpr unsigned total_bandwidth_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_TOTAL_BANDWIDTH];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using TotalBandwidthAccessor =
    FieldAccessor<
      HYPERION_TYPE_DOUBLE,
      total_bandwidth_rank,
      MODE,
      CHECK_BOUNDS>;

  bool
  has_total_bandwidth() const {
    return
      m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_TOTAL_BANDWIDTH) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  TotalBandwidthAccessor<MODE, CHECK_BOUNDS>
  total_bandwidth() const {
    return
      TotalBandwidthAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_TOTAL_BANDWIDTH),
        C::fid(C::col_t::MS_SPECTRAL_WINDOW_COL_TOTAL_BANDWIDTH));
  }

  //
  // NET_SIDEBAND
  //
  static const constexpr unsigned net_sideband_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_NET_SIDEBAND];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using NetSidebandAccessor =
    FieldAccessor<HYPERION_TYPE_INT, net_sideband_rank, MODE, CHECK_BOUNDS>;

  bool
  has_net_sideband() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_NET_SIDEBAND) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  NetSidebandAccessor<MODE, CHECK_BOUNDS>
  net_sideband() const {
    return
      NetSidebandAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_NET_SIDEBAND),
        C::fid(C::col_t::MS_SPECTRAL_WINDOW_COL_NET_SIDEBAND));
  }

  //
  // BBC_NO
  //
  static const constexpr unsigned bbc_no_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_BBC_NO];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using BbcNoAccessor =
    FieldAccessor<HYPERION_TYPE_INT, bbc_no_rank, MODE, CHECK_BOUNDS>;

  bool
  has_bbc_no() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_BBC_NO) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  BbcNoAccessor<MODE, CHECK_BOUNDS>
  bbc_no() const {
    return
      BbcNoAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_BBC_NO),
        C::fid(C::col_t::MS_SPECTRAL_WINDOW_COL_BBC_NO));
  }

  //
  // BBC_SIDEBAND
  //
  static const constexpr unsigned bbc_sideband_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_BBC_SIDEBAND];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using BbcSidebandAccessor =
    FieldAccessor<HYPERION_TYPE_INT, bbc_sideband_rank, MODE, CHECK_BOUNDS>;

  bool
  has_bbc_sideband() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_BBC_SIDEBAND) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  BbcSidebandAccessor<MODE, CHECK_BOUNDS>
  bbc_sideband() const {
    return
      BbcSidebandAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_BBC_SIDEBAND),
        C::fid(C::col_t::MS_SPECTRAL_WINDOW_COL_BBC_SIDEBAND));
  }

  //
  // IF_CONV_CHAIN
  //
  static const constexpr unsigned if_conv_chain_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_IF_CONV_CHAIN];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using IfConvChainAccessor =
    FieldAccessor<HYPERION_TYPE_INT, if_conv_chain_rank, MODE, CHECK_BOUNDS>;

  bool
  has_if_conv_chain() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_IF_CONV_CHAIN) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  IfConvChainAccessor<MODE, CHECK_BOUNDS>
  if_conv_chain() const {
    return
      IfConvChainAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_IF_CONV_CHAIN),
        C::fid(C::col_t::MS_SPECTRAL_WINDOW_COL_IF_CONV_CHAIN));
  }

  //
  // RECEIVER_ID
  //
  static const constexpr unsigned receiver_id_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_RECEIVER_ID];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using ReceiverIdAccessor =
   FieldAccessor<HYPERION_TYPE_INT, receiver_id_rank, MODE, CHECK_BOUNDS>;

  bool
  has_receiver_id() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_RECEIVER_ID) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  ReceiverIdAccessor<MODE, CHECK_BOUNDS>
  receiver_id() const {
    return
      ReceiverIdAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_RECEIVER_ID),
        C::fid(C::col_t::MS_SPECTRAL_WINDOW_COL_RECEIVER_ID));
  }

  //
  // FREQ_GROUP
  //
  static const constexpr unsigned freq_group_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_FREQ_GROUP];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using FreqGroupAccessor =
    FieldAccessor<HYPERION_TYPE_INT, freq_group_rank, MODE, CHECK_BOUNDS>;

  bool
  has_freq_group() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_FREQ_GROUP) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  FreqGroupAccessor<MODE, CHECK_BOUNDS>
  freq_group() const {
    return
      FreqGroupAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_FREQ_GROUP),
        C::fid(C::col_t::MS_SPECTRAL_WINDOW_COL_FREQ_GROUP));
  }

  //
  // FREQ_GROUP_NAME
  //
  static const constexpr unsigned freq_group_name_rank =
    row_rank
    + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_FREQ_GROUP_NAME];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using FreqGroupNameAccessor =
    FieldAccessor<HYPERION_TYPE_STRING, freq_group_name_rank, MODE, CHECK_BOUNDS>;

  bool
  has_freq_group_name() const {
    return
      m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_FREQ_GROUP_NAME) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  FreqGroupNameAccessor<MODE, CHECK_BOUNDS>
  freq_group_name() const {
    return
      FreqGroupNameAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_FREQ_GROUP_NAME),
        C::fid(C::col_t::MS_SPECTRAL_WINDOW_COL_FREQ_GROUP_NAME));
  }

  //
  // DOPPLER_ID
  //
  static const constexpr unsigned doppler_id_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_DOPPLER_ID];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using DopplerIdAccessor =
    FieldAccessor<HYPERION_TYPE_INT, doppler_id_rank, MODE, CHECK_BOUNDS>;

  bool
  has_doppler_id() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_DOPPLER_ID) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  DopplerIdAccessor<MODE, CHECK_BOUNDS>
  doppler_id() const {
    return
      DopplerIdAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_DOPPLER_ID),
        C::fid(C::col_t::MS_SPECTRAL_WINDOW_COL_DOPPLER_ID));
  }

  //
  // ASSOC_SPW_ID
  //
  static const constexpr unsigned assoc_spw_id_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_ASSOC_SPW_ID];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using AssocSpwIdAccessor =
    FieldAccessor<HYPERION_TYPE_INT, assoc_spw_id_rank, MODE, CHECK_BOUNDS>;

  bool
  has_assoc_spw_id() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_ASSOC_SPW_ID) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  AssocSpwIdAccessor<MODE, CHECK_BOUNDS>
  assoc_spw_id() const {
    return
      AssocSpwIdAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_ASSOC_SPW_ID),
        C::fid(C::col_t::MS_SPECTRAL_WINDOW_COL_ASSOC_SPW_ID));
  }

  //
  // ASSOC_NATURE
  //
  static const constexpr unsigned assoc_nature_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_ASSOC_NATURE];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using AssocNatureAccessor =
    FieldAccessor<HYPERION_TYPE_STRING, assoc_nature_rank, MODE, CHECK_BOUNDS>;

  bool
  has_assoc_nature() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_ASSOC_NATURE) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  AssocNatureAccessor<MODE, CHECK_BOUNDS>
  assoc_nature() const {
    return
      AssocNatureAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_ASSOC_NATURE),
        C::fid(C::col_t::MS_SPECTRAL_WINDOW_COL_ASSOC_NATURE));
  }

  //
  // FLAG_ROW
  //
  static const constexpr unsigned flag_row_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_FLAG_ROW];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using FlagRowAccessor =
    FieldAccessor<HYPERION_TYPE_BOOL, flag_row_rank, MODE, CHECK_BOUNDS>;

  bool
  has_flag_row() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_FLAG_ROW) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  FlagRowAccessor<MODE, CHECK_BOUNDS>
  flag_row() const {
    return
      FlagRowAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_FLAG_ROW),
        C::fid(C::col_t::MS_SPECTRAL_WINDOW_COL_FLAG_ROW));
  }

private:

  Legion::DomainT<row_rank> m_rows;

  std::unordered_map<C::col_t, Legion::PhysicalRegion> m_regions;

#ifdef HYPERION_USE_CASACORE
  // the values of this map are of type mr_t<M> for some M
  std::unordered_map<C::col_t, std::any> m_mrs;
#endif
};

} // end namespace hyperion

#endif // HYPERION_MS_SP_WINDOW_COLUMNS_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
