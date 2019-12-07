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
# include <memory>
# include <optional>
# include <unordered_map>
# include <vector>
#pragma GCC visibility pop

namespace hyperion {

class HYPERION_API MSSpWindowColumns {
public:

  typedef MSTableColumns<MS_SPECTRAL_WINDOW> C;

  MSSpWindowColumns(
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

  static const constexpr unsigned row_rank = 1;

  Legion::DomainT<row_rank>
  rows(Legion::Runtime* rt) const {
    return
      rt->get_index_space_domain(
        m_rows_requirement.region.get_index_space());
  }

  //
  // NUM_CHAN
  //
  static const constexpr unsigned numChan_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_NUM_CHAN];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using NumChanAccessor =
    FieldAccessor<HYPERION_TYPE_INT, numChan_rank, MODE, CHECK_BOUNDS>;

  bool
  has_numChan() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_NUM_CHAN) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  NumChanAccessor<MODE, CHECK_BOUNDS>
  numChan() const {
    return
      NumChanAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_NUM_CHAN),
        Column::VALUE_FID);
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
        Column::VALUE_FID);
  }

  //
  // REF_FREQUENCY
  //
  static const constexpr unsigned refFrequency_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_REF_FREQUENCY];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using RefFrequencyAccessor =
    FieldAccessor<HYPERION_TYPE_DOUBLE, refFrequency_rank, MODE, CHECK_BOUNDS>;

  bool
  has_refFrequency() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_REF_FREQUENCY) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  RefFrequencyAccessor<MODE, CHECK_BOUNDS>
  refFrequency() const {
    return
      RefFrequencyAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_REF_FREQUENCY),
        Column::VALUE_FID);
  }

#ifdef HYPERION_USE_CASACORE
  template <typename T>
  class RefFrequencyMeasWriterMixin
    : public T {
  public:
    using T::T;

    void
    write(
      const Legion::Point<refFrequency_rank, Legion::coord_t>& pt,
      const casacore::MFrequency& val) {

      auto f = T::m_convert(val);
      T::m_ref_frequency[pt] = f.get(T::m_units).getValue();
    }
  };

  template <typename T>
  class RefFrequencyMeasReaderMixin
    : public T {
  public:
    using T::T;

    casacore::MFrequency
    read(const Legion::Point<refFrequency_rank, Legion::coord_t>& pt) const {
      const DataType<HYPERION_TYPE_DOUBLE>::ValueType& f =
        T::m_ref_frequency[pt];
      return
        casacore::MFrequency(casacore::Quantity(f, T::m_units), *T::m_mr);
    }
  };

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  class RefFrequencyMeasAccessorBase {
  public:
    RefFrequencyMeasAccessorBase(
      const Legion::PhysicalRegion& region,
      const std::shared_ptr<casacore::MeasRef<casacore::MFrequency>>& mr)
      : m_ref_frequency(region, Column::VALUE_FID)
      , m_mr(mr)
      , m_units(C::units.at(C::col_t::MS_SPECTRAL_WINDOW_COL_REF_FREQUENCY)) {
      m_convert.setOut(*m_mr);
    }

  protected:

    RefFrequencyAccessor<MODE, CHECK_BOUNDS> m_ref_frequency;

    std::shared_ptr<casacore::MeasRef<casacore::MFrequency>> m_mr;

    const char* m_units;

    casacore::MFrequency::Convert m_convert;
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
  has_refFrequencyMeas() const {
    return has_refFrequency() && m_ref_frequency_mr;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  RefFrequencyMeasAccessor<MODE, CHECK_BOUNDS>
  refFrequencyMeas() const {
    return
      RefFrequencyMeasAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_REF_FREQUENCY),
        m_ref_frequency_mr);
  }
#endif // HYPERION_USE_CASACORE

  //
  // MEAS_FREQ_REF
  //
  static const constexpr unsigned measFreqRef_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_MEAS_FREQ_REF];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using MeasFreqRefAccessor =
    FieldAccessor<HYPERION_TYPE_INT, measFreqRef_rank, MODE, CHECK_BOUNDS>;

  bool
  has_measFreqRef() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_MEAS_FREQ_REF) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  MeasFreqRefAccessor<MODE, CHECK_BOUNDS>
  measFreqRef() const {
    return
      MeasFreqRefAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_MEAS_FREQ_REF),
        Column::VALUE_FID);
  }

  //
  // CHAN_FREQ
  //
  static const constexpr unsigned chanFreq_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_CHAN_FREQ];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using ChanFreqAccessor =
    FieldAccessor<HYPERION_TYPE_DOUBLE, chanFreq_rank, MODE, CHECK_BOUNDS>;

  bool
  has_chanFreq() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_CHAN_FREQ) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  ChanFreqAccessor<MODE, CHECK_BOUNDS>
  chanFreq() const {
    return
      ChanFreqAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_CHAN_FREQ),
        Column::VALUE_FID);
  }

#ifdef HYPERION_USE_CASACORE
  template <typename T>
  class ChanFreqMeasWriterMixin
    : public T {
  public:
    using T::T;

    void
    write(
      const Legion::Point<chanFreq_rank, Legion::coord_t>& pt,
      const casacore::MFrequency& val) {

      auto m = 0; // FIXME: T::m_meas_freq_mr[pt[0]];
      auto f = T::m_convert[m](val);
      T::m_chan_freq[pt] = f.get(T::m_units).getValue();
    }
  };

  template <typename T>
  class ChanFreqMeasReaderMixin
    : public T {
  public:
    using T::T;

    casacore::MFrequency
    read(const Legion::Point<chanFreq_rank, Legion::coord_t>& pt) const {
      const DataType<HYPERION_TYPE_DOUBLE>::ValueType& f =
        T::m_chan_freq[pt];
      auto m = 0; // FIXME: T::m_meas_freq_mr[pt[0]];
      return
        casacore::MFrequency(casacore::Quantity(f, T::m_units), *T::m_mr[m]);
    }
  };

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  class ChanFreqMeasAccessorBase {
  public:
    ChanFreqMeasAccessorBase(
      const Legion::PhysicalRegion& chan_freq_region,
      const Legion::PhysicalRegion& meas_freq_ref_region,
      const std::vector<
        std::shared_ptr<casacore::MeasRef<casacore::MFrequency>>>& mr)
      : m_chan_freq(chan_freq_region, Column::VALUE_FID)
      , m_meas_freq_ref(meas_freq_ref_region)
      , m_mr(mr)
      , m_units(C::units.at(C::col_t::MS_SPECTRAL_WINDOW_COL_CHAN_FREQ)){
      m_convert.resize(m_mr.size());
      for (size_t i = 0; i < m_mr.size(); ++i)
        m_convert[i].setOut(*m_mr[i]);
    }

  protected:

    ChanFreqAccessor<MODE, CHECK_BOUNDS> m_chan_freq;

    MeasFreqRefAccessor<READ_ONLY, CHECK_BOUNDS> m_meas_freq_ref;

    std::vector<std::shared_ptr<casacore::MeasRef<casacore::MFrequency>>>
    m_mr;

    const char* m_units;

    std::vector<casacore::MFrequency::Convert> m_convert;
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
  has_chanFreqMeas() const {
    return has_chanFreq() && m_chan_freq_mr.size() > 0
      && has_measFreqRef();
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  ChanFreqMeasAccessor<MODE, CHECK_BOUNDS>
  chanFreqMeas() const {
    return
      ChanFreqMeasAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_CHAN_FREQ),
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_MEAS_FREQ_REF),
        m_chan_freq_mr);
  }
#endif

  //
  // CHAN_WIDTH
  //
  static const constexpr unsigned chanWidth_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_CHAN_WIDTH];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using ChanWidthAccessor =
    FieldAccessor<HYPERION_TYPE_DOUBLE, chanWidth_rank, MODE, CHECK_BOUNDS>;

  bool
  has_chanWidth() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_CHAN_WIDTH) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  ChanWidthAccessor<MODE, CHECK_BOUNDS>
  chanWidth() const {
    return
      ChanWidthAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_CHAN_WIDTH),
        Column::VALUE_FID);
  }

  //
  // EFFECTIVE_BW
  //
  static const constexpr unsigned effectiveBW_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_EFFECTIVE_BW];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using EffectiveBWAccessor =
    FieldAccessor<HYPERION_TYPE_DOUBLE, effectiveBW_rank, MODE, CHECK_BOUNDS>;

  bool
  has_effectiveBW() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_EFFECTIVE_BW) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  EffectiveBWAccessor<MODE, CHECK_BOUNDS>
  effectiveBW() const {
    return
      EffectiveBWAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_EFFECTIVE_BW),
        Column::VALUE_FID);
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
        Column::VALUE_FID);
  }

  //
  // TOTAL_BANDWIDTH
  //
  static const constexpr unsigned totalBandwidth_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_TOTAL_BANDWIDTH];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using TotalBandwidthAccessor =
    FieldAccessor<
      HYPERION_TYPE_DOUBLE,
      totalBandwidth_rank,
      MODE,
      CHECK_BOUNDS>;

  bool
  has_totalBandwidth() const {
    return
      m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_TOTAL_BANDWIDTH) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  TotalBandwidthAccessor<MODE, CHECK_BOUNDS>
  totalBandwidth() const {
    return
      TotalBandwidthAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_TOTAL_BANDWIDTH),
        Column::VALUE_FID);
  }

  //
  // NET_SIDEBAND
  //
  static const constexpr unsigned netSideband_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_NET_SIDEBAND];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using NetSidebandAccessor =
    FieldAccessor<HYPERION_TYPE_INT, netSideband_rank, MODE, CHECK_BOUNDS>;

  bool
  has_netSideband() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_NET_SIDEBAND) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  NetSidebandAccessor<MODE, CHECK_BOUNDS>
  netSideband() const {
    return
      NetSidebandAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_NET_SIDEBAND),
        Column::VALUE_FID);
  }

  //
  // BBC_NO
  //
  static const constexpr unsigned bbcNo_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_BBC_NO];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using BbcNoAccessor =
    FieldAccessor<HYPERION_TYPE_INT, bbcNo_rank, MODE, CHECK_BOUNDS>;

  bool
  has_bbcNo() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_BBC_NO) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  BbcNoAccessor<MODE, CHECK_BOUNDS>
  bbcNo() const {
    return
      BbcNoAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_BBC_NO),
        Column::VALUE_FID);
  }

  //
  // BBC_SIDEBAND
  //
  static const constexpr unsigned bbcSideband_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_BBC_SIDEBAND];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using BbcSidebandAccessor =
    FieldAccessor<HYPERION_TYPE_INT, bbcSideband_rank, MODE, CHECK_BOUNDS>;

  bool
  has_bbcSideband() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_BBC_SIDEBAND) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  BbcSidebandAccessor<MODE, CHECK_BOUNDS>
  bbcSideband() const {
    return
      BbcSidebandAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_BBC_SIDEBAND),
        Column::VALUE_FID);
  }

  //
  // IF_CONV_CHAIN
  //
  static const constexpr unsigned ifConvChain_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_IF_CONV_CHAIN];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using IfConvChainAccessor =
    FieldAccessor<HYPERION_TYPE_INT, ifConvChain_rank, MODE, CHECK_BOUNDS>;

  bool
  has_ifConvChain() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_IF_CONV_CHAIN) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  IfConvChainAccessor<MODE, CHECK_BOUNDS>
  ifConvChain() const {
    return
      IfConvChainAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_IF_CONV_CHAIN),
        Column::VALUE_FID);
  }

  //
  // RECEIVER_ID
  //
  static const constexpr unsigned receiverId_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_RECEIVER_ID];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using ReceiverIdAccessor =
   FieldAccessor<HYPERION_TYPE_INT, receiverId_rank, MODE, CHECK_BOUNDS>;

  bool
  has_receiverId() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_RECEIVER_ID) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  ReceiverIdAccessor<MODE, CHECK_BOUNDS>
  receiverId() const {
    return
      ReceiverIdAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_RECEIVER_ID),
        Column::VALUE_FID);
  }

  //
  // FREQ_GROUP
  //
  static const constexpr unsigned freqGroup_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_FREQ_GROUP];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using FreqGroupAccessor =
    FieldAccessor<HYPERION_TYPE_INT, freqGroup_rank, MODE, CHECK_BOUNDS>;

  bool
  has_freqGroup() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_FREQ_GROUP) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  FreqGroupAccessor<MODE, CHECK_BOUNDS>
  freqGroup() const {
    return
      FreqGroupAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_FREQ_GROUP),
        Column::VALUE_FID);
  }

  //
  // FREQ_GROUP_NAME
  //
  static const constexpr unsigned freqGroupName_rank =
    row_rank
    + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_FREQ_GROUP_NAME];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using FreqGroupNameAccessor =
    FieldAccessor<HYPERION_TYPE_STRING, freqGroupName_rank, MODE, CHECK_BOUNDS>;

  bool
  has_freqGroupName() const {
    return
      m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_FREQ_GROUP_NAME) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  FreqGroupNameAccessor<MODE, CHECK_BOUNDS>
  freqGroupName() const {
    return
      FreqGroupNameAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_FREQ_GROUP_NAME),
        Column::VALUE_FID);
  }

  //
  // DOPPLER_ID
  //
  static const constexpr unsigned dopplerId_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_DOPPLER_ID];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using DopplerIdAccessor =
    FieldAccessor<HYPERION_TYPE_INT, dopplerId_rank, MODE, CHECK_BOUNDS>;

  bool
  has_dopplerId() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_DOPPLER_ID) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  DopplerIdAccessor<MODE, CHECK_BOUNDS>
  dopplerId() const {
    return
      DopplerIdAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_DOPPLER_ID),
        Column::VALUE_FID);
  }

  //
  // ASSOC_SPW_ID
  //
  static const constexpr unsigned assocSpwId_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_ASSOC_SPW_ID];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using AssocSpwIdAccessor =
    FieldAccessor<HYPERION_TYPE_INT, assocSpwId_rank, MODE, CHECK_BOUNDS>;

  bool
  has_assocSpwId() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_ASSOC_SPW_ID) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  AssocSpwIdAccessor<MODE, CHECK_BOUNDS>
  assocSpwId() const {
    return
      AssocSpwIdAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_ASSOC_SPW_ID),
        Column::VALUE_FID);
  }

  //
  // ASSOC_NATURE
  //
  static const constexpr unsigned assocNature_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_ASSOC_NATURE];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using AssocNatureAccessor =
    FieldAccessor<HYPERION_TYPE_STRING, assocNature_rank, MODE, CHECK_BOUNDS>;

  bool
  has_assocNature() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_ASSOC_NATURE) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  AssocNatureAccessor<MODE, CHECK_BOUNDS>
  assocNature() const {
    return
      AssocNatureAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_ASSOC_NATURE),
        Column::VALUE_FID);
  }

  //
  // FLAG_ROW
  //
  static const constexpr unsigned flagRow_rank =
    row_rank + C::element_ranks[C::col_t::MS_SPECTRAL_WINDOW_COL_FLAG_ROW];

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using FlagRowAccessor =
    FieldAccessor<HYPERION_TYPE_BOOL, flagRow_rank, MODE, CHECK_BOUNDS>;

  bool
  has_flagRow() const {
    return m_regions.count(C::col_t::MS_SPECTRAL_WINDOW_COL_FLAG_ROW) > 0;
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  FlagRowAccessor<MODE, CHECK_BOUNDS>
  flagRow() const {
    return
      FlagRowAccessor<MODE, CHECK_BOUNDS>(
        m_regions.at(C::col_t::MS_SPECTRAL_WINDOW_COL_FLAG_ROW),
        Column::VALUE_FID);
  }

private:

  Legion::RegionRequirement m_rows_requirement;

  std::unordered_map<C::col_t, Legion::PhysicalRegion> m_regions;

#ifdef HYPERION_USE_CASACORE
  std::shared_ptr<casacore::MeasRef<casacore::MFrequency>>
  m_ref_frequency_mr;
  std::vector<std::shared_ptr<casacore::MeasRef<casacore::MFrequency>>>
  m_chan_freq_mr;
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
