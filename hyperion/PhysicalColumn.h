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
#ifndef HYPERION_PHYSICAL_COLUMN_H_
#define HYPERION_PHYSICAL_COLUMN_H_

#include <hyperion/hyperion.h>
#include <hyperion/Keywords.h>
#include <hyperion/ColumnSpace.h>
#ifdef HYPERION_USE_CASACORE
# include <hyperion/MeasRef.h>
#endif // HYPERION_USE_CASACORE

#pragma GCC visibility push(default)
# include <any>
# include <memory>
# include <optional>
# include <unordered_map>
# include <variant>
# include <vector>
#pragma GCC visibility pop

namespace hyperion {

class Column;

#ifdef HYPERION_USE_CASACORE
// the following mixin classes assume that the column layout has the measure
// components in a contiguous sequence in offset range 0..MV_SIZE-1
template <
  typename T,
  hyperion::TypeTag DT,
  hyperion::MClass MC,
  unsigned INDEX_RANK,
  unsigned COLUMN_RANK,
  unsigned MV_SIZE,
  typename COORD_T>
class MeasReaderMixin
  : public T {
public:
  using T::T;

  static const constexpr unsigned M_RANK = MClassT<MC>::m_rank;

  static_assert(M_RANK <= 1);
  static_assert(MV_SIZE > 0);
  static_assert((M_RANK == 1) || (MV_SIZE == 1));

  typename MClassT<MC>::type
  read(const Legion::Point<COLUMN_RANK - M_RANK, COORD_T>& pt) const {

    return
      MClassT<MC>::load<DT>(
        std::experimental::mdspan<typename DataType<DT>::ValueType, MV_SIZE>(
          T::m_value.ptr(pt)),
        T::m_units,
        T::m_meas_ref.meas_ref_at(pt));
  }
};

template <
  typename T,
  hyperion::TypeTag DT,
  hyperion::MClass MC,
  unsigned INDEX_RANK,
  unsigned COLUMN_RANK,
  unsigned MV_SIZE,
  typename COORD_T>
class MeasWriterMixin
  : public T {
public:
  using T::T;

  static const constexpr unsigned M_RANK = MClassT<MC>::m_rank;

  static_assert(M_RANK <= 1);
  static_assert(MV_SIZE > 0);
  static_assert((M_RANK == 1) || (MV_SIZE == 1));

  void
  write(
    const Legion::Point<COLUMN_RANK - M_RANK, COORD_T>& pt,
    const typename MClassT<MC>::type& val) {

    MClassT<MC>::store<DT>(
      T::m_meas_ref.convert_at(pt)(val),
      T::m_units,
      std::experimental::mdspan<typename DataType<DT>::ValueType, MV_SIZE>(
        T::m_value.ptr(pt)));
  }
};
#endif //HYPERION_USE_CASACORE

class HYPERION_API PhysicalColumn {
public:

  PhysicalColumn(
    Legion::Runtime* rt,
    hyperion::TypeTag dt,
    Legion::FieldID fid,
    unsigned index_rank,
    const Legion::PhysicalRegion& metadata,
    const Legion::LogicalRegion& parent,
    const std::variant<Legion::PhysicalRegion, Legion::LogicalRegion>& values,
    const std::optional<Keywords::pair<Legion::PhysicalRegion>>& kws
#ifdef HYPERION_USE_CASACORE
    , const std::optional<MeasRef::DataRegions>& mr_drs
    , const std::optional<
        std::tuple<std::string, std::shared_ptr<PhysicalColumn>>>& refcol
#endif // HYPERION_USE_CASACORE
    )
    : m_dt(dt)
    , m_fid(fid)
    , m_index_rank(index_rank)
    , m_metadata(metadata)
    , m_domain(
      (parent != Legion::LogicalRegion::NO_REGION)
      ? rt->get_index_space_domain(parent.get_index_space())
      : Legion::Domain::NO_DOMAIN)
    , m_parent(parent)
    , m_values(values)
    , m_kws(kws)
#ifdef HYPERION_USE_CASACORE
    , m_mr_drs(mr_drs)
    , m_refcol(refcol)
#endif // HYPERION_USE_CASACORE
    {}

  Column
  column() const;

  ColumnSpace::AXIS_VECTOR_TYPE
  axes() const;

  template <
    Legion::PrivilegeMode MODE,
    typename FT,
    int N,
    typename COORD_T = Legion::coord_t,
    typename A = Legion::GenericAccessor<FT, N, COORD_T>,
    bool CHECK_BOUNDS = false>
  Legion::FieldAccessor<MODE, FT, N, COORD_T, A, CHECK_BOUNDS>
  accessor() const {
    return
      Legion::FieldAccessor<MODE, FT, N, COORD_T, A, CHECK_BOUNDS>(
        std::get<Legion::PhysicalRegion>(m_values),
        m_fid);
  }

  hyperion::TypeTag
  dt() const {
    return m_dt;
  }

  Legion::FieldID
  fid() const {
    return m_fid;
  };

  unsigned
  index_rank() const {
    return m_index_rank;
  };

  const Legion::PhysicalRegion&
  metadata() const {
    return m_metadata;
  };


  const Legion::Domain&
  domain() const {
    return m_domain;
  }

  const Legion::LogicalRegion&
  parent() const {
    return m_parent;
  };

  const std::variant<Legion::PhysicalRegion, Legion::LogicalRegion>&
  values() const {
    return m_values;
  };

  const std::optional<Keywords::pair<Legion::PhysicalRegion>>&
  kws() const {
    return m_kws;
  };

#ifdef HYPERION_USE_CASACORE

  const std::optional<MeasRef::DataRegions>
  mr_drs() const {
    return m_mr_drs;
  };

  const std::optional<std::tuple<std::string, std::shared_ptr<PhysicalColumn>>>&
  refcol() const {
    return m_refcol;
  }

  std::vector<std::shared_ptr<casacore::MRBase>>
  mrbases(Legion::Runtime *rt) const;

  typedef std::shared_ptr<casacore::MRBase> simple_mrb_t;

  typedef std::tuple<
    std::vector<std::shared_ptr<casacore::MRBase>>,
    std::unordered_map<unsigned, unsigned>,
    Legion::PhysicalRegion,
    Legion::FieldID> ref_mrb_t;

  typedef std::variant<simple_mrb_t, ref_mrb_t> mrb_t;

  std::optional<mrb_t>
  mrb(Legion::Runtime* rt) const;
#endif // HYPERION_USE_CASACORE

  Legion::LogicalRegion
  create_index(Legion::Context ctx, Legion::Runtime* rt) const;

protected:

  PhysicalColumn(const PhysicalColumn& from)
  : m_dt(from.m_dt)
  , m_fid(from.m_fid)
  , m_index_rank(from.m_index_rank)
  , m_metadata(from.m_metadata)
  , m_domain(from.m_domain)
  , m_parent(from.m_parent)
  , m_values(from.m_values)
  , m_kws(from.m_kws)
#ifdef HYPERION_USE_CASACORE
  , m_mr_drs(from.m_mr_drs)
  , m_refcol(from.m_refcol)
#endif // HYPERION_USE_CASACORE
  {}

protected:

  friend class PhysicalTable;
  void
  set_refcol(const std::string& name, const std::shared_ptr<PhysicalColumn>& col) {
    m_refcol = std::make_tuple(name, col);
  }

protected:

  hyperion::TypeTag m_dt;

  Legion::FieldID m_fid;

  unsigned m_index_rank;

  Legion::PhysicalRegion m_metadata;

  // maintain m_domain in order to allow checks in narrowing copy constructors to
  // PhysicalColumn variants (w.o. needing Legion::Runtime instance)
  Legion::Domain m_domain;

  Legion::LogicalRegion m_parent;

  // allow an optional values region, to support a PhysicalColumn without values
  // (e.g, some Table index column spaces)
  std::variant<Legion::PhysicalRegion, Legion::LogicalRegion> m_values;

  std::optional<Keywords::pair<Legion::PhysicalRegion>> m_kws;

#ifdef HYPERION_USE_CASACORE
  std::optional<MeasRef::DataRegions> m_mr_drs;

  std::optional<std::tuple<std::string, std::shared_ptr<PhysicalColumn>>>
  m_refcol;
#endif // HYPERION_USE_CASACORE
};

template <
  hyperion::TypeTag DT,
  template <typename, int, typename> typename A = Legion::GenericAccessor,
  typename COORD_T = Legion::coord_t>
class PhysicalColumnT
  : public PhysicalColumn {
public:

  PhysicalColumnT(
    Legion::Runtime* rt,
    Legion::FieldID fid,
    unsigned index_rank,
    const Legion::PhysicalRegion& metadata,
    const Legion::LogicalRegion& parent,
    const std::variant<Legion::PhysicalRegion, Legion::LogicalRegion>& values,
    const std::optional<Keywords::pair<Legion::PhysicalRegion>>& kws
#ifdef HYPERION_USE_CASACORE
    , const std::optional<MeasRef::DataRegions>& mr_drs
    , const std::optional<
        std::tuple<std::string, std::shared_ptr<PhysicalColumn>>>& refcol
#endif // HYPERION_USE_CASACORE
    )
    : PhysicalColumn(
      rt,
      fid,
      DT,
      index_rank,
      metadata,
      parent,
      values,
      kws
#ifdef HYPERION_USE_CASACORE
      , mr_drs
      , refcol
#endif // HYPERION_USE_CASACORE
      ) {}

  PhysicalColumnT(const PhysicalColumn& from)
    : PhysicalColumn(from) {
    // FIXME: change assertion to exception
    assert(DT == m_dt);
  }

  template <Legion::PrivilegeMode MODE, int N, bool CHECK_BOUNDS = false>
  Legion::FieldAccessor<
    MODE,
    typename DataType<DT>::ValueType,
    N,
    COORD_T,
    A<typename DataType<DT>::ValueType, N, COORD_T>,
    CHECK_BOUNDS>
  accessor() const {
    return
      PhysicalColumn::accessor<
        MODE,
        typename DataType<DT>::ValueType,
        N,
        COORD_T,
        A<typename DataType<DT>::ValueType, N, COORD_T>,
        CHECK_BOUNDS>();
  }
};

template <
  hyperion::TypeTag DT,
  unsigned INDEX_RANK,
  unsigned COLUMN_RANK,
  template <typename, int, typename> typename A = Legion::GenericAccessor,
  typename COORD_T = Legion::coord_t>
class PhysicalColumnTD
  : public PhysicalColumn {
public:

  static_assert(INDEX_RANK <= COLUMN_RANK);

  PhysicalColumnTD(
    Legion::Runtime* rt,
    Legion::FieldID fid,
    const Legion::PhysicalRegion& metadata,
    const Legion::LogicalRegion& parent,
    const std::variant<Legion::PhysicalRegion, Legion::LogicalRegion>& values,
    const std::optional<Keywords::pair<Legion::PhysicalRegion>>& kws
#ifdef HYPERION_USE_CASACORE
    , const std::optional<MeasRef::DataRegions>& mr_drs
    , const std::optional<
        std::tuple<std::string, std::shared_ptr<PhysicalColumn>>>& refcol
#endif // HYPERION_USE_CASACORE
    )
    : PhysicalColumn(
      rt,
      fid,
      DT,
      INDEX_RANK,
      metadata,
      parent,
      values,
      kws
#ifdef HYPERION_USE_CASACORE
      , mr_drs
      , refcol
#endif // HYPERION_USE_CASACORE
      ) {}

  PhysicalColumnTD(const PhysicalColumn& from)
    : PhysicalColumn(from) {
    // FIXME: change assertions to exceptions
    assert(DT == m_dt);
    assert(INDEX_RANK == m_index_rank);
    assert(COLUMN_RANK == m_parent.get_index_space().get_dim());
  }

  template <Legion::PrivilegeMode MODE, bool CHECK_BOUNDS = false>
  Legion::FieldAccessor<
    MODE,
    typename DataType<DT>::ValueType,
    COLUMN_RANK,
    COORD_T,
    A<typename DataType<DT>::ValueType, COLUMN_RANK, COORD_T>,
    CHECK_BOUNDS>
  accessor() const {
    return
      PhysicalColumn::accessor<
        MODE,
        typename DataType<DT>::ValueType,
        COLUMN_RANK,
        COORD_T,
        A<typename DataType<DT>::ValueType, COLUMN_RANK, COORD_T>,
        CHECK_BOUNDS>();
  }

  Legion::DomainT<COLUMN_RANK>
  domain() const {
    return PhysicalColumn::domain();
  }
};

#ifdef HYPERION_USE_CASACORE
template <
  hyperion::TypeTag DT,
  hyperion::MClass MC,
  template <typename, int, typename> typename A = Legion::GenericAccessor,
  typename COORD_T = Legion::coord_t>
class PhysicalColumnTM
  : public PhysicalColumn {
public:

  PhysicalColumnTM(
    Legion::Runtime* rt,
    Legion::FieldID fid,
    unsigned index_rank,
    const Legion::PhysicalRegion& metadata,
    const Legion::LogicalRegion& parent,
    const std::variant<Legion::PhysicalRegion, Legion::LogicalRegion>& values,
    const std::optional<Keywords::pair<Legion::PhysicalRegion>>& kws,
    const std::optional<MeasRef::DataRegions>& mr_drs,
    const std::optional<
      std::tuple<std::string, std::shared_ptr<PhysicalColumn>>>& refcol)
    : PhysicalColumn(
      rt,
      fid,
      DT,
      index_rank,
      metadata,
      parent,
      values,
      kws,
      mr_drs,
      refcol) {}

  PhysicalColumnTM(const PhysicalColumn& from)
    : PhysicalColumn(from) {
    // FIXME: change assertion to exception
    assert(DT == m_dt);
    assert(m_mr_drs);
    assert(MeasRef::mclass(m_mr_drs.value().metadata) == MC);
  }

  template <Legion::PrivilegeMode MODE, int N, bool CHECK_BOUNDS = false>
  Legion::FieldAccessor<
    MODE,
    typename DataType<DT>::ValueType,
    N,
    COORD_T,
    A<typename DataType<DT>::ValueType, N, COORD_T>,
    CHECK_BOUNDS>
  accessor() const {
    return
      PhysicalColumn::accessor<
        MODE,
        typename DataType<DT>::ValueType,
        N,
        COORD_T,
        A<typename DataType<DT>::ValueType, N, COORD_T>,
        CHECK_BOUNDS>();
  }

  typedef casacore::MeasRef<typename MClassT<MC>::type> MR_t;

  std::vector<std::shared_ptr<MR_t>>
  mrs(Legion::Runtime* rt) {
    auto mrbv = mrbases(rt);
    std::vector<std::shared_ptr<MR_t>> result;
    result.reserve(mrbv.size());
    for (auto& mrb : mrbv)
      result.push_back(std::dynamic_pointer_cast<MR_t>(mrb));
    return result;
  }
};

template <
  hyperion::TypeTag DT,
  hyperion::MClass MC,
  unsigned INDEX_RANK,
  unsigned COLUMN_RANK,
  unsigned MV_SIZE,
  template <typename, int, typename> typename A = Legion::GenericAccessor,
  typename COORD_T = Legion::coord_t>
class PhysicalColumnTMD
  : public PhysicalColumn {
public:

  static const constexpr unsigned M_RANK = MClassT<MC>::m_rank;

  static_assert(MV_SIZE > 0);
  static_assert(M_RANK <= 1);

  PhysicalColumnTMD(
    Legion::Runtime* rt,
    Legion::FieldID fid,
    const Legion::PhysicalRegion& metadata,
    const Legion::LogicalRegion& parent,
    const std::variant<Legion::PhysicalRegion, Legion::LogicalRegion>& values,
    const std::optional<Keywords::pair<Legion::PhysicalRegion>>& kws,
    const std::optional<MeasRef::DataRegions>& mr_drs,
    const std::optional<
      std::tuple<std::string, std::shared_ptr<PhysicalColumn>>>& refcol)
    : PhysicalColumn(
      rt,
      fid,
      DT,
      INDEX_RANK,
      metadata,
      parent,
      values,
      kws,
      mr_drs,
      refcol) {}

  PhysicalColumnTMD(const PhysicalColumn& from)
    : PhysicalColumn(from) {
    // FIXME: change assertions to exceptions
    assert(DT == m_dt);
    assert(INDEX_RANK == m_index_rank);
    assert(COLUMN_RANK == m_parent.get_index_space().get_dim());
    assert(m_mr_drs);
    assert(MeasRef::mclass(m_mr_drs.value().metadata) == MC);
    if (M_RANK == 1) {
      Legion::PointInDomainIterator<COLUMN_RANK> pid(m_domain, false);
      Legion::Point<COLUMN_RANK> pt = *pid;
      for (COORD_T i = 0; i < MV_SIZE; ++i) {
        pt[COLUMN_RANK - 1] = i;
        assert(m_domain.contains(pt));
      }
      pt[COLUMN_RANK - 1] = MV_SIZE;
      assert(!m_domain.contains(pt));
    }
  }

  typedef casacore::MeasRef<typename MClassT<MC>::type> MR_t;

  typedef typename MClassT<MC>::type::Convert MC_t;

  template <Legion::PrivilegeMode MODE, bool CHECK_BOUNDS = false>
  class MeasRefAccessor {
  private:

    typedef Legion::FieldAccessor<
      MODE,
      DataType<HYPERION_TYPE_INT>::ValueType,
      INDEX_RANK,
      COORD_T,
      A<DataType<HYPERION_TYPE_INT>::ValueType, INDEX_RANK, COORD_T>,
      CHECK_BOUNDS> RefcodeAccessor;

  protected:
    friend class
    PhysicalColumnTMD<DT, MC, INDEX_RANK, COLUMN_RANK, MV_SIZE, A, COORD_T>;

    MeasRefAccessor(const mrb_t* mr) {
      std::visit(overloaded {
          [this](simple_mrb_t& mr) {
            m_mr = std::dynamic_pointer_cast<MR_t>(mr);
            m_convert.setOut(*m_mr);
          },
          [this](ref_mrb_t& mr) {
            auto& [mrs, rmap, rcodes_pr, fid] = mr;
            m_mrv =
              std::make_tuple(
                mrs,
                rmap,
                RefcodeAccessor(rcodes_pr, fid));
          }
        },
        *mr);
    }

  public:

    MC_t&
    convert_at(const Legion::Point<COLUMN_RANK - M_RANK, Legion::coord_t>& pt) {
      if (m_mrv) {
        auto& [mrs, rmap, rcodes] = m_mrv.value();
        m_convert.setOut(
          *mrs[
            rmap.at(
              rcodes.read(
                reinterpret_cast<Legion::Point<INDEX_RANK, Legion::coord_t>&>(
                  pt)))]);
      }
      return m_convert;
    }

    MR_t&
    meas_ref_at(const Legion::Point<COLUMN_RANK - M_RANK, Legion::coord_t>& pt)
      const {
      if (m_mrv) {
        auto& [mrs, rmap, rcodes] = m_mrv.value();
        m_mr =
          mrs[
            rmap.at(
              rcodes.read(
                reinterpret_cast<Legion::Point<INDEX_RANK, Legion::coord_t>&>(
                  pt)))];
      }
      return *m_mr;
    }

  private:
    std::shared_ptr<MR_t> m_mr;

    std::optional<
      std::tuple<
        std::vector<std::shared_ptr<MR_t>>,
        std::unordered_map<unsigned, unsigned>,
        RefcodeAccessor>>
    m_mrv;

    MC_t m_convert;
  };

  template <Legion::PrivilegeMode MODE, bool CHECK_BOUNDS = false>
  Legion::FieldAccessor<
    MODE,
    typename DataType<DT>::ValueType,
    COLUMN_RANK,
    COORD_T,
    A<typename DataType<DT>::ValueType, COLUMN_RANK, COORD_T>,
    CHECK_BOUNDS>
  accessor() const {
    return
      PhysicalColumn::accessor<
        MODE,
        typename DataType<DT>::ValueType,
        COLUMN_RANK,
        COORD_T,
        A<typename DataType<DT>::ValueType, COLUMN_RANK, COORD_T>,
        CHECK_BOUNDS>();
  }

  Legion::DomainT<COLUMN_RANK>
  domain() const {
    return PhysicalColumn::domain();
  }

  template <Legion::PrivilegeMode MODE, bool CHECK_BOUNDS = false>
  MeasRefAccessor<MODE, CHECK_BOUNDS>
  meas_ref_accessor(Legion::Runtime* rt) const {
    auto omrb = mrb(rt);
    return MeasRefAccessor<MODE, CHECK_BOUNDS>(&omrb.value());
  }

  template <Legion::PrivilegeMode MODE, bool CHECK_BOUNDS = false>
  class MeasAccessorBase {
  public:
    MeasAccessorBase(
      Legion::Runtime* rt,
      const PhysicalColumnTMD<
        DT,
        MC,
        INDEX_RANK,
        COLUMN_RANK,
        MV_SIZE,
        A,
        COORD_T>* col,
      const char* units)
      : m_units(units)
      , m_value(col->accessor<MODE, CHECK_BOUNDS>())
      , m_meas_ref(col->meas_ref_accessor<MODE, CHECK_BOUNDS>(rt))
      {}

  protected:
    const char* m_units;

    Legion::FieldAccessor<
      MODE,
      typename DataType<DT>::ValueType,
      COLUMN_RANK,
      COORD_T,
      A<typename DataType<DT>::ValueType, COLUMN_RANK, COORD_T>,
      CHECK_BOUNDS> m_value;

    MeasRefAccessor<MODE, CHECK_BOUNDS> m_meas_ref;
  };

  template <Legion::PrivilegeMode MODE, bool CHECK_BOUNDS>
  class MeasAccessor
    : public MeasWriterMixin<
        MeasAccessorBase<MODE, CHECK_BOUNDS>,
        DT,
        MC,
        INDEX_RANK,
        COLUMN_RANK,
        MV_SIZE,
        COORD_T> {
    typedef MeasWriterMixin<
      MeasAccessorBase<MODE, CHECK_BOUNDS>,
      DT,
      MC,
      INDEX_RANK,
      COLUMN_RANK,
      MV_SIZE,
      COORD_T> T;
  public:
    using T::T;
  };

  template <bool CHECK_BOUNDS>
  class MeasAccessor<READ_ONLY, CHECK_BOUNDS>
    : public MeasReaderMixin<
        MeasAccessorBase<READ_ONLY, CHECK_BOUNDS>,
        DT,
        MC,
        INDEX_RANK,
        COLUMN_RANK,
        MV_SIZE,
        COORD_T> {
    typedef MeasReaderMixin<
      MeasAccessorBase<READ_ONLY, CHECK_BOUNDS>,
      DT,
      MC,
      INDEX_RANK,
      COLUMN_RANK,
      MV_SIZE,
      COORD_T> T;
  public:
    using T::T;
  };

  template <bool CHECK_BOUNDS>
  class MeasAccessor<READ_WRITE, CHECK_BOUNDS>
    : public MeasReaderMixin<
        MeasWriterMixin<
          MeasAccessorBase<READ_WRITE, CHECK_BOUNDS>,
          DT,
          MC,
          INDEX_RANK,
          COLUMN_RANK,
          MV_SIZE,
          COORD_T>,
        DT,
        MC,
        INDEX_RANK,
        COLUMN_RANK,
        MV_SIZE,
        COORD_T> {
    typedef MeasReaderMixin<
      MeasWriterMixin<
        MeasAccessorBase<READ_WRITE, CHECK_BOUNDS>,
        DT,
        MC,
        INDEX_RANK,
        COLUMN_RANK,
        MV_SIZE,
        COORD_T>,
      DT,
      MC,
      INDEX_RANK,
      COLUMN_RANK,
      MV_SIZE,
      COORD_T> T;
  public:
    using T::T;
  };

  template <Legion::PrivilegeMode MODE, bool CHECK_BOUNDS = false>
  MeasAccessor<MODE, CHECK_BOUNDS>
  meas_accessor(Legion::Runtime* rt, const char* units) const {
    return MeasAccessor<MODE, CHECK_BOUNDS>(rt, this, units);
  }
};

#endif // HYPERION_USE_CASACORE

} // end namespace hyperion

#endif // HYPERION_PHYSICAL_COLUMN_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
