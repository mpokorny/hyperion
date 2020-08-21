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

#include <any>
#include <memory>
#include CXX_OPTIONAL_HEADER
#include <unordered_map>
#if HAVE_CXX17
#include <variant>
#endif // HAVE_CXX17
#include <vector>

#ifdef HYPERION_USE_KOKKOS
# include <Kokkos_Core.hpp>
# include <Kokkos_OffsetView.hpp>
#endif
#include <experimental/mdspan>

namespace hyperion {

class Column;

#ifdef HYPERION_USE_KOKKOS

template <typename T, unsigned N>
struct view_element {
  typedef typename view_element<T*, N - 1>::type type;
};

template <typename T>
struct view_element<T, 1> {
  typedef T* type;
};
#endif // HYPERION_USE_KOKKOS

#ifdef HYPERION_USE_CASACORE
// the following mixin classes assume that the column layout has the measure
// components in a contiguous sequence in offset range 0..MV_SIZE-1
template <
  typename T,
  hyperion::TypeTag DT,
  hyperion::MClass MC,
  unsigned INDEX_RANK,
  unsigned COLUMN_RANK,
  unsigned M_RANK,
  unsigned MV_SIZE,
  typename COORD_T>
class MeasReaderMixin
  : public T {
};

template <
  typename T,
  hyperion::TypeTag DT,
  hyperion::MClass MC,
  unsigned INDEX_RANK,
  unsigned COLUMN_RANK,
  typename COORD_T>
class MeasReaderMixin<T, DT, MC, INDEX_RANK, COLUMN_RANK, 0, 1, COORD_T>
  : public T {
public:
  using T::T;

  typename MClassT<MC>::type
  read(const Legion::Point<COLUMN_RANK, COORD_T>& pt) {

    return
      MClassT<MC>::template load<DT>(
        T::m_value.read(pt),
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
class MeasReaderMixin<T, DT, MC, INDEX_RANK, COLUMN_RANK, 1, MV_SIZE, COORD_T>
  : public T {
public:
  using T::T;

  static_assert(MV_SIZE > 0);

  typename MClassT<MC>::type
  read(const Legion::Point<COLUMN_RANK - 1, COORD_T>& pt) {

    Legion::Point<COLUMN_RANK, COORD_T> ept;
    for (size_t i = 0; i < COLUMN_RANK - 1; ++i)
      ept[i] = pt[i];
    casacore::Vector<typename DataType<DT>::ValueType> vs(MV_SIZE);
    for (size_t i = 0; i < MV_SIZE; ++i) {
      ept[COLUMN_RANK - 1] = i;
      vs[i] = T::m_value.read(ept);
    }
    return
      MClassT<MC>::template load<DT>(
        vs,
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
  unsigned M_RANK,
  unsigned MV_SIZE,
  typename COORD_T>
class MeasWriterMixin
  : public T {
public:
  using T::T;
};

template <
  typename T,
  hyperion::TypeTag DT,
  hyperion::MClass MC,
  unsigned INDEX_RANK,
  unsigned COLUMN_RANK,
  typename COORD_T>
class MeasWriterMixin<T, DT, MC, INDEX_RANK, COLUMN_RANK, 0, 1, COORD_T>
  : public T {
public:
  using T::T;

  void
  write(
    const Legion::Point<COLUMN_RANK, COORD_T>& pt,
    const typename MClassT<MC>::type& val) {

    typename DataType<DT>::ValueType v;
    MClassT<MC>::template store<DT>(
      T::m_meas_ref.convert_at(pt)(val),
      T::m_units,
      v);
    T::m_value.write(pt, v);
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
class MeasWriterMixin<T, DT, MC, INDEX_RANK, COLUMN_RANK, 1, MV_SIZE, COORD_T>
  : public T {
public:
  using T::T;

  static_assert(MV_SIZE > 0);

  void
  write(
    const Legion::Point<COLUMN_RANK - 1, COORD_T>& pt,
    const typename MClassT<MC>::type& val) {

    casacore::Vector<typename DataType<DT>::ValueType> vs(MV_SIZE);
    MClassT<MC>::template store<DT>(
      T::m_meas_ref.convert_at(pt)(val),
      T::m_units,
      vs);
    Legion::Point<COLUMN_RANK, COORD_T> ept;
    for (size_t i = 0; i < COLUMN_RANK - 1; ++i)
      ept[i] = pt[i];
    for (size_t i = 0; i < MV_SIZE; ++i) {
      ept[COLUMN_RANK - 1] = i;
      T::m_value.write(ept, vs[i]);
    }
  }
};
#endif //HYPERION_USE_CASACORE

template <typename T, int N, ptrdiff_t...Args>
struct dynamic_mdspan_ {
  typedef typename
  dynamic_mdspan_<T, N - 1, std::experimental::dynamic_extent, Args...>::type
  type;
};

template <typename T, ptrdiff_t...Args>
struct dynamic_mdspan_<T, 1, Args...> {
  typedef std::experimental::mdspan<T, std::experimental::dynamic_extent, Args...> type;
};

template <typename T, int N>
struct dynamic_mdspan_t {
  typedef void type;
};

template <typename T>
struct dynamic_mdspan_t<T, 1> {
  typedef typename dynamic_mdspan_<T, 1>::type type;
  static type create(T* ptr, const Legion::Rect<1>& r) {
    return type(ptr, r.hi[0] - r.lo[0] + 1);
  }
};

template <typename T>
struct dynamic_mdspan_t<T, 2> {
  typedef typename dynamic_mdspan_<T, 2>::type type;
  static type create(T* ptr, const Legion::Rect<2>& r) {
    return type(
      ptr,
      r.hi[0] - r.lo[0] + 1,
      r.hi[1] - r.lo[1] + 1);
  }
};

template <typename T>
struct dynamic_mdspan_t<T, 3> {
  typedef typename dynamic_mdspan_<T, 3>::type type;
  static type create(T* ptr, const Legion::Rect<3>& r) {
    return type(
      ptr,
      r.hi[0] - r.lo[0] + 1,
      r.hi[1] - r.lo[1] + 1,
      r.hi[2] - r.lo[2] + 1);
  }
};

template <typename T>
struct dynamic_mdspan_t<T, 4> {
  typedef typename dynamic_mdspan_<T, 4>::type type;
  static type create(T* ptr, const Legion::Rect<4>& r) {
    return type(
      ptr,
      r.hi[0] - r.lo[0] + 1,
      r.hi[1] - r.lo[1] + 1,
      r.hi[2] - r.lo[2] + 1,
      r.hi[3] - r.lo[3] + 1);
  }
};

template <typename T>
struct dynamic_mdspan_t<T, 5> {
  typedef typename dynamic_mdspan_<T, 5>::type type;
  static type create(T* ptr, const Legion::Rect<5>& r) {
    return type(
      ptr,
      r.hi[0] - r.lo[0] + 1,
      r.hi[1] - r.lo[1] + 1,
      r.hi[2] - r.lo[2] + 1,
      r.hi[3] - r.lo[3] + 1,
      r.hi[4] - r.lo[4] + 1);
  }
};

template <typename T>
struct dynamic_mdspan_t<T, 6> {
  typedef typename dynamic_mdspan_<T, 6>::type type;
  static type create(T* ptr, const Legion::Rect<6>& r) {
    return type(
      ptr,
      r.hi[0] - r.lo[0] + 1,
      r.hi[1] - r.lo[1] + 1,
      r.hi[2] - r.lo[2] + 1,
      r.hi[3] - r.lo[3] + 1,
      r.hi[4] - r.lo[4] + 1,
      r.hi[5] - r.lo[5] + 1);
  }
};

template <typename T>
struct dynamic_mdspan_t<T, 7> {
  typedef typename dynamic_mdspan_<T, 7>::type type;
  static type create(T* ptr, const Legion::Rect<7>& r) {
    return type(
      ptr,
      r.hi[0] - r.lo[0] + 1,
      r.hi[1] - r.lo[1] + 1,
      r.hi[2] - r.lo[2] + 1,
      r.hi[3] - r.lo[3] + 1,
      r.hi[4] - r.lo[4] + 1,
      r.hi[5] - r.lo[5] + 1,
      r.hi[6] - r.lo[6] + 1);
  }
};

template <typename T>
struct dynamic_mdspan_t<T, 8> {
  typedef typename dynamic_mdspan_<T, 8>::type type;
  static type create(T* ptr, const Legion::Rect<8>& r) {
    return type(
      ptr,
      r.hi[0] - r.lo[0] + 1,
      r.hi[1] - r.lo[1] + 1,
      r.hi[2] - r.lo[2] + 1,
      r.hi[3] - r.lo[3] + 1,
      r.hi[4] - r.lo[4] + 1,
      r.hi[5] - r.lo[5] + 1,
      r.hi[6] - r.lo[6] + 1,
      r.hi[7] - r.lo[7] + 1);
  }
};

template <typename T>
struct dynamic_mdspan_t<T, 9> {
  typedef typename dynamic_mdspan_<T, 9>::type type;
  static type create(T* ptr, const Legion::Rect<9>& r) {
    return type(
      ptr,
      r.hi[0] - r.lo[0] + 1,
      r.hi[1] - r.lo[1] + 1,
      r.hi[2] - r.lo[2] + 1,
      r.hi[3] - r.lo[3] + 1,
      r.hi[4] - r.lo[4] + 1,
      r.hi[5] - r.lo[5] + 1,
      r.hi[6] - r.lo[6] + 1,
      r.hi[7] - r.lo[7] + 1,
      r.hi[8] - r.lo[8] + 1);
  }
};

class HYPERION_EXPORT PhysicalColumn {
public:

  PhysicalColumn(
    Legion::Runtime* rt,
    hyperion::TypeTag dt,
    Legion::FieldID fid,
    unsigned index_rank,
    const Legion::PhysicalRegion& metadata,
    const Legion::LogicalRegion& region,
    const Legion::LogicalRegion& parent,
    const CXX_OPTIONAL_NAMESPACE::optional<Legion::PhysicalRegion>& values,
    const CXX_OPTIONAL_NAMESPACE::optional<
      Keywords::pair<Legion::PhysicalRegion>>& kws
#ifdef HYPERION_USE_CASACORE
    , const CXX_OPTIONAL_NAMESPACE::optional<MeasRef::DataRegions>& mr_drs
    , const CXX_OPTIONAL_NAMESPACE::optional<
        std::tuple<std::string, std::shared_ptr<PhysicalColumn>>>& refcol
#endif // HYPERION_USE_CASACORE
    )
    : m_dt(dt)
    , m_fid(fid)
    , m_index_rank(index_rank)
    , m_metadata(metadata)
    , m_domain(
      (region != Legion::LogicalRegion::NO_REGION)
      ? rt->get_index_space_domain(region.get_index_space())
      : Legion::Domain::NO_DOMAIN)
    , m_region(region)
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

  ColumnSpace
  column_space() const;

  ColumnSpace::AXIS_VECTOR_TYPE
  axes() const;

  template <
    Legion::PrivilegeMode MODE,
    typename FT,
    int N,
    typename COORD_T = Legion::coord_t,
    template<typename, int, typename> typename A = Legion::GenericAccessor,
    bool CHECK_BOUNDS = HYPERION_CHECK_BOUNDS>
  Legion::FieldAccessor<MODE, FT, N, COORD_T, A<FT, N, COORD_T>, CHECK_BOUNDS>
  accessor() const {
    return
      Legion::FieldAccessor<
        MODE,
        FT,
        N,
        COORD_T,
        A<FT, N, COORD_T>,
        CHECK_BOUNDS>(
        m_values.value(),
        m_fid);
  }

#ifdef HYPERION_USE_KOKKOS
  template <
    typename execution_space,
    Legion::PrivilegeMode MODE,
    typename FT,
    int N,
    typename COORD_T = Legion::coord_t,
    template<typename, int, typename> typename A = Legion::AffineAccessor,
    bool CHECK_BOUNDS = HYPERION_CHECK_BOUNDS>
  typename std::conditional<
    N == 1,
    Kokkos::View<
      typename std::conditional<
        MODE == READ_ONLY,
        typename view_element<const FT, N>::type,
        typename view_element<FT, N>::type>::type,
      typename execution_space::memory_space>,
    Kokkos::View<
      typename std::conditional<
        MODE == READ_ONLY,
        typename view_element<const FT, N>::type,
        typename view_element<FT, N>::type>::type,
      Kokkos::LayoutStride,
      typename execution_space::memory_space>>::type
  view() const {
    return accessor<MODE, FT, N, COORD_T, A, CHECK_BOUNDS>().accessor;
  }

  template <
    typename execution_space,
    Legion::PrivilegeMode MODE,
    typename FT,
    int N,
    typename COORD_T = Legion::coord_t,
    template<typename, int, typename> typename A = Legion::AffineAccessor,
    bool CHECK_BOUNDS = HYPERION_CHECK_BOUNDS>
  Kokkos::Experimental::OffsetView<
    typename std::conditional<
      MODE == READ_ONLY,
      typename view_element<const FT, N>::type,
      typename view_element<FT, N>::type>::type,
    Kokkos::LayoutStride,
    typename execution_space::memory_space>
  offset_view() const {
    return accessor<MODE, FT, N, COORD_T, A, CHECK_BOUNDS>().accessor;
  }
#endif // HYPERION_USE_KOKKOS

  template <
    Legion::PrivilegeMode MODE,
    typename FT,
    int N,
    typename COORD_T = Legion::coord_t,
    template<typename, int, typename> typename A = Legion::AffineAccessor,
    bool CHECK_BOUNDS = HYPERION_CHECK_BOUNDS>
  typename dynamic_mdspan_t<
    typename std::conditional<MODE == READ_ONLY, const FT, FT>::type,
    N>::type
  span() const {
    // FIXME: this does not account for layout!
    typedef dynamic_mdspan_t<
      typename std::conditional<MODE == READ_ONLY, const FT, FT>::type,
      N> mds;
    Legion::Rect<N> bounds(domain());
    return mds::create(
      accessor<MODE, FT, N, COORD_T, A, CHECK_BOUNDS>().ptr(bounds),
      bounds);
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
  region() const {
    return m_region;
  };

  const Legion::LogicalRegion&
  parent() const {
    return m_parent;
  };

  const CXX_OPTIONAL_NAMESPACE::optional<Legion::PhysicalRegion>&
  values() const {
    return m_values;
  };

  Legion::LogicalRegion
  values_lr() const {
    return
      map(
        m_values,
        [](const auto& pr) { return pr.get_logical_region(); })
      .value_or(m_region);
  }

  const CXX_OPTIONAL_NAMESPACE::optional<
    Keywords::pair<Legion::PhysicalRegion>>&
  kws() const {
    return m_kws;
  };

#ifdef HYPERION_USE_CASACORE

  const CXX_OPTIONAL_NAMESPACE::optional<MeasRef::DataRegions>&
  mr_drs() const {
    return m_mr_drs;
  };

  const CXX_OPTIONAL_NAMESPACE::optional<
    std::tuple<std::string, std::shared_ptr<PhysicalColumn>>>&
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

#if HAVE_CXX17
  typedef std::variant<simple_mrb_t, ref_mrb_t> mrb_t;
#else // !HAVE_CXX17
  typedef struct {
    bool is_simple;
    ref_mrb_t ref;
  } mrb_t;
#endif // HAVE_CXX17

  CXX_OPTIONAL_NAMESPACE::optional<mrb_t>
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
  , m_region(from.m_region)
  , m_parent(from.m_parent)
  , m_values(from.m_values)
  , m_kws(from.m_kws)
#ifdef HYPERION_USE_CASACORE
  , m_mr_drs(from.m_mr_drs)
  , m_refcol(from.m_refcol)
#endif // HYPERION_USE_CASACORE
  {}

protected:

  friend class Table;
  friend class PhysicalTable;

#ifdef HYPERION_USE_CASACORE
  void
  set_refcol(const std::string& name, const std::shared_ptr<PhysicalColumn>& col) {
    m_refcol = std::make_tuple(name, col);
  }
#endif // HYPERION_USE_CASACORE

  hyperion::TypeTag m_dt;

  Legion::FieldID m_fid;

  unsigned m_index_rank;

  Legion::PhysicalRegion m_metadata;

  // maintain m_domain in order to allow checks in narrowing copy constructors
  // to PhysicalColumn variants (w.o. needing Legion::Runtime instance)
  Legion::Domain m_domain;

  Legion::LogicalRegion m_region;

  Legion::LogicalRegion m_parent;

  // allow an optional values region, to support a PhysicalColumn without values
  // (e.g, some Table index column spaces)
  CXX_OPTIONAL_NAMESPACE::optional<Legion::PhysicalRegion> m_values;

  CXX_OPTIONAL_NAMESPACE::optional<Keywords::pair<Legion::PhysicalRegion>> m_kws;

#ifdef HYPERION_USE_CASACORE
  CXX_OPTIONAL_NAMESPACE::optional<MeasRef::DataRegions> m_mr_drs;

  CXX_OPTIONAL_NAMESPACE::optional<
    std::tuple<std::string, std::shared_ptr<PhysicalColumn>>> m_refcol;
#endif // HYPERION_USE_CASACORE
};

template <
  hyperion::TypeTag DT,
  template <typename, int, typename> typename A = Legion::GenericAccessor,
  typename COORD_T = Legion::coord_t>
class PhysicalColumnT
  : public PhysicalColumn {
public:

  typedef typename DataType<DT>::ValueType value_t;

  PhysicalColumnT(
    Legion::Runtime* rt,
    Legion::FieldID fid,
    unsigned index_rank,
    const Legion::PhysicalRegion& metadata,
    const Legion::LogicalRegion& region,
    const Legion::LogicalRegion& parent,
    const CXX_OPTIONAL_NAMESPACE::optional<Legion::PhysicalRegion>& values,
    const CXX_OPTIONAL_NAMESPACE::optional<
      Keywords::pair<Legion::PhysicalRegion>>& kws
#ifdef HYPERION_USE_CASACORE
    , const CXX_OPTIONAL_NAMESPACE::optional<MeasRef::DataRegions>& mr_drs
    , const CXX_OPTIONAL_NAMESPACE::optional<
        std::tuple<std::string, std::shared_ptr<PhysicalColumn>>>& refcol
#endif // HYPERION_USE_CASACORE
    )
    : PhysicalColumn(
      rt,
      fid,
      DT,
      index_rank,
      metadata,
      region,
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

  template <
    Legion::PrivilegeMode MODE,
    int N,
    bool CHECK_BOUNDS = HYPERION_CHECK_BOUNDS>
  Legion::FieldAccessor<
    MODE,
    value_t,
    N,
    COORD_T,
    A<typename DataType<DT>::ValueType, N, COORD_T>,
    CHECK_BOUNDS>
  accessor() const {
    return
      PhysicalColumn::accessor<MODE, value_t, N, COORD_T, A, CHECK_BOUNDS>();
  }

#ifdef HYPERION_USE_KOKKOS
  template <
    typename execution_space,
    Legion::PrivilegeMode MODE,
    int N,
    bool CHECK_BOUNDS = HYPERION_CHECK_BOUNDS>
  typename std::conditional<
    N == 1,
    Kokkos::View<
      typename std::conditional<
        MODE == READ_ONLY,
        typename view_element<const value_t, N>::type,
        typename view_element<value_t, N>::type>::type,
      typename execution_space::memory_space>,
    Kokkos::View<
      typename std::conditional<
        MODE == READ_ONLY,
        typename view_element<const value_t, N>::type,
        typename view_element<value_t, N>::type>::type,
      Kokkos::LayoutStride,
      typename execution_space::memory_space>>::type
  view() const {
    return accessor<MODE, N, CHECK_BOUNDS>().accessor;
  }

  template <
    typename execution_space,
    Legion::PrivilegeMode MODE,
    int N,
    bool CHECK_BOUNDS = HYPERION_CHECK_BOUNDS>
  Kokkos::Experimental::OffsetView<
    typename std::conditional<
      MODE == READ_ONLY,
      typename view_element<const value_t, N>::type,
      typename view_element<value_t, N>::type>::type,
    Kokkos::LayoutStride,
    typename execution_space::memory_space>
  offset_view() const {
    return accessor<MODE, N, CHECK_BOUNDS>().accessor;
  }
#endif // HYPERION_USE_KOKKOS

  template <
    Legion::PrivilegeMode MODE,
    int N,
    bool CHECK_BOUNDS = HYPERION_CHECK_BOUNDS>
  typename dynamic_mdspan_t<
    typename std::conditional<MODE == READ_ONLY, const value_t, value_t>::type,
    N>::type
  span() const {
    // FIXME: this does not account for layout!
    typedef dynamic_mdspan_t<
      typename std::conditional<MODE == READ_ONLY, const value_t, value_t>::type,
      N> mds;
    Legion::Rect<N> bounds(domain());
    return mds::create(accessor<MODE, N, CHECK_BOUNDS>().ptr(bounds), bounds);
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

  typedef typename DataType<DT>::ValueType value_t;

  static_assert(INDEX_RANK <= COLUMN_RANK);

  PhysicalColumnTD(
    Legion::Runtime* rt,
    Legion::FieldID fid,
    const Legion::PhysicalRegion& metadata,
    const Legion::LogicalRegion& region,
    const Legion::LogicalRegion& parent,
    const CXX_OPTIONAL_NAMESPACE::optional<Legion::PhysicalRegion>& values,
    const CXX_OPTIONAL_NAMESPACE::optional<
      Keywords::pair<Legion::PhysicalRegion>>& kws
#ifdef HYPERION_USE_CASACORE
    , const CXX_OPTIONAL_NAMESPACE::optional<MeasRef::DataRegions>& mr_drs
    , const CXX_OPTIONAL_NAMESPACE::optional<
        std::tuple<std::string, std::shared_ptr<PhysicalColumn>>>& refcol
#endif // HYPERION_USE_CASACORE
    )
    : PhysicalColumn(
      rt,
      fid,
      DT,
      INDEX_RANK,
      metadata,
      region,
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
    assert(COLUMN_RANK == m_region.get_index_space().get_dim());
  }

  template <
    Legion::PrivilegeMode MODE,
    bool CHECK_BOUNDS = HYPERION_CHECK_BOUNDS>
  Legion::FieldAccessor<
    MODE,
    value_t,
    COLUMN_RANK,
    COORD_T,
    A<typename DataType<DT>::ValueType, COLUMN_RANK, COORD_T>,
    CHECK_BOUNDS>
  accessor() const {
    return
      PhysicalColumn::accessor<
        MODE,
        value_t,
        COLUMN_RANK,
        COORD_T,
        A,
        CHECK_BOUNDS>();
  }

#ifdef HYPERION_USE_KOKKOS
  template <
    typename execution_space,
    Legion::PrivilegeMode MODE,
    bool CHECK_BOUNDS = HYPERION_CHECK_BOUNDS>
  typename std::conditional<
    COLUMN_RANK == 1,
    Kokkos::View<
      typename std::conditional<
        MODE == READ_ONLY,
        typename view_element<const value_t, COLUMN_RANK>::type,
        typename view_element<value_t, COLUMN_RANK>::type>::type,
      typename execution_space::memory_space>,
    Kokkos::View<
      typename std::conditional<
        MODE == READ_ONLY,
        typename view_element<const value_t, COLUMN_RANK>::type,
        typename view_element<value_t, COLUMN_RANK>::type>::type,
      Kokkos::LayoutStride,
      typename execution_space::memory_space>>::type
  view() const {
    return accessor<MODE, CHECK_BOUNDS>().accessor;
  }

  template <
    typename execution_space,
    Legion::PrivilegeMode MODE,
    bool CHECK_BOUNDS = HYPERION_CHECK_BOUNDS>
  Kokkos::Experimental::OffsetView<
    typename std::conditional<
      MODE == READ_ONLY,
      typename view_element<const value_t, COLUMN_RANK>::type,
      typename view_element<value_t, COLUMN_RANK>::type>::type,
    Kokkos::LayoutStride,
    typename execution_space::memory_space>
  offset_view() const {
    return accessor<MODE, CHECK_BOUNDS>().accessor;
  }
#endif // HYPERION_USE_KOKKOS

  template <
    Legion::PrivilegeMode MODE,
    bool CHECK_BOUNDS = HYPERION_CHECK_BOUNDS>
  typename dynamic_mdspan_t<
    typename std::conditional<MODE == READ_ONLY, const value_t, value_t>::type,
    COLUMN_RANK>::type
  span() const {
    // FIXME: this does not account for layout!
    typedef dynamic_mdspan_t<
      typename std::conditional<MODE == READ_ONLY, const value_t, value_t>::type,
      COLUMN_RANK> mds;
    return mds::create(accessor<MODE, CHECK_BOUNDS>().ptr(rect()), rect());
  }

  Legion::DomainT<COLUMN_RANK>
  domain() const {
    return PhysicalColumn::domain();
  }

  Legion::Rect<COLUMN_RANK>
  rect() const {
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

  typedef typename DataType<DT>::ValueType value_t;

  PhysicalColumnTM(
    Legion::Runtime* rt,
    Legion::FieldID fid,
    unsigned index_rank,
    const Legion::PhysicalRegion& metadata,
    const Legion::LogicalRegion& region,
    const Legion::LogicalRegion& parent,
    const CXX_OPTIONAL_NAMESPACE::optional<Legion::PhysicalRegion>& values,
    const CXX_OPTIONAL_NAMESPACE::optional<
      Keywords::pair<Legion::PhysicalRegion>>& kws,
    const CXX_OPTIONAL_NAMESPACE::optional<MeasRef::DataRegions>& mr_drs,
    const CXX_OPTIONAL_NAMESPACE::optional<
      std::tuple<std::string, std::shared_ptr<PhysicalColumn>>>& refcol)
    : PhysicalColumn(
      rt,
      fid,
      DT,
      index_rank,
      metadata,
      region,
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

  template <
    Legion::PrivilegeMode MODE,
    int N,
    bool CHECK_BOUNDS = HYPERION_CHECK_BOUNDS>
  Legion::FieldAccessor<
    MODE,
    value_t,
    N,
    COORD_T,
    A<typename DataType<DT>::ValueType, N, COORD_T>,
    CHECK_BOUNDS>
  accessor() const {
    return
      PhysicalColumn::accessor<MODE, value_t, N, COORD_T, A, CHECK_BOUNDS>();
  }

#ifdef HYPERION_USE_KOKKOS
  template <
    typename execution_space,
    Legion::PrivilegeMode MODE,
    int N,
    bool CHECK_BOUNDS = HYPERION_CHECK_BOUNDS>
  typename std::conditional<
    N == 1,
    Kokkos::View<
      typename std::conditional<
        MODE == READ_ONLY,
        typename view_element<const value_t, N>::type,
        typename view_element<value_t, N>::type>::type,
      typename execution_space::memory_space>,
    Kokkos::View<
      typename std::conditional<
        MODE == READ_ONLY,
        typename view_element<const value_t, N>::type,
        typename view_element<value_t, N>::type>::type,
      Kokkos::LayoutStride,
      typename execution_space::memory_space>>::type
  view() const {
    return accessor<MODE, N, CHECK_BOUNDS>().accessor;
  }

  template <
    typename execution_space,
    Legion::PrivilegeMode MODE,
    int N,
    bool CHECK_BOUNDS = HYPERION_CHECK_BOUNDS>
  Kokkos::Experimental::OffsetView<
    typename std::conditional<
      MODE == READ_ONLY,
      typename view_element<const value_t, N>::type,
      typename view_element<value_t, N>::type>::type,
    Kokkos::LayoutStride,
    typename execution_space::memory_space>
  offset_view() const {
    return accessor<MODE, N, CHECK_BOUNDS>().accessor;
  }
#endif // HYPERION_USE_KOKKOS

  template <
    Legion::PrivilegeMode MODE,
    int N,
    bool CHECK_BOUNDS = HYPERION_CHECK_BOUNDS>
  typename dynamic_mdspan_t<
    typename std::conditional<MODE == READ_ONLY, const value_t, value_t>::type,
    N>::type
  span() const {
    // FIXME: this does not account for layout!
    typedef dynamic_mdspan_t<
      typename std::conditional<MODE == READ_ONLY, const value_t, value_t>::type,
      N> mds;
    Legion::Rect<N> bounds(domain());
    return mds::create(accessor<MODE, N, CHECK_BOUNDS>().ptr(bounds), bounds);
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

  typedef typename DataType<DT>::ValueType value_t;

  static const constexpr unsigned M_RANK = MClassT<MC>::mrank;

  static_assert(MV_SIZE > 0);
  static_assert(M_RANK <= 1);

  PhysicalColumnTMD(
    Legion::Runtime* rt,
    Legion::FieldID fid,
    const Legion::PhysicalRegion& metadata,
    const Legion::LogicalRegion& region,
    const Legion::LogicalRegion& parent,
    const CXX_OPTIONAL_NAMESPACE::optional<Legion::PhysicalRegion>& values,
    const CXX_OPTIONAL_NAMESPACE::optional<
      Keywords::pair<Legion::PhysicalRegion>>& kws,
    const CXX_OPTIONAL_NAMESPACE::optional<MeasRef::DataRegions>& mr_drs,
    const CXX_OPTIONAL_NAMESPACE::optional<
      std::tuple<std::string, std::shared_ptr<PhysicalColumn>>>& refcol)
    : PhysicalColumn(
      rt,
      fid,
      DT,
      INDEX_RANK,
      metadata,
      region,
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
    assert(COLUMN_RANK == m_region.get_index_space().get_dim());
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

  template <
    Legion::PrivilegeMode MODE,
    bool CHECK_BOUNDS = HYPERION_CHECK_BOUNDS>
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
#if HAVE_CXX17
      std::visit(overloaded {
          [this](const simple_mrb_t& mr) {
            m_mr = std::dynamic_pointer_cast<MR_t>(mr);
            m_convert.setOut(*m_mr);
          },
          [this](const ref_mrb_t& mr) {
            auto& [mrs, rmap, rcodes_pr, fid] = mr;
            std::vector<std::shared_ptr<MR_t>> tmrs;
            for (auto& m : mrs)
              tmrs.push_back(std::dynamic_pointer_cast<MR_t>(m));
            m_mrv =
              std::make_tuple(
                tmrs,
                rmap,
                RefcodeAccessor(rcodes_pr, fid));
          }
        },
        *mr);
#else // !HAVE_CXX17
      auto& mrs = std::get<0>(mr->ref);
      auto& rmap = std::get<1>(mr->ref);
      auto& rcodes_pr = std::get<2>(mr->ref);
      auto& fid = std::get<3>(mr->ref);
      if (mr->is_simple) {
        m_mr = std::dynamic_pointer_cast<MR_t>(mrs[0]);
        m_convert.setOut(*m_mr);
      } else {
        std::vector<std::shared_ptr<MR_t>> tmrs;
        for (auto& m : mrs)
          tmrs.push_back(std::dynamic_pointer_cast<MR_t>(m));
        m_mrv =
          std::make_tuple(
            tmrs,
            rmap,
            RefcodeAccessor(rcodes_pr, fid));
      }
#endif
    }

  public:

    MC_t&
    convert_at(const Legion::Point<COLUMN_RANK - M_RANK, Legion::coord_t>& pt) {
      if (m_mrv) {
#if HAVE_CXX17
        auto& [mrs, rmap, rcodes] = m_mrv.value();
#else // !HAVE_CXX17
        auto& mrs = std::get<0>(m_mrv.value());
        auto& rmap = std::get<1>(m_mrv.value());
        auto& rcodes = std::get<2>(m_mrv.value());
#endif // HAVE_CXX17
        m_convert.setOut(
          *mrs[
            rmap.at(
              rcodes.read(
                reinterpret_cast<const Legion::Point<INDEX_RANK, Legion::coord_t>&>(
                  pt)))]);
      }
      return m_convert;
    }

    MR_t&
    meas_ref_at(const Legion::Point<COLUMN_RANK - M_RANK, Legion::coord_t>& pt) {
      if (m_mrv) {
#if HAVE_CXX17
        auto& [mrs, rmap, rcodes] = m_mrv.value();
#else // !HAVE_CXX17
        auto& mrs = std::get<0>(m_mrv.value());
        auto& rmap = std::get<1>(m_mrv.value());
        auto& rcodes = std::get<2>(m_mrv.value());
#endif // HAVE_CXX17
        m_mr =
          mrs[
            rmap.at(
              rcodes.read(
                reinterpret_cast<const Legion::Point<INDEX_RANK, Legion::coord_t>&>(
                  pt)))];
      }
      return *m_mr;
    }

  private:
    std::shared_ptr<MR_t> m_mr;

    CXX_OPTIONAL_NAMESPACE::optional<
      std::tuple<
        std::vector<std::shared_ptr<MR_t>>,
        std::unordered_map<unsigned, unsigned>,
        RefcodeAccessor>>
    m_mrv;

    MC_t m_convert;
  };

  template <
    Legion::PrivilegeMode MODE,
    bool CHECK_BOUNDS = HYPERION_CHECK_BOUNDS>
  Legion::FieldAccessor<
    MODE,
    value_t,
    COLUMN_RANK,
    COORD_T,
    A<typename DataType<DT>::ValueType, COLUMN_RANK, COORD_T>,
    CHECK_BOUNDS>
  accessor() const {
    return
      PhysicalColumn::accessor<
        MODE,
        value_t,
        COLUMN_RANK,
        COORD_T,
        A,
        CHECK_BOUNDS>();
  }

#ifdef HYPERION_USE_KOKKOS
  template <
    typename execution_space,
    Legion::PrivilegeMode MODE,
    bool CHECK_BOUNDS = HYPERION_CHECK_BOUNDS>
  typename std::conditional<
    COLUMN_RANK == 1,
    Kokkos::View<
      typename std::conditional<
        MODE == READ_ONLY,
        typename view_element<const value_t, COLUMN_RANK>::type,
        typename view_element<value_t, COLUMN_RANK>::type>::type,
      typename execution_space::memory_space>,
    Kokkos::View<
      typename std::conditional<
        MODE == READ_ONLY,
        typename view_element<const value_t, COLUMN_RANK>::type,
        typename view_element<value_t, COLUMN_RANK>::type>::type,
      Kokkos::LayoutStride,
      typename execution_space::memory_space>>::type
  view() const {
    return accessor<MODE, CHECK_BOUNDS>().accessor;
  }

  template <
    typename execution_space,
    Legion::PrivilegeMode MODE,
    bool CHECK_BOUNDS = HYPERION_CHECK_BOUNDS>
  Kokkos::Experimental::OffsetView<
    typename std::conditional<
      MODE == READ_ONLY,
      typename view_element<const value_t, COLUMN_RANK>::type,
      typename view_element<value_t, COLUMN_RANK>::type>::type,
    Kokkos::LayoutStride,
    typename execution_space::memory_space>
  offset_view() const {
    return accessor<MODE, CHECK_BOUNDS>().accessor;
  }

#endif // HYPERION_USE_KOKKOS

  template <
    Legion::PrivilegeMode MODE,
    bool CHECK_BOUNDS = HYPERION_CHECK_BOUNDS>
  typename dynamic_mdspan_t<
    typename std::conditional<MODE == READ_ONLY, const value_t, value_t>::type,
    COLUMN_RANK>::type
  span() const {
    // FIXME: this does not account for layout!
    typedef dynamic_mdspan_t<
      typename std::conditional<MODE == READ_ONLY, const value_t, value_t>::type,
      COLUMN_RANK> mds;
    return mds::create(accessor<MODE, CHECK_BOUNDS>().ptr(rect()), rect());
  }

  Legion::DomainT<COLUMN_RANK>
  domain() const {
    return PhysicalColumn::domain();
  }

  Legion::Rect<COLUMN_RANK>
  rect() const {
    return PhysicalColumn::domain();
  }

  template <
    Legion::PrivilegeMode MODE,
    bool CHECK_BOUNDS = HYPERION_CHECK_BOUNDS>
  MeasRefAccessor<MODE, CHECK_BOUNDS>
  meas_ref_accessor(Legion::Runtime* rt) const {
    auto omrb = mrb(rt);
    return MeasRefAccessor<MODE, CHECK_BOUNDS>(&omrb.value());
  }

  template <
    Legion::PrivilegeMode MODE,
    bool CHECK_BOUNDS = HYPERION_CHECK_BOUNDS>
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

  template <
    Legion::PrivilegeMode MODE,
    bool CHECK_BOUNDS = HYPERION_CHECK_BOUNDS>
  class MeasAccessor
    : public MeasWriterMixin<
        MeasAccessorBase<MODE, CHECK_BOUNDS>,
        DT,
        MC,
        INDEX_RANK,
        COLUMN_RANK,
        MClassT<MC>::mrank,
        MV_SIZE,
        COORD_T> {
    typedef MeasWriterMixin<
      MeasAccessorBase<MODE, CHECK_BOUNDS>,
      DT,
      MC,
      INDEX_RANK,
      COLUMN_RANK,
      MClassT<MC>::mrank,
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
        MClassT<MC>::mrank,
        MV_SIZE,
        COORD_T> {
    typedef MeasReaderMixin<
      MeasAccessorBase<READ_ONLY, CHECK_BOUNDS>,
      DT,
      MC,
      INDEX_RANK,
      COLUMN_RANK,
      MClassT<MC>::mrank,
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
          MClassT<MC>::mrank,
          MV_SIZE,
          COORD_T>,
        DT,
        MC,
        INDEX_RANK,
        COLUMN_RANK,
        MClassT<MC>::mrank,
        MV_SIZE,
        COORD_T> {
    typedef MeasReaderMixin<
      MeasWriterMixin<
        MeasAccessorBase<READ_WRITE, CHECK_BOUNDS>,
        DT,
        MC,
        INDEX_RANK,
        COLUMN_RANK,
        MClassT<MC>::mrank,
        MV_SIZE,
        COORD_T>,
      DT,
      MC,
      INDEX_RANK,
      COLUMN_RANK,
      MClassT<MC>::mrank,
      MV_SIZE,
      COORD_T> T;
  public:
    using T::T;
  };

  template <
    Legion::PrivilegeMode MODE,
    bool CHECK_BOUNDS = HYPERION_CHECK_BOUNDS>
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
