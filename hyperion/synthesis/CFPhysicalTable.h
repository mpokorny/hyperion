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
#ifndef HYPERION_SYNTHESIS_CF_PHYSICAL_TABLE_H_
#define HYPERION_SYNTHESIS_CF_PHYSICAL_TABLE_H_

#include <hyperion/synthesis/CFTableBase.h>
#include <hyperion/PhysicalTable.h>

namespace hyperion {
namespace synthesis {

template <cf_table_axes_t ...AXES>
class CFPhysicalTable
  : public hyperion::PhysicalTable {
public:

  CFPhysicalTable(const hyperion::PhysicalTable& pt)
    : hyperion::PhysicalTable(pt) {
    assert(verify_axes());
  }

  CFPhysicalTable(hyperion::PhysicalTable&& pt)
    : hyperion::PhysicalTable(std::move(pt)) {
    assert(verify_axes());
  }

  bool
  verify_axes() const {
    return
      axes_uid()
      && (axes_uid().value() == hyperion::Axes<cf_table_axes_t>::uid)
      && (index_axes() == std::vector<int>{static_cast<int>(AXES)...});
  }

  static const constexpr unsigned row_rank = sizeof...(AXES);

  template <cf_table_axes_t A>
  static constexpr bool
  has_index_axis() {
    return cf_indexing::includes<A, AXES...>;
  }

  template <cf_table_axes_t A>
  static constexpr int
  index_axis_index() {
    return std::conditional<
      cf_indexing::includes<A, AXES...>,
      typename cf_indexing::index_of<A, AXES...>::type,
      std::integral_constant<int, -1>>::type::value;
  }

  size_t
  grid_size() const {
    return value().rect().hi[value_rank - 1] + 1;
  }

  //
  // PS_SCALE
  //
  static const constexpr unsigned ps_scale_rank =
    std::conditional<
      cf_indexing::includes<CF_PS_SCALE, AXES...>,
      std::integral_constant<unsigned, 1>,
      std::integral_constant<unsigned, row_rank>>::type::value;

  bool
  has_ps_scale() const {
    return m_columns.count(cf_table_axis<CF_PS_SCALE>::name) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    ValueType<typename cf_table_axis<CF_PS_SCALE>::type>::DataType,
    ps_scale_rank,
    ps_scale_rank,
    A,
    COORD_T>
  ps_scale() const {
    return
      decltype(ps_scale<A, COORD_T>())(
        *m_columns.at(cf_table_axis<CF_PS_SCALE>::name));
  }

  template <int N>
  Legion::Point<ps_scale_rank>
  ps_scale_index(const Legion::Point<N>& pt) const {
    return cf_indexing::Pt<row_rank, N, CF_PS_SCALE, AXES...>(pt).pt;
  }

  //
  // BASELINE_CLASS
  //
  static const constexpr unsigned baseline_class_rank =
    std::conditional<
      cf_indexing::includes<CF_BASELINE_CLASS, AXES...>,
      std::integral_constant<unsigned, 1>,
      std::integral_constant<unsigned, row_rank>>::type::value;

  bool
  has_baseline_class() const {
    return m_columns.count(cf_table_axis<CF_BASELINE_CLASS>::name) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    ValueType<typename cf_table_axis<CF_BASELINE_CLASS>::type>::DataType,
    baseline_class_rank,
    baseline_class_rank,
    A,
    COORD_T>
  baseline_class() const {
    return
      decltype(baseline_class<A, COORD_T>())(
        *m_columns.at(cf_table_axis<CF_BASELINE_CLASS>::name));
  }

  template <int N>
  Legion::Point<baseline_class_rank>
  baseline_class_index(const Legion::Point<N>& pt) const {
    return cf_indexing::Pt<row_rank, N, CF_BASELINE_CLASS, AXES...>(pt).pt;
  }

  //
  // FREQUENCY
  //
  static const constexpr unsigned frequency_rank =
    std::conditional<
      cf_indexing::includes<CF_FREQUENCY, AXES...>,
      std::integral_constant<unsigned, 1>,
      std::integral_constant<unsigned, row_rank>>::type::value;

  bool
  has_frequency() const {
    return m_columns.count(cf_table_axis<CF_FREQUENCY>::name) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    ValueType<typename cf_table_axis<CF_FREQUENCY>::type>::DataType,
    frequency_rank,
    frequency_rank,
    A,
    COORD_T>
  frequency() const {
    return
      decltype(frequency<A, COORD_T>())(
        *m_columns.at(cf_table_axis<CF_FREQUENCY>::name));
  }

  template <int N>
  Legion::Point<frequency_rank>
  frequency_index(const Legion::Point<N>& pt) const {
    return cf_indexing::Pt<row_rank, N, CF_FREQUENCY, AXES...>(pt).pt;
  }

  //
  // W
  //
  static const constexpr unsigned w_rank =
    std::conditional<
      cf_indexing::includes<CF_W, AXES...>,
      std::integral_constant<unsigned, 1>,
      std::integral_constant<unsigned, row_rank>>::type::value;

  bool
  has_w() const {
    return m_columns.count(cf_table_axis<CF_W>::name) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    ValueType<typename cf_table_axis<CF_W>::type>::DataType,
    w_rank,
    w_rank,
    A,
    COORD_T>
  w() const {
    return decltype(w<A, COORD_T>())(*m_columns.at(cf_table_axis<CF_W>::name));
  }

  template <int N>
  Legion::Point<w_rank>
  w_index(const Legion::Point<N>& pt) const {
    return cf_indexing::Pt<row_rank, N, CF_W, AXES...>(pt).pt;
  }

  //
  // PARALLACTIC_ANGLE
  //
  static const constexpr unsigned parallactic_angle_rank =
    std::conditional<
      cf_indexing::includes<CF_PARALLACTIC_ANGLE, AXES...>,
      std::integral_constant<unsigned, 1>,
      std::integral_constant<unsigned, row_rank>>::type::value;

  bool
  has_parallactic_angle() const {
    return m_columns.count(cf_table_axis<CF_PARALLACTIC_ANGLE>::name) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    ValueType<typename cf_table_axis<CF_PARALLACTIC_ANGLE>::type>::DataType,
    parallactic_angle_rank,
    parallactic_angle_rank,
    A,
    COORD_T>
  parallactic_angle() const {
    return
      decltype(parallactic_angle<A, COORD_T>())(
        *m_columns.at(cf_table_axis<CF_PARALLACTIC_ANGLE>::name));
  }

  template <int N>
  Legion::Point<parallactic_angle_rank>
  parallactic_angle_index(const Legion::Point<N>& pt) const {
    return cf_indexing::Pt<row_rank, N, CF_PARALLACTIC_ANGLE, AXES...>(pt).pt;
  }

  //
  // STOKES_OUT
  //
  static const constexpr unsigned stokes_out_rank =
    std::conditional<
      cf_indexing::includes<CF_STOKES_OUT, AXES...>,
      std::integral_constant<unsigned, 1>,
      std::integral_constant<unsigned, row_rank>>::type::value;

  bool
  has_stokes_out() const {
    return m_columns.count(cf_table_axis<CF_STOKES_OUT>::name) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    ValueType<typename cf_table_axis<CF_STOKES_OUT>::type>::DataType,
    stokes_out_rank,
    stokes_out_rank,
    A,
    COORD_T>
  stokes_out() const {
    return
      decltype(stokes_out<A, COORD_T>())(
        *m_columns.at(cf_table_axis<CF_STOKES_OUT>::name));
  }

  template <int N>
  Legion::Point<stokes_out_rank>
  stokes_out_index(const Legion::Point<N>& pt) const {
    return cf_indexing::Pt<row_rank, N, CF_STOKES_OUT, AXES...>(pt).pt;
  }

  //
  // STOKES_IN
  //
  static const constexpr unsigned stokes_in_rank =
    std::conditional<
    cf_indexing::includes<CF_STOKES_IN, AXES...>,
    std::integral_constant<unsigned, 1>,
    std::integral_constant<unsigned, row_rank>>::type::value;

  bool
  has_stokes_in() const {
    return m_columns.count(cf_table_axis<CF_STOKES_IN>::name) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    ValueType<typename cf_table_axis<CF_STOKES_IN>::type>::DataType,
    stokes_in_rank,
    stokes_in_rank,
    A,
    COORD_T>
  stokes_in() const {
    return
      decltype(stokes_in<A, COORD_T>())(
        *m_columns.at(cf_table_axis<CF_STOKES_IN>::name));
  }

  template <int N>
  Legion::Point<stokes_in_rank>
  stokes_in_index(const Legion::Point<N>& pt) const {
    return cf_indexing::Pt<row_rank, N, CF_STOKES_IN, AXES...>(pt).pt;
  }

  //
  // STOKES
  //
  static const constexpr unsigned stokes_rank =
    std::conditional<
    cf_indexing::includes<CF_STOKES, AXES...>,
    std::integral_constant<unsigned, 1>,
    std::integral_constant<unsigned, row_rank>>::type::value;

  bool
  has_stokes() const {
    return m_columns.count(cf_table_axis<CF_STOKES>::name) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    ValueType<typename cf_table_axis<CF_STOKES>::type>::DataType,
    stokes_rank,
    stokes_rank,
    A,
    COORD_T>
  stokes() const {
    return decltype(stokes<A, COORD_T>())(
      *m_columns.at(cf_table_axis<CF_STOKES>::name));
  }

  template <int N>
  Legion::Point<stokes_rank>
  stokes_index(const Legion::Point<N>& pt) const {
    return cf_indexing::Pt<row_rank, N, CF_STOKES, AXES...>(pt).pt;
  }

  //
  // VALUE
  //
  static const constexpr unsigned value_rank = row_rank + 2;

  bool
  has_value() const {
    return m_columns.count(CFTableBase::CF_VALUE_COLUMN_NAME) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    ValueType<CFTableBase::cf_value_t>::DataType,
    row_rank,
    value_rank,
    A,
    COORD_T>
  value() const {
    return decltype(value<A, COORD_T>())(
      *m_columns.at(CFTableBase::CF_VALUE_COLUMN_NAME));
  }

  const Legion::Point<row_rank>&
  value_row_index(const Legion::Point<value_rank>& pt) const {
    return reinterpret_cast<const Legion::Point<row_rank>&>(pt);
  }

  //
  // WEIGHT
  //
  static const constexpr unsigned weight_rank = row_rank + 2;

  bool
  has_weight() const {
    return m_columns.count(CFTableBase::CF_WEIGHT_COLUMN_NAME) > 0;
  }

  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  PhysicalColumnTD<
    ValueType<CFTableBase::cf_value_t>::DataType,
    row_rank,
    weight_rank,
    A,
    COORD_T>
  weight() const {
    return decltype(weight<A, COORD_T>())(
      *m_columns.at(CFTableBase::CF_WEIGHT_COLUMN_NAME));
  }

  const Legion::Point<row_rank>&
  weight_row_index(const Legion::Point<weight_rank>& pt) const {
    return reinterpret_cast<const Legion::Point<row_rank>&>(pt);
  }
};

template <cf_table_axes_t A>
struct IndexAxisHelper {

  template <cf_table_axes_t...T>
  static CFTableBase::Axis<A>
  index_axis(const CFPhysicalTable<T...>& pt);
};

template <>
struct IndexAxisHelper<CF_PS_SCALE> {

  template <cf_table_axes_t...T>
  static CFTableBase::Axis<CF_PS_SCALE>
  index_axis(const CFPhysicalTable<T...>& pt) {
    std::vector<typename cf_table_axis<CF_PS_SCALE>::type> values;
    if (pt.has_ps_scale()
        && CFPhysicalTable<T...>::ps_scale_rank == 1) {
      auto col = pt.template ps_scale<Legion::AffineAccessor>();
      auto acc = col.template accessor<LEGION_READ_ONLY>();
      for (Legion::PointInRectIterator<1> pir(col.rect()); pir(); pir++) {
        values.push_back(acc[*pir]);
      }
    }
    return CFTableBase::Axis<CF_PS_SCALE>(values);
  }
};

template <>
struct IndexAxisHelper<CF_BASELINE_CLASS> {

  template <cf_table_axes_t...T>
  static CFTableBase::Axis<CF_BASELINE_CLASS>
  index_axis(const CFPhysicalTable<T...>& pt) {
    std::vector<typename cf_table_axis<CF_BASELINE_CLASS>::type> values;
    if (pt.has_baseline_class()
        && CFPhysicalTable<T...>::baseline_class_rank == 1) {
      auto col = pt.template baseline_class<Legion::AffineAccessor>();
      auto acc = col.template accessor<LEGION_READ_ONLY>();
      for (Legion::PointInRectIterator<1> pir(col.rect()); pir(); pir++) {
        values.push_back(acc[*pir]);
      }
    }
    return CFTableBase::Axis<CF_BASELINE_CLASS>(values);
  }
};

template <>
struct IndexAxisHelper<CF_FREQUENCY> {

  template <cf_table_axes_t...T>
  static CFTableBase::Axis<CF_FREQUENCY>
  index_axis(const CFPhysicalTable<T...>& pt) {
    std::vector<typename cf_table_axis<CF_FREQUENCY>::type> values;
    if (pt.has_frequency()
        && CFPhysicalTable<T...>::frequency_rank == 1) {
      auto col = pt.template frequency<Legion::AffineAccessor>();
      auto acc = col.template accessor<LEGION_READ_ONLY>();
      for (Legion::PointInRectIterator<1> pir(col.rect()); pir(); pir++) {
        values.push_back(acc[*pir]);
      }
    }
    return CFTableBase::Axis<CF_FREQUENCY>(values);
  }
};

template <>
struct IndexAxisHelper<CF_W> {

  template <cf_table_axes_t...T>
  static CFTableBase::Axis<CF_W>
  index_axis(const CFPhysicalTable<T...>& pt) {
    std::vector<typename cf_table_axis<CF_W>::type> values;
    if (pt.has_w() && CFPhysicalTable<T...>::w_rank == 1) {
      auto col = pt.template w<Legion::AffineAccessor>();
      auto acc = col.template accessor<LEGION_READ_ONLY>();
      for (Legion::PointInRectIterator<1> pir(col.rect()); pir(); pir++) {
        values.push_back(acc[*pir]);
      }
    }
    return CFTableBase::Axis<CF_W>(values);
  }
};

template <>
struct IndexAxisHelper<CF_PARALLACTIC_ANGLE> {

  template <cf_table_axes_t...T>
  static CFTableBase::Axis<CF_PARALLACTIC_ANGLE>
  index_axis(const CFPhysicalTable<T...>& pt) {
    std::vector<typename cf_table_axis<CF_PARALLACTIC_ANGLE>::type> values;
    if (pt.has_parallactic_angle() &&
        CFPhysicalTable<T...>::parallactic_angle_rank == 1) {
      auto col = pt.template parallactic_angle<Legion::AffineAccessor>();
      auto acc = col.template accessor<LEGION_READ_ONLY>();
      for (Legion::PointInRectIterator<1> pir(col.rect()); pir(); pir++) {
        values.push_back(acc[*pir]);
      }
    }
    return CFTableBase::Axis<CF_PARALLACTIC_ANGLE>(values);
  }
};

template <>
struct IndexAxisHelper<CF_STOKES_OUT> {

  template <cf_table_axes_t...T>
  static CFTableBase::Axis<CF_STOKES_OUT>
  index_axis(const CFPhysicalTable<T...>& pt) {
    std::vector<typename cf_table_axis<CF_STOKES_OUT>::type> values;
    if (pt.has_stokes_out()
        && CFPhysicalTable<T...>::stokes_out_rank == 1) {
      auto col = pt.template stokes_out<Legion::AffineAccessor>();
      auto acc = col.template accessor<LEGION_READ_ONLY>();
      for (Legion::PointInRectIterator<1> pir(col.rect()); pir(); pir++) {
        values.push_back(acc[*pir]);
      }
    }
    return CFTableBase::Axis<CF_STOKES_OUT>(values);
  }
};

template <>
struct IndexAxisHelper<CF_STOKES_IN> {

  template <cf_table_axes_t...T>
  static CFTableBase::Axis<CF_STOKES_IN>
  index_axis(const CFPhysicalTable<T...>& pt) {
    std::vector<typename cf_table_axis<CF_STOKES_IN>::type> values;
    if (pt.has_stokes_in()
        && CFPhysicalTable<T...>::stokes_in_rank == 1) {
      auto col = pt.template stokes_in<Legion::AffineAccessor>();
      auto acc = col.template accessor<LEGION_READ_ONLY>();
      for (Legion::PointInRectIterator<1> pir(col.rect()); pir(); pir++) {
        values.push_back(acc[*pir]);
      }
    }
    return CFTableBase::Axis<CF_STOKES_IN>(values);
  }
};

template <>
struct IndexAxisHelper<CF_STOKES> {

  template <cf_table_axes_t...T>
  static CFTableBase::Axis<CF_STOKES>
  index_axis(const CFPhysicalTable<T...>& pt) {
    std::vector<typename cf_table_axis<CF_STOKES>::type> values;
    if (pt.has_stokes()
        && CFPhysicalTable<T...>::stokes_rank == 1) {
      auto col = pt.template stokes<Legion::AffineAccessor>();
      auto acc = col.template accessor<LEGION_READ_ONLY>();
      for (Legion::PointInRectIterator<1> pir(col.rect()); pir(); pir++) {
        values.push_back(acc[*pir]);
      }
    }
    return CFTableBase::Axis<CF_STOKES>(values);
  }
};

/**
 * hlist represents a compile-time, heterogeneous list
 *
 * This type is useful for iterating over the arguments of a variadic template
 * function, with arguments that are themselves variadic template types.
 *
 * @todo move this class to hyperion namespace, and use it elsewhere
 */
template <typename...Ts>
struct hlist {};

struct hnil
  : public hlist<> {
  hnil() {}
};

template <typename T, typename...Ts>
struct hcons;
template <typename...Ts>
struct hlist_tail;
template <typename T, typename...Ts>
struct hlist_tail<T, Ts...> {
  typedef hcons<Ts...> type;
};
template <typename T>
struct hlist_tail<T> {
  typedef hnil type;
};

template <typename T, typename...Ts>
struct hcons
  : public hlist<T, Ts...> {
  hcons(const T& t, const Ts&...ts)
    : hd(t)
    , tl(ts...) {}

  const T& hd;
  typename hlist_tail<T, Ts...>::type tl;
};

/**
 * get the first index axis of a given type from a list of tables
 *
 * see index_axis() for a version without any \ref hlist arguments
 */
template <
  cf_table_axes_t A,
  cf_table_axes_t...A0,
  typename...PTs,
  std::enable_if_t<cf_indexing::includes<A, A0...>, int> = 0>
CFTableBase::Axis<A>
index_axis_h(
  const CFPhysicalTable<A0...>& pt0,
  const hlist<PTs...>&) {

  return IndexAxisHelper<A>::template index_axis<A0...>(pt0);
}

template <
  cf_table_axes_t A,
  cf_table_axes_t...A0,
  typename...PTs,
  std::enable_if_t<!cf_indexing::includes<A, A0...>, int> = 0>
CFTableBase::Axis<A>
index_axis_h(
  const CFPhysicalTable<A0...>& pt0,
  const hcons<PTs...>& hc) {

  return index_axis_h<A>(hc.hd, hc.tl);
}

/**
 * get the first index axis of a given type from a list of tables
 *
 * see index_axis_h() for a version with an \ref hlist argument
 */
template <
  cf_table_axes_t A,
  typename PT0,
  typename...PTs,
  std::enable_if_t<(sizeof...(PTs) > 0), int> = 0>
CFTableBase::Axis<A>
index_axis(const PT0& pt0, const PTs&... rest) {
  return index_axis_h<A>(pt0, hcons<PTs...>(rest...));
}

template <cf_table_axes_t A, typename PT0>
CFTableBase::Axis<A>
index_axis(const PT0& pt0) {
  return index_axis_h<A>(pt0, hnil());
}

} // end namespace synthesis

} // end namespace hyperion

#endif // HYPERION_SYNTHESIS_CF_PHYSICAL_TABLE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
