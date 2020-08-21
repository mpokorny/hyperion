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
#ifndef HYPERION_SYNTHESIS_CF_TABLE_BASE_H_
#define HYPERION_SYNTHESIS_CF_TABLE_BASE_H_

#include <hyperion/hyperion.h>
#include <hyperion/utility.h>
#include <hyperion/Table.h>
#include <hyperion/PhysicalColumn.h>

#include <array>
#include <vector>

namespace hyperion {

namespace synthesis {

typedef enum cf_table_axes_t {
  CF_PS_SCALE,
  CF_BASELINE_CLASS,
  CF_FREQUENCY,
  CF_W,
  CF_PARALLACTIC_ANGLE,
  CF_STOKES_OUT,
  CF_STOKES_IN,
  CF_STOKES,
  CF_X,
  CF_Y,
  // the semantics of the ORDER axes are intentionally ambiguous; if no
  // partitions use these values, everything should be OK
  CF_ORDER0,
  CF_ORDER1,
  CF_LAST_AXIS = CF_ORDER1
} cf_table_axes_t;
}

typedef hyperion::table_indexing<hyperion::synthesis::cf_table_axes_t>
  cf_indexing;

template <>
struct Axes<synthesis::cf_table_axes_t> {
  static const constexpr char* uid = "hyperion:cf";
  static const std::vector<std::string> names;
  static const constexpr unsigned num_axes =
    synthesis::cf_table_axes_t::CF_LAST_AXIS - 1;
#ifdef HYPERION_USE_HDF5
  static const hid_t h5_datatype;
#endif
};

namespace synthesis {

template <cf_table_axes_t T>
struct cf_table_axis {
  //typedef ... type;
};
template <>
struct cf_table_axis<CF_PS_SCALE> {
  typedef float type;
  static const constexpr char* name = "PS_SCALE";
};
template <>
struct cf_table_axis<CF_BASELINE_CLASS> {
  typedef unsigned type;
  static const constexpr char* name = "BASELINE_CLASS";
};
template <>
struct cf_table_axis<CF_FREQUENCY> {
  typedef float type;
  static const constexpr char* name = "FREQUENCY";
};
template <>
struct cf_table_axis<CF_W> {
  typedef float type;
  static const constexpr char* name = "W";
};
template <>
struct cf_table_axis<CF_PARALLACTIC_ANGLE> {
  typedef float type;
  static const constexpr char* name = "PARALLACTIC_ANGLE";
};
template <>
struct cf_table_axis<CF_STOKES_OUT> {
  typedef stokes_t type;
  static const constexpr char* name = "STOKES_OUT";
};
template <>
struct cf_table_axis<CF_STOKES_IN> {
  typedef stokes_t type;
  static const constexpr char* name = "STOKES_IN";
};
template <>
struct cf_table_axis<CF_STOKES> {
  typedef stokes_t type;
  static const constexpr char* name = "STOKES";
};
template <>
struct cf_table_axis<CF_X> {
  typedef void type; // not a value index axis
  static const constexpr char* name = "X";
};
template <>
struct cf_table_axis<CF_Y> {
  typedef void type; // not a value index axis
  static const constexpr char* name = "Y";
};
template <>
struct cf_table_axis<CF_ORDER0> {
  typedef void type; // not a value index axis
  static const constexpr char* name = "ORDER0";
};
template <>
struct cf_table_axis<CF_ORDER1> {
  typedef void type; // not a value index axis
  static const constexpr char* name = "ORDER1";
};

HYPERION_EXPORT const char*
cf_table_axis_name(cf_table_axes_t ax);

class HYPERION_EXPORT CFTableBase
  : public hyperion::Table {
public:

  template <cf_table_axes_t T>
  struct Axis {
    Axis(const std::vector<typename cf_table_axis<T>::type>& v)
      : values(v) {}
    Axis(std::vector<typename cf_table_axis<T>::type>&& v)
      : values(v) {}

    std::vector<typename cf_table_axis<T>::type> values;

    Legion::Rect<1> bounds() const {
      return Legion::Rect<1>(0, values.size() -1);
    }
  };

  typedef float cf_fp_t;

  static const constexpr Legion::FieldID INDEX_VALUE_FID = 14;

  static const constexpr Legion::FieldID CF_VALUE_FID = 24;
  static const constexpr char* CF_VALUE_COLUMN_NAME = "VALUE";
  typedef hyperion::complex<cf_fp_t> cf_value_t;

  static const constexpr Legion::FieldID CF_WEIGHT_FID = 34;
  static const constexpr char* CF_WEIGHT_COLUMN_NAME = "WEIGHT";
  typedef hyperion::complex<cf_fp_t> cf_weight_t;

  static Legion::TaskID init_index_column_task_id;
  static const constexpr char* init_index_column_task_name =
    "CFTableBase::init_index_column_task";

  struct HYPERION_EXPORT InitIndexColumnTaskArgs {
    hyperion::Table::Desc desc;

    std::vector<typename cf_table_axis<CF_PS_SCALE>::type>
      ps_scales;
    std::vector<typename cf_table_axis<CF_BASELINE_CLASS>::type>
      baseline_classes;
    std::vector<typename cf_table_axis<CF_FREQUENCY>::type>
      frequencies;
    std::vector<typename cf_table_axis<CF_W>::type>
      w_values;
    std::vector<typename cf_table_axis<CF_PARALLACTIC_ANGLE>::type>
      parallactic_angles;
    std::vector<typename cf_table_axis<CF_STOKES_OUT>::type>
      stokes_out_values;
    std::vector<typename cf_table_axis<CF_STOKES_IN>::type>
      stokes_in_values;
    std::vector<typename cf_table_axis<CF_STOKES>::type>
      stokes_values;

    size_t serialized_size() const;
    size_t serialize(void*) const;
    size_t deserialize(const void*);

    template <cf_table_axes_t T>
    constexpr std::vector<typename cf_table_axis<T>::type>& values();

    template <cf_table_axes_t...Axes>
    struct initializer {
      static void init(InitIndexColumnTaskArgs& args, const Axis<Axes>&...);
    };
  };

  static void
  init_index_column_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* rt);

  CFTableBase(Table&& t)
    : Table(std::move(t)) {}

  CFTableBase()
    : Table() {}

  static void preregister_all();

  template <int N>
  static KOKKOS_INLINE_FUNCTION Kokkos::Array<long, N>
  rect_lo(const Legion::Rect<N, Legion::coord_t>& r) {
    Kokkos::Array<long, N> result;
    for (size_t i = 0; i < N; ++i)
      result[i] = r.lo[i];
    return result;
  }

  template <int N>
  static KOKKOS_INLINE_FUNCTION Kokkos::Array<long, N>
  rect_hi(const Legion::Rect<N, Legion::coord_t>& r) {
    Kokkos::Array<long, N> result;
    for (size_t i = 0; i < N; ++i)
      result[i] = r.hi[i] + 1;
    return result;
  }

  template <int N>
  static KOKKOS_INLINE_FUNCTION Kokkos::Array<long, N>
  rect_zero(const Legion::Rect<N, Legion::coord_t>&) {
    Kokkos::Array<long, N> result;
    for (size_t i = 0; i < N; ++i)
      result[i] = 0;
    return result;
  }

  template <int N>
  static KOKKOS_INLINE_FUNCTION Kokkos::Array<long, N>
  rect_size(const Legion::Rect<N, Legion::coord_t>& r) {
    Kokkos::Array<long, N> result;
    for (size_t i = 0; i < N; ++i)
      result[i] = r.hi[i] - r.lo[i] + 1;
    return result;
  }

  template <int N, typename T>
  static T
  linearized_index(
    const array<T, N>& pt,
    const Legion::Rect<N, T>& bounds) {
    T result = 0;
    T stride = 1;
    for (size_t i = N - 1; i > 0; --i) {
      result += stride * (pt[i] - bounds.lo[i]);
      stride *= bounds.hi[i] - bounds.lo[i] + 1;
    }
    return result + stride * (pt[0] - bounds.lo[0]);
  }

  template <int N, typename T>
  static T
  linearized_index_range(const Legion::Rect<N, T>& bounds) {
    T result = 1;
    for (size_t i = 0; i < N; ++i)
      result *= bounds.hi[i] - bounds.lo[i] + 1;
    return result;
  }

  template <int N, typename T>
  static KOKKOS_INLINE_FUNCTION array<T, N>
  multidimensional_index(T pt, const Legion::Rect<N, T>& bounds) {
    array<T, N> result;
    T stride = 1;
    for (size_t i = 1; i < N; ++i)
      stride *= bounds.hi[i] - bounds.lo[i] + 1;
    result[0] = pt / stride + bounds.lo[0];
    for (size_t i = 1; i < N; ++i) {
      stride /= bounds.hi[i] - bounds.lo[i] + 1;
      result[i] = pt / stride + bounds.lo[i];
      pt = pt % stride;
    }
    return result;
  }

  template <int N, typename T>
  static KOKKOS_INLINE_FUNCTION array<long, N>
  multidimensional_index_l(T pt, const Legion::Rect<N, T>& bounds) {
    array<long, N> sz = rect_size(bounds);
    array<long, N> result;
    T stride = 1;
    for (size_t i = 1; i < N; ++i)
      stride *= sz[i];
    result[0] = pt / stride;
    for (size_t i = 1; i < N; ++i) {
      stride /= sz[i];
      result[i] = pt / stride;
      pt = pt % stride;
    }
    return result;
  }

  void
  apply_fft(
    Legion::Context ctx,
    Legion::Runtime* rt,
    int sign,
    bool rotate_in,
    bool rotate_out,
    unsigned flags,
    double seconds,
    const ColumnSpacePartition& partition = ColumnSpacePartition()) const;

  void
  show_cf_values(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::string& title) const;

  struct ShowValuesTaskArgs {
    Table::Desc tdesc;
    hyperion::string title;
  };

  static Legion::TaskID show_values_task_id;

  static const constexpr char* show_values_task_name =
    "CFTableBase::show_values";

  static void
  show_values_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* rt);

  // TODO: this should probably live elsewhere
  static void
  show_index_value(const PhysicalColumn& col, Legion::coord_t i);
};

template <>
constexpr std::vector<typename cf_table_axis<CF_PS_SCALE>::type>&
CFTableBase::InitIndexColumnTaskArgs::values<CF_PS_SCALE>() {
  return ps_scales;
}
template <>
constexpr std::vector<typename cf_table_axis<CF_BASELINE_CLASS>::type>&
CFTableBase::InitIndexColumnTaskArgs::values<CF_BASELINE_CLASS>() {
  return baseline_classes;
}
template <>
constexpr std::vector<typename cf_table_axis<CF_FREQUENCY>::type>&
CFTableBase::InitIndexColumnTaskArgs::values<CF_FREQUENCY>() {
  return frequencies;
}
template <>
constexpr std::vector<typename cf_table_axis<CF_W>::type>&
CFTableBase::InitIndexColumnTaskArgs::values<CF_W>() {
  return w_values;
}
template <>
constexpr std::vector<typename cf_table_axis<CF_PARALLACTIC_ANGLE>::type>&
CFTableBase::InitIndexColumnTaskArgs::values<CF_PARALLACTIC_ANGLE>() {
  return parallactic_angles;
}
template <>
constexpr std::vector<typename cf_table_axis<CF_STOKES_OUT>::type>&
CFTableBase::InitIndexColumnTaskArgs::values<CF_STOKES_OUT>() {
  return stokes_out_values;
}
template <>
constexpr std::vector<typename cf_table_axis<CF_STOKES_IN>::type>&
CFTableBase::InitIndexColumnTaskArgs::values<CF_STOKES_IN>() {
  return stokes_in_values;
}
template <>
constexpr std::vector<typename cf_table_axis<CF_STOKES>::type>&
CFTableBase::InitIndexColumnTaskArgs::values<CF_STOKES>() {
  return stokes_values;
}
template <cf_table_axes_t T>
struct CFTableBase::InitIndexColumnTaskArgs::initializer<T> {

  static void
  init(
    InitIndexColumnTaskArgs& args,
    const CFTableBase::Axis<T>& ax) {

    args.values<T>() = ax.values;
  }
};
template <cf_table_axes_t T0, cf_table_axes_t T1, cf_table_axes_t...Axes>
struct CFTableBase::InitIndexColumnTaskArgs::initializer<T0, T1, Axes...> {
  static void
  init(
    InitIndexColumnTaskArgs& args,
    const CFTableBase::Axis<T0>& ax0,
    const CFTableBase::Axis<T1>& ax1,
    const CFTableBase::Axis<Axes>&...axs) {

    args.values<T0>() = ax0.values;
    initializer<T1, Axes...>::init(args, ax1, axs...);
  }
};

#if LEGION_MAX_DIM >= 1
template <typename V, typename XS, typename YS>
KOKKOS_INLINE_FUNCTION auto
cf_subview(const V& v, const array<coord_t, 1>& pt, XS&& xs, YS&& ys) {

  return Kokkos::subview(
    v,
    pt[0],
    std::forward<XS>(xs), std::forward<YS>(ys));
}
#endif

#if LEGION_MAX_DIM >= 2
template <typename V, typename XS, typename YS>
KOKKOS_INLINE_FUNCTION auto
cf_subview(const V& v, const array<coord_t, 2>& pt, XS&& xs, YS&& ys) {

  return Kokkos::subview(
    v,
    pt[0], pt[1],
    std::forward<XS>(xs), std::forward<YS>(ys));
}
#endif

#if LEGION_MAX_DIM >= 3
template <typename V, typename XS, typename YS>
KOKKOS_INLINE_FUNCTION auto
cf_subview(const V& v, const array<coord_t, 3>& pt, XS&& xs, YS&& ys) {

  return Kokkos::subview(
    v,
    pt[0], pt[1], pt[2],
    std::forward<XS>(xs), std::forward<YS>(ys));
}
#endif

#if LEGION_MAX_DIM >= 4
template <typename V, typename XS, typename YS>
KOKKOS_INLINE_FUNCTION auto
cf_subview(const V& v, const array<coord_t, 4>& pt, XS&& xs, YS&& ys) {

  return Kokkos::subview(
    v,
    pt[0], pt[1], pt[2], pt[3],
    std::forward<XS>(xs), std::forward<YS>(ys));
}
#endif

#if LEGION_MAX_DIM >= 5
template <typename V, typename XS, typename YS>
KOKKOS_INLINE_FUNCTION auto
cf_subview(const V& v, const array<coord_t, 5>& pt, XS&& xs, YS&& ys) {

  return Kokkos::subview(
    v,
    pt[0], pt[1], pt[2], pt[3], pt[4],
    std::forward<XS>(xs), std::forward<YS>(ys));
}
#endif

#if LEGION_MAX_DIM >= 6
template <typename V, typename XS, typename YS>
KOKKOS_INLINE_FUNCTION auto
cf_subview(const V& v, const array<coord_t, 6>& pt, XS&& xs, YS&& ys) {

  return Kokkos::subview(
    v,
    pt[0], pt[1], pt[2], pt[3], pt[4], pt[5],
    std::forward<XS>(xs), std::forward<YS>(ys));
}
#endif

} // end namespace synthesis
} // end namespace hyperion

#endif // HYPERION_SYNTHESIS_CF_TABLE_BASE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
