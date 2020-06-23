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

#include <array>
#include <vector>

namespace hyperion {

namespace synthesis {
typedef enum cf_table_axes_t {
  CF_PB_SCALE,
  CF_BASELINE_CLASS,
  CF_FREQUENCY,
  CF_W,
  CF_PARALLACTIC_ANGLE,
  CF_STOKES,
  CF_X,
  CF_Y,
  CF_LAST_AXIS = CF_Y
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
struct cf_table_axis<CF_PB_SCALE> {
  typedef float type;
  static const constexpr char* name = "PB_SCALE";
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
struct cf_table_axis<CF_STOKES> {
  typedef int type;
  static const constexpr char* name = "STOKES";
};
template <>
struct cf_table_axis<CF_X> {
  typedef int type;
  static const constexpr char* name = "X";
};
template <>
struct cf_table_axis<CF_Y> {
  typedef int type;
  static const constexpr char* name = "Y";
};

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

  static const constexpr Legion::FieldID INDEX_VALUE_FID = 14;

  static const constexpr Legion::FieldID CF_VALUE_FID = 24;
  static const constexpr char* CF_VALUE_COLUMN_NAME = "VALUE";
  typedef hyperion::complex<double> cf_value_t;

  static const constexpr Legion::FieldID CF_WEIGHT_FID = 34;
  static const constexpr char* CF_WEIGHT_COLUMN_NAME = "WEIGHT";
  typedef hyperion::complex<double> cf_weight_t;

  static Legion::TaskID init_index_column_task_id;
  static const constexpr char* init_index_column_task_name =
    "CFTable::init_index_column_task";

  struct HYPERION_EXPORT InitIndexColumnTaskArgs {
    hyperion::Table::Desc desc;

    std::vector<typename cf_table_axis<CF_PB_SCALE>::type>
      pb_scales;
    std::vector<typename cf_table_axis<CF_BASELINE_CLASS>::type>
      baseline_classes;
    std::vector<typename cf_table_axis<CF_FREQUENCY>::type>
      frequencies;
    std::vector<typename cf_table_axis<CF_W>::type>
      w_values;
    std::vector<typename cf_table_axis<CF_PARALLACTIC_ANGLE>::type>
      parallactic_angles;
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

  struct ComputeCFSTaskArgs {
    hyperion::Table::Desc desc;
    size_t serialized_size() const;
    void serialize(void* buffer) const;
    void deserialize(const void* buffer);
  };

  static void preregister_all();

#ifdef HYPERION_USE_KOKKOS
  template <int N>
  static inline Kokkos::Array<long, N>
  rect_lo(const Legion::Rect<N, Legion::coord_t>& r) {
    Kokkos::Array<long, N> result;
    for (size_t i = 0; i < N; ++i)
      result[i] = r.lo[i];
    return result;
  }

  template <int N>
  static inline Kokkos::Array<long, N>
  rect_hi(const Legion::Rect<N, Legion::coord_t>& r) {
    Kokkos::Array<long, N> result;
    for (size_t i = 0; i < N; ++i)
      result[i] = r.hi[i];
    return result;
  }
#endif // !HYPERION_USE_KOKKOS
};

template <>
constexpr std::vector<typename cf_table_axis<CF_PB_SCALE>::type>&
CFTableBase::InitIndexColumnTaskArgs::values<CF_PB_SCALE>() {
  return pb_scales;
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

} // end namespace synthesis
} // end namespace hyperion

#endif // HYPERION_SYNTHESIS_CF_TABLE_BASE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
