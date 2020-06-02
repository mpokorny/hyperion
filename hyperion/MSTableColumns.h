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
#ifndef HYPERION_MS_TABLE_COLUMNS_H_
#define HYPERION_MS_TABLE_COLUMNS_H_

#include <hyperion/MSTableColumns_c.h>
#include <hyperion/MSTable.h>
#ifdef HYPERION_USE_CASACORE
# include <hyperion/MeasRef.h>
#endif
#include <hyperion/Table.h>

#ifdef HYPERION_USE_CASACORE
# include <casacore/measures/Measures/MeasRef.h>
#endif

#include <algorithm>
#include <array>
#include <cstring>
#include <iterator>
#include CXX_OPTIONAL_HEADER
#include <unordered_map>
#include <vector>

namespace hyperion {

template <MSTables T>
struct MSTableDefs {
  typedef void col_t;
  static const unsigned num_cols;
  static const char* column_names[];
  static const unsigned element_ranks[];
  static const unsigned fid_base;
  static const std::unordered_map<col_t, const char*> units;
  static const std::map<col_t, const char*> measure_names;
};

#define MSTDEF(T, t)                                                    \
  template <>                                                           \
  struct HYPERION_API MSTableDefs<MS_##T> {                             \
    typedef ms_##t##_col_t col_t;                                       \
    static const constexpr unsigned num_cols =                          \
      MS_##T##_NUM_COLS;                                                \
    static const constexpr std::array<const char*, num_cols> column_names = \
      MS_##T##_COLUMN_NAMES;                                            \
    static const constexpr std::array<unsigned, num_cols> element_ranks = \
      MS_##T##_COLUMN_ELEMENT_RANKS;                                    \
    static const constexpr unsigned fid_base =                          \
      MS_##T##_COL_FID_BASE;                                            \
    static const constexpr unsigned user_fid_base =                     \
      MS_##T##_COL_USER_FID_BASE;                                       \
    static const std::unordered_map<col_t, const char*> units;          \
    static const std::map<col_t, const char*> measure_names;            \
  };

HYPERION_FOREACH_MS_TABLE_Tt(MSTDEF);

template <MSTables T>
struct MSTableColumns {
  typedef typename MSTableDefs<T>::col_t col_t;

  static const constexpr std::array<const char*, MSTableDefs<T>::num_cols>
    column_names =
    MSTableDefs<T>::column_names;

  static const constexpr std::array<unsigned, MSTableDefs<T>::num_cols>
    element_ranks =
    MSTableDefs<T>::element_ranks;

  static constexpr Legion::FieldID fid(col_t c) {
    return c + MSTableDefs<T>::fid_base;
  }

  static const constexpr Legion::FieldID user_fid_base =
    MSTableDefs<T>::user_fid_base;

  static const std::unordered_map<col_t, const char*> units;

  static const std::map<col_t, const char*> measure_names;

  static CXX_OPTIONAL_NAMESPACE::optional<col_t>
  lookup_col(const std::string& nm) {
    auto col =
      std::find_if(
        column_names.begin(),
        column_names.end(),
        [&nm](std::string cn) {
          return cn == nm;
        });
    if (col != column_names.end())
      return static_cast<col_t>(std::distance(column_names.begin(), col));
    return CXX_OPTIONAL_NAMESPACE::nullopt;
  }
};

template <MSTables T>
const std::unordered_map<typename MSTableColumns<T>::col_t, const char*>
MSTableColumns<T>::units =
  MSTableDefs<T>::units;

template <MSTables T>
const std::map<typename MSTableColumns<T>::col_t, const char*>
MSTableColumns<T>::measure_names =
  MSTableDefs<T>::measure_names;

#if __cplusplus < 201703L
#define MS_TABLE_COLUMNS(T)                                             \
  template <>                                                           \
  struct MSTableColumns<MS_##T> {                                       \
    typedef typename MSTableDefs<MS_##T>::col_t col_t;                  \
                                                                        \
    static const constexpr std::array<const char*, MSTableDefs<MS_##T>::num_cols> \
    column_names =                                                      \
      MSTableDefs<MS_##T>::column_names;                                \
                                                                        \
    static const constexpr std::array<unsigned, MSTableDefs<MS_##T>::num_cols> \
    element_ranks =                                                     \
      MSTableDefs<MS_##T>::element_ranks;                               \
                                                                        \
    static constexpr Legion::FieldID fid(col_t c) {                     \
      return c + MSTableDefs<MS_##T>::fid_base;                         \
    }                                                                   \
                                                                        \
    static const constexpr Legion::FieldID user_fid_base =              \
      MSTableDefs<MS_##T>::user_fid_base;                               \
                                                                        \
    static const std::unordered_map<col_t, const char*> units;          \
                                                                        \
    static const std::map<col_t, const char*> measure_names;            \
                                                                        \
    static CXX_OPTIONAL_NAMESPACE::optional<col_t>                      \
    lookup_col(const std::string& nm) {                                 \
      auto col =                                                        \
        std::find_if(                                                   \
          column_names.begin(),                                         \
          column_names.end(),                                           \
          [&nm](std::string cn) {                                       \
            return cn == nm;                                            \
          });                                                           \
      if (col != column_names.end())                                    \
        return static_cast<col_t>(std::distance(column_names.begin(), col)); \
      return CXX_OPTIONAL_NAMESPACE::nullopt;                           \
    }                                                                   \
  };
HYPERION_FOREACH_MS_TABLE(MS_TABLE_COLUMNS)
#endif // !c++17

#define HYPERION_COLUMN_NAME(T, C)                    \
  MSTableColumns<MS_##T>::column_names[               \
    MSTableColumns<MS_##T>::col_t::MS_##T##_COL_##C]

} // end namespace hyperion

#endif // HYPERION_MS_TABLE_COLUMNS_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
