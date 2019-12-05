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
#ifndef HYPERION_MS_TABLE_COLUMNS_H_
#define HYPERION_MS_TABLE_COLUMNS_H_

#include <hyperion/MSTableColumns_c.h>
#include <hyperion/MSTable.h>

#pragma GCC visibility push(default)
# include <algorithm>
# include <array>
# include <iterator>
# include <optional>
#pragma GCC visibility pop

namespace hyperion {

template <MSTables T>
struct MSTableColumns {
  //typedef ColEnums;
  static const std::array<const char*, 0> column_names;
};

template <MSTables T>
const std::array<const char*, 0> MSTableColumns<T>::column_names;

#define MSTC(T, t)                                                      \
  template <>                                                           \
  struct HYPERION_API MSTableColumns<MS_##T> {                          \
    typedef ms_##t##_col_t col_t;                                       \
    static const constexpr std::array<const char*, MS_##T##_NUM_COLS>   \
      column_names = MS_##T##_COL_NAMES;                                \
    static const constexpr std::array<unsigned, MS_##T##_NUM_COLS>      \
      element_ranks = MS_##T##_COL_ELEMENT_RANKS;                       \
    static const std::unordered_map<col_t, const char*> units;          \
    static const std::map<col_t, const char*> measure_names;            \
    static std::optional<col_t>                                         \
    lookup_col(const std::string& nm) {                                 \
      auto col =                                                        \
        std::find_if(                                                   \
          column_names.begin(),                                         \
          column_names.end(),                                           \
          [&nm](std::string cn) {                                       \
            return cn == nm;                                            \
          });                                                           \
      if (col != column_names.end())                                    \
        return                                                          \
          static_cast<col_t>(std::distance(column_names.begin(), col)); \
      return std::nullopt;                                              \
    }                                                                   \
    static std::optional<col_t>                                         \
    lookup_measure_col(const std::string& nm) {                         \
      auto cm =                                                         \
        std::find_if(                                                   \
          measure_names.begin(),                                        \
          measure_names.end(),                                          \
          [&nm](auto& c_m){                                             \
            std::string m = std::get<1>(c_m);                           \
            return m == nm;                                             \
          });                                                           \
      if (cm != measure_names.end())                                    \
        return std::get<0>(*cm);                                        \
      return std::nullopt;                                              \
    }                                                                   \
  };

HYPERION_FOREACH_MS_TABLE_Tt(MSTC);

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
