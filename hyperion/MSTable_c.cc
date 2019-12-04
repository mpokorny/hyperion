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
#include <hyperion/MSTable_c.h>
#include <hyperion/MSTable.h>
#include <hyperion/c_util.h>

#pragma GCC visibility push(default)
# include <algorithm>
# include <array>
# include <cassert>
# include <cstring>
# include <mutex>
#pragma GCC visibility pop

using namespace hyperion;

#define TABLE_FUNCTION_DEFNS(T, t)                                      \
  const char*                                                           \
  t##_table_name() {                                                    \
    return MSTable<MS_##T>::name;                                       \
  }                                                                     \
                                                                        \
  const ms_##t##_column_axes_t*                                         \
  t##_table_element_axes() {                                            \
    static std::once_flag initialized;                                  \
    static std::vector<ms_##t##_column_axes_t> result;                  \
    std::call_once(                                                     \
      initialized,                                                      \
      []() {                                                            \
        std::transform(                                                 \
          MSTable<MS_##T>::element_axes.begin(),                        \
          MSTable<MS_##T>::element_axes.end(),                          \
          std::back_inserter(result),                                   \
          [](auto& nm_axs) {                                            \
            auto& [nm, axs] = nm_axs;                                   \
            return                                                      \
              ms_##t##_column_axes_t {                                  \
              nm.c_str(),                                               \
                axs.data(),                                             \
                static_cast<unsigned>(axs.size())};                     \
          });                                                           \
      });                                                               \
    return result.data();                                               \
  }                                                                     \
                                                                        \
  unsigned                                                              \
  t##_table_num_columns() {                                             \
    return static_cast<unsigned>(MSTable<MS_##T>::element_axes.size()); \
  }                                                                     \
                                                                        \
  const char* const*                                                    \
  t##_table_axis_names() {                                              \
    typedef typename MSTable<MS_##T>::Axes Axes;                        \
    static std::once_flag initialized;                                  \
    static std::vector<const char*> result;                             \
    std::call_once(                                                     \
      initialized,                                                      \
      []() {                                                            \
        auto axis_names = MSTable<MS_##T>::axis_names();                \
        result.resize(axis_names.size());                               \
        for (int i = 0; i <= static_cast<int>(MSTable<MS_##T>::LAST_AXIS); ++i) { \
          result[i] = axis_names[static_cast<Axes>(i)].c_str();         \
        }                                                               \
      });                                                               \
    return result.data();                                               \
  }                                                                     \
                                                                        \
  unsigned                                                              \
  t##_table_num_axes() {                                                \
    return static_cast<unsigned>(MSTable<MS_##T>::LAST_AXIS) + 1;       \
  }

HYPERION_FOREACH_MS_TABLE_Tt(TABLE_FUNCTION_DEFNS);

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
