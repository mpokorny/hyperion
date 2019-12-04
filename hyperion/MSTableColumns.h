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

namespace hyperion {

template <MSTables T>
struct MSTableColumns {
  //typedef ColEnums;
  static const std::array<const char*, 0> column_names;
};

template <MSTables T>
const std::array<const char*, 0> MSTableColumns<T>::column_names;

#define MSTC(T, t)                                                    \
  template <>                                                         \
  struct MSTableColumns<MS_##T> {                                     \
    typedef ms_##t##_col_t col_t;                                     \
    static const constexpr std::array<const char*, MS_##T##_NUM_COLS> \
    column_names = MS_##T##_COL_NAMES;                                \
    static const constexpr std::array<unsigned, MS_##T##_NUM_COLS>    \
    element_ranks = MS_##T##_COL_ELEMENT_RANKS;                       \
  };

HYPERION_FOREACH_MS_TABLE_Tt(MSTC);

} // end namespace hyperion

#endif // HYPERION_MS_TABLE_COLUMNS_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
