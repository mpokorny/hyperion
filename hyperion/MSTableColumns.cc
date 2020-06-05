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
#include <hyperion/MSTableColumns.h>

using namespace hyperion;

#define DEFS(T)                                                         \
  const std::unordered_map<MSTableDefs<MS_##T>::col_t, const char*>     \
  MSTableDefs<MS_##T>::units = MS_##T##_COLUMN_UNITS;                   \
  const std::map<MSTableDefs<MS_##T>::col_t, const char*>               \
  MSTableDefs<MS_##T>::measure_names = MS_##T##_COLUMN_MEASURE_NAMES;

HYPERION_FOREACH_MS_TABLE(DEFS);

#if !HAVE_CXX17

#define COLCONST(T)                                                     \
  const constexpr std::array<const char*, MSTableDefs<MS_##T>::num_cols> \
  MSTableColumns<MS_##T>::column_names;                                 \
  const std::unordered_map<typename MSTableColumns<MS_##T>::col_t, const char*> \
  MSTableColumns<MS_##T>::units =                                       \
                  MSTableDefs<MS_##T>::units;                           \
  const std::map<typename MSTableColumns<MS_##T>::col_t, const char*>   \
  MSTableColumns<MS_##T>::measure_names =                               \
                  MSTableDefs<MS_##T>::measure_names;

HYPERION_FOREACH_MS_TABLE(COLCONST);
#endif // !HAVE_CXX17

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
