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
#include <hyperion/MSAntennaColumns.h>

using namespace hyperion;
using namespace Legion;

namespace cc = casacore;

MSAntennaColumns::MSAntennaColumns(
  Runtime* rt,
  const RegionRequirement& rows_requirement,
  const std::unordered_map<std::string, Regions>& regions)
  : m_rows_requirement(rows_requirement) {

  for (auto& [nm, rgs] : regions) {
    auto col = C::lookup_col(nm);
    if (col) {
      m_regions[col.value()] = rgs.values;
      switch (col.value()) {
      case C::col_t::MS_ANTENNA_COL_POSITION:
      case C::col_t::MS_ANTENNA_COL_OFFSET:
#ifdef HYPERION_USE_CASACORE
        m_mrs[col.value()] = create_mr<cc::MPosition>(rt, rgs);
#endif
        break;
      default:
        break;
      }
    }
  }
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
