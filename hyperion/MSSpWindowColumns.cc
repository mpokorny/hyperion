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
#include <hyperion/MSSpWindowColumns.h>
#include <hyperion/MeasRef.h>

using namespace hyperion;
using namespace Legion;

namespace cc = casacore;

MSSpWindowColumns::MSSpWindowColumns(
  Legion::Runtime* rt,
  const Legion::RegionRequirement& rows_requirement,
  const std::unordered_map<std::string, Regions>& regions)
  : m_rows(
    rt->get_index_space_domain(rows_requirement.region.get_index_space())) {

  for (auto& [nm, rgs] : regions) {
    auto ocol = C::lookup_col(nm);
    if (ocol) {
      auto& col = ocol.value();
      m_regions[col] = rgs.values;
#ifdef HYPERION_USE_CASACORE
      switch (col) {
      case C::col_t::MS_SPECTRAL_WINDOW_COL_REF_FREQUENCY:
      case C::col_t::MS_SPECTRAL_WINDOW_COL_CHAN_FREQ:
        m_mrs[col] = create_mr<cc::MFrequency>(rt, rgs);
        break;
      default:
        break;
      }
#endif
    }
  }
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
