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
#include <hyperion/TableField.h>

using namespace hyperion;
using namespace Legion;

std::vector<RegionRequirement>
TableField::requirements(
  Runtime* rt,
  PrivilegeMode privilege,
  bool mapped) const {

  std::vector<RegionRequirement> result;
  if (!kw.is_empty()) {
    std::vector<FieldID> fids(kw.size(rt));
    std::iota(fids.begin(), fids.end(), 0);
    auto ppr = kw.requirements(rt, fids, privilege, mapped).value();
    result.push_back(ppr.type_tags);
    result.push_back(ppr.values);
  }
#ifdef HYPERION_USE_CASACORE
  if (!mr.is_empty()) {
#if HAVE_CXX17
    auto [mreq, vreq, oireq] = mr.requirements(privilege, mapped);
#else // !HAVE_CXX17
    auto rqs = mr.requirements(privilege, mapped);
    auto& mreq = std::get<0>(rqs);
    auto& vreq = std::get<1>(rqs);
    auto& oireq = std::get<2>(rqs);
#endif // HAVE_CXX17
    result.push_back(mreq);
    result.push_back(vreq);
    if (oireq)
      result.push_back(oireq.value());
  }
#endif
  return result;
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
