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
#include <hyperion/MeasRefDict.h>
#include <algorithm>

using namespace hyperion;
using namespace Legion;

std::unordered_set<std::string>
MeasRefDict::names() const {

  std::unordered_set<std::string> result;
  std::transform(
    m_meas_refs.begin(),
    m_meas_refs.end(),
    std::inserter(result, result.end()),
    [](auto& nm_r) { return std::get<0>(nm_r); });
  return result;
}

std::optional<MeasRefDict::Ref>
MeasRefDict::get(const std::string& name) const {
  std::optional<Ref> result;
  if (m_meas_refs.count(name) > 0) {
    if (m_refs.count(name) == 0) {
      Ref ref =
        std::visit(overloaded {
            [this](const MeasRef* mr) -> Ref {
              switch (mr->mclass(m_ctx, m_rt)) {
#define MK(M)                                                           \
                case M:                                                 \
                  return mr->make<MClassT<M>::type>(m_ctx, m_rt).value(); \
                  break;
                HYPERION_FOREACH_MCLASS(MK)
#undef MK
                default:
                  assert(false);
                  break;
              }
            },
            [this](const MeasRef::DataRegions& prs) -> Ref {
              switch (MeasRef::mclass(prs.metadata)) {
#define MK(M)                                                           \
                case M:                                                 \
                  return                                                \
                    MeasRef::make<MClassT<M>::type>(m_rt, prs).value(); \
                  break;
                HYPERION_FOREACH_MCLASS(MK)
#undef MK
                default:
                  assert(false);
                  break;
              }
            }
          },
          m_meas_refs.at(name));
      m_refs.insert(std::make_pair(name, ref));
    }
    result = m_refs.at(name);
  }
  return result;
}

std::optional<const MeasRef*>
MeasRefDict::get_mr(const std::string& name) const {
  std::optional<const MeasRef*> result;
  if (!m_has_physical_regions
      && m_meas_refs.count(name) > 0)
    result = std::get<const MeasRef*>(m_meas_refs.at(name));
  return result;
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
