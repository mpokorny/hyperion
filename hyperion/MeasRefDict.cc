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

std::unordered_set<std::string>
MeasRefDict::tags() const {

  std::unordered_set<std::string> result;
  std::for_each(
    m_meas_refs.begin(),
    m_meas_refs.end(),
    [&result](auto& nm_r) {
      auto& nm = std::get<0>(nm_r);
      result.insert(nm.substr(MeasRef::find_tag(nm)));
    });
  return result;
}

std::optional<MeasRefDict::Ref>
MeasRefDict::get(const std::string& name) const {
  std::optional<Ref> result;
  if (m_meas_refs.count(name) > 0) {
    if (m_refs.count(name) == 0) {
      const auto& mr = m_meas_refs.at(name);
      switch (mr->mclass(m_ctx, m_rt)) {
#define MK(M)                                           \
        case M:                                         \
          m_refs.insert(                                \
            std::make_pair(                             \
              name,                                     \
              mr->make<MClassT<M>::type>(m_ctx, m_rt)   \
              .value()));                               \
          break;
        HYPERION_FOREACH_MCLASS(MK)
#undef MK
      default:
          assert(false);
        break;
      }
    }
    result = m_refs.at(name);
  }
  return result;
}

void
MeasRefDict::add_tags() {
  std::unordered_map<std::string, std::map<unsigned, const MeasRef*>> tag_refs;
  for (auto& [nm, mr] : m_meas_refs) {
    auto tg = nm.substr(MeasRef::find_tag(nm));
    if (tag_refs.count(tg) == 0)
      tag_refs[tg] = std::map<unsigned, const MeasRef*>();
    tag_refs[tg][std::count(nm.begin(), nm.end(), '/')] = mr;
  }
  for (auto& [tg, refs] : tag_refs)
    m_meas_refs[tg] = refs.rbegin()->second;
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
