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
#include <hyperion/PhysicalColumn.h>

using namespace hyperion;
using namespace Legion;

std::optional<std::any>
PhysicalColumn::kw(const std::string& key) const {
  std::optional<std::any> result;
  if (m_kws.count(key) > 0)
    result = m_kws.at(key);
  return result;
}

#ifdef HYPERION_USE_CASACORE
std::vector<std::shared_ptr<casacore::MRBase>>
PhysicalColumn::mrbs() const {
  std::vector<std::shared_ptr<casacore::MRBase>> result;
  if (m_mrb) {
    mrb_t mrb = m_mrb.value();
    std::visit(overloaded {
        [&result](simple_mrb_t& smrb) {
          result.push_back(smrb);
        },
        [&result](ref_mrb_t& rmrb) {
          result = std::get<0>(rmrb);
        }
      },
      mrb);
  }
  return result;
}
#endif

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
