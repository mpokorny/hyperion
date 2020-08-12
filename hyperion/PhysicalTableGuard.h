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
#ifndef HYPERION_PHYSICAL_TABLE_GUARD_H_
#define HYPERION_PHYSICAL_TABLE_GUARD_H_

#include <hyperion/hyperion.h>

namespace hyperion {

template <typename PT>
class PhysicalTableGuard {
public:

  typedef PT table_t;

  PhysicalTableGuard(
    Legion::Context& ctx,
    Legion::Runtime*& rt,
    PT&& pt)
    : m_ctx(ctx)
    , m_rt(rt)
    , m_pt(std::move(pt)) {}

  virtual ~PhysicalTableGuard() {
    m_pt.unmap_regions(m_ctx, m_rt);
  }

  const PT&
  operator*() const {
    return m_pt;
  }

  const PT*
  operator->() const {
    return &m_pt;
  }

private:

  Legion::Context& m_ctx;

  Legion::Runtime*& m_rt;

  PT m_pt;
};


}  // namespace hyperion

#endif /* HYPERION_PHYSICAL_TABLE_GUARD_H_ */
// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
