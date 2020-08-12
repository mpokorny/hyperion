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
#ifndef HYPERION_SYNTHESIS_PRODUCT_CF_TABLE_H_
#define HYPERION_SYNTHESIS_PRODUCT_CF_TABLE_H_

#include <hyperion/synthesis/CFTable.h>
#include <hyperion/PhysicalTableGuard.h>

namespace hyperion {
namespace synthesis {

template <cf_table_axes_t...Axes>
class ProductCFTable
  : public CFTable<Axes...> {
public:

  template <typename...Ts>
  ProductCFTable(Legion::Context ctx, Legion::Runtime* rt, const Ts&...ts)
    : CFTable<Axes...>(
      product(
        ctx,
        rt,
        PhysicalTableGuard(
          ctx,
          rt,
          typename Ts::physical_table_t(
            ts.map_inline(
              ctx,
              rt,
              {},
              Column::default_requirements_mapped)))...)) {}

protected:

  template <typename PT0, typename...PTs>
  static CFTable<Axes...>
  product(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const PT0& pt0,
    const PTs&...pts) {

    auto grid_sizes =
      std::array<size_t, sizeof...(PTs) + 1>{
      pt0->grid_size(),
      pts->grid_size()...};
    size_t grid_size = 0;
    for (auto& s : grid_sizes)
      grid_size = std::max(grid_size, s);

    return
      CFTable<Axes...>(
        ctx,
        rt,
        grid_size,
        index_axis_h<Axes>(*pt0, hcons<typename PTs::table_t...>(*pts...))...);
  }
};

}  // synthesis

}  // hyperion

#endif /* HYPERION_SYNTHESIS_PRODUCT_CF_TABLE_H_ */
// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
