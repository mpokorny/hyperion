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
#ifndef HYPERION_SYNTHESIS_PS_TERM_TABLE_H_
#define HYPERION_SYNTHESIS_PS_TERM_TABLE_H_

#include <hyperion/synthesis/CFTable.h>

#include <array>

namespace hyperion {
namespace synthesis {

class HYPERION_EXPORT PSTermTable
  : public CFTable<CF_PB_SCALE> {
public:

  using CFTable<CF_PB_SCALE>::CFTable;

  PSTermTable(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::array<Legion::coord_t, 2>& cf_bounds_lo,
    const std::array<Legion::coord_t, 2>& cf_bounds_hi,
    const std::vector<typename cf_table_axis<CF_PB_SCALE>::type>& pb_scales);

  PSTermTable(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const Legion::coord_t& cf_x_radius,
    const Legion::coord_t& cf_y_radius,
    const std::vector<typename cf_table_axis<CF_PB_SCALE>::type>& pb_scales);

  void
  compute_cfs(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const ColumnSpacePartition& partition = ColumnSpacePartition()) const;

  static const constexpr char* compute_cfs_task_name =
    "PSTermTable::compute_cfs_task";

  static Legion::TaskID compute_cfs_task_id;

  static void
  compute_cfs_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* rt);

  static void
  preregister_tasks();
};

} // end namespace synthesis
} // end namespace hyperion

#endif // HYPERION_SYNTHESIS_PS_TERM_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
