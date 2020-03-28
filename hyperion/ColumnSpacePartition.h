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
#ifndef HYPERION_COLUMN_SPACE_PARTITION_H_
#define HYPERION_COLUMN_SPACE_PARTITION_H_

#include <hyperion/hyperion.h>
#include <hyperion/utility.h>
#include <hyperion/ColumnSpace.h>

#pragma GCC visibility push(default)
# include <array>
# include <string>
# include <vector>
#pragma GCC visibility pop

namespace hyperion {

struct HYPERION_API ColumnSpacePartition {

  ColumnSpacePartition() {}

  ColumnSpacePartition(
    const ColumnSpace& column_space_,
    const Legion::IndexPartition column_ip_,
    const std::array<AxisPartition, ColumnSpace::MAX_DIM>& partition_)
  : column_space(column_space_)
  , column_ip(column_ip_)
  , partition(partition_) {
  }

  bool
  is_valid() const {
    return column_ip != Legion::IndexPartition::NO_PART
    && column_space.is_valid();
  }

  void
  destroy(
    Legion::Context ctx,
    Legion::Runtime* rt,
    bool destroy_column_space=false,
    bool destroy_column_space_index_space=false);

  int
  color_dim(Legion::Runtime* rt) const;

  static Legion::Future /* ColumnSpacePartition */
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const ColumnSpace& column_space,
    const std::vector<AxisPartition>& partition);

  static Legion::Future /* ColumnSpacePartition */
  create(
    Legion::Context ctx,
    Legion::Runtime *rt,
    const ColumnSpace& column_space,
    const std::string& block_axes_uid,
    const std::vector<std::pair<int, Legion::coord_t>>& block_sizes);

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  static Legion::Future /* ColumnSpacePartition */
  create(
    Legion::Context ctx,
    Legion::Runtime *rt,
    const ColumnSpace& column_space,
    const std::vector<std::pair<D, Legion::coord_t>>& block_sizes) {

    std::vector<std::pair<int, Legion::coord_t>> bsz;
    bsz.reserve(block_sizes.size());
    for (auto& [d, sz] : block_sizes)
      bsz.emplace_back(static_cast<int>(d), sz);
    return create(ctx, rt, column_space, Axes<D>::uid, bsz);
  }

  static ColumnSpacePartition
  create(
    Legion::Context ctx,
    Legion::Runtime *rt,
    const Legion::IndexSpace& column_space_is,
    const std::vector<AxisPartition>& partition,
    const Legion::PhysicalRegion& column_space_metadata_pr);

  static ColumnSpacePartition
  create(
    Legion::Context ctx,
    Legion::Runtime *rt,
    const Legion::IndexSpace& column_space_is,
    const std::string& block_axes_uid,
    const std::vector<std::pair<int, Legion::coord_t>>& block_sizes,
    const Legion::PhysicalRegion& column_space_metadata_pr);

  Legion::Future /* ColumnSpacePartition */
  project_onto(
    Legion::Context ctx,
    Legion::Runtime *rt,
    const ColumnSpace& tgt_column_space) const;

  ColumnSpacePartition
  project_onto(
    Legion::Context ctx,
    Legion::Runtime *rt,
    const Legion::IndexSpace& tgt_column_is,
    const Legion::PhysicalRegion& tgt_column_md) const;

  static void
  preregister_tasks();

  ColumnSpace column_space;

  Legion::IndexPartition column_ip;

  std::array<AxisPartition, ColumnSpace::MAX_DIM> partition;

// protected:

//   friend class Legion::LegionTaskWrapper;

  static ColumnSpacePartition
  create_task_ap(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *rt);

  static ColumnSpacePartition
  create_task_bs(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *rt);

  static Legion::TaskID create_task_ap_id;

  static const char* create_task_ap_name;

  static Legion::TaskID create_task_bs_id;

  static const char* create_task_bs_name;

};
} // end namespace hyperion

#endif // HYPERION_COLUMN_SPACE_PARTITION_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
