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
#ifndef HYPERION_X_COLUMN_SPACE_PARTITION_H_
#define HYPERION_X_COLUMN_SPACE_PARTITION_H_

#include <hyperion/hyperion.h>
#include <hyperion/utility.h>
#include <hyperion/x/ColumnSpace.h>

#pragma GCC visibility push(default)
# include <vector>
#pragma GCC visibility pop

namespace hyperion {
namespace x {

struct HYPERION_API ColumnSpacePartition {

  ColumnSpacePartition() {}

  ColumnSpacePartition(
    const ColumnSpace& column_space_,
    const Legion::IndexPartition column_ip_)
    : column_space(column_space_)
    , column_ip(column_ip_) {
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

  static Legion::Future /* ColumnSpacePartition */
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const ColumnSpace& column_space,
    const std::vector<hyperion::AxisPartition>& partition);

  typedef ColumnSpacePartition create_result_t;

  static create_result_t
  create(
    Legion::Context ctx,
    Legion::Runtime *rt,
    const Legion::IndexSpace& column_space_is,
    const std::vector<hyperion::AxisPartition>& partition,
    const Legion::PhysicalRegion& column_space_metadata_pr);

  Legion::Future /* ColumnSpacePartition */
  project_onto(
    Legion::Context ctx,
    Legion::Runtime *rt,
    const ColumnSpace& tgt_column_space) const;

  typedef ColumnSpacePartition project_onto_result_t;

  static project_onto_result_t
  project_onto(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const Legion::IndexPartition& csp_column_ip,
    const Legion::IndexSpace& tgt_cs_column_is,
    const Legion::PhysicalRegion& csp_cs_metadata_pr,
    const Legion::PhysicalRegion& tgt_cs_metadata_pr);

  static void
  preregister_tasks();

  ColumnSpace column_space;

  Legion::IndexPartition column_ip;

private:

  static Legion::TaskID create_task_id;

  static const char* create_task_name;

  static create_result_t
  create_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *rt);

  static Legion::TaskID project_onto_task_id;

  static const char* project_onto_task_name;

  static project_onto_result_t
  project_onto_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *rt);

};
} // end namespace x
} // end namespace hyperion


#endif // HYPERION_X_COLUMN_SPACE_PARTITION_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
