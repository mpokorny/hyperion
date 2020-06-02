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
#ifndef HYPERION_COLUMN_SPACE_PARTITION_H_
#define HYPERION_COLUMN_SPACE_PARTITION_H_

#include <hyperion/hyperion.h>
#include <hyperion/utility.h>
#include <hyperion/ColumnSpace.h>

#include <array>
#include <string>
#include <vector>

namespace hyperion {

/**
 * Partition of a ColumnSpace
 *
 * Basically a Legion::IndexPartition of a ColumnSpace.column_is member, with
 * the addition of some fields that describe the partitioning (through an array
 * of AxisPartition values).
 */
struct HYPERION_API ColumnSpacePartition {

  /**
   * Construct an empty ColumnSpacePartition
   */
  ColumnSpacePartition() {}

  /**
   * Construct a ColumnSpacePartition
   */
  ColumnSpacePartition(
    const ColumnSpace& column_space_,
    const Legion::IndexPartition column_ip_,
    const std::array<AxisPartition, ColumnSpace::MAX_DIM>& partition_)
  : column_space(column_space_)
  , column_ip(column_ip_)
  , partition(partition_) {
  }

  /**
   * Test for a valid partition
   */
  bool
  is_valid() const {
    return column_ip != Legion::IndexPartition::NO_PART
    && column_space.is_valid();
  }

  /**
   * Destroy the resources
   *
   * @param[in] destroy_column_space
   *            whether to destroy the referenced ColumnSpace
   * @param[in] destroy_column_space_index_space
   *            whether to destroy the referenced ColumnSpace IndexSpace
   */
  void
  destroy(
    Legion::Context ctx,
    Legion::Runtime* rt,
    bool destroy_column_space=false,
    bool destroy_column_space_index_space=false);

  /**
   * The dimension (rank) of the partition color space
   */
  int
  color_dim(Legion::Runtime* rt) const;

  /**
   * Create a ColumnSpacePartition from a vector of AxisPartition
   *
   * @param[in] partition axis partitions (order is not required to follow the
   *            ColumnSpace axis order, nor does the value have to name every
   *            axis in the ColumnSpace)
   */
  static Legion::Future /* ColumnSpacePartition */
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const ColumnSpace& column_space,
    const std::vector<AxisPartition>& partition);

  /**
   * Create a ColumnSpacePartition from a vector of block sizes
   *
   * @param[in] block_axes_uid UID string for axes enumerated in block_sizes
   * @param[in] block_sizes block sizes of partition axes (axes identified by
   *            integer in enumeration block_axes_uid, unlisted axes are not
   *            partitioned, order of axes is unconstrained)
   */
  static Legion::Future /* ColumnSpacePartition */
  create(
    Legion::Context ctx,
    Legion::Runtime *rt,
    const ColumnSpace& column_space,
    const std::string& block_axes_uid,
    const std::vector<std::pair<int, Legion::coord_t>>& block_sizes);

  /**
   * Create a ColumnSpacePartition from a vector of block sizes
   *
   * @param[in] block_sizes block sizes of partition axes (axes identified by
   *            enumeration values)
   */
  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  static Legion::Future /* ColumnSpacePartition */
  create(
    Legion::Context ctx,
    Legion::Runtime *rt,
    const ColumnSpace& column_space,
    const std::vector<std::pair<D, Legion::coord_t>>& block_sizes) {

    std::vector<std::pair<int, Legion::coord_t>> bsz;
    bsz.reserve(block_sizes.size());
#if __cplusplus >= 201703L
    for (auto& [d, sz] : block_sizes)
      bsz.emplace_back(static_cast<int>(d), sz);
#else // !c++17
    for (auto& d_sz : block_sizes) {
      auto& d = std::get<0>(d_sz);
      auto& sz = std::get<1>(d_sz);
      bsz.emplace_back(static_cast<int>(d), sz);
    }
#endif // c++17
    return create(ctx, rt, column_space, Axes<D>::uid, bsz);
  }

  /**
   * Create a ColumnSpacePartition from an IndexSpace and PhysicalRegion
   *
   * @param[in] column_space_is IndexSpace of a ColumnSpace
   * @param[in] partition axis partitions
   * @param[in] column_space_metadata_pr PhysicalRegion of ColumnSpace metadata
   */
  static ColumnSpacePartition
  create(
    Legion::Context ctx,
    Legion::Runtime *rt,
    const Legion::IndexSpace& column_space_is,
    const std::vector<AxisPartition>& partition,
    const Legion::PhysicalRegion& column_space_metadata_pr);


  /**
   * Create a ColumnSpacePartition from an IndexSpace and PhysicalRegion
   *
   * @param[in] column_space_is IndexSpace of a ColumnSpace
   * @param[in] block_axes_uid UID string for axes enumerated in block_sizes
   * @param[in] block_sizes block sizes of partition axes (axes identified by
   *            integer in enumeration block_axes_uid, unlisted axes are not
   *            partitioned, order of axes is unconstrained)
   * @param[in] column_space_metadata_pr PhysicalRegion of ColumnSpace metadata
   */
  static ColumnSpacePartition
  create(
    Legion::Context ctx,
    Legion::Runtime *rt,
    const Legion::IndexSpace& column_space_is,
    const std::string& block_axes_uid,
    const std::vector<std::pair<int, Legion::coord_t>>& block_sizes,
    const Legion::PhysicalRegion& column_space_metadata_pr);

  /**
   * Project partition onto another ColumnSpace
   */
  Legion::Future /* ColumnSpacePartition */
  project_onto(
    Legion::Context ctx,
    Legion::Runtime *rt,
    const ColumnSpace& tgt_column_space) const;

  /**
   * Project partition onto another ColumnSpace (regions)
   */
  ColumnSpacePartition
  project_onto(
    Legion::Context ctx,
    Legion::Runtime *rt,
    const Legion::IndexSpace& tgt_column_is,
    const Legion::PhysicalRegion& tgt_column_md) const;

  /**
   * Preregister tasks used by ColumnSpacePartition
   *
   * Must be called before Legion runtime starts
   */
  static void
  preregister_tasks();

  ColumnSpace column_space; /**< column space */

  Legion::IndexPartition column_ip; /**< partition of column_space column_is */

  /** partition description
   */
  std::array<AxisPartition, ColumnSpace::MAX_DIM> partition;

// protected:

//   friend class Legion::LegionTaskWrapper;

  /**
   * Task to create ColumnSpacePartition from an AxisPartition vector
   */
  static ColumnSpacePartition
  create_task_ap(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *rt);

  /**
   * Task to create ColumnSpacePartition from block size vector
   */
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
