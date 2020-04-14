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
#ifndef HYPERION_COLUMN_SPACE_H_
#define HYPERION_COLUMN_SPACE_H_

#include <hyperion/hyperion.h>
#include <hyperion/utility.h>

#pragma GCC visibility push(default)
# include <algorithm>
# include <array>
# include <string>
# include <vector>
#pragma GCC visibility pop

namespace hyperion {

/**
 * Column index space
 *
 * A ColumnSpace encapsulates a Legion::IndexSpace along with a small
 * Legion::LogicalRegion to hold various bits of metadata about the axes of the
 * index space. Additionally, a flag is maintained to indicate whether the
 * ColumnSpace belongs to an index column of its parent Table.
 */
struct HYPERION_API ColumnSpace {

  /**
   * Maximum supported dimension
   */
  static const constexpr size_t MAX_DIM = LEGION_MAX_DIM;

  /**
   * FieldID of the axis vector
   */
  static const constexpr Legion::FieldID AXIS_VECTOR_FID = 0;
  typedef std::array<int, MAX_DIM> AXIS_VECTOR_TYPE;
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using AxisVectorAccessor =
    Legion::FieldAccessor<
      MODE,
      AXIS_VECTOR_TYPE,
      1,
      Legion::coord_t,
      Legion::AffineAccessor<AXIS_VECTOR_TYPE, 1, Legion::coord_t>,
      CHECK_BOUNDS>;

  /**
   * FieldID of the axes UID
   */
  static const constexpr Legion::FieldID AXIS_SET_UID_FID = 1;
  typedef string AXIS_SET_UID_TYPE;
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using AxisSetUIDAccessor =
    Legion::FieldAccessor<
      MODE,
      AXIS_SET_UID_TYPE,
      1,
      Legion::coord_t,
      Legion::AffineAccessor<AXIS_SET_UID_TYPE, 1, Legion::coord_t>,
      CHECK_BOUNDS>;

  /**
   * FieldID of the table index column flag
   */
  static const constexpr Legion::FieldID INDEX_FLAG_FID = 2;
  typedef bool INDEX_FLAG_TYPE;
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using IndexFlagAccessor =
    Legion::FieldAccessor<
      MODE,
      INDEX_FLAG_TYPE,
      1,
      Legion::coord_t,
      Legion::AffineAccessor<INDEX_FLAG_TYPE, 1, Legion::coord_t>,
      CHECK_BOUNDS>;

  /**
   * Construct an empty ColumnSpace
   */
  ColumnSpace() {}

  /**
   * Construct a ColumnSpac
   */
  ColumnSpace(
    const Legion::IndexSpace& column_is,
    const Legion::LogicalRegion& metadata_lr);

  /**
   * Clone a ColumnSpace
   *
   * Create a ColumnSpace with duplicated index space and metadata
   */
  ColumnSpace
  clone(Legion::Context ctx, Legion::Runtime* rt) const;

  /**
   * Clone a ColumnSpace
   *
   * Create a ColumnSpace with duplicated index space and metadata using a
   * metadata Legion::PhysicalRegion
   */
  static ColumnSpace
  clone(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const Legion::IndexSpace& column_is,
    const Legion::PhysicalRegion& metadata_pr);

  /**
   * Is the ColumnSpace valid?
   *
   * @return true, if and only the metadata region exists
   */
  bool
  is_valid() const;

  /**
   * Is the ColumnSpace empty?
   *
   * @return true, if and only if the index space is not empty
   */
  bool
  is_empty() const;

  /**
   * Get the axis vector
   *
   * @return axis vector (integer-valued)
   */
  std::vector<int>
  axes(Legion::Context ctx, Legion::Runtime* rt) const;

  /**
   * Get the axis vector from a metadata Legion::PhysicalRegion
   *
   * @return axis vector (integer-valued)
   */
  static AXIS_VECTOR_TYPE
  axes(Legion::PhysicalRegion pr);

  /**
   * Get the axes UID
   *
   * @return axes UID string
   */
  std::string
  axes_uid(Legion::Context ctx, Legion::Runtime* rt) const;

  /**
   * Get the axes UID from a metadata Legion::PhysicalRegion
   *
   * @return axes UID string
   */
  static AXIS_SET_UID_TYPE
  axes_uid(Legion::PhysicalRegion pr);

  /**
   * Get the (table) index column flag
   *
   * @return flag value
   */
  bool
  is_index(Legion::Context ctx, Legion::Runtime* rt) const;

  /**
   * Get the (table) index column flag from a metadata Legion::PhysicalRegion
   *
   * @return flag value
   */
  static INDEX_FLAG_TYPE
  is_index(Legion::PhysicalRegion pr);

  /**
   * Release resources used by a ColumnSpace
   */
  void
  destroy(
    Legion::Context ctx,
    Legion::Runtime* rt,
    bool destroy_index_space=false /**< destroy the IndexSpace */);

  bool
  operator<(const ColumnSpace& rhs) const;

  bool
  operator==(const ColumnSpace& rhs) const;

  bool
  operator!=(const ColumnSpace& rhs) const;

  /**
   * Create a ColumnsSpace
   *
   * Metadata will be initialized.
   */
  static ColumnSpace
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<int>& axes, /**< integer-valued axis vector */
    const std::string& axis_set_uid /**< axes UID */,
    const Legion::IndexSpace& column_is /**< index space */,
    bool is_index /**< table index column flag */);

  /**
   * Create a ColumnSpace from typed axes
   */
  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  static ColumnSpace
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<D>& axes /**< axis-type-valued axis vector */,
    const Legion::IndexSpace& column_is /**< index space */,
    bool is_index /**< table index column flag */) {

    return create(ctx, rt, map_to_int(axes), Axes<D>::uid, column_is, is_index);
  }

  /**
   * Helper function to convert from std::vector to AXIS_VECTOR_TYPE
   *
   * Accounts for encoding of empty values in AXIS_VECTOR_TYPE value.
   */
  static AXIS_VECTOR_TYPE
  to_axis_vector(const std::vector<int>& v) {
    assert(v.size() <= MAX_DIM);
    AXIS_VECTOR_TYPE result;
    auto e = std::copy(v.begin(), v.end(), result.begin());
    std::fill(e, result.end(), -1);
    return result;
  }

  /**
   * Helper function to get size of axis vector
   */
  static unsigned
  size(const AXIS_VECTOR_TYPE& v) {
    return
      std::distance(
        v.begin(),
        std::find_if(v.begin(), v.end(), [](int i) { return i < 0; }));
  }

  /**
   * Helper function to convert from AXIS_VECTOR_TYPE to std::vector
   *
   * Accounts for encoding of empty values in AXIS_VECTOR_TYPE value.
   */
  static std::vector<int>
  from_axis_vector(const AXIS_VECTOR_TYPE& av) {
    return std::vector<int>(av.begin(), av.begin() + size(av));
  }

  /**
   * Get the RegionRequirement for mapping the metadata
   */
  Legion::RegionRequirement
  requirements(
    Legion::PrivilegeMode privilege,
    Legion::CoherenceProperty coherence) const;

  // reindexed_result_t fields:
  // - new ColumnSpace
  // - mapping from "row" in old index space to regions in new index space
  typedef std::tuple<ColumnSpace, Legion::LogicalRegion> reindexed_result_t;

  /**
   * FieldID for "row rectangles" used during reindexing
   */
  static constexpr const Legion::FieldID REINDEXED_ROW_RECTS_FID = 22;

  /**
   * Reindex the ColumnSpace
   *
   * This method mainly exists to support Table reindexing.
   *
   * \sa Table::reindexed()
   *
   * @return Future value of type reindexed_result_t
   */
  Legion::Future
  reindexed(
    Legion::Context ctx,
    Legion::Runtime* rt,
    unsigned element_rank, /**< column element rank */
    /** ordered vector of pairs of axis ids and index column value regions */
    const std::vector<std::pair<int, Legion::LogicalRegion>>& index_columns,
    bool allow_rows /**< allow result to maintain a "row" axis */) const;

  /**
   * Reindex the ColumnSpace from a metadata Legion::PhysicalRegion
   *
   * @sa reindexed(Legion::Context, Legion::Runtime*, unsigned, const
   * std::vector<std::pair<int, Legion::LogicalRegion>&, bool)
   */
  static reindexed_result_t
  reindexed(
    Legion::Context ctx,
    Legion::Runtime* rt,
    unsigned element_rank,
    const std::vector<std::pair<int, Legion::LogicalRegion>>& index_column_lrs,
    bool allow_rows,
    const Legion::IndexSpace& column_is,
    const Legion::PhysicalRegion& metadata_pr);

  /**
   * Compute mapping of index column values to rows
   *
   * Compute a mapping from the "row" index space to a rectangle in the
   * reindexed column space.
   */
  static void
  compute_row_mapping(
    Legion::Context ctx,
    Legion::Runtime* rt,
    bool allow_rows,
    Legion::IndexPartition row_partition,
    const std::vector<Legion::LogicalRegion>& index_column_lrs,
    const Legion::LogicalRegion& row_map_lr);

  /**
   * Preregister tasks used by ColumnSpace
   *
   * Must be called before Legion runtime starts
   */
  static void
  preregister_tasks();

  Legion::IndexSpace column_is; /**< index space */

  Legion::LogicalRegion metadata_lr; /**< metadata region */

// protected:

//   friend class Legion::LegionTaskWrapper;

  static void
  init_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *rt);

  static reindexed_result_t
  reindexed_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *rt);

  static void
  compute_row_mapping_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *rt);

  static Legion::TaskID init_task_id;

  static const char* init_task_name;

  static Legion::TaskID reindexed_task_id;

  static const char* reindexed_task_name;

  static Legion::TaskID compute_row_mapping_task_id;

  static const char* compute_row_mapping_task_name;
};

} // end namespace hyperion

#endif // HYPERION_COLUMN_SPACE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
