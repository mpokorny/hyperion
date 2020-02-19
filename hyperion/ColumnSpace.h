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
#ifndef HYPERION_X_COLUMN_SPACE_H_
#define HYPERION_X_COLUMN_SPACE_H_

#include <hyperion/hyperion.h>
#include <hyperion/utility.h>

#pragma GCC visibility push(default)
# include <algorithm>
# include <array>
# include <string>
# include <vector>
#pragma GCC visibility pop

namespace hyperion {
namespace x {

struct HYPERION_API ColumnSpace {

  static const constexpr size_t MAX_DIM = LEGION_MAX_DIM;

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

  static const constexpr Legion::FieldID AXIS_SET_UID_FID = 1;
  typedef hyperion::string AXIS_SET_UID_TYPE;
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using AxisSetUIDAccessor =
    Legion::FieldAccessor<
      MODE,
      AXIS_SET_UID_TYPE,
      1,
      Legion::coord_t,
      Legion::AffineAccessor<AXIS_SET_UID_TYPE, 1, Legion::coord_t>,
      CHECK_BOUNDS>;

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

  ColumnSpace() {}

  ColumnSpace(
    const Legion::IndexSpace& column_is,
    const Legion::LogicalRegion& metadata_lr);

  bool
  is_valid() const;

  bool
  is_empty() const;

  std::vector<int>
  axes(Legion::Context ctx, Legion::Runtime* rt) const;

  static AXIS_VECTOR_TYPE
  axes(Legion::PhysicalRegion pr);

  std::string
  axes_uid(Legion::Context ctx, Legion::Runtime* rt) const;

  static AXIS_SET_UID_TYPE
  axes_uid(Legion::PhysicalRegion pr);

  bool
  is_index(Legion::Context ctx, Legion::Runtime* rt) const;

  static INDEX_FLAG_TYPE
  is_index(Legion::PhysicalRegion pr);

  void
  destroy(
    Legion::Context ctx,
    Legion::Runtime* rt,
    bool destroy_index_space=false);

  bool
  operator<(const ColumnSpace& rhs) const;

  bool
  operator==(const ColumnSpace& rhs) const;

  bool
  operator!=(const ColumnSpace& rhs) const;

  static ColumnSpace
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<int>& axes,
    const std::string& axis_set_uid,
    const Legion::IndexSpace& column_is,
    bool is_index);

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  static ColumnSpace
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<D>& axes,
    const Legion::IndexSpace& column_is,
    bool is_index) {

    return
      create(
        ctx,
        rt,
        hyperion::map_to_int(axes),
        Axes<D>::uid,
        column_is,
        is_index);
  }

  static AXIS_VECTOR_TYPE
  to_axis_vector(const std::vector<int>& v) {
    assert(v.size() <= MAX_DIM);
    AXIS_VECTOR_TYPE result;
    auto e = std::copy(v.begin(), v.end(), result.begin());
    std::fill(e, result.end(), -1);
    return result;
  }

  static unsigned
  size(const AXIS_VECTOR_TYPE& v) {
    return
      std::distance(
        v.begin(),
        std::find_if(v.begin(), v.end(), [](int i) { return i < 0; }));
  }

  static std::vector<int>
  from_axis_vector(const AXIS_VECTOR_TYPE& av) {
    return std::vector<int>(av.begin(), av.begin() + size(av));
  }

  // reindexed_result_t fields:
  // - new ColumnSpace
  // - mapping from "row" in old index space to regions in new index space
  typedef std::tuple<ColumnSpace, Legion::LogicalRegion> reindexed_result_t;

  static constexpr const Legion::FieldID REINDEXED_ROW_RECTS_FID = 22;

  Legion::Future
  reindexed(
    Legion::Context ctx,
    Legion::Runtime* rt,
    unsigned element_rank,
    const std::vector<std::pair<int, Legion::LogicalRegion>>& index_columns,
    bool allow_rows) const;

  static reindexed_result_t
  reindexed(
    Legion::Context ctx,
    Legion::Runtime* rt,
    unsigned element_rank,
    const std::vector<std::pair<int, Legion::LogicalRegion>>& index_column_lrs,
    bool allow_rows,
    const Legion::IndexSpace& column_is,
    const Legion::PhysicalRegion& metadata_pr);

  static void
  compute_row_mapping(
    Legion::Context ctx,
    Legion::Runtime* rt,
    bool allow_rows,
    Legion::IndexPartition row_partition,
    const std::vector<Legion::LogicalRegion>& index_column_lrs,
    const Legion::LogicalRegion& row_map_lr);

  static void
  preregister_tasks();

  Legion::IndexSpace column_is;

  Legion::LogicalRegion metadata_lr;

protected:

  friend class Legion::LegionTaskWrapper;

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

private:

  static Legion::TaskID init_task_id;

  static const char* init_task_name;

  static Legion::TaskID reindexed_task_id;

  static const char* reindexed_task_name;

  static Legion::TaskID compute_row_mapping_task_id;

  static const char* compute_row_mapping_task_name;
};

} // end namespace x
} // end namespace hyperion

#endif // HYPERION_X_COLUMN_SPACE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
