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
#ifndef HYPERION_COLUMN_PARTITION_H_
#define HYPERION_COLUMN_PARTITION_H_

#include <hyperion/hyperion.h>
#include <hyperion/utility.h>
#include <hyperion/ColumnPartition_c.h>
#include <hyperion/c_util.h>

#pragma GCC visibility push(default)
# include <algorithm>
# include <vector>
#pragma GCC visibility pop

namespace hyperion {

class HYPERION_API ColumnPartition {
public:
  static const constexpr Legion::FieldID AXES_UID_FID = 0;
  Legion::LogicalRegion axes_uid_lr;
  static const constexpr Legion::FieldID AXES_FID = 0;
  Legion::LogicalRegion axes_lr;

  Legion::IndexPartition index_partition;

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using AxesUidAccessor =
    Legion::FieldAccessor<
    MODE,
    hyperion::string,
    1,
    Legion::coord_t,
    Legion::AffineAccessor<hyperion::string, 1, Legion::coord_t>,
    CHECK_BOUNDS>;

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using AxesAccessor =
    Legion::FieldAccessor<
    MODE,
    int,
    1,
    Legion::coord_t,
    Legion::AffineAccessor<int, 1, Legion::coord_t>,
    CHECK_BOUNDS>;

  ColumnPartition() {}

  ColumnPartition(
    Legion::LogicalRegion axes_uid,
    Legion::LogicalRegion axes,
    Legion::IndexPartition index_partition);

  static ColumnPartition
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::string& axes_uid,
    const std::vector<int>& axes,
    const Legion::IndexPartition& ip);

  static ColumnPartition
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::string& axes_uid,
    const std::vector<int>& axes,
    Legion::IndexSpace is,
    const std::vector<AxisPartition>& parts);

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  static ColumnPartition
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<D>& axes,
    const Legion::IndexPartition& ip) {

    return create(ctx, rt, Axes<D>::uid, ip, map_to_int(axes));
  }

  void
  destroy(
    Legion::Context ctx,
    Legion::Runtime* rt,
    bool destroy_color_space=false);

  std::string
  axes_uid(Legion::Context ctx, Legion::Runtime* rt) const;

  static const char*
  axes_uid(const Legion::PhysicalRegion& metadata);

  std::vector<int>
  axes(Legion::Context ctx, Legion::Runtime* rt) const;
};

template <>
struct CObjectWrapper::Wrapper<ColumnPartition> {

  typedef column_partition_t t;
  static column_partition_t
  wrap(const ColumnPartition& cp) {
    return
      column_partition_t{
      Legion::CObjectWrapper::wrap(cp.axes_uid_lr),
        Legion::CObjectWrapper::wrap(cp.axes_lr),
        Legion::CObjectWrapper::wrap(cp.index_partition)};
  }
};

template <>
struct CObjectWrapper::Unwrapper<column_partition_t> {

  typedef ColumnPartition t;
  static ColumnPartition
  unwrap(const column_partition_t& cp) {
    return
      ColumnPartition(
        Legion::CObjectWrapper::unwrap(cp.axes_uid),
        Legion::CObjectWrapper::unwrap(cp.axes),
        Legion::CObjectWrapper::unwrap(cp.index_partition));
  }
};

} // end namespace hyperion

#endif // HYPERION_COLUMN_PARTITION_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
