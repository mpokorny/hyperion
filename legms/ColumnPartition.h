#ifndef LEGMS_COLUMN_PARTITION_H_
#define LEGMS_COLUMN_PARTITION_H_

#pragma GCC visibility push(default)
#include <algorithm>
#include <vector>
#pragma GCC visibility pop

#include "legms.h"
#include "utility.h"
#include "ColumnPartition_c.h"

#include "c_util.h"

namespace legms {

class LEGMS_API ColumnPartition {
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
    legms::string,
    1,
    Legion::coord_t,
    Legion::AffineAccessor<legms::string, 1, Legion::coord_t>,
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
struct CObjectWrapper::UniqueWrapper<ColumnPartition> {
  typedef column_partition_t type_t;
};

template <>
struct CObjectWrapper::UniqueWrapped<column_partition_t> {
  typedef ColumnPartition type_t;
  typedef std::unique_ptr<type_t> impl_t;
};

} // end namespace legms

#endif // LEGMS_COLUMN_PARTITION_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
