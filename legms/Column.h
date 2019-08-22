#ifndef LEGMS_COLUMN_H_
#define LEGMS_COLUMN_H_

#pragma GCC visibility push(default)
#include <cassert>
#include <functional>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#pragma GCC visibility pop

#include "legms.h"
#include "utility.h"

#include "Column_c.h"

#include "Keywords.h"
#include "IndexTree.h"
#include "ColumnPartition.h"

#include "c_util.h"

namespace legms {

class Column;

class LEGMS_API Column {

public:

  static const constexpr Legion::FieldID METADATA_NAME_FID = 0;
  static const constexpr Legion::FieldID METADATA_AXES_UID_FID = 1;
  static const constexpr Legion::FieldID METADATA_DATATYPE_FID = 2;
  Legion::LogicalRegion metadata_lr;
  static const constexpr Legion::FieldID AXES_FID = 0;
  Legion::LogicalRegion axes_lr;
  static const constexpr Legion::FieldID VALUE_FID = 0;
  Legion::LogicalRegion values_lr;
  Keywords keywords;

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using NameAccessor =
    Legion::FieldAccessor<
    MODE,
    legms::string,
    1,
    Legion::coord_t,
    Legion::AffineAccessor<legms::string, 1, Legion::coord_t>,
    CHECK_BOUNDS>;

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
  using DatatypeAccessor =
    Legion::FieldAccessor<
    MODE,
    legms::TypeTag,
    1,
    Legion::coord_t,
    Legion::AffineAccessor<legms::TypeTag, 1, Legion::coord_t>,
    CHECK_BOUNDS>;

  typedef std::function<Column(Legion::Context, Legion::Runtime*)> Generator;

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using AxesAccessor =
    Legion::FieldAccessor<
    MODE,
    int,
    1,
    Legion::coord_t,
    Legion::AffineAccessor<int, 1, Legion::coord_t>,
    CHECK_BOUNDS>;

  Column();

  Column(
    Legion::LogicalRegion metadata,
    Legion::LogicalRegion axes,
    Legion::LogicalRegion values,
    const Keywords& keywords);

  Column(
    Legion::LogicalRegion metadata,
    Legion::LogicalRegion axes,
    Legion::LogicalRegion values,
    Keywords&& keywords);

  static Column
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::string& name,
    const std::string& axes_uid,
    const std::vector<int>& axes,
    legms::TypeTag datatype,
    const IndexTreeL& index_tree,
    const Keywords::kw_desc_t& kws = Keywords::kw_desc_t());

  void
  destroy(Legion::Context ctx, Legion::Runtime* rt);

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  static Column
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::string& name,
    const std::vector<D>& axes,
    legms::TypeTag datatype,
    const IndexTreeL& index_tree,
    const Keywords::kw_desc_t& kws = Keywords::kw_desc_t()) {
    return
      create(
        ctx,
        rt,
        name,
        Axes<D>::uid,
        map_to_int(axes),
        datatype,
        index_tree,
        kws);
  }

  std::string
  name(Legion::Context ctx, Legion::Runtime* rt) const;

  static const char*
  name(const Legion::PhysicalRegion& metadata);

  std::string
  axes_uid(Legion::Context ctx, Legion::Runtime* rt) const;

  static const char*
  axes_uid(const Legion::PhysicalRegion& metadata);

  legms::TypeTag
  datatype(Legion::Context ctx, Legion::Runtime* rt) const;

  static legms::TypeTag
  datatype(const Legion::PhysicalRegion& metadata);

  std::vector<int>
  axes(Legion::Context ctx, Legion::Runtime* rt) const;

  unsigned
  rank(Legion::Runtime* rt) const;

  bool
  is_empty() const;

  IndexTreeL
  index_tree(Legion::Runtime* rt) const;

  ColumnPartition
  partition_on_axes(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<AxisPartition>& parts) const;

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  ColumnPartition
  partition_on_axes(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<D>& ds) const {
    assert(axes_uid(ctx, rt) == Axes<D>::uid);
    return partition_on_iaxes(ctx, rt, map_to_int(ds));
  }

  ColumnPartition
  partition_on_axes(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<int>& ds) const {
    assert(AxesRegistrar::in_range(axes_uid(ctx, rt), ds));
    return partition_on_iaxes(ctx, rt, ds);
  }

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  ColumnPartition
  partition_on_axes(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<std::tuple<D, Legion::coord_t>>& ds)
    const {
    assert(axes_uid(ctx, rt) == Axes<D>::uid);
    std::vector<std::tuple<int, Legion::coord_t>> is =
      map(
        ds,
        [](const auto& d) {
          return
            std::make_tuple(static_cast<int>(std::get<0>(d)), std::get<1>(d));
        });
    return partition_on_iaxes(ctx, rt, is);
  }

  ColumnPartition
  partition_on_axes(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<std::tuple<int, Legion::coord_t>>& ds)
    const {
    std::vector<int> is = map(ds, [](const auto& d) { return std::get<0>(d); });
    assert(AxesRegistrar::in_range(axes_uid(ctx, rt), is));
    return partition_on_iaxes(ctx, rt, ds);
  }

  ColumnPartition
  projected_column_partition(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const ColumnPartition& cp) const;

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  static Generator
  generator(
    const std::string& name,
    const std::vector<D>& axes,
    legms::TypeTag datatype,
    const IndexTreeL& index_tree,
    const Keywords::kw_desc_t& kws = Keywords::kw_desc_t()) {

    return
      [=](Legion::Context ctx, Legion::Runtime* rt) {
        return create(ctx, rt, name, axes, datatype, index_tree, kws);
      };
  }

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  static Generator
  generator(
    const std::string& name,
    const std::vector<D>& axes,
    legms::TypeTag datatype,
    const IndexTreeL& row_index_pattern,
    unsigned num_rows,
    const Keywords::kw_desc_t& kws = Keywords::kw_desc_t()) {

    return
      generator(
        name,
        axes,
        datatype,
        IndexTreeL(row_index_pattern, num_rows),
        kws);
  }

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  static Generator
  generator(
    const std::string& name,
    const std::vector<D>& axes,
    legms::TypeTag datatype,
    const IndexTreeL& row_index_pattern,
    const IndexTreeL& row_pattern,
    unsigned num_rows,
    const Keywords::kw_desc_t& kws = Keywords::kw_desc_t()) {

    return
      generator(
        name,
        axes,
        datatype,
        IndexTreeL(
          row_pattern,
          num_rows * row_pattern.size() / row_index_pattern.size()),
        kws);
  }

protected:

  ColumnPartition
  partition_on_iaxes(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<int>& ds) const;

  ColumnPartition
  partition_on_iaxes(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<std::tuple<int, Legion::coord_t>>& ds) const;
};

template <>
struct CObjectWrapper::Wrapper<Column> {

  typedef column_t t;
  static column_t
  wrap(const Column& c) {
    return
      column_t{
      Legion::CObjectWrapper::wrap(c.metadata_lr),
        Legion::CObjectWrapper::wrap(c.axes_lr),
        Legion::CObjectWrapper::wrap(c.values_lr),
        Legion::CObjectWrapper::wrap(c.keywords.type_tags_lr),
        Legion::CObjectWrapper::wrap(c.keywords.values_lr)};
  }
};

template <>
struct CObjectWrapper::Unwrapper<column_t> {

  typedef Column t;
  static Column
  unwrap(const column_t& c) {
    return
      Column(
        Legion::CObjectWrapper::unwrap(c.metadata),
        Legion::CObjectWrapper::unwrap(c.axes),
        Legion::CObjectWrapper::unwrap(c.values),
        Keywords(
          Keywords::pair<Legion::LogicalRegion>{
            Legion::CObjectWrapper::unwrap(c.keyword_type_tags),
              Legion::CObjectWrapper::unwrap(c.keyword_values)}));
  }
};

} // end namespace legms

#endif // LEGMS_COLUMN_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
