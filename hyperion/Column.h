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
#ifndef HYPERION_COLUMN_H_
#define HYPERION_COLUMN_H_

#include <hyperion/hyperion.h>
#include <hyperion/utility.h>
#include <hyperion/Column_c.h>
#include <hyperion/Keywords.h>
#include <hyperion/IndexTree.h>
#include <hyperion/ColumnPartition.h>
#include <hyperion/c_util.h>

#pragma GCC visibility push(default)
# include <cassert>
# include <functional>
# include <memory>
# include <mutex>
# include <tuple>
# include <unordered_map>
#pragma GCC visibility pop

#ifdef HYPERION_USE_HDF5
# include <hyperion/hdf5.h>
#endif // HYPERION_USE_HDF5

#ifdef HYPERION_USE_CASACORE
# include <hyperion/MeasRef.h>
#endif

namespace hyperion {

class HYPERION_API Column {
public:

  typedef hyperion::string METADATA_NAME_TYPE;
  static const constexpr Legion::FieldID METADATA_NAME_FID = 0;
  typedef hyperion::string METADATA_AXES_UID_TYPE;
  static const constexpr Legion::FieldID METADATA_AXES_UID_FID = 1;
  typedef hyperion::TypeTag METADATA_DATATYPE_TYPE;
  static const constexpr Legion::FieldID METADATA_DATATYPE_FID = 2;
  typedef hyperion::string METADATA_REF_COL_TYPE;
  static const constexpr Legion::FieldID METADATA_REF_COL_FID = 3;
  Legion::LogicalRegion metadata_lr;

  typedef int AXES_VALUE_TYPE;
  static const constexpr Legion::FieldID AXES_FID = 0;
  Legion::LogicalRegion axes_lr;

  static const constexpr Legion::FieldID VALUE_FID = 0;
  Legion::LogicalRegion values_lr;

#ifdef HYPERION_USE_CASACORE
  MeasRef meas_ref;
#endif
  Keywords keywords;

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using NameAccessor =
    Legion::FieldAccessor<
    MODE,
    METADATA_NAME_TYPE,
    1,
    Legion::coord_t,
    Legion::AffineAccessor<METADATA_NAME_TYPE, 1, Legion::coord_t>,
    CHECK_BOUNDS>;

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using AxesUidAccessor =
    Legion::FieldAccessor<
    MODE,
    METADATA_AXES_UID_TYPE,
    1,
    Legion::coord_t,
    Legion::AffineAccessor<METADATA_AXES_UID_TYPE, 1, Legion::coord_t>,
    CHECK_BOUNDS>;

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using DatatypeAccessor =
    Legion::FieldAccessor<
    MODE,
    METADATA_DATATYPE_TYPE,
    1,
    Legion::coord_t,
    Legion::AffineAccessor<METADATA_DATATYPE_TYPE, 1, Legion::coord_t>,
    CHECK_BOUNDS>;

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using RefColAccessor =
    Legion::FieldAccessor<
      MODE,
      METADATA_REF_COL_TYPE,
      1,
      Legion::coord_t,
      Legion::AffineAccessor<METADATA_REF_COL_TYPE, 1, Legion::coord_t>,
      CHECK_BOUNDS>;

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using AxesAccessor =
    Legion::FieldAccessor<
    MODE,
    AXES_VALUE_TYPE,
    1,
    Legion::coord_t,
    Legion::AffineAccessor<AXES_VALUE_TYPE, 1, Legion::coord_t>,
    CHECK_BOUNDS>;

  typedef std::function<
    Column(Legion::Context, Legion::Runtime*, const std::string&)> Generator;

  Column();

  Column(
    Legion::LogicalRegion metadata,
    Legion::LogicalRegion axes,
    Legion::LogicalRegion values,
#ifdef HYPERION_USE_CASACORE
    const MeasRef& meas_ref,
#endif
    const Keywords& keywords);

  Column(
    Legion::LogicalRegion metadata,
    Legion::LogicalRegion axes,
    Legion::LogicalRegion values,
#ifdef HYPERION_USE_CASACORE
    const MeasRef& meas_ref,
#endif
    Keywords&& keywords);

  static Column
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::string& name,
    const std::string& axes_uid,
    const std::vector<int>& axes,
    hyperion::TypeTag datatype,
    const IndexTreeL& index_tree,
#ifdef HYPERION_USE_CASACORE
    const MeasRef& meas_ref,
    const std::optional<std::string>& ref_column,
#endif
    const Keywords::kw_desc_t& kws = Keywords::kw_desc_t(),
    const std::string& name_prefix = "");

  static Column
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::string& name,
    const std::string& axes_uid,
    const std::vector<int>& axes,
    hyperion::TypeTag datatype,
    const Legion::LogicalRegion& values,
#ifdef HYPERION_USE_CASACORE
    const MeasRef& meas_ref,
    const std::optional<std::string>& ref_column,
#endif
    const Keywords& kws);

  void
  destroy(Legion::Context ctx, Legion::Runtime* rt);

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  static Column
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::string& name,
    const std::vector<D>& axes,
    hyperion::TypeTag datatype,
    const IndexTreeL& index_tree,
#ifdef HYPERION_USE_CASACORE
    const MeasRef& meas_ref,
    const std::optional<std::string>& ref_column,
#endif
    const Keywords::kw_desc_t& kws = Keywords::kw_desc_t(),
    const std::string& name_prefix = "") {
    return
      create(
        ctx,
        rt,
        name,
        Axes<D>::uid,
        map_to_int(axes),
        datatype,
        index_tree,
#ifdef HYPERION_USE_CASACORE
        meas_ref,
        ref_column,
#endif
        kws,
        name_prefix);
  }

  std::string
  name(Legion::Context ctx, Legion::Runtime* rt) const;

  static const char*
  name(const Legion::PhysicalRegion& metadata);

  std::string
  axes_uid(Legion::Context ctx, Legion::Runtime* rt) const;

  static const char*
  axes_uid(const Legion::PhysicalRegion& metadata);

  hyperion::TypeTag
  datatype(Legion::Context ctx, Legion::Runtime* rt) const;

  static hyperion::TypeTag
  datatype(const Legion::PhysicalRegion& metadata);

  std::vector<int>
  axes(Legion::Context ctx, Legion::Runtime* rt) const;

  std::optional<std::string>
  ref_column(Legion::Context ctx, Legion::Runtime* rt) const;

  static hyperion::string
  ref_column(const Legion::PhysicalRegion& metadata);

  unsigned
  rank(Legion::Runtime* rt) const;

  bool
  is_empty() const;

  IndexTreeL
  index_tree(Legion::Runtime* rt) const;

#ifdef HYPERION_USE_HDF5
  template <
    typename FN,
    std::enable_if_t<
      !std::is_void_v<
        std::invoke_result_t<FN, Legion::Context, Legion::Runtime*, Column&>>,
      int> = 0>
  std::invoke_result_t<FN, Legion::Context, Legion::Runtime*, Column&>
  with_attached(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const CXX_FILESYSTEM_NAMESPACE::path& file_path,
    const std::string& table_root,
    FN f,
    bool mapped = false,
    bool read_write = false) {

    typedef
      std::invoke_result_t<FN, Legion::Context, Legion::Runtime*, Column&> RET;

    Legion::PhysicalRegion pr =
      with_attached_prologue(
        ctx,
        rt,
        file_path,
        table_root,
        mapped,
        read_write);
    RET result = f(ctx, rt, *this);
    with_attached_epilogue(ctx, rt, pr);
    return result;
  }

  template <
    typename FN,
    std::enable_if_t<
      std::is_void_v<
      std::invoke_result_t<FN, Legion::Context, Legion::Runtime*, Column&>>,
      int> = 0>
  void
  with_attached(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const CXX_FILESYSTEM_NAMESPACE::path& file_path,
    const std::string& table_root,
    FN f,
    bool mapped = false,
    bool read_write = false) {

    Legion::PhysicalRegion pr =
      with_attached_prologue(
        ctx,
        rt,
        file_path,
        table_root,
        mapped,
        read_write);
    f(ctx, rt, *this);
    with_attached_epilogue(ctx, rt, pr);
  }
#endif // HYPERION_USE_HDF5

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
    hyperion::TypeTag datatype,
    const IndexTreeL& index_tree,
#ifdef HYPERION_USE_CASACORE
    const MeasRef& meas_ref,
    const std::optional<std::string>& ref_column,
#endif
    const Keywords::kw_desc_t& kws = Keywords::kw_desc_t()) {

    return
      [=]
      (Legion::Context ctx,
       Legion::Runtime* rt,
       const std::string& name_prefix) {
        return
          create(
            ctx,
            rt,
            name,
            axes,
            datatype,
            index_tree,
#ifdef HYPERION_USE_CASACORE
            meas_ref,
            ref_column,
#endif
            kws,
            name_prefix);
      };
  }

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  static Generator
  generator(
    const std::string& name,
    const std::vector<D>& axes,
    hyperion::TypeTag datatype,
    const IndexTreeL& row_index_pattern,
    unsigned num_rows,
#ifdef HYPERION_USE_CASACORE
    const MeasRef& meas_ref,
    const std::optional<std::string>& ref_column,
#endif
    const Keywords::kw_desc_t& kws = Keywords::kw_desc_t()) {

    return
      generator(
        name,
        axes,
        datatype,
        IndexTreeL(row_index_pattern, num_rows),
#ifdef HYPERION_USE_CASACORE
        meas_ref,
        ref_column,
#endif
        kws);
  }

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  static Generator
  generator(
    const std::string& name,
    const std::vector<D>& axes,
    hyperion::TypeTag datatype,
    const IndexTreeL& row_index_pattern,
    const IndexTreeL& row_pattern,
    unsigned num_rows,
#ifdef HYPERION_USE_CASACORE
    const MeasRef& meas_ref,
    const std::optional<std::string>& ref_column,
#endif
    const Keywords::kw_desc_t& kws = Keywords::kw_desc_t()) {

    return
      generator(
        name,
        axes,
        datatype,
        IndexTreeL(
          row_pattern,
          num_rows * row_pattern.size() / row_index_pattern.size()),
#ifdef HYPERION_USE_CASACORE
        meas_ref,
        ref_column,
#endif
        kws);
  }

protected:
#ifdef HYPERION_USE_HDF5
  Legion::PhysicalRegion
  with_attached_prologue(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const CXX_FILESYSTEM_NAMESPACE::path& file_path,
    const std::string& table_root,
    bool mapped,
    bool read_write);

  void
  with_attached_epilogue(
    Legion::Context ctx,
    Legion::Runtime* rt,
    Legion::PhysicalRegion pr);
#endif // HYPERION_USE_HDF5

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
#ifdef HYPERION_USE_CASACORE
        MeasRef(), // FIXME
#endif
        Keywords(
          Keywords::pair<Legion::LogicalRegion>{
            Legion::CObjectWrapper::unwrap(c.keyword_type_tags),
              Legion::CObjectWrapper::unwrap(c.keyword_values)}));
  }
};

} // end namespace hyperion

#endif // HYPERION_COLUMN_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
