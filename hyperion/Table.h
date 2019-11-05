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
#ifndef HYPERION_TABLE_H_
#define HYPERION_TABLE_H_

#include <hyperion/hyperion.h>
#include <hyperion/Table_c.h>
#include <hyperion/utility.h>
#include <hyperion/Keywords.h>
#include <hyperion/Column.h>
#include <hyperion/IndexTree.h>
#include <hyperion/ColumnPartition.h>
#include <hyperion/MSTable.h>
#include <hyperion/c_util.h>

#ifdef HYPERION_USE_HDF5
# include <hyperion/hdf5.h>
#endif // HYPERION_USE_HDF5

#ifdef HYPERION_USE_CASACORE
# include <hyperion/MeasRefContainer.h>
#endif

#pragma GCC visibility push(default)
# include <algorithm>
# include <cassert>
# include CXX_FILESYSTEM_HEADER
# include <memory>
# include <optional>
# include <unordered_map>
# include <unordered_set>
# include <vector>
#pragma GCC visibility pop

namespace hyperion {

class HYPERION_API Table {
public:

  static const constexpr Legion::FieldID METADATA_NAME_FID = 0;
  static const constexpr Legion::FieldID METADATA_AXES_UID_FID = 1;
  Legion::LogicalRegion metadata_lr;
  static const constexpr Legion::FieldID AXES_FID = 0;
  Legion::LogicalRegion axes_lr;
  static const constexpr Legion::FieldID COLUMNS_FID = 0;
  Legion::LogicalRegion columns_lr;
#ifdef HYPERION_USE_CASACORE
  MeasRefContainer meas_refs;
#endif
  Keywords keywords;

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using NameAccessor =
    Legion::FieldAccessor<
    MODE,
    hyperion::string,
    1,
    Legion::coord_t,
    Legion::AffineAccessor<hyperion::string, 1, Legion::coord_t>,
    CHECK_BOUNDS>;

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

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using ColumnsAccessor =
    Legion::FieldAccessor<
    MODE,
    Column,
    1,
    Legion::coord_t,
    Legion::AffineAccessor<Column, 1, Legion::coord_t>,
    CHECK_BOUNDS>;

  Table();

  Table(
    Legion::LogicalRegion metadata,
    Legion::LogicalRegion axes,
    Legion::LogicalRegion columns,
#ifdef HYPERION_USE_CASACORE
    const MeasRefContainer& meas_refs,
#endif
    const Keywords& keywords);

  Table(
    Legion::LogicalRegion metadata,
    Legion::LogicalRegion axes,
    Legion::LogicalRegion columns,
#ifdef HYPERION_USE_CASACORE
    const MeasRefContainer& meas_refs,
#endif
    Keywords&& keywords);

  std::string
  name(Legion::Context ctx, Legion::Runtime* rt) const;

  static const char*
  name(const Legion::PhysicalRegion& metadata);

  std::string
  axes_uid(Legion::Context ctx, Legion::Runtime* rt) const;

  static const char*
  axes_uid(const Legion::PhysicalRegion& metadata);

  std::vector<int>
  index_axes(Legion::Context ctx, Legion::Runtime* rt) const;

  static Table
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::string& name,
    const std::string& axes_uid,
    const std::vector<int>& index_axes,
    const std::vector<Column>& columns_,
#ifdef HYPERION_USE_CASACORE
    const MeasRefContainer& meas_refs,
#endif
    const Keywords::kw_desc_t& kws = Keywords::kw_desc_t(),
    const std::string& name_prefix = "");

  template <
    typename D,
    std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  static Table
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::string& name,
    const std::vector<D>& index_axes,
    const std::vector<Column>& columns,
#ifdef HYPERION_USE_CASACORE
    const MeasRefContainer& meas_refs,
#endif
    const Keywords::kw_desc_t& kws = Keywords::kw_desc_t(),
    const std::string& name_prefix = "") {

    return
      create(
        ctx,
        rt,
        name,
        Axes<D>::uid,
        map_to_int(index_axes),
        columns,
#ifdef HYPERION_USE_CASACORE
        meas_refs,
#endif
        kws,
        name_prefix);
  }

  static Table
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::string& name,
    const std::string& axes_uid,
    const std::vector<int>& index_axes,
    const std::vector<Column::Generator>& column_generators,
#ifdef HYPERION_USE_CASACORE
    const MeasRefContainer& meas_refs,
#endif
    const Keywords::kw_desc_t& kws = Keywords::kw_desc_t(),
    const std::string& name_prefix = "");

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  static Table
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::string& name,
    const std::vector<D>& index_axes,
    const std::vector<Column::Generator>& column_generators,
#ifdef HYPERION_USE_CASACORE
    const MeasRefContainer& meas_refs,
#endif
    const Keywords::kw_desc_t& kws = Keywords::kw_desc_t(),
    const std::string& name_prefix = "")  {

    return
      create(
        ctx,
        rt,
        name,
        Axes<D>::uid,
        map_to_int(index_axes),
        column_generators,
#ifdef HYPERION_USE_CASACORE
        meas_refs,
#endif
        kws,
        name_prefix);
  }


  void
  destroy(Legion::Context ctx, Legion::Runtime* rt, bool destroy_columns=true);

  bool
  is_empty(Legion::Context ctx, Legion::Runtime* rt) const;

  static bool
  is_empty(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const Legion::PhysicalRegion& columns);

  std::vector<std::string>
  column_names(Legion::Context ctx, Legion::Runtime* rt) const;

  static std::vector<std::string>
  column_names(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const Legion::PhysicalRegion& columns);

  Column
  column(Legion::Context ctx, Legion::Runtime* rt, const std::string& name)
    const;

  static Column
  column(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const Legion::PhysicalRegion& columns,
    const std::string& name);

  Column
  min_rank_column(Legion::Context ctx, Legion::Runtime* rt) const;

  static Column
  min_rank_column(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const Legion::PhysicalRegion& columns);

  template <typename FN>
  std::vector<std::invoke_result_t<FN,Legion::Context,Legion::Runtime*,Column>>
  map_columns(Legion::Context ctx, Legion::Runtime* rt, FN f) const {
    Legion::RegionRequirement req(columns_lr, READ_ONLY, EXCLUSIVE, columns_lr);
    req.add_field(COLUMNS_FID);
    auto pr = rt->map_region(ctx, req);
    std::vector<std::invoke_result_t<FN,Legion::Context,Legion::Runtime*,Column>>
      result;
    for (auto& colname : column_names(ctx, rt, pr)) {
      auto col = column(ctx, rt, pr, colname);
      result.push_back(f(ctx, rt, col));
    }

    rt->unmap_region(ctx, pr);
    return result;
  }

  template <typename FN>
  void
  foreach_column(Legion::Context ctx, Legion::Runtime* rt, FN f) const {
    Legion::RegionRequirement req(columns_lr, READ_ONLY, EXCLUSIVE, columns_lr);
    req.add_field(COLUMNS_FID);
    auto pr = rt->map_region(ctx, req);
    for (auto& colname : column_names(ctx, rt, pr)) {
      auto col = column(ctx, rt, pr, colname);
      f(ctx, rt, col);
    }
    rt->unmap_region(ctx, pr);
  }

#ifdef HYPERION_USE_HDF5
  template <
    typename FN,
    std::enable_if_t<
      !std::is_void_v<
        std::invoke_result_t<FN, Legion::Context, Legion::Runtime*, Table*>>,
      int> = 0>
  std::invoke_result_t<FN, Legion::Context, Legion::Runtime*, Table*>
  with_columns_attached(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const CXX_FILESYSTEM_NAMESPACE::path& file_path,
    const std::string& root_path,
    const std::unordered_set<std::string> read_only,
    const std::unordered_set<std::string> read_write,
    FN f) {

    typedef
      std::invoke_result_t<FN, Legion::Context, Legion::Runtime*, Table*> RET;

    std::vector<Legion::PhysicalRegion> prs =
      with_columns_attached_prologue(
        ctx,
        rt,
        file_path,
        root_path,
        {this, read_only, read_write});
    try {
      RET result = f(ctx, rt, this);
      with_columns_attached_epilogue(ctx, rt, prs);
      return result;
    } catch (...) {
      with_columns_attached_epilogue(ctx, rt, prs);
      throw;
    }
  }

  template <
    typename FN,
    std::enable_if_t<
      std::is_void_v<
        std::invoke_result_t<FN, Legion::Context, Legion::Runtime*, Table*>>,
      int> = 0>
  void
  with_columns_attached(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const CXX_FILESYSTEM_NAMESPACE::path& file_path,
    const std::string& root_path,
    const std::unordered_set<std::string> read_only,
    const std::unordered_set<std::string> read_write,
    FN f) {

    std::vector<Legion::PhysicalRegion> prs =
      with_columns_attached_prologue(
        ctx,
        rt,
        file_path,
        root_path,
        {this, read_only, read_write});
    try {
      f(ctx, rt, this);
      with_columns_attached_epilogue(ctx, rt, prs);
    } catch (...) {
      with_columns_attached_epilogue(ctx, rt, prs);
      throw;
    }
  }

  template <
    typename FN,
    std::enable_if_t<
      !std::is_void_v<
        std::invoke_result_t<
          FN,
          Legion::Context,
          Legion::Runtime*,
          std::unordered_map<std::string,Table*>&>>,
    int> = 0>
  static std::invoke_result_t<
    FN,
    Legion::Context,
    Legion::Runtime*,
    std::unordered_map<std::string,Table*>&>
  with_columns_attached(
    Legion::Context ctx,
    Legion::Runtime *rt,
    const CXX_FILESYSTEM_NAMESPACE::path& file_path,
    const std::string& root_path,
    const std::vector<
      std::tuple<
        Table*,
        std::unordered_set<std::string>,
        std::unordered_set<std::string>>>& table_columns,
    FN f) {

    typedef
      std::invoke_result_t<
        FN,
        Legion::Context,
        Legion::Runtime*,
        std::unordered_map<std::string,Table*>&> RET;

    std::unordered_map<std::string,Table*> tables;
    std::vector<Legion::PhysicalRegion> prs;
    std::for_each(
      table_columns.begin(),
      table_columns.end(),
      [&ctx, rt, &file_path, &root_path, &tables, &prs](auto& t_ro_rw) {
        auto tprs =
          with_columns_attached_prologue(
            ctx,
            rt,
            file_path,
            root_path,
            t_ro_rw);
        std::copy(tprs.begin(), tprs.end(), std::back_inserter(prs));
        auto t = std::get<0>(t_ro_rw);
        tables[t->name(ctx, rt)] = t;
      });
    try {
      RET result = f(ctx, rt, tables);
      with_columns_attached_epilogue(ctx, rt, prs);
      return result;
    } catch(...) {
      with_columns_attached_epilogue(ctx, rt, prs);
      throw;
    }
  }

  template <
    typename FN,
    std::enable_if_t<
      std::is_void_v<
        std::invoke_result_t<
          FN,
          Legion::Context,
          Legion::Runtime*,
          std::unordered_map<std::string,Table*>&>>,
    int> = 0>
  static void
  with_columns_attached(
    Legion::Context ctx,
    Legion::Runtime *rt,
    const CXX_FILESYSTEM_NAMESPACE::path& file_path,
    const std::string& root_path,
    const std::vector<
      std::tuple<
        Table*,
        std::unordered_set<std::string>,
        std::unordered_set<std::string>>>& table_columns,
    FN f) {

    std::unordered_map<std::string,Table*> tables;
    std::vector<Legion::PhysicalRegion> prs;
    std::for_each(
      table_columns.begin(),
      table_columns.end(),
      [&ctx, rt, &file_path, &root_path, &tables, &prs](auto& t_ro_rw) {
        auto tprs =
          with_columns_attached_prologue(
            ctx,
            rt,
            file_path,
            root_path,
            t_ro_rw);
        std::copy(tprs.begin(), tprs.end(), std::back_inserter(prs));
        auto t = std::get<0>(t_ro_rw);
        tables[t->name(ctx, rt)] = t;
      });
    try {
      f(ctx, rt, tables);
      with_columns_attached_epilogue(ctx, rt, prs);
    } catch(...) {
      with_columns_attached_epilogue(ctx, rt, prs);
      throw;
    }
  }

#endif // HYPERION_USE_HDF5

#ifndef NO_REINDEX
  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  Legion::Future/* Table */
  reindexed(const std::vector<D>& axes, bool allow_rows = true) const {
    assert(Axes<D>::uid == m_axes_uid);
    return ireindexed(Axes<D>::names, map_to_int(axes), allow_rows);
  }

  Legion::Future/* Table */
  reindexed(const std::vector<int>& axes, bool allow_rows = true) const {
    auto axs = AxesRegistrar::axes(axes_uid()).value();
    assert(
      std::all_of(
        axes.begin(),
        axes.end(),
        [m=axs.names.size()](auto& a) {
          return 0 <= a && static_cast<unsigned>(a) < m;
        }));
    return ireindexed(axs.names, axes, allow_rows);
  }
#endif // !NO_REINDEX

  // the returned Futures contain a LogicalRegion with two fields: at
  // IndexColumnTask::VALUE_FID, the column values (sorted in ascending order);
  // and at IndexColumnTask::ROWS_FID, a sorted vector of DomainPoints in the
  // original column. The LogicalRegions, along with their IndexSpaces and
  // FieldSpaces, should eventually be reclaimed.
  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  std::unordered_map<D, Legion::Future>
  index_by_value(
    Legion::Context ctx,
    Legion::Runtime *rt,
    const std::unordered_set<D>& axes) const {
    assert(Axes<D>::uid == axes_uid(ctx, rt));
    auto ia = iindex_by_value(ctx, rt, Axes<D>::names, map_to_int(axes));
    std::unordered_map<D, Legion::Future> result =
      hyperion::map(
        ia,
        [](auto& a_f) {
          auto& [a, f] = a_f;
          return std::make_pair(static_cast<D>(a), f);
        });
    return result;
  }

  // the returned Futures contain a LogicalRegion with two fields: at
  // IndexColumnTask::VALUE_FID, the column values (sorted in ascending order);
  // and at IndexColumnTask::ROWS_FID, a sorted vector of DomainPoints in the
  // original column. The LogicalRegions, along with their IndexSpaces and
  // FieldSpaces, should eventually be reclaimed.
  std::unordered_map<int, Legion::Future>
  index_by_value(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::unordered_set<int>& axes) const {
    auto axs = AxesRegistrar::axes(axes_uid(ctx, rt)).value();
    assert(
      std::all_of(
        axes.begin(),
        axes.end(),
        [m=axs.names.size()](auto& a) {
          return 0 <= a && static_cast<unsigned>(a) < m;
        }));
    return iindex_by_value(ctx, rt, axs.names, axes);
  }

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  std::unordered_map<std::string, Legion::Future /*ColumnPartition*/>
  partition_by_value(
    Legion::Context context,
    Legion::Runtime* runtime,
    const std::vector<D>& axes) const {

    assert(Axes<D>::uid == axes_uid(context, runtime));
    return
      ipartition_by_value(context, runtime, Axes<D>::names, map_to_int(axes));
  }

  std::unordered_map<std::string, Legion::Future /*ColumnPartition*/>
  partition_by_value(
    Legion::Context context,
    Legion::Runtime* runtime,
    const std::vector<int>& axes) const {

    auto axs = AxesRegistrar::axes(axes_uid(context, runtime)).value();
    assert(
      std::all_of(
        axes.begin(),
        axes.end(),
        [m=axs.names.size()](auto& a) {
          return 0 <= a && static_cast<unsigned>(a) < m;
        }));
    return ipartition_by_value(context, runtime, axs.names, axes);
  }

#ifdef HYPERION_USE_CASACORE
  static Table
  from_ms(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const CXX_FILESYSTEM_NAMESPACE::path& path,
    const std::unordered_set<std::string>& column_selections);
#endif // HYPERION_USE_CASACORE

  static void
  register_tasks(Legion::Context context, Legion::Runtime* runtime);

  static void
  preregister_tasks();

protected:

  static Legion::LogicalRegion
  create_metadata(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::string& name,
    const std::string& axes_uid,
    const std::string& name_prefix);

  static Legion::LogicalRegion
  create_axes(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<int>& index_axes,
    const std::string& name_prefix);

#ifdef HYPERION_USE_HDF5
  static std::vector<Legion::PhysicalRegion>
  with_columns_attached_prologue(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const CXX_FILESYSTEM_NAMESPACE::path& file_path,
    const std::string& root_path,
    const std::tuple<
      Table*,
      std::unordered_set<std::string>,
      std::unordered_set<std::string>>& table_columns);

  static void
  with_columns_attached_epilogue(
    Legion::Context ctx,
    Legion::Runtime* rt,
    std::vector<Legion::PhysicalRegion>& prs);
#endif // HYPERION_USE_HDF5

#ifndef NO_REINDEX
  Legion::Future/* Table */
  ireindexed(
    const std::vector<std::string>& axis_names,
    const std::vector<int>& axes,
    bool allow_rows = true) const;
#endif // !NO_REINDEX

  // the returned Futures contain a LogicalRegion with two fields: at
  // IndexColumnTask::VALUE_FID, the column values (sorted in ascending order);
  // and at IndexColumnTask::ROWS_FID, a sorted vector of DomainPoints in the
  // original column. The LogicalRegions, along with their IndexSpaces and
  // FieldSpaces, should eventually be reclaimed.
  std::unordered_map<int, Legion::Future>
  iindex_by_value(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<std::string>& axis_names,
    const std::unordered_set<int>& axes) const;

  std::unordered_map<std::string, Legion::Future /*ColumnPartition*/>
  ipartition_by_value(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<std::string>& axis_names,
    const std::vector<int>& axes) const;
};

class HYPERION_API IndexColumnTask
  : public Legion::TaskLauncher {
public:

  static Legion::TaskID TASK_ID;
  static const constexpr char* TASK_NAME = "IndexColumnTask";
  static constexpr Legion::FieldID VALUE_FID = Column::VALUE_FID;
  static constexpr Legion::FieldID ROWS_FID = Column::VALUE_FID + 10;

  IndexColumnTask(const Column& column);

  static void
  preregister_task();

  Legion::Future
  dispatch(Legion::Context ctx, Legion::Runtime* runtime);

  static Legion::LogicalRegion
  base_impl(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *runtime);
};

#ifndef NO_REINDEX
class HYPERION_API ReindexColumnTask {
public:

  static Legion::TaskID TASK_ID;
  static const char* TASK_NAME;

  ReindexColumnTask(
    const std::shared_ptr<Column>& col,
    ssize_t row_axis_offset,
    const std::vector<std::shared_ptr<Column>>& ixcols,
    bool allow_rows);

  static void
  preregister_task();

  Legion::Future
  dispatch(Legion::Context ctx, Legion::Runtime* runtime);

  static Column
  base_impl(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *runtime);

  static constexpr const Legion::FieldID row_rects_fid = 0;

private:

  struct TaskArgs {
    bool allow_rows;
    std::vector<int> index_axes;
    Legion::IndexPartition row_partition;
    Column col;

    size_t
    serialized_size() const;

    size_t
    serialize(void* buffer) const;

    static size_t
    deserialize(TaskArgs& val, const void* buffer);
  };

  std::unique_ptr<char[]> m_args_buffer;

  std::unique_ptr<ColumnPartition> m_partition;

  Legion::TaskLauncher m_launcher;
};

class HYPERION_API ReindexedTableTask {
public:

  static Legion::TaskID TASK_ID;
  static const char* TASK_NAME;

  ReindexedTableTask(
    const std::string& name,
    const std::string& axes_uid,
    const std::vector<int>& index_axes,
    Legion::LogicalRegion keywords_region,
    const std::vector<Legion::Future>& reindexed);

  static void
  preregister_task();

  Legion::Future
  dispatch(Legion::Context ctx, Legion::Runtime* runtime);

  static Table
  base_impl(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *runtime);

private:

  Legion::TaskLauncher m_launcher;

  std::unique_ptr<char[]> m_args;
};
#endif // !NO_REINDEX

template <>
struct CObjectWrapper::Wrapper<Table> {

  typedef table_t t;
  static table_t
  wrap(const Table& tb) {
    return
      table_t{
      Legion::CObjectWrapper::wrap(tb.metadata_lr),
        Legion::CObjectWrapper::wrap(tb.axes_lr),
        Legion::CObjectWrapper::wrap(tb.columns_lr),
        Legion::CObjectWrapper::wrap(tb.keywords.type_tags_lr),
        Legion::CObjectWrapper::wrap(tb.keywords.values_lr)};
  }
};

template <>
struct CObjectWrapper::Unwrapper<table_t> {

  typedef Table t;
  static Table
  unwrap(const table_t& tb) {
    return
      Table(
        Legion::CObjectWrapper::unwrap(tb.metadata),
        Legion::CObjectWrapper::unwrap(tb.axes),
        Legion::CObjectWrapper::unwrap(tb.columns),
#ifdef HYPERION_USE_CASACORE
        MeasRefContainer(), // FIXME
#endif
        Keywords(
          Keywords::pair<Legion::LogicalRegion>{
            Legion::CObjectWrapper::unwrap(tb.keyword_type_tags),
              Legion::CObjectWrapper::unwrap(tb.keyword_values)}));
  }
};

} // end namespace hyperion

#endif // HYPERION_TABLE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
