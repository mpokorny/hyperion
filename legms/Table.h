#ifndef LEGMS_TABLE_H_
#define LEGMS_TABLE_H_

#include <algorithm>
#include <cassert>
#include <experimental/filesystem>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "legms.h"

#include "Table_c.h"

#include "utility.h"
#include "WithKeywords.h"
#include "Column.h"
#include "IndexTree.h"
#include "ColumnPartition.h"
#include "MSTable.h"

#ifdef LEGMS_USE_HDF5
# include <hdf5.h>
#endif // LEGMS_USE_HDF5

#include "c_util.h"

namespace legms {

class Table;

struct TableGenArgs {

  TableGenArgs() {}

  TableGenArgs(
    const std::string& name,
    const std::string& axes_uid,
    const std::vector<int>& index_axes,
    const std::vector<ColumnGenArgs>& col_genargs,
    Legion::LogicalRegion keywords,
    const std::vector<TypeTag> keyword_datatypes)
    : name(name)
    , axes_uid(axes_uid)
    , index_axes(index_axes)
    , col_genargs(col_genargs)
    , keywords(keywords)
    , keyword_datatypes(keyword_datatypes) {}

  TableGenArgs(const table_t& table);

  std::string name;
  std::string axes_uid;
  std::vector<int> index_axes;
  std::vector<ColumnGenArgs> col_genargs;
  Legion::LogicalRegion keywords;
  std::vector<TypeTag> keyword_datatypes;

  std::unique_ptr<Table>
  operator()(Legion::Context ctx, Legion::Runtime* runtime) const;

  size_t
  legion_buffer_size(void) const;

  size_t
  legion_serialize(void *buffer) const;

  size_t
  legion_deserialize(const void *buffer);

  table_t
  to_table_t() const;
};

class Table
  : public WithKeywords {
public:

  Table(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    const std::string& axes_uid,
    const std::vector<int>& index_axes,
    const kw_desc_t& kws = kw_desc_t())
    : WithKeywords(ctx, runtime, kws)
    , m_name(name)
    , m_axes_uid(axes_uid)
    , m_index_axes(index_axes) {
  }

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  Table(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    const std::vector<D>& index_axes,
    const kw_desc_t& kws = kw_desc_t())
    : Table(
      ctx,
      runtime,
      name,
      Axes<D>::uid,
      map_to_int(index_axes),
      kws) {}

  Table(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    const std::string& axes_uid,
    const std::vector<int>& index_axes,
    Legion::LogicalRegion keywords,
    const std::vector<TypeTag>& datatypes)
    : WithKeywords(ctx, runtime, keywords, datatypes)
    , m_name(name)
    , m_axes_uid(axes_uid)
    , m_index_axes(index_axes) {
  }

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  Table(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    const std::vector<D>& index_axes,
    Legion::LogicalRegion keywords,
    const std::vector<TypeTag>& datatypes)
    : Table(
      ctx,
      runtime,
      name,
      Axes<D>::uid,
      map_to_int(index_axes),
      keywords,
      datatypes) {}

  template <typename GeneratorIter>
  Table(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    const std::string& axes_uid,
    const std::vector<int>& index_axes,
    GeneratorIter generator_first,
    GeneratorIter generator_last,
    const kw_desc_t& kws = kw_desc_t())
    : Table(ctx, runtime, name, axes_uid, index_axes, kws) {

    std::transform(
      generator_first,
      generator_last,
      std::inserter(m_columns, m_columns.end()),
      [&ctx, runtime](auto gen) {
        std::shared_ptr<Column> col(gen(ctx, runtime));
        return std::make_pair(col->name(), col);
      });

    assert(m_columns.size() > 0);

    set_min_max_rank();
  }

  template <
    typename D,
    typename GeneratorIter,
    std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  Table(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    const std::vector<D>& index_axes,
    GeneratorIter generator_first,
    GeneratorIter generator_last,
    const kw_desc_t& kws = kw_desc_t())
    : Table(
      ctx,
      runtime,
      name,
      Axes<D>::uid,
      map_to_int(index_axes),
      generator_first,
      generator_last,
      kws) {}

  Table(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    const std::string& axes_uid,
    const std::vector<int>& index_axes,
    const std::vector<Column::Generator>& column_generators,
    const kw_desc_t& kws = kw_desc_t())
    : Table(
      ctx,
      runtime,
      name,
      axes_uid,
      index_axes,
      column_generators.begin(),
      column_generators.end(),
      kws) {}

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  Table(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    const std::vector<D>& index_axes,
    const std::vector<Column::Generator>& column_generators,
    const kw_desc_t& kws = kw_desc_t())
    : Table(
      ctx,
      runtime,
      name,
      Axes<D>::uid,
      map_to_int(index_axes),
      column_generators,
      kws) {}

  Table(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    const std::string& axes_uid,
    const std::vector<int>& index_axes,
    const std::vector<ColumnGenArgs>& col_genargs,
    Legion::LogicalRegion keywords,
    const std::vector<TypeTag>& kw_datatypes)
    : Table(
      ctx,
      runtime,
      name,
      axes_uid,
      index_axes,
      keywords,
      kw_datatypes) {

    std::transform(
      col_genargs.begin(),
      col_genargs.end(),
      std::inserter(m_columns, m_columns.end()),
      [&ctx, runtime](auto gen) {
        std::shared_ptr<Column>
          col(gen.template operator()(ctx, runtime));
        return std::make_pair(col->name(), col);
      });

    set_min_max_rank();
  }

  virtual ~Table() {
  }

  const std::string&
  name() const {
    return m_name;
  }

  const std::string&
  axes_uid() const {
    return m_axes_uid;
  }

  const std::vector<int>&
  index_axes() const {
    return m_index_axes;
  }

  bool
  is_empty() const {
    return
      column_names().empty()
      || column(min_rank_column_name().value())->index_tree() == IndexTreeL();
  }

  std::unordered_set<std::string>
  column_names() const {
    std::unordered_set<std::string> result;
    std::transform(
      m_columns.begin(),
      m_columns.end(),
      std::inserter(result, result.end()),
      [](auto& col) {
        return col.first;
      });
    return result;
  }

  bool
  has_column(const std::string& name) const {
    return column_names().count(name) > 0;
  }

  std::shared_ptr<Column>
  column(const std::string& name) const {
    return m_columns.at(name);
  }

  const std::optional<std::string>&
  min_rank_column_name() const {
    return m_min_rank_colname;
  }

  const std::optional<std::string>&
  max_rank_column_name() const {
    return m_max_rank_colname;
  }

  TableGenArgs
  generator_args() const {
    std::vector<ColumnGenArgs> col_genargs;
    std::transform(
      m_columns.begin(),
      m_columns.end(),
      std::back_inserter(col_genargs),
      [](auto& nm_cp) { return std::get<1>(nm_cp)->generator_args(); });
    return TableGenArgs(
      name(),
      axes_uid(),
      index_axes(),
      col_genargs, // TODO: std::move?
      keywords_region(),
      keyword_datatypes());
  }

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  Legion::Future/* TableGenArgs */
  reindexed(const std::vector<D>& axes, bool allow_rows = true) const {
    assert(Axes<D>::uid == m_axes_uid);
    return ireindexed(Axes<D>::names, map_to_int(axes), allow_rows);
  }

  Legion::Future/* TableGenArgs */
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

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  std::unordered_map<D, Legion::Future>
  index_by_value(const std::unordered_set<D>& axes) const {
    assert(Axes<D>::uid == m_axes_uid);
    auto ia = iindex_by_value(Axes<D>::names, map_to_int(axes));
    std::unordered_map<D, Legion::Future> result =
      legms::map(
        ia,
        [](auto& a_f) {
          auto& [a, f] = a_f;
          return std::make_pair(static_cast<D>(a), f);
        });
    return result;
  }

  std::unordered_map<int, Legion::Future>
  index_by_value(const std::unordered_set<int>& axes) const {
    auto axs = AxesRegistrar::axes(axes_uid()).value();
    assert(
      std::all_of(
        axes.begin(),
        axes.end(),
        [m=axs.names.size()](auto& a) {
          return 0 <= a && static_cast<unsigned>(a) < m;
        }));
    return iindex_by_value(axs.names, axes);
  }

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  std::unordered_map<std::string, Legion::Future /*IndexPartition*/>
  partition_by_value(
    Legion::Context context,
    Legion::Runtime* runtime,
    const std::vector<D>& axes) const {

    assert(Axes<D>::uid == m_axes_uid);
    return
      ipartition_by_value(context, runtime, Axes<D>::names, map_to_int(axes));
  }

  std::unordered_map<std::string, Legion::Future /*IndexPartition*/>
  partition_by_value(
    Legion::Context context,
    Legion::Runtime* runtime,
    const std::vector<int>& axes) const {

    auto axs = AxesRegistrar::axes(axes_uid()).value();
    assert(
      std::all_of(
        axes.begin(),
        axes.end(),
        [m=axs.names.size()](auto& a) {
          return 0 <= a && static_cast<unsigned>(a) < m;
        }));
    return ipartition_by_value(context, runtime, axs.names, axes);
  }

#ifdef LEGMS_USE_CASACORE
  static std::unique_ptr<Table>
  from_ms(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::experimental::filesystem::path& path,
    const std::unordered_set<std::string>& column_selections);
#endif // LEGMS_USE_CASACORE

  Legion::Context
  context() const { return m_context; }

  Legion::Runtime*
  runtime() const { return m_runtime; }

  static void
  register_tasks(Legion::Runtime* runtime);

protected:

  Legion::Future/* TableGenArgs */
  ireindexed(
    const std::vector<std::string>& axis_names,
    const std::vector<int>& axes,
    bool allow_rows = true) const;

  std::unordered_map<int, Legion::Future>
  iindex_by_value(
    const std::vector<std::string>& axis_names,
    const std::unordered_set<int>& axes) const;

  std::unordered_map<std::string, Legion::Future /*IndexPartition*/>
  ipartition_by_value(
    Legion::Context context,
    Legion::Runtime* runtime,
    const std::vector<std::string>& axis_names,
    const std::vector<int>& axes) const;

  void
  set_min_max_rank() {
    if (!m_columns.empty()) {
      auto col0 = (*m_columns.begin()).second;
      std::tie(std::ignore, m_min_rank_colname, m_max_rank_colname) =
        std::accumulate(
          m_columns.begin(),
          m_columns.end(),
          std::make_tuple(col0->rank(), col0->name(), col0->name()),
          [](auto &acc, auto& nc) {
            auto& [mrank, mincol, maxcol] = acc;
            auto& [name, col] = nc;
            if (col->rank() < mrank)
              return std::make_tuple(col->rank(), name, maxcol);
            if (col->rank() > mrank)
              return std::make_tuple(col->rank(), mincol, name);
            return acc;
          });
    }
  }

private:

  std::string m_name;

  std::string m_axes_uid;

  std::vector<int> m_index_axes;

  std::unordered_map<std::string, std::shared_ptr<Column>> m_columns;

  std::optional<std::string> m_min_rank_colname;

  std::optional<std::string> m_max_rank_colname;
};

class IndexColumnTask {
public:

  static Legion::TaskID TASK_ID;
  static const char* TASK_NAME;
  static constexpr Legion::FieldID value_fid = Column::value_fid;
  static constexpr Legion::FieldID indices_fid = Column::value_fid + 10;

  IndexColumnTask(
    const std::shared_ptr<Column>& column,
    int axis);

  static void
  register_task(Legion::Runtime* runtime);

  Legion::Future
  dispatch(Legion::Context ctx, Legion::Runtime* runtime);

  static ColumnGenArgs
  base_impl(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *runtime);

private:

  Legion::TaskLauncher m_launcher;

  std::unique_ptr<char[]> m_args;
};

class ReindexColumnTask {
public:

  static Legion::TaskID TASK_ID;
  static const char* TASK_NAME;

  ReindexColumnTask(
    const std::shared_ptr<Column>& col,
    ssize_t row_axis_offset,
    const std::vector<std::shared_ptr<Column>>& ixcols,
    bool allow_rows);

  static void
  register_task(Legion::Runtime* runtime);

  Legion::Future
  dispatch(Legion::Context ctx, Legion::Runtime* runtime);

  static ColumnGenArgs
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
    ColumnGenArgs col;

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

class ReindexedTableTask {
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
  register_task(Legion::Runtime* runtime);

  Legion::Future
  dispatch(Legion::Context ctx, Legion::Runtime* runtime);

  static TableGenArgs
  base_impl(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *runtime);

private:

  Legion::TaskLauncher m_launcher;

  std::unique_ptr<char[]> m_args;
};

template <>
struct CObjectWrapper::UniqueWrapper<Table> {
  typedef table_t type_t;
};

template <>
struct CObjectWrapper::UniqueWrapped<table_t> {
  typedef Table type_t;
  typedef std::unique_ptr<type_t> impl_t;
};

} // end namespace legms

#endif // LEGMS_TABLE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
