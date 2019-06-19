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

#if USE_HDF5
# include <hdf5.h>
#endif // USE_HDF5

#include "c_util.h"

namespace legms {

class Table
  : public WithKeywords {
public:

  Table(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    const std::vector<int>& index_axes,
    const kw_desc_t& kws = kw_desc_t())
    : WithKeywords(ctx, runtime, kws)
    , m_name(name)
    , m_index_axes(index_axes) {
  };

  Table(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    const std::vector<int>& index_axes,
    Legion::LogicalRegion keywords,
    const std::vector<TypeTag>& datatypes)
    : WithKeywords(ctx, runtime, keywords, datatypes)
    , m_name(name)
    , m_index_axes(index_axes) {
  };

  virtual ~Table() {
  }

  const std::string&
  name() const {
    return m_name;
  }

  bool
  is_empty() const {
    return
      column_names().empty()
      || column(min_rank_column_name().value())->index_tree() == IndexTreeL();
  }

  virtual std::unordered_set<std::string>
  column_names() const = 0;

  bool
  has_column(const std::string& name) const {
    return column_names().count(name) > 0;
  }

  virtual std::shared_ptr<Column>
  column(const std::string& name) const = 0;

  virtual const std::optional<std::string>&
  min_rank_column_name() const = 0;

  virtual const std::optional<std::string>&
  max_rank_column_name() const = 0;

  virtual const char*
  axes_uid() const = 0;

  const std::vector<int>&
  index_axes() const {
    return m_index_axes;
  }

#if USE_HDF5
  virtual hid_t
  h5_axes_datatype() const = 0;
#endif // USE_HDF5

#ifdef USE_CASACORE
  static std::unique_ptr<Table>
  from_ms(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::experimental::filesystem::path& path,
    const std::unordered_set<std::string>& column_selections);
#endif // USE_CASACORE

  static void
  register_tasks(Legion::Runtime* runtime);

private:

  std::string m_name;

  std::vector<int> m_index_axes;
};

template <typename D> class TableT;

class IndexColumnTask {
public:

  static Legion::TaskID TASK_ID;
  static constexpr const char* TASK_NAME = "index_column_task";
  static constexpr Legion::FieldID rows_fid = Column::value_fid + 10;

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
  static constexpr const char* TASK_NAME = "reindex_column_task";

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

struct TableGenArgs {
  std::string name;
  std::string axes_uid;
  std::vector<int> index_axes;
  std::vector<ColumnGenArgs> col_genargs;
  Legion::LogicalRegion keywords;
  std::vector<TypeTag> keyword_datatypes;

  template <typename D>
  std::unique_ptr<TableT<D>>
  operator()(Legion::Context ctx, Legion::Runtime* runtime) const;

  size_t
  legion_buffer_size(void) const;

  size_t
  legion_serialize(void *buffer) const;

  size_t
  legion_deserialize(const void *buffer);
};

class ReindexedTableTask {
public:

  static Legion::TaskID TASK_ID;
  static constexpr const char* TASK_NAME = "reindexed_table_task";

  ReindexedTableTask(
    const std::string& name,
    const char* axes_uid,
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

template <typename D> class TableT;

template <typename D>
class TableT

  : public Table {
public:

  TableT(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    const std::vector<int>& index_axes,
    const std::vector<typename ColumnT<D>::Generator>& column_generators,
    const kw_desc_t& kws = kw_desc_t())
    : TableT(
      ctx,
      runtime,
      name,
      index_axes,
      column_generators.begin(),
      column_generators.end(),
      kws) {}

  template <typename GeneratorIter>
  TableT(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    const std::vector<int>& index_axes,
    GeneratorIter generator_first,
    GeneratorIter generator_last,
    const kw_desc_t& kws = kw_desc_t())
    : Table(ctx, runtime, name, index_axes, kws) {

    std::transform(
      generator_first,
      generator_last,
      std::inserter(m_columns, m_columns.end()),
      [&ctx, runtime](auto gen) {
        std::shared_ptr<ColumnT<D>> col(gen(ctx, runtime));
        return std::make_pair(col->name(), col);
      });

    assert(m_columns.size() > 0);

    set_min_max_rank();
  }

  TableT(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    const std::vector<int>& index_axes,
    const std::vector<ColumnGenArgs>& col_genargs,
    Legion::LogicalRegion keywords,
    const std::vector<TypeTag>& kw_datatypes)
    : Table(ctx, runtime, name, index_axes, keywords, kw_datatypes) {

    std::transform(
      col_genargs.begin(),
      col_genargs.end(),
      std::inserter(m_columns, m_columns.end()),
      [&ctx, runtime](auto gen) {
        std::shared_ptr<ColumnT<D>>
          col(gen.template operator()<D>(ctx, runtime));
        return std::make_pair(col->name(), col);
      });

    set_min_max_rank();
  }

  virtual ~TableT() {
  }

  const char*
  axes_uid() const override {
    return AxesUID<D>::id;
  }

#if USE_HDF5
  hid_t
  h5_axes_datatype() const override {
    return m_h5_axes_datatype;
  }

  static hid_t
  h5_axes() {
    return m_h5_axes_datatype;
  }
#endif // USE_HDF5

  std::unordered_set<std::string>
  column_names() const override {
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

  std::shared_ptr<Column>
  column(const std::string& name) const override {
    return m_columns.at(name);
  }

  std::shared_ptr<ColumnT<D>>
  columnT(const std::string& name) const {
    return m_columns.at(name);
  }

  TableGenArgs
  generator_args() const {
    std::vector<ColumnGenArgs> col_genargs;
    std::transform(
      column_names().begin(),
      column_names().end(),
      std::back_inserter(col_genargs),
      [](auto& nm) { return columnT(nm)->generator_args(); });
    return TableGenArgs {
      name(),
        axes_uid(),
        index_axes(),
        col_genargs, // TODO: std::move?
        keywords_region(),
        keywords_datatypes()};
  }

  std::optional<Legion::Future/*TableGenArgs*/>
  reindexed(const std::vector<D>& axes, bool allow_rows = true) const {

    // 'allow_rows' is intended to support the case where the reindexing may not
    // result in a single value in a column per aggregate index, necessitating the
    // maintenance of a row index. A value of 'true' for this argument is always
    // safe, but may result in a degenerate axis when an aggregate index always
    // identifies a single value in a column. If the value is 'false' and a
    // non-degenerate axis is required by the reindexing, this method will return
    // an empty value. TODO: remove degenerate axes after the fact, and do that
    // automatically in this method, which would allow us to remove the
    // 'allow_rows' argument.

    // can only reindex along an axis if table has a column with the associated
    // name
    //
    // TODO: add support for index columns that already exist in the table
    if ((index_axes().size() > 1)
        || (index_axes().back() != static_cast<int>(D::ROW)))
      return std::nullopt;

    // for every column in table, determine which axes need indexing
    std::unordered_map<std::string, std::vector<D>> col_reindex_axes;
    std::transform(
      column_names().begin(),
      column_names().end(),
      std::inserter(col_reindex_axes, col_reindex_axes.end()),
      [this, &axes](auto& nm) {
        std::vector<D> ax;
        auto col_axes = columnT(nm)->axesT();
        // skip the column if it does not have a "row" axis
        if (col_axes.back() == D::ROW) {
          // if column is a reindexing axis, reindexing depends only on itself
          auto myaxis = column_is_axis(nm, axes);
          if (myaxis) {
            ax.push_back(myaxis.value());
          } else {
            // select those axes in "axes" that are not already an axis of the
            // column
            std::for_each(
              axes.begin(),
              axes.end(),
              [&col_axes, &ax](auto& d) {
                if (find(col_axes.begin(), col_axes.end(), d) == col_axes.end())
                  ax.push_back(d);
              });
          }
        }
        return std::pair(nm, std::move(ax));
      });

    // index associated columns; the Future in "index_cols" below contains a
    // ColumnGenArgs of a LogicalRegion with two fields: at Column::value_fid, the
    // column values (sorted in ascending order); and at Column::value_fid +
    // IndexColumnTask::rows_fid, a sorted vector of DomainPoints in the original
    // column.
    auto& axis_names = Axes<D>::names;
    std::unordered_map<D, Legion::Future> index_cols;
    std::for_each(
      col_reindex_axes.begin(),
      col_reindex_axes.end(),
      [this, &axis_names, &index_cols](auto& nm_ds) {
        const std::vector<D>& ds = std::get<1>(nm_ds);
        std::for_each(
          ds.begin(),
          ds.end(),
          [this, &axis_names, &index_cols](auto& d) {
            if (index_cols.count(d) == 0) {
              auto col = columnT(axis_names.at(d));
              IndexColumnTask task(col, static_cast<int>(d));
              index_cols[d] = task.dispatch(context(), runtime());
            }
          });
      });

    // do reindexing of columns
    std::vector<Legion::Future> reindexed;
    std::transform(
      col_reindex_axes.begin(),
      col_reindex_axes.end(),
      std::back_inserter(reindexed),
      [this, &index_cols, &allow_rows](auto& nm_ds) {
        auto& [nm, ds] = nm_ds;
        // if this column is an index column, we've already launched a task to
        // create its logical region, so we can use that
        if (ds.size() == 1 && index_cols.count(ds[0]) > 0)
          return index_cols.at(ds[0]);

        // create reindexing task launcher
        // TODO: start intermediary task dependent on Futures of index columns
        std::vector<std::shared_ptr<Column>> ixcols;
        for (auto d : ds) {
          ixcols.push_back(
            index_cols.at(d)
            .template get_result<ColumnGenArgs>()
            .template operator()<D>(context(), runtime()));
        }
        auto col = columnT(nm);
        auto col_axes = col->axesT();
        auto row_axis_offset =
          std::distance(
            col_axes.begin(),
            find(col_axes.begin(), col_axes.end(), D::ROW));
        ReindexColumnTask task(col, row_axis_offset, ixcols, allow_rows);
        return task.dispatch(context(), runtime());
      });

    // launch task that creates the reindexed table
    std::vector<int> iaxes;
    std::transform(
      axes.begin(),
      axes.end(),
      std::back_inserter(iaxes),
      [](auto& d) {
        return static_cast<int>(d);
      });
    if (allow_rows)
      iaxes.push_back(static_cast<int>(D::ROW));
    ReindexedTableTask
      task(
        name(),
        axes_uid(),
        iaxes,
        keywords_region(),
        reindexed);
    return task.dispatch(context(), runtime());
  }

protected:

  const std::optional<std::string>&
  min_rank_column_name() const override {
    return m_min_rank_colname;
  }

  const std::optional<std::string>&
  max_rank_column_name() const override {
    return m_max_rank_colname;
  }

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

  std::unordered_map<std::string, std::shared_ptr<ColumnT<D>>> m_columns;

  std::optional<std::string> m_min_rank_colname;

  std::optional<std::string> m_max_rank_colname;

#if USE_HDF5
  static hid_t m_h5_axes_datatype;
#endif // USE_HDF5
};

template <typename D>
std::unique_ptr<TableT<D>>
TableGenArgs::operator()(
  Legion::Context ctx,
  Legion::Runtime* runtime) const {

  // TODO: convert this assertion to an exception
  assert(std::string(AxesUID<D>::id) == axes_uid);

  return
    std::make_unique<TableT<D>>(
      ctx,
      runtime,
      name,
      index_axes,
      col_genargs,
      keywords,
      keyword_datatypes);
}

template <>
struct CObjectWrapper::UniqueWrapper<Table> {
  typedef legms_table_t type_t;
};

template <>
struct CObjectWrapper::UniqueWrapped<legms_table_t> {
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
