#include <algorithm>
#include <array>
#include <numeric>
#include <tuple>

#include "Column.h"
#include "Table.h"

using namespace legms;
using namespace std;
using namespace Legion;

std::unique_ptr<Table>
Table::from_ms(
  Context ctx,
  Runtime* runtime,
  const std::experimental::filesystem::path& path,
  const std::unordered_set<std::string>& column_selections) {

  std::string table_name = path.filename();

#define FROM_MS_TABLE(N)                                \
  do {                                                  \
    if (table_name == MSTable<MSTables::N>::name)       \
      return legms:: template from_ms<MSTables::N>( \
        ctx, runtime, path, column_selections);         \
  } while (0)

  FROM_MS_TABLE(MAIN);
  FROM_MS_TABLE(ANTENNA);
  FROM_MS_TABLE(DATA_DESCRIPTION);
  FROM_MS_TABLE(DOPPLER);
  FROM_MS_TABLE(FEED);
  FROM_MS_TABLE(FIELD);
  FROM_MS_TABLE(FLAG_CMD);
  FROM_MS_TABLE(FREQ_OFFSET);
  FROM_MS_TABLE(HISTORY);
  FROM_MS_TABLE(OBSERVATION);
  FROM_MS_TABLE(POINTING);
  FROM_MS_TABLE(POLARIZATION);
  FROM_MS_TABLE(PROCESSOR);
  FROM_MS_TABLE(SOURCE);
  FROM_MS_TABLE(SPECTRAL_WINDOW);
  FROM_MS_TABLE(STATE);
  FROM_MS_TABLE(SYSCAL);
  FROM_MS_TABLE(WEATHER);
  // try to read as main table
  return
    legms:: template from_ms<MSTables::MAIN>(
      ctx,
      runtime,
      path,
      column_selections);

#undef FROM_MS_TABLE
}

size_t
TableGenArgs::legion_buffer_size(void) const {
  size_t result =
    accumulate(
      col_genargs.begin(),
      col_genargs.end(),
      name.size() + 1 + sizeof(size_t),
      [](auto& acc, auto& cg) {
        return acc + cg.legion_buffer_size();
      });
  return result + 2 * sizeof(LogicalRegion);
}

size_t
TableGenArgs::legion_serialize(void *buffer) const {
  char* buff = static_cast<char*>(buffer);

  size_t s = name.size() + 1;
  memcpy(buff, name.c_str(), s);
  buff += s;

  size_t csz = col_genargs.size();
  s = sizeof(csz);
  memcpy(buff, &csz, s);
  buff += s;

  buff =
    accumulate(
      col_genargs.begin(),
      col_genargs.end(),
      buff,
      [](auto& bf, auto& cg) {
        return bf + cg.legion_serialize(bf);
      });

  s = sizeof(LogicalRegion);
  memcpy(buff, &keywords, s);
  buff += s;

  return buff - static_cast<char*>(buffer);
}

size_t
TableGenArgs::legion_deserialize(const void *buffer) {
  const char *buff = static_cast<const char*>(buffer);

  name = *buff;
  buff += name.size() + 1;

  size_t ncg = *reinterpret_cast<const size_t *>(buff);
  buff += sizeof(ncg);

  col_genargs.clear();
  for (size_t i = 0; i < ncg; ++i) {
    ColumnGenArgs genargs;
    buff += genargs.legion_deserialize(buff);
    col_genargs.push_back(genargs);
  }

  keywords = *(const decltype(keywords) *)buff;
  buff += sizeof(keywords);

  return buff - static_cast<const char*>(buffer);
}

ReindexedTableTask::ReindexedTableTask(
  const std::string& name,
  LogicalRegion keywords_region,
  const std::vector<Legion::Future>& reindexed) {

  // reuse TableGenArgsSerializer to pass task arguments
  TableGenArgs args;
  args.name = name;
  args.keywords = keywords_region;

  size_t buffsz = args.legion_buffer_size();
  m_args = std::make_unique<char[]>(buffsz);
  args.legion_serialize(m_args.get());
  m_launcher =
    TaskLauncher(
      ReindexedTableTask::TASK_ID,
      TaskArgument(m_args.get(), buffsz));
  std::for_each(
    reindexed.begin(),
    reindexed.end(),
    [this](auto& f) { m_launcher.add_future(f); });
}

void
ReindexedTableTask::register_task(Runtime* runtime) {
  TASK_ID = runtime->generate_library_task_ids("legms::ReindexedTableTask", 1);
  TaskVariantRegistrar registrar(TASK_ID, TASK_NAME);
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  registrar.set_leaf();
  registrar.set_inner();
  registrar.set_idempotent();
  registrar.set_replicable();
  runtime->register_task_variant<TableGenArgs,base_impl>(registrar);
}

Future
ReindexedTableTask::dispatch(Context ctx, Runtime* runtime) {
  return runtime->execute_task(ctx, m_launcher);
}

TableGenArgs
ReindexedTableTask::base_impl(
  const Task* task,
  const std::vector<PhysicalRegion>&,
  Context,
  Runtime *) {

  TableGenArgs result;
  result.legion_deserialize(task->args);

  transform(
    task->futures.begin(),
    task->futures.end(),
    inserter(result.col_genargs, result.col_genargs.end()),
    [](auto& f) {
      return f.template get_result<ColumnGenArgs>();
    });
  return result;
}

template <typename T>
struct IndexAcc {
  typedef std::vector<std::tuple<T, std::vector<DomainPoint>>> field_t;
};

template <typename T>
using acc_field_t = typename IndexAcc<T>::field_t;

template <typename T>
class IndexAccumulateTask {
public:

  typedef DataType<ValueType<T>::DataType> DT;

  static Legion::TaskID TASK_ID;
  static char TASK_NAME[40];

  IndexAccumulateTask(RegionRequirement req, LogicalRegion acc_lr) {

    m_launcher =
      IndexTaskLauncher(
        TASK_ID,
        req.region.get_index_space(),
        TaskArgument(NULL, 0),
        ArgumentMap());
    m_launcher.add_region_requirement(req);
    RegionRequirement acc_req(
      acc_lr,
      DT::af_redop_id,
      EXCLUSIVE,
      acc_lr);
    acc_req.add_field(0);
    m_launcher.add_region_requirement(acc_req);
  }

  void
  dispatch(Context ctx, Runtime* runtime) {
    runtime->execute_index_space(ctx, m_launcher);
  }

  static void
  register_task(Runtime* runtime, int tid0) {
    TASK_ID = tid0 + DT::id;
    std::string tname =
      std::string("index_column_task<") + DT::s + std::string(">");
    strncpy(TASK_NAME, tname.c_str(), sizeof(TASK_NAME));
    TASK_NAME[sizeof(TASK_NAME) - 1] = '\0';
    TaskVariantRegistrar registrar(TASK_ID, TASK_NAME);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    registrar.set_idempotent();
    registrar.set_replicable();
    runtime->register_task_variant<base_impl>(registrar);
  }

  static void
  base_impl(
    const Task* task,
    const std::vector<PhysicalRegion>& regions,
    Context,
    Runtime*) {

    const FieldAccessor<
      READ_ONLY,
      T,
      1,
      coord_t,
      AffineAccessor<T, 1, coord_t>,
      false> values(regions[0], 0);

    const ReductionAccessor<
      acc_field_redop<T>,
      true,
      1,
      coord_t,
      AffineAccessor<
        std::vector<std::tuple<T, std::vector<DomainPoint>>>,
        1,
        coord_t>,
      false> acc(regions[1], 0, DT::af_redop_id);

    std::vector<DomainPoint> pt { task->index_point };
    acc[0] <<= {std::make_tuple(values[0], pt)};
  }

private:

  IndexTaskLauncher m_launcher;
};

class IndexAccumulateTasks {
public:

  static void
  register_tasks(Runtime *runtime) {
    auto tid0 =
      runtime->generate_library_task_ids(
        "legms::IndexAccumulateTasks",
        NUM_CASACORE_DATATYPES);

#define REG_TASK(DT) \
    IndexAccumulateTask<DataType<DT>::ValueType>::register_task(runtime, tid0);

    FOREACH_DATATYPE(REG_TASK);
  }
};

IndexColumnTask::IndexColumnTask(std::shared_ptr<Column>& column) {

  ColumnGenArgs args = column->generator_args();
  args.axes.clear();
  args.axes.push_back(-1);
  size_t buffsz = args.legion_buffer_size();
  m_args = std::make_unique<char[]>(buffsz);
  args.legion_serialize(m_args.get());
  m_launcher =
    TaskLauncher(
      TASK_ID,
      TaskArgument(m_args.get(), buffsz));
}

void
IndexColumnTask::register_task(Runtime* runtime) {
  TASK_ID = runtime->generate_dynamic_task_id();
  TaskVariantRegistrar registrar(TASK_ID, TASK_NAME);
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  registrar.set_idempotent();
  registrar.set_replicable();
  runtime->register_task_variant<ColumnGenArgs,base_impl>(registrar);
}

Future
IndexColumnTask::dispatch(Context ctx, Runtime* runtime) {
  return runtime->execute_task(ctx, m_launcher);
}

template <typename T>
LogicalRegion
index_column(
  const Task* task,
  Context ctx,
  Runtime *runtime,
  casacore::DataType dt) {

  // create accumulator lr
  auto acc_is = runtime->create_index_space(ctx, Rect<1>(0, 0));
  auto acc_fs = runtime->create_field_space(ctx);
  {
    auto fa = runtime->create_field_allocator(ctx, acc_fs);
    fa.allocate_field(
      sizeof(acc_field_t<T>),
      0,
      DataType<ValueType<T>::DataType>::af_serdez_id);
  }
  auto acc_lr = runtime->create_logical_region(ctx, acc_is, acc_fs);

  // launch index space task on input region to write to accumulator lr
  IndexAccumulateTask<T> acc_index_task(task->regions[0], acc_lr);
  acc_index_task.dispatch(ctx, runtime);

  // create result lr
  auto result_fs = runtime->create_field_space(ctx);
  {
    auto fa = runtime->create_field_allocator(ctx, result_fs);
    add_field(dt, fa, Column::value_fid);
    fa.allocate_field(
      sizeof(std::vector<DomainPoint>),
      IndexColumnTask::rows_fid,
      OpsManager::V_DOMAIN_POINT_SID);
  }
  // need results of acc_index_task to create the result lr
  RegionRequirement acc_req(acc_lr, READ_ONLY, EXCLUSIVE, acc_lr);
  acc_req.add_field(0);
  InlineLauncher acc_task(acc_req);
  PhysicalRegion acc_pr = runtime->map_region(ctx, acc_task);
  acc_pr.wait_until_valid();

  const FieldAccessor<
    READ_ONLY,
    acc_field_t<T>,
    1,
    coord_t,
    AffineAccessor<acc_field_t<T>, 1, coord_t>,
    false> rows_acc(acc_pr, 0);
  const acc_field_t<T>& acc_field = rows_acc[0];
  assert(acc_field.size() > 0); // FIXME: should return empty lr
  auto result_is =
    runtime->create_index_space(ctx, Rect<1>(0, acc_field.size() - 1));
  auto result_lr =
    runtime->create_logical_region(ctx, result_is, result_fs);

  // transfer values and row numbers from acc_lr to result_lr
  RegionRequirement result_req(result_lr, WRITE_DISCARD, EXCLUSIVE, result_lr);
  result_req.add_field(Column::value_fid);
  result_req.add_field(IndexColumnTask::rows_fid);
  InlineLauncher result_task(result_req);
  PhysicalRegion result_pr = runtime->map_region(ctx, result_task);
  const FieldAccessor<
    WRITE_DISCARD,
    T,
    1,
    coord_t,
    AffineAccessor<T, 1, coord_t>,
    false> values(result_pr, Column::value_fid);
  const FieldAccessor<
    WRITE_DISCARD,
    std::vector<DomainPoint>,
    1,
    coord_t,
    AffineAccessor<std::vector<DomainPoint>, 1, coord_t>,
    false> rns(result_pr, IndexColumnTask::rows_fid);
  for (auto i = 0; i < acc_field.size(); ++i)
    std::tie(values[i], rns[i]) = acc_field[i];

  runtime->destroy_logical_region(ctx, acc_lr);
  runtime->destroy_field_space(ctx, acc_fs);
  runtime->destroy_index_space(ctx, acc_is);
  runtime->destroy_field_space(ctx, result_fs);
  runtime->destroy_index_space(ctx, result_is);
  return result_lr;
}

ColumnGenArgs
IndexColumnTask::base_impl(
  const Task* task,
  const std::vector<PhysicalRegion>&,
  Context ctx,
  Runtime *runtime) {

  ColumnGenArgs result;
  result.legion_deserialize(task->args);

#define ICR(DT)                                                       \
  case DT:                                                            \
    result.values =                                                   \
      index_column<DataType<DT>::ValueType>(task, ctx, runtime, DT);  \
    break;

  switch (result.datatype) {
    FOREACH_DATATYPE(ICR);
  default:
    assert(false);
    break;
  }
  return result;
}

class ComputeRectanglesTask {
public:

  static TaskID TASK_ID;
  static constexpr const char* TASK_NAME = "compute_rectangles_task";

  ComputeRectanglesTask(
    LogicalRegion new_rects,
    bool allow_rows,
    IndexPartition row_partition,
    const std::vector<LogicalRegion>& ix_columns,
    const std::vector<coord_t>& ix0,
    const std::vector<DomainPoint>& rows)
    : m_args{new_rects, allow_rows, row_partition, ix_columns, ix0, rows} {
  };

  void
  dispatch(Context ctx, Runtime* runtime) {

    auto i = m_args.ix0.size();
    std::unique_ptr<char[]> args_buffer =
      std::make_unique<char[]>(m_args.serialized_size());
    m_args.serialize(args_buffer.get());
    IndexTaskLauncher launcher(
      TASK_ID,
      m_args.ix_columns[i].get_index_space(),
      TaskArgument(args_buffer.get(), m_args.serialized_size()),
      ArgumentMap());

    RegionRequirement ixc_req(
      m_args.ix_columns[i],
      0,
      READ_ONLY,
      EXCLUSIVE,
      m_args.ix_columns[i]);
    ixc_req.add_field(IndexColumnTask::rows_fid);
    launcher.add_region_requirement(ixc_req);
    if (i == m_args.ix_columns.size() - 1) {
      RegionRequirement new_rects_req(
        m_args.new_rects,
        WRITE_DISCARD,
        EXCLUSIVE, // TODO: ATOMIC?
        m_args.new_rects);
      new_rects_req.add_field(ReindexColumnTask::row_rects_fid);
      launcher.add_region_requirement(new_rects_req);
    }
    runtime->execute_index_space(ctx, launcher);
  }

  static void
  base_impl(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *runtime) {

    TaskArgs args;
    TaskArgs::deserialize(args, static_cast<const void *>(task->args));

    const FieldAccessor<
      READ_ONLY,
      std::vector<DomainPoint>,
      1,
      coord_t,
      AffineAccessor<std::vector<DomainPoint>, 1, coord_t>,
      false> rows(regions[0], IndexColumnTask::rows_fid);

    args.ix0.push_back(task->index_point[0]);
    if (args.ix0.size() == 1)
      args.rows = rows[0];
    else
      args.rows = intersection(args.rows, rows[0]);
    if (args.rows.size() > 0) {
      if (regions.size() == 1) {
        // start task at next index level
        ComputeRectanglesTask task(
          args.new_rects,
          args.allow_rows,
          args.row_partition,
          args.ix_columns,
          args.ix0,
          args.rows);
        task.dispatch(ctx, runtime);
      } else {
        // at bottom of indexes, write results to "new_rects" region
        if (args.allow_rows || args.rows.size() == 1) {

          auto rowdim = args.rows[0].get_dim();
          auto rectdim =
            args.ix_columns.size() + args.row_partition.get_dim() - rowdim
            + (args.allow_rows ? 1 : 0);

#define WRITE_RECTS(ROWDIM, RECTDIM)                                    \
          case (ROWDIM * LEGMS_MAX_DIM + RECTDIM): {                    \
            const FieldAccessor<                                        \
              WRITE_DISCARD, \
              Rect<RECTDIM>, \
              ROWDIM, \
              coord_t, \
              AffineAccessor<Rect<RECTDIM>, ROWDIM, coord_t>, \
              false> rects(regions[1], ReindexColumnTask::row_rects_fid); \
                                                                        \
            for (size_t i = 0; i < args.rows.size(); ++i) {             \
              Domain row_d =                                            \
                runtime->get_index_space_domain(                        \
                  ctx,                                                  \
                  runtime->get_index_subspace(                          \
                    ctx,                                                \
                    args.row_partition,                                 \
                    args.rows[i]));                                     \
              Rect<RECTDIM> row_rect;                                   \
              size_t j = 0;                                             \
              for (; j < args.ix0.size(); ++j) {                        \
                row_rect.lo[j] = args.ix0[j];                           \
                row_rect.hi[j] = args.ix0[j];                           \
              }                                                         \
              if (args.allow_rows) {                                    \
                row_rect.lo[j] = i;                                     \
                row_rect.hi[j] = i;                                     \
                ++j;                                                    \
              }                                                         \
              for (; j < RECTDIM; ++j) {                                \
                row_rect.lo[j] = row_d.lo()[j - (RECTDIM - ROWDIM)];    \
                row_rect.hi[j] = row_d.hi()[j - (RECTDIM - ROWDIM)];    \
              }                                                         \
              rects[args.rows[i]] = row_rect;                           \
            }                                                           \
            break;                                                      \
          }

        switch (rowdim * LEGMS_MAX_DIM + rectdim) {
          LEGMS_FOREACH_MN(WRITE_RECTS);
        default:
          assert(false);
          break;
        }
#undef WRITE_RECTS

        } else {
          // TODO: FAIL
        }
      }
    }
  }

  static void
  register_task(Runtime* runtime) {
    TASK_ID =
      runtime->generate_library_task_ids("legms::ComputeRectanglesTask", 1);
    TaskVariantRegistrar registrar(TASK_ID, TASK_NAME);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    registrar.set_replicable();
    runtime->register_task_variant<base_impl>(registrar);
  }

private:

  struct TaskArgs {
    LogicalRegion new_rects;
    bool allow_rows;
    IndexPartition row_partition;
    std::vector<LogicalRegion> ix_columns;
    std::vector<coord_t> ix0;
    std::vector<DomainPoint> rows;

    size_t
    serialized_size() const {
      return
        sizeof(new_rects) + sizeof(allow_rows) + sizeof(row_partition)
        + vector_serdez<decltype(ix_columns)::value_type>::serialized_size(
          ix_columns)
        + vector_serdez<decltype(ix0)::value_type>::serialized_size(ix0)
        + vector_serdez<decltype(rows)::value_type>::serialized_size(rows);
    }

    size_t
    serialize(void *buffer) const {
      size_t result = 0;
      char* buff = static_cast<char*>(buffer);
      memcpy(buff, &new_rects, sizeof(new_rects));
      result += sizeof(new_rects);
      memcpy(buff + result, &allow_rows, sizeof(allow_rows));
      result += sizeof(allow_rows);
      memcpy(buff + result, &row_partition, sizeof(row_partition));
      result += sizeof(row_partition);
      result +=
        vector_serdez<decltype(ix_columns)::value_type>::serialize(
          ix_columns,
          buff + result);
      result +=
        vector_serdez<decltype(ix0)::value_type>::serialize(
          ix0,
          buff + result);
      result +=
        vector_serdez<decltype(rows)::value_type>::serialize(
          rows,
          buff + result);
      return result;
    }

    static size_t
    deserialize(TaskArgs& val, const void *buffer) {
      size_t result = 0;
      const char* buff = static_cast<const char*>(buffer);
      memcpy(&val.new_rects, buff, sizeof(new_rects));
      result += sizeof(new_rects);
      memcpy(&val.allow_rows, buff + result, sizeof(allow_rows));
      result += sizeof(allow_rows);
      memcpy(&val.row_partition, buff + result, sizeof(row_partition));
      result += sizeof(row_partition);
      result +=
        vector_serdez<decltype(ix_columns)::value_type>::deserialize(
          val.ix_columns,
          buff + result);
      result +=
        vector_serdez<decltype(ix0)::value_type>::deserialize(
          val.ix0,
          buff + result);
      result +=
        vector_serdez<decltype(rows)::value_type>::deserialize(
          val.rows,
          buff + result);
      return result;
    }
  };

  TaskArgs m_args;

  static std::vector<DomainPoint>
  intersection(
    const std::vector<DomainPoint>& first,
    const std::vector<DomainPoint>& second) {

    std::vector<DomainPoint> result;
    std::set_intersection(
      first.begin(),
      first.end(),
      second.begin(),
      second.end(),
      std::back_inserter(result));
    return result;
  }
};

class ReindexColumnCopyTask {
public:

  ReindexColumnCopyTask(
    LogicalRegion column,
    IndexPartition row_partition,
    LogicalRegion new_rects_lr,
    LogicalRegion new_col_lr)
    : m_column(column)
    , m_row_partition(row_partition)
    , m_new_rects_lr(new_rects_lr)
    , m_new_col_lr(new_col_lr) {
  }

  void
  dispatch(Context ctx, Runtime* runtime) {

    IndexSpace row_colors =
      runtime->get_index_partition_color_space_name(m_row_partition);

    // use partition of m_new_rects_lr by m_row_partition to get partition of
    // m_new_col_lr index space
    LogicalPartition new_rects_lp =
      runtime->get_logical_partition(ctx, m_new_rects_lr, m_row_partition);

    IndexPartition new_col_ip =
      runtime->create_partition_by_image_range(
        ctx,
        m_new_col_lr.get_index_space(),
        new_rects_lp,
        m_new_rects_lr,
        ReindexColumnTask::row_rects_fid,
        row_colors);

    LogicalPartition new_col_lp =
      runtime->get_logical_partition(ctx, m_new_col_lr, new_col_ip);

    // we now have partitions over the same color space on both m_column and
    // m_new_col_lr

    RegionRequirement src_req(
      runtime->get_logical_partition(ctx, m_column, m_row_partition),
      0,
      READ_ONLY,
      EXCLUSIVE,
      m_column);
    src_req.add_field(Column::value_fid);

    RegionRequirement dst_req(
      new_col_lp,
      0,
      WRITE_DISCARD,
      EXCLUSIVE,
      m_new_col_lr);
    dst_req.add_field(Column::value_fid);

    IndexCopyLauncher copier(row_colors);
    copier.add_copy_requirements(src_req, dst_req);

    runtime->issue_copy_operation(ctx, copier);
  }

private:

  LogicalRegion m_column;

  IndexPartition m_row_partition;

  LogicalRegion m_new_rects_lr;

  LogicalRegion m_new_col_lr;
};

size_t
ReindexColumnTask::TaskArgs::serialized_size() const {
  return
    sizeof(allow_rows) + sizeof(row_partition) + col.legion_buffer_size();
}

size_t
ReindexColumnTask::TaskArgs::serialize(void* buffer) const {
  char* buff = static_cast<char*>(buffer);
  memcpy(buff, &allow_rows, sizeof(allow_rows));
  buff += sizeof(allow_rows);
  memcpy(buff, &row_partition, sizeof(row_partition));
  buff += sizeof(row_partition);
  buff += col.legion_serialize(buff);
  return buff - static_cast<char*>(buffer);
}

size_t
ReindexColumnTask::TaskArgs::deserialize(
  ReindexColumnTask::TaskArgs& val,
  const void* buffer) {

  const char* buff = static_cast<const char*>(buffer);
  val.allow_rows = *reinterpret_cast<const decltype(val.allow_rows)*>(buff);
  buff += sizeof(val.allow_rows);
  val.row_partition =
    *reinterpret_cast<const decltype(val.row_partition)*>(buff);
  buff += sizeof(val.row_partition);
  buff += val.col.legion_deserialize(buff);
  return buff - static_cast<const char*>(buffer);
}

ReindexColumnTask::ReindexColumnTask(
  const std::shared_ptr<Column>& col,
  ssize_t row_axis_offset,
  const std::vector<Future>& ixcol_futures,
  bool allow_rows) {

  // get column partition down to row axis
  assert(row_axis_offset >= 0);
  std::vector<int> col_part_axes;
  std::copy(
    col->axes().begin(),
    col->axes().begin() + row_axis_offset,
    std::back_inserter(col_part_axes));
  m_partition = col->partition_on_axes(col_part_axes);

  TaskArgs args {allow_rows, m_partition->index_partition(),
                 col->generator_args()};
  m_args_buffer = std::make_unique<char[]>(args.serialized_size());
  args.serialize(m_args_buffer.get());
  m_launcher =
    Legion::TaskLauncher(
      TASK_ID,
      TaskArgument(m_args_buffer.get(), sizeof(args.serialized_size())));

  // add the futures for all the index columns at once (Legion Futures are not
  // allowed to escape the context in which they were created)
  std::for_each(
    ixcol_futures.begin(),
    ixcol_futures.end(),
    [this](auto& f) { m_launcher.add_future(f); });
}

Future
ReindexColumnTask::dispatch(Legion::Context ctx, Legion::Runtime* runtime) {
  return runtime->execute_task(ctx, m_launcher);
}

template <int OLDDIM, int NEWDIM>
ColumnGenArgs
reindex_column(
  const ReindexColumnTask::TaskArgs& args,
  const std::vector<ColumnGenArgs>& ixcols,
  Context ctx,
  Runtime *runtime) {

  Rect<OLDDIM> col_domain =
    runtime->get_index_space_domain(ctx, args.col.values.get_index_space());
  // we use the name "rows_is" for the index space at or above the "ROW" axis
  IndexSpace rows_is =
    runtime->get_index_partition_color_space_name(ctx, args.row_partition);
  // logical region over rows_is with a field for the rectangle in the new
  // column index space for each value in row_is
  auto new_rects_fs = runtime->create_field_space(ctx);
  {
    auto fa = runtime->create_field_allocator(ctx, new_rects_fs);
    fa.allocate_field(sizeof(Rect<NEWDIM>), ReindexColumnTask::row_rects_fid);
  }
  auto new_rects_lr =
    runtime->create_logical_region(ctx, rows_is, new_rects_fs);

  // task to compute new index space rectangle for each row in column
  std::vector<LogicalRegion> ixcs;
  std::transform(
    ixcols.begin(),
    ixcols.end(),
    std::back_inserter(ixcs),
    [](auto& ic){ return ic.values; });
  ComputeRectanglesTask
    new_rects_task(
      new_rects_lr,
      args.allow_rows,
      args.row_partition,
      ixcs,
      {},
      {});
  new_rects_task.dispatch(ctx, runtime);

  // create the new index space via create_partition_by_image_range based on
  // rows_rect_lr; for this, we need the bounding index space first
  Rect<NEWDIM> new_bounds;
  std::vector<int> new_axes(NEWDIM);
  {
    // start with axes above original row axis
    auto d = args.row_partition.get_dim() - 1;
    int i = 0; // index in new_bounds
    int j = 0; // index in col arrays
    while (i < d) {
      new_bounds.lo[i] = col_domain.lo[j];
      new_bounds.hi[i] = col_domain.hi[j];
      new_axes[i] = args.col.axes[j];
      ++i; ++j;
    }
    // append new index axes
    for (size_t k = 0; k < ixcols.size(); ++k) {
      Rect<1> ix_domain =
        runtime->get_index_space_domain(ixcols[k].values.get_index_space());
      new_bounds.lo[i] = ix_domain.lo[0];
      new_bounds.hi[i] = ix_domain.hi[0];
      new_axes[i] = ixcols[k].axes[0];
      ++i;
    }
    // append row axis, if allowed
    if (args.allow_rows) {
      new_bounds.lo[i] = col_domain.lo[j];
      new_bounds.hi[i] = col_domain.hi[j];
      assert(args.col.axes[j] == 0);
      new_axes[i] = 0;
      ++i;
    }
    ++j;
    // append remaining (ctds element-level) axes
    while (i < NEWDIM) {
      assert(j < OLDDIM);
      new_bounds.lo[i] = col_domain.lo[j];
      new_bounds.hi[i] = col_domain.hi[j];
      new_axes[i] = args.col.axes[j];
      ++i; ++j;
    }
  }
  auto new_bounds_is = runtime->create_index_space(ctx, new_bounds);

  // to do this, we need a logical partition of new_rects_lr, which will
  // comprise a single index subspace
  IndexSpaceT<1> unitary_cs = runtime->create_index_space(ctx, Rect<1>(0, 0));
  // auto row_rects_ip =
  //   runtime->create_pending_partition(ctx, bounds_is, row_rects_cs);
  // runtime->create_index_space_union(
  //   ctx,
  //   row_rects_ip,
  //   Point<1>(0),
  //   {rows_is});
  auto unitary_rows_ip =
    runtime->create_equal_partition(ctx, rows_is, unitary_cs);
  auto unitary_new_rects_lp =
    runtime->get_logical_partition(ctx, new_rects_lr, unitary_rows_ip);
  IndexPartitionT<NEWDIM> new_bounds_ip(
    runtime->create_partition_by_image_range(
      ctx,
      new_bounds_is,
      unitary_new_rects_lp,
      new_rects_lr,
      ReindexColumnTask::row_rects_fid,
      unitary_cs));
  IndexSpaceT<NEWDIM> new_col_is(
    runtime->get_index_subspace(new_bounds_ip, 0));

  // finally, we create the new column logical region
  auto new_col_fs = runtime->create_field_space(ctx);
  {
    auto fa = runtime->create_field_allocator(ctx, new_col_fs);
    add_field(args.col.datatype, fa, Column::value_fid);
  }
  auto new_col_lr = runtime->create_logical_region(ctx, new_col_is, new_col_fs);

  // copy values from the column logical region to new_col_lr
  ReindexColumnCopyTask
    copy_task(args.col.values, args.row_partition, new_rects_lr, new_col_lr);
  copy_task.dispatch(ctx, runtime);

  ColumnGenArgs
    result {args.col.name, args.col.datatype, new_axes, new_col_lr,
            args.col.keywords};

  // FIXME: clean up
  return result;
}

ColumnGenArgs
ReindexColumnTask::base_impl(
  const Legion::Task* task,
  const std::vector<Legion::PhysicalRegion>&,
  Legion::Context ctx,
  Legion::Runtime *runtime) {

  TaskArgs args;
  ReindexColumnTask::TaskArgs::deserialize(args, task->args);
  std::vector<ColumnGenArgs> ixcols;
  std::transform(
    task->futures.begin(),
    task->futures.end(),
    std::back_inserter(ixcols),
    [](auto& f) { return f.template get_result<ColumnGenArgs>(); });

  auto olddim = args.row_partition.get_dim();
  auto eltdim =
    olddim
    - runtime->get_index_partition_color_space(ctx, args.row_partition)
      .get_dim();
  auto newdim = ixcols.size() + eltdim + (args.allow_rows ? 1 : 0);

#define REINDEX_COLUMN(OLDDIM, NEWDIM) \
  case (OLDDIM * LEGMS_MAX_DIM + NEWDIM): { \
    return \
      reindex_column<OLDDIM, NEWDIM>(           \
        args,                                   \
        ixcols,                                 \
        ctx,                                    \
        runtime);                               \
    break;                                      \
  }

  switch (olddim * LEGMS_MAX_DIM + newdim) {
    LEGMS_FOREACH_MN(REINDEX_COLUMN);
  default:
    assert(false);
    return ColumnGenArgs {}; // keep compiler happy
    break;
  }
}

void
ReindexColumnTask::register_task(Legion::Runtime* runtime) {
  TASK_ID = runtime->generate_library_task_ids("legms::ReindexColumnTask", 1);
  TaskVariantRegistrar registrar(TASK_ID, TASK_NAME);
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  registrar.set_inner();
  registrar.set_idempotent();
  registrar.set_replicable();
  runtime->register_task_variant<ColumnGenArgs,base_impl>(registrar);
}

void
Table::register_tasks(Legion::Runtime* runtime) {
  IndexColumnTask::register_task(runtime);
  ReindexedTableTask::register_task(runtime);
  IndexAccumulateTasks::register_tasks(runtime);
  ComputeRectanglesTask::register_task(runtime);
  ReindexColumnTask::register_task(runtime);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
