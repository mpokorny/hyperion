#include <algorithm>
#include <array>
#include <numeric>
#include <tuple>
#include <vector>

#include "legms.h"
#include "Column.h"
#include "Table.h"
#include "TableBuilder.h"

#if USE_HDF5
# include <hdf5.h>
#endif

using namespace legms;
using namespace std;
using namespace Legion;

#undef HIERARCHICAL_COMPUTE_RECTANGLES
#undef WORKAROUND

size_t
TableGenArgs::legion_buffer_size(void) const {
  size_t result = name.size() + 1;
  result += axes_uid.size() + 1;
  result += vector_serdez<int>::serialized_size(index_axes);
  result =
    accumulate(
      col_genargs.begin(),
      col_genargs.end(),
      result + sizeof(size_t),
      [](auto& acc, auto& cg) {
        return acc + cg.legion_buffer_size();
      });
  result += sizeof(LogicalRegion);
  result += vector_serdez<legms::TypeTag>::serialized_size(keyword_datatypes);
  return result;
}

size_t
TableGenArgs::legion_serialize(void *buffer) const {
  char* buff = static_cast<char*>(buffer);

  size_t s = name.size() + 1;
  memcpy(buff, name.c_str(), s);
  buff += s;

  s = axes_uid.size() + 1;
  memcpy(buff, axes_uid.c_str(), s);
  buff += s;

  buff += vector_serdez<int>::serialize(index_axes, buff);

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

  buff += vector_serdez<legms::TypeTag>::serialize(keyword_datatypes, buff);

  return buff - static_cast<char*>(buffer);
}

size_t
TableGenArgs::legion_deserialize(const void *buffer) {
  const char *buff = static_cast<const char*>(buffer);

  name = buff;
  buff += name.size() + 1;

  axes_uid = buff;
  buff += axes_uid.size() + 1;

  buff += vector_serdez<int>::deserialize(index_axes, buff);

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

  buff += vector_serdez<legms::TypeTag>::deserialize(keyword_datatypes, buff);

  return buff - static_cast<const char*>(buffer);
}

std::unique_ptr<Table>
TableGenArgs::operator()(Legion::Context ctx, Legion::Runtime* runtime) const {

  return
    std::make_unique<Table>(
      ctx,
      runtime,
      name,
      axes_uid,
      index_axes,
      col_genargs,
      keywords,
      keyword_datatypes);
}

TaskID ReindexedTableTask::TASK_ID;

ReindexedTableTask::ReindexedTableTask(
  const std::string& name,
  const std::string& axes_uid,
  const vector<int>& index_axes,
  LogicalRegion keywords_region,
  const vector<Future>& reindexed) {

  // reuse TableGenArgsSerializer to pass task arguments
  TableGenArgs args;
  args.name = name;
  args.axes_uid = axes_uid;
  args.index_axes = index_axes;
  args.keywords = keywords_region;

  size_t buffsz = args.legion_buffer_size();
  m_args = make_unique<char[]>(buffsz);
  args.legion_serialize(m_args.get());
  m_launcher =
    TaskLauncher(
      ReindexedTableTask::TASK_ID,
      TaskArgument(m_args.get(), buffsz));
  for_each(
    reindexed.begin(),
    reindexed.end(),
    [this](auto& f) { m_launcher.add_future(f); });
}

void
ReindexedTableTask::register_task(Runtime* runtime) {
  TASK_ID = runtime->generate_library_task_ids("legms::ReindexedTableTask", 1);
  TaskVariantRegistrar registrar(TASK_ID, TASK_NAME, false);
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  // registrar.set_inner();
  // registrar.set_idempotent();
  // registrar.set_replicable();
  runtime->register_task_variant<TableGenArgs,base_impl>(registrar);
}

Future
ReindexedTableTask::dispatch(Context ctx, Runtime* runtime) {
  return runtime->execute_task(ctx, m_launcher);
}

TableGenArgs
ReindexedTableTask::base_impl(
  const Task* task,
  const vector<PhysicalRegion>&,
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

template <typename T, int DIM>
using ROAccessor =
  FieldAccessor<
  READ_ONLY,
  T,
  DIM,
  coord_t,
  AffineAccessor<T, DIM, coord_t>,
  false>;

template <typename T, int DIM>
using WDAccessor =
  FieldAccessor<
  WRITE_DISCARD,
  T,
  DIM,
  coord_t,
  AffineAccessor<T, DIM, coord_t>,
  false>;

template <typename T, int DIM>
using WOAccessor =
  FieldAccessor<
  WRITE_ONLY,
  T,
  DIM,
  coord_t,
  AffineAccessor<T, DIM, coord_t>,
  false>;

template <typename T>
class IndexAccumulateTask {
public:

  typedef DataType<ValueType<T>::DataType> DT;

  static TaskID TASK_ID;
  static char TASK_NAME[40];

  IndexAccumulateTask(const RegionRequirement& col_req) {

    m_launcher =
      IndexTaskLauncher(
        TASK_ID,
        col_req.region.get_index_space(),
        TaskArgument(NULL, 0),
        ArgumentMap());
    m_launcher.add_region_requirement(
      RegionRequirement(
        col_req.region,
        0,
        READ_ONLY,
        EXCLUSIVE,
        col_req.region));
    m_launcher.add_field(0, Column::value_fid);
  }

  Future
  dispatch(Context ctx, Runtime* runtime) {
    return runtime->execute_index_space(ctx, m_launcher, DT::af_redop_id);
  }

  static void
  register_task(Runtime* runtime, int tid0) {
    TASK_ID = tid0 + DT::id;
    std::string tname =
      std::string("index_accumulate_task<") + DT::s + std::string(">");
    strncpy(TASK_NAME, tname.c_str(), sizeof(TASK_NAME));
    TASK_NAME[sizeof(TASK_NAME) - 1] = '\0';
    TaskVariantRegistrar registrar(TASK_ID, TASK_NAME, false);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    // registrar.set_leaf();
    // registrar.set_idempotent();
    // registrar.set_replicable();
    runtime->register_task_variant<acc_field_redop_rhs<T>, base_impl>(
      registrar);
  }

  static acc_field_redop_rhs<T>
  base_impl(
    const Task* task,
    const vector<PhysicalRegion>& regions,
    Context,
    Runtime*) {

    const FieldAccessor<
      READ_ONLY,
      T,
      1,
      coord_t,
      AffineAccessor<T, 1, coord_t>,
      false> values(regions[0], Column::value_fid);

    vector<DomainPoint> pt { task->index_point };
    return acc_field_redop_rhs<T>{{make_tuple(values[pt[0][0]], pt)}};
  }

private:

  IndexTaskLauncher m_launcher;
};

template <typename T>
TaskID IndexAccumulateTask<T>::TASK_ID;

template <typename T>
char IndexAccumulateTask<T>::TASK_NAME[40];

class IndexAccumulateTasks {
public:

  static void
  register_tasks(Runtime *runtime) {
    auto tid0 =
      runtime->generate_library_task_ids(
        "legms::IndexAccumulateTasks",
        NUM_LEGMS_DATATYPES);

#define REG_TASK(DT)                                                    \
    IndexAccumulateTask<DataType<DT>::ValueType>::register_task(runtime, tid0);

    LEGMS_FOREACH_DATATYPE(REG_TASK);
#undef REG_TASK
  }
};

TaskID IndexColumnTask::TASK_ID;

IndexColumnTask::IndexColumnTask(const shared_ptr<Column>& column, int axis) {

  ColumnGenArgs args = column->generator_args();
  args.axes.clear();
  args.axes.push_back(axis);
  size_t buffsz = args.legion_buffer_size();
  m_args = make_unique<char[]>(buffsz);
  args.legion_serialize(m_args.get());
  m_launcher =
    TaskLauncher(
      TASK_ID,
      TaskArgument(m_args.get(), buffsz));
  m_launcher.add_region_requirement(
    RegionRequirement(
      column->logical_region(),
      READ_ONLY,
      EXCLUSIVE,
      column->logical_region()));
  m_launcher.add_field(0, Column::value_fid);
}

void
IndexColumnTask::register_task(Runtime* runtime) {
  TASK_ID = runtime->generate_dynamic_task_id();
  TaskVariantRegistrar registrar(TASK_ID, TASK_NAME, false);
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  // registrar.set_idempotent();
  // registrar.set_replicable();
  runtime->register_task_variant<ColumnGenArgs,base_impl>(registrar);
}

Future
IndexColumnTask::dispatch(Context ctx, Runtime* runtime) {
  return runtime->execute_task(ctx, m_launcher);
}

template <typename T>
LogicalRegion
index_column(
  const Task*,
  Context ctx,
  Runtime *runtime,
  legms::TypeTag dt,
  const RegionRequirement& col_req) {

  // launch index space task on input region to compute accumulator value
  IndexAccumulateTask<T> acc_index_task(col_req);
  Future acc_future = acc_index_task.dispatch(ctx, runtime);
  // TODO: create and initialize the resulting LogicalRegion in another task, so
  // we don't have to wait on this future explicitly
  auto acc =
    acc_future.get_result<typename acc_field_redop<T>::LHS>();

  LogicalRegionT<1> result_lr;
  if (acc.size() > 0) {
    auto result_fs = runtime->create_field_space(ctx);
    {
      auto fa = runtime->create_field_allocator(ctx, result_fs);
      add_field(dt, fa, IndexColumnTask::value_fid);
      fa.allocate_field(
        sizeof(vector<DomainPoint>),
        IndexColumnTask::indices_fid,
        OpsManager::V_DOMAIN_POINT_SID);
    }
    IndexSpaceT<1> result_is =
      runtime->create_index_space(ctx, Rect<1>(0, acc.size() - 1));
    result_lr = runtime->create_logical_region(ctx, result_is, result_fs);

    // transfer values and row numbers from acc_lr to result_lr
    RegionRequirement result_req(result_lr, WRITE_ONLY, EXCLUSIVE, result_lr);
    result_req.add_field(IndexColumnTask::value_fid);
    result_req.add_field(IndexColumnTask::indices_fid);
    PhysicalRegion result_pr = runtime->map_region(ctx, result_req);
    const WOAccessor<T, 1> values(result_pr, IndexColumnTask::value_fid);
    const WOAccessor<vector<DomainPoint>, 1>
      rns(result_pr, IndexColumnTask::indices_fid);
    for (size_t i = 0; i < acc.size(); ++i) {
      ::new (rns.ptr(i)) vector<DomainPoint>;
      tie(values[i], rns[i]) = acc[i];
    }

    runtime->unmap_region(ctx, result_pr);
    // TODO: keep?
    //runtime->destroy_field_space(ctx, result_fs);
    //runtime->destroy_index_space(ctx, result_is);
  }
  return result_lr;
}

ColumnGenArgs
IndexColumnTask::base_impl(
  const Task* task,
  const vector<PhysicalRegion>&,
  Context ctx,
  Runtime *runtime) {

  ColumnGenArgs result;
  result.legion_deserialize(task->args);

#define ICR(DT)                                 \
  case DT:                                      \
    result.values =                             \
      index_column<DataType<DT>::ValueType>(    \
        task,                                   \
        ctx,                                    \
        runtime,                                \
        DT,                                     \
        task->regions[0]);                      \
    break;

  switch (result.datatype) {
    LEGMS_FOREACH_DATATYPE(ICR);
  default:
    assert(false);
    break;
  }
  return result;
}

#ifdef HIERARCHICAL_COMPUTE_RECTANGLES

class ComputeRectanglesTask {
public:

  static TaskID TASK_ID;
  static constexpr const char* TASK_NAME = "compute_rectangles_task";

  ComputeRectanglesTask(
    bool allow_rows,
    IndexPartition row_partition,
    const vector<LogicalRegion>& ix_columns,
    LogicalRegion new_rects,
    const vector<PhysicalRegion>& parent_regions,
    const vector<coord_t>& ix0,
    const vector<DomainPoint>& rows) {

    TaskArgs args{allow_rows, row_partition, ix0, rows};
    auto idx = args.ix0.size();
    m_args_buffer = make_unique<char[]>(args.serialized_size());
    args.serialize(m_args_buffer.get());
    m_launcher =
      IndexTaskLauncher(
        TASK_ID,
        ix_columns[idx].get_index_space(),
        TaskArgument(m_args_buffer.get(), args.serialized_size()),
        ArgumentMap());

    bool has_parent = false/*parent_regions.size() > 0*/;
    cout << "ix_columns " << ix_columns.size()
         << "; idx " << idx
         << " (";
    for_each(
      ix0.begin(),
      ix0.end(),
      [](auto& d) { cout << d << " "; });
    cout << ")" << endl;

    for (size_t i = 0; i < ix_columns.size(); ++i) {
      RegionRequirement req;
      if (i == idx)
        req =
          RegionRequirement(
            ix_columns[i],
            0,
            READ_ONLY,
            EXCLUSIVE,
            (has_parent
             ? parent_regions[i].get_logical_region()
             : ix_columns[i]));
      else
        req =
          RegionRequirement(
            ix_columns[i],
            READ_ONLY,
            EXCLUSIVE,
            (has_parent
             ? parent_regions[i].get_logical_region()
             : ix_columns[i]));
      req.add_field(IndexColumnTask::indices_fid);
      m_launcher.add_region_requirement(req);
    }

    RegionRequirement req(
      new_rects,
      WRITE_DISCARD,
      SIMULTANEOUS,
      (has_parent ? parent_regions.back().get_logical_region() : new_rects));
    req.add_field(ReindexColumnTask::row_rects_fid);
    m_launcher.add_region_requirement(req);
  };

  void
  dispatch(Context ctx, Runtime* runtime) {
    runtime->execute_index_space(ctx, m_launcher);
  }

  static void
  base_impl(
    const Task* task,
    const vector<PhysicalRegion>& regions,
    Context ctx,
    Runtime *runtime) {

    TaskArgs args;
    TaskArgs::deserialize(args, static_cast<const void *>(task->args));

    const FieldAccessor<
      READ_ONLY,
      vector<DomainPoint>,
      1,
      coord_t,
      AffineAccessor<vector<DomainPoint>, 1, coord_t>,
      false> rows(regions[args.ix0.size()], IndexColumnTask::indices_fid);

    auto pt = task->index_point[0];
    args.ix0.push_back(pt);
    if (args.ix0.size() == 1)
      args.rows = rows[pt];
    else
      args.rows = intersection(args.rows, rows[pt]);
    if (args.rows.size() > 0) {
      if (args.ix0.size() < regions.size() - 1) {
        // start task at next index level
        vector<LogicalRegion> col_lrs;
        for (size_t i = 0; i < regions.size() - 1; ++i)
          col_lrs.push_back(regions[i].get_logical_region());
        ComputeRectanglesTask task(
          args.allow_rows,
          args.row_partition,
          col_lrs,
          regions.back().get_logical_region(),
          regions,
          args.ix0,
          args.rows);
        task.dispatch(ctx, runtime);
      } else {
        // at bottom of indexes, write results to "new_rects" region

        auto rowdim = args.rows[0].get_dim();
        auto rectdim =
          regions.size() - 1 + args.row_partition.get_dim() - rowdim
          + (args.allow_rows ? 1 : 0);

        if (args.allow_rows || args.rows.size() == 1) {

#define WRITE_RECTS(ROWDIM, RECTDIM)                                    \
          case (ROWDIM * LEGION_MAX_DIM + RECTDIM): {                   \
            const FieldAccessor<                                        \
              WRITE_DISCARD, \
              Rect<RECTDIM>, \
              ROWDIM, \
              coord_t, \
              AffineAccessor<Rect<RECTDIM>, ROWDIM, coord_t>, \
              false> rects(regions.back(), ReindexColumnTask::row_rects_fid); \
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

          switch (rowdim * LEGION_MAX_DIM + rectdim) {
            LEGMS_FOREACH_MN(WRITE_RECTS);
          default:
            assert(false);
            break;
          }
#undef WRITE_RECTS

        }
      }
    }
  }

  static void
  register_task(Runtime* runtime) {
    TASK_ID =
      runtime->generate_library_task_ids("legms::ComputeRectanglesTask", 1);
    TaskVariantRegistrar registrar(TASK_ID, TASK_NAME, false);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    // registrar.set_idempotent();
    // registrar.set_replicable();
    runtime->register_task_variant<base_impl>(registrar);
  }

private:

  struct TaskArgs {
    bool allow_rows;
    IndexPartition row_partition;
    vector<coord_t> ix0;
    vector<DomainPoint> rows;

    size_t
    serialized_size() const {
      return
        sizeof(allow_rows) + sizeof(row_partition)
        + vector_serdez<decltype(ix0)::value_type>::serialized_size(ix0)
        + vector_serdez<decltype(rows)::value_type>::serialized_size(rows);
    }

    size_t
    serialize(void *buffer) const {
      size_t result = 0;
      char* buff = static_cast<char*>(buffer);
      memcpy(buff + result, &allow_rows, sizeof(allow_rows));
      result += sizeof(allow_rows);
      memcpy(buff + result, &row_partition, sizeof(row_partition));
      result += sizeof(row_partition);
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
      memcpy(&val.allow_rows, buff + result, sizeof(allow_rows));
      result += sizeof(allow_rows);
      memcpy(&val.row_partition, buff + result, sizeof(row_partition));
      result += sizeof(row_partition);
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

  unique_ptr<char[]> m_args_buffer;

  IndexTaskLauncher m_launcher;

  static vector<DomainPoint>
  intersection(
    const vector<DomainPoint>& first,
    const vector<DomainPoint>& second) {

    vector<DomainPoint> result;
    set_intersection(
      first.begin(),
      first.end(),
      second.begin(),
      second.end(),
      back_inserter(result));
    return result;
  }
};

#else // !HIERARCHICAL_COMPUTE_RECTANGLES

class ComputeRectanglesTask {
public:

  static TaskID TASK_ID;
  static constexpr const char* TASK_NAME = "compute_rectangles_task";

  ComputeRectanglesTask(
    bool allow_rows,
    IndexPartition row_partition,
    const vector<LogicalRegion>& ix_columns,
    LogicalRegion new_rects)
    : m_allow_rows(allow_rows)
    , m_row_partition(row_partition)
    , m_ix_columns(ix_columns)
    , m_new_rects_lr(new_rects) {
  };

  void
  dispatch(Context ctx, Runtime* runtime) {

    IndexTaskLauncher launcher;
    unique_ptr<char[]> args_buffer;
    TaskArgs args{m_allow_rows, m_row_partition};
    args_buffer = make_unique<char[]>(args.serialized_size());
    args.serialize(args_buffer.get());

    //LogicalRegion new_rects_lr = m_new_rects.get_logical_region();
    // AcquireLauncher acquire(new_rects_lr, new_rects_lr, m_new_rects);
    // acquire.add_field(ReindexColumnTask::row_rects_fid);
    // PhaseBarrier acquired = runtime->create_phase_barrier(ctx, 1);
    // acquire.add_arrival_barrier(acquired);
    // runtime->issue_acquire(ctx, acquire);

    //PhaseBarrier released;
    Domain bounds;

#define INIT_LAUNCHER(DIM)                                      \
    case DIM: {                                                 \
      Rect<DIM> rect;                                           \
      for (size_t i = 0; i < DIM; ++i) {                        \
        IndexSpaceT<1> cis(m_ix_columns[i].get_index_space());  \
        Rect<1> dom = runtime->get_index_space_domain(cis).bounds;  \
        rect.lo[i] = dom.lo[0];                                 \
        rect.hi[i] = dom.hi[0];                                 \
      }                                                         \
      bounds = rect;                                            \
      break;                                                    \
    }

    switch (m_ix_columns.size()) {
      LEGMS_FOREACH_N(INIT_LAUNCHER);
    default:
      assert(false);
      break;
    }
#undef INIT_LAUNCHER

    /*released = runtime->create_phase_barrier(ctx, rect.volume());*/ 
    launcher = 
      IndexTaskLauncher(                                          
        TASK_ID, 
        bounds,                                                   
        TaskArgument(args_buffer.get(), args.serialized_size()), 
        ArgumentMap()); 
    /*launcher.add_wait_barrier(acquired);                       */
    /*launcher.add_arrival_barrier(released);                    */

    for_each(
      m_ix_columns.begin(),
      m_ix_columns.end(),
      [&launcher](auto& lr) {
        RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
        req.add_field(IndexColumnTask::indices_fid);
        launcher.add_region_requirement(req);
      });

    RegionRequirement
      req(m_new_rects_lr, WRITE_DISCARD, SIMULTANEOUS, m_new_rects_lr);
    req.add_field(ReindexColumnTask::row_rects_fid);
    launcher.add_region_requirement(req);

    runtime->execute_index_space(ctx, launcher);

    // PhaseBarrier complete = runtime->advance_phase_barrier(ctx, released);
    // ReleaseLauncher release(new_rects_lr, new_rects_lr, m_new_rects);
    // release.add_field(ReindexColumnTask::row_rects_fid);
    // release.add_wait_barrier(complete);
    // runtime->issue_release(ctx, release);
  }

  static void
  base_impl(
    const Task* task,
    const vector<PhysicalRegion>& regions,
    Context ctx,
    Runtime *runtime) {

    TaskArgs args;
    TaskArgs::deserialize(args, static_cast<const void *>(task->args));

    typedef const FieldAccessor<
      READ_ONLY,
      vector<DomainPoint>,
      1,
      coord_t,
      AffineAccessor<vector<DomainPoint>, 1, coord_t>,
      false> rows_acc_t;

    auto ixdim = regions.size() - 1;

    vector<DomainPoint> common_rows;
    {
      rows_acc_t rows(regions[0], IndexColumnTask::indices_fid);
      common_rows = rows[task->index_point[0]];
    }
    for (size_t i = 1; i < ixdim; ++i) {
      rows_acc_t rows(regions[i], IndexColumnTask::indices_fid);
      common_rows = intersection(common_rows, rows[task->index_point[i]]);
    }

#define WRITE_RECTS(ROWDIM, RECTDIM)                                    \
    case (ROWDIM * LEGION_MAX_DIM + RECTDIM): {                         \
      const FieldAccessor<                                              \
        WRITE_DISCARD, \
        Rect<RECTDIM>, \
        ROWDIM, \
        coord_t, \
        AffineAccessor<Rect<RECTDIM>, ROWDIM, coord_t>, \
        false> rects(regions.back(), ReindexColumnTask::row_rects_fid); \
                                                                        \
      for (size_t i = 0; i < common_rows.size(); ++i) {                 \
        Domain row_d =                                                  \
          runtime->get_index_space_domain(                              \
            ctx,                                                        \
            runtime->get_index_subspace(                                \
              ctx,                                                      \
              args.row_partition,                                       \
              common_rows[i]));                                         \
        Rect<RECTDIM> row_rect;                                         \
        size_t j = 0;                                                   \
        for (; j < ixdim; ++j) {                                        \
          row_rect.lo[j] = task->index_point[j];                        \
          row_rect.hi[j] = task->index_point[j];                        \
        }                                                               \
        if (args.allow_rows) {                                          \
          row_rect.lo[j] = i;                                           \
          row_rect.hi[j] = i;                                           \
          ++j;                                                          \
        }                                                               \
        for (; j < RECTDIM; ++j) {                                      \
          row_rect.lo[j] = row_d.lo()[j - (RECTDIM - ROWDIM)];          \
          row_rect.hi[j] = row_d.hi()[j - (RECTDIM - ROWDIM)];          \
        }                                                               \
        rects[common_rows[i]] = row_rect;                               \
      }                                                                 \
      break;                                                            \
    }
    if (common_rows.size() > 0) {
      auto rowdim = common_rows[0].get_dim();
      auto rectdim =
        ixdim + args.row_partition.get_dim()
        - rowdim + (args.allow_rows ? 1 : 0);
      if (args.allow_rows || common_rows.size() == 1) {
        switch (rowdim * LEGION_MAX_DIM + rectdim) {
          LEGMS_FOREACH_MN(WRITE_RECTS);
        default:
          assert(false);
          break;
        }
      }
    }

#undef WRITE_RECTS

  }

  static void
  register_task(Runtime* runtime) {
    TASK_ID =
      runtime->generate_library_task_ids("legms::ComputeRectanglesTask", 1);
    TaskVariantRegistrar registrar(TASK_ID, TASK_NAME, false);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    // registrar.set_idempotent();
    // registrar.set_replicable();
    // registrar.set_leaf();
    runtime->register_task_variant<base_impl>(registrar);
  }

private:

  struct TaskArgs {
    bool allow_rows;
    IndexPartition row_partition;

    size_t
    serialized_size() const {
      return sizeof(allow_rows) + sizeof(row_partition);
    }

    size_t
    serialize(void *buffer) const {
      size_t result = 0;
      char* buff = static_cast<char*>(buffer);
      memcpy(buff + result, &allow_rows, sizeof(allow_rows));
      result += sizeof(allow_rows);
      memcpy(buff + result, &row_partition, sizeof(row_partition));
      result += sizeof(row_partition);
      return result;
    }

    static size_t
    deserialize(TaskArgs& val, const void *buffer) {
      size_t result = 0;
      const char* buff = static_cast<const char*>(buffer);
      memcpy(&val.allow_rows, buff + result, sizeof(allow_rows));
      result += sizeof(allow_rows);
      memcpy(&val.row_partition, buff + result, sizeof(row_partition));
      result += sizeof(row_partition);
      return result;
    }
  };

  bool m_allow_rows;

  IndexPartition m_row_partition;

  vector<LogicalRegion> m_ix_columns;

  LogicalRegion m_new_rects_lr;

  static vector<DomainPoint>
  intersection(
    const vector<DomainPoint>& first,
    const vector<DomainPoint>& second) {

    vector<DomainPoint> result;
    set_intersection(
      first.begin(),
      first.end(),
      second.begin(),
      second.end(),
      back_inserter(result));
    return result;
  }
};

#endif

TaskID ComputeRectanglesTask::TASK_ID;

class ReindexColumnCopyTask {
public:

  static TaskID TASK_ID;
  static constexpr const char* TASK_NAME = "reindex_column_copy_task";

  ReindexColumnCopyTask(
    LogicalRegion column,
    legms::TypeTag column_dt,
    IndexPartition row_partition,
    LogicalRegion new_rects_lr,
    LogicalRegion new_col_lr)
    : m_column(column)
    , m_column_dt(column_dt)
    , m_row_partition(row_partition)
    , m_new_rects_lr(new_rects_lr)
    , m_new_col_lr(new_col_lr) {
  }

  template <legms::TypeTag DT, int DIM>
  using SA = FieldAccessor<
    READ_ONLY,
    typename DataType<DT>::ValueType,
    DIM,
    coord_t,
    AffineAccessor<typename DataType<DT>::ValueType, DIM, coord_t>,
    true>;

  template <legms::TypeTag DT, int DIM>
  using DA = FieldAccessor<
    WRITE_ONLY,
    typename DataType<DT>::ValueType,
    DIM,
    coord_t,
    AffineAccessor<typename DataType<DT>::ValueType, DIM, coord_t>,
    true>;

  template <legms::TypeTag DT>
  static void
  copy(const PhysicalRegion& src, const PhysicalRegion& dst) {

#define CPY(SRCDIM, DSTDIM)                             \
    case (SRCDIM * LEGION_MAX_DIM + DSTDIM): {          \
      const SA<DT,SRCDIM> from(src, Column::value_fid); \
      const DA<DT,DSTDIM> to(dst, Column::value_fid);   \
      DomainT<SRCDIM> src_bounds(src);                  \
      DomainT<DSTDIM> dst_bounds(dst);                  \
      PointInDomainIterator<SRCDIM> s(src_bounds);      \
      PointInDomainIterator<DSTDIM> d(dst_bounds);      \
      while (s()) {                                     \
        to[*d] = from[*s];                              \
        d++; s++;                                       \
      }                                                 \
      break;                                            \
    }

    int srcdim = src.get_logical_region().get_dim();
    int dstdim = dst.get_logical_region().get_dim();
    switch (srcdim * LEGION_MAX_DIM + dstdim) {
      LEGMS_FOREACH_MN(CPY)
    default:
        assert(false);
      break;
    }
  }

  void
  dispatch(Context ctx, Runtime* runtime) {

    // use partition of m_new_rects_lr by m_row_partition to get partition of
    // m_new_col_lr index space: FIXME
    IndexSpace new_rects_is = m_new_rects_lr.get_index_space();
    IndexPartition new_rects_ip =
      runtime->create_equal_partition(ctx, new_rects_is, new_rects_is);
    LogicalPartition new_rects_lp =
      runtime->get_logical_partition(ctx, m_new_rects_lr, new_rects_ip);

    IndexPartition new_col_ip =
      runtime->create_partition_by_image_range(
        ctx,
        m_new_col_lr.get_index_space(),
        new_rects_lp,
        m_new_rects_lr,
        ReindexColumnTask::row_rects_fid,
        new_rects_is);

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
      WRITE_ONLY,
      EXCLUSIVE,
      m_new_col_lr);
    dst_req.add_field(Column::value_fid);

    IndexTaskLauncher
      copier(
        TASK_ID,
        new_rects_is,
        TaskArgument(&m_column_dt, sizeof(m_column_dt)),
        ArgumentMap());
    copier.add_region_requirement(src_req);
    copier.add_region_requirement(dst_req);
    runtime->execute_index_space(ctx, copier);

    // IndexCopyLauncher copier(new_rects_is);
    // copier.add_copy_requirements(src_req, dst_req);

    // runtime->issue_copy_operation(ctx, copier);
  }

  static void
  base_impl(
    const Task* task,
    const vector<PhysicalRegion>& regions,
    Context,
    Runtime*) {

    legms::TypeTag dt = *static_cast<legms::TypeTag*>(task->args);

#define CPYDT(DT)                                       \
    case (DT): copy<DT>(regions[0], regions[1]); break;

    switch (dt) {
      LEGMS_FOREACH_DATATYPE(CPYDT)
    default:
        assert(false);
      break;
    }
#undef CPYDT
  }

  static void
  register_task(Runtime* runtime) {
    TASK_ID =
      runtime->generate_library_task_ids("legms::ReindexColumnCopyTask", 1);
    TaskVariantRegistrar registrar(TASK_ID, TASK_NAME, false);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    // registrar.set_idempotent();
    // registrar.set_replicable();
    runtime->register_task_variant<base_impl>(registrar);
  }

private:

  LogicalRegion m_column;

  legms::TypeTag m_column_dt;

  IndexPartition m_row_partition;

  LogicalRegion m_new_rects_lr;

  LogicalRegion m_new_col_lr;
};

TaskID ReindexColumnCopyTask::TASK_ID;

TaskID ReindexColumnTask::TASK_ID;

size_t
ReindexColumnTask::TaskArgs::serialized_size() const {
  return
    sizeof(allow_rows)
    + vector_serdez<int>::serialized_size(index_axes)
    + sizeof(row_partition)
    + col.legion_buffer_size();
}

size_t
ReindexColumnTask::TaskArgs::serialize(void* buffer) const {
  char* buff = static_cast<char*>(buffer);
  memcpy(buff, &allow_rows, sizeof(allow_rows));
  buff += sizeof(allow_rows);
  buff += vector_serdez<int>::serialize(index_axes, buff);
  *reinterpret_cast<decltype(row_partition)*>(buff) = row_partition;
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
  buff += vector_serdez<int>::deserialize(val.index_axes, buff);
  val.row_partition =
    *reinterpret_cast<const decltype(val.row_partition)*>(buff);
  buff += sizeof(val.row_partition);
  buff += val.col.legion_deserialize(buff);
  return buff - static_cast<const char*>(buffer);
}

ReindexColumnTask::ReindexColumnTask(
  const shared_ptr<Column>& col,
  ssize_t row_axis_offset,
  const vector<shared_ptr<Column>>& ixcols,
  bool allow_rows) {

  // get column partition down to row axis
  assert(row_axis_offset >= 0);
  vector<int> col_part_axes;
  copy_n(
    col->axes().begin(),
    row_axis_offset + 1,
    back_inserter(col_part_axes));
  m_partition = col->partition_on_axes(col_part_axes);

  vector<int> index_axes =
    map(
      ixcols,
      [](const auto& ixc) {
        assert(ixc->axes().size() == 1);
        return ixc->axes()[0];
      });
  TaskArgs args {allow_rows, index_axes, m_partition->index_partition(),
                 col->generator_args()};
  m_args_buffer = make_unique<char[]>(args.serialized_size());
  args.serialize(m_args_buffer.get());
  m_launcher =
    TaskLauncher(
      TASK_ID,
      TaskArgument(m_args_buffer.get(), args.serialized_size()));
  RegionRequirement
    col_req(
      col->logical_region(),
      READ_ONLY,
      EXCLUSIVE,
      col->logical_region());
  col_req.add_field(Column::value_fid);
  m_launcher.add_region_requirement(col_req);
  for_each(
    ixcols.begin(),
    ixcols.end(),
    [this](auto& ixc) {
      auto lr = ixc->logical_region();
      assert(lr != LogicalRegion::NO_REGION);
      RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
      req.add_field(IndexColumnTask::indices_fid);
      m_launcher.add_region_requirement(req);
    });
}

Future
ReindexColumnTask::dispatch(Context ctx, Runtime* runtime) {
  return runtime->execute_task(ctx, m_launcher);
}

template <int OLDDIM, int NEWDIM>
ColumnGenArgs
reindex_column(
  const ReindexColumnTask::TaskArgs& args,
  const vector<PhysicalRegion>& regions,
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

  // initialize new_rects_lr values to empty rectangles
  Rect<NEWDIM> empty;
  empty.lo[0] = 0;
  empty.hi[0] = -1;
  assert(empty.empty());
  runtime->fill_field(
    ctx,
    new_rects_lr,
    new_rects_lr,
    ReindexColumnTask::row_rects_fid,
    empty);

  // Set RegionRequirements for this task and its children
  // RegionRequirement
  //   new_rects_req(new_rects_lr, WRITE_DISCARD, SIMULTANEOUS, new_rects_lr);
  // cout << "new_rects_lr " << new_rects_lr << endl;
  // new_rects_req.add_field(ReindexColumnTask::row_rects_fid);
  // PhysicalRegion new_rects_pr = runtime->map_region(ctx, new_rects_req);

  vector<LogicalRegion> ix_lrs;
  ix_lrs.reserve(regions.size() - 1);
  std::transform(
    regions.begin() + 1,
    regions.end(),
    std::back_inserter(ix_lrs),
    [](const auto& rg) { return rg.get_logical_region(); });

  // task to compute new index space rectangle for each row in column
#ifdef HIERARCHICAL_COMPUTE_RECTANGLES
  ComputeRectanglesTask
    new_rects_task(
      args.allow_rows,
      args.row_partition,
      ix_lrs,
      new_rects_lr,
      {},
      {},
      {});
#else
  ComputeRectanglesTask
    new_rects_task(
      args.allow_rows,
      args.row_partition,
      ix_lrs,
      new_rects_lr);
#endif
  new_rects_task.dispatch(ctx, runtime);

  // create the new index space via create_partition_by_image_range based on
  // rows_rect_lr; for this, we need the bounding index space first
  Rect<NEWDIM> new_bounds;
  vector<int> new_axes(NEWDIM);
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
    for (size_t k = 0; k < ix_lrs.size(); ++k) {
      Rect<1> ix_domain =
        runtime->get_index_space_domain(ix_lrs[k].get_index_space());
      new_bounds.lo[i] = ix_domain.lo[0];
      new_bounds.hi[i] = ix_domain.hi[0];
      new_axes[i] = args.index_axes[k];
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

#ifdef WORKAROUND
  {
    RegionRequirement req(new_rects_lr, READ_ONLY, EXCLUSIVE, new_rects_lr);
    req.add_field(ReindexColumnTask::row_rects_fid);
    PhysicalRegion pr = runtime->map_region(ctx, req);

#define PRINTIT(N)     \
    case (N): { \
      const FieldAccessor< \
        READ_ONLY,\
        Rect<NEWDIM>,\
        N,\
        coord_t,\
        AffineAccessor<Rect<NEWDIM>, N, coord_t>, \
        true> rr(pr, ReindexColumnTask::row_rects_fid);\
      for (PointInDomainIterator<N> pid(runtime->get_index_space_domain(ctx, rows_is)); \
           pid();                                                       \
           pid++)                                                       \
        cout << *pid << ": " << rr[*pid] << endl;                       \
      break;                                                            \
    }
    switch (rows_is.get_dim()) {
      LEGMS_FOREACH_N(PRINTIT)
    default:
      assert(false);
      break;
    }
#undef PRINTIT
  }
  return ColumnGenArgs {};
#endif
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

  ColumnGenArgs
    result {
    args.col.name,
    args.col.axes_uid,
    args.col.datatype,
    new_axes,
    LogicalRegion::NO_REGION,
    args.col.keywords,
    args.col.keyword_datatypes};

  // if reindexing failed, new_col_is should be empty
  if (!runtime->get_index_space_domain(ctx, new_col_is).empty()) {
    // finally, we create the new column logical region
    auto new_col_fs = runtime->create_field_space(ctx);
    {
      auto fa = runtime->create_field_allocator(ctx, new_col_fs);
      add_field(args.col.datatype, fa, Column::value_fid);
    }
    auto new_col_lr =
      runtime->create_logical_region(ctx, new_col_is, new_col_fs);

    // copy values from the column logical region to new_col_lr
    ReindexColumnCopyTask
      copy_task(
        args.col.values,
        args.col.datatype,
        args.row_partition,
        new_rects_lr,
        new_col_lr);
    copy_task.dispatch(ctx, runtime);

    result.values = new_col_lr;

    // TODO: is the following OK? does new_col_lr retain needed reference?
    //runtime->destroy_field_space(ctx, new_col_fs);
  }

  //runtime->destroy_field_space(ctx, new_rects_fs);

  // runtime->destroy_index_space(ctx, unitary_cs);
  // runtime->destroy_index_partition(ctx, unitary_rows_ip);

  // TODO: are the following OK? does new_col_lr retain needed references?
  // runtime->destroy_index_space(ctx, new_bounds_is);
  // runtime->destroy_index_partition(ctx, new_bounds_ip);
  // runtime->destroy_index_space(ctx, new_col_is);

  return result;
}

ColumnGenArgs
ReindexColumnTask::base_impl(
  const Task* task,
  const vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *runtime) {

  TaskArgs args;
  ReindexColumnTask::TaskArgs::deserialize(args, task->args);

  auto olddim = args.row_partition.get_dim();
  auto eltdim =
    olddim
    - runtime->get_index_partition_color_space(ctx, args.row_partition)
    .get_dim();
  auto newdim = (regions.size() - 1) + eltdim + (args.allow_rows ? 1 : 0);

#define REINDEX_COLUMN(OLDDIM, NEWDIM)          \
  case (OLDDIM * LEGION_MAX_DIM + NEWDIM): {    \
    return                                      \
      reindex_column<OLDDIM, NEWDIM>(           \
        args,                                   \
        regions,                                \
        ctx,                                    \
        runtime);                               \
    break;                                      \
  }

  switch (olddim * LEGION_MAX_DIM + newdim) {
    LEGMS_FOREACH_MN(REINDEX_COLUMN);
  default:
    assert(false);
    return ColumnGenArgs {}; // keep compiler happy
    break;
  }
}

void
ReindexColumnTask::register_task(Runtime* runtime) {
  TASK_ID = runtime->generate_library_task_ids("legms::ReindexColumnTask", 1);
  TaskVariantRegistrar registrar(TASK_ID, TASK_NAME, false);
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  // registrar.set_inner();
  // registrar.set_idempotent();
  // registrar.set_replicable();
  runtime->register_task_variant<ColumnGenArgs,base_impl>(registrar);
}

Legion::Future/*TableGenArgs*/
Table::ireindexed(
  const std::vector<std::string>& axis_names,
  const std::vector<int>& axes,
  bool allow_rows) const {

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
  if ((index_axes().size() > 1) || (index_axes().back() != 0)) {
    TableGenArgs empty{name(), axes_uid()};
    return Legion::Future::from_value(runtime(), empty);
  }

  // for every column in table, determine which axes need indexing
  std::unordered_map<std::string, std::vector<int>> col_reindex_axes;
  std::transform(
    column_names().begin(),
    column_names().end(),
    std::inserter(col_reindex_axes, col_reindex_axes.end()),
    [this, &axis_names, &axes](auto& nm) {
      std::vector<int> ax;
      auto col_axes = column(nm)->axes();
      // skip the column if it does not have a "row" axis
      if (col_axes.back() == 0) {
        // if column is a reindexing axis, reindexing depends only on itself
        auto myaxis = column_is_axis(axis_names, nm, axes);
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
  // column values (sorted in ascending order); and at
  // IndexColumnTask::indices_fid, a sorted vector of DomainPoints in the original
  // column.
  std::unordered_map<int, Legion::Future> index_cols;
  std::for_each(
    col_reindex_axes.begin(),
    col_reindex_axes.end(),
    [this, &axis_names, &index_cols](auto& nm_ds) {
      const std::vector<int>& ds = std::get<1>(nm_ds);
      std::for_each(
        ds.begin(),
        ds.end(),
        [this, &axis_names, &index_cols](auto& d) {
          if (index_cols.count(d) == 0) {
            auto col = column(axis_names[d]);
            IndexColumnTask task(col, d);
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
          .operator()(context(), runtime()));
      }
      auto col = column(nm);
      auto col_axes = col->axes();
      auto row_axis_offset =
        std::distance(
          col_axes.begin(),
          find(col_axes.begin(), col_axes.end(), 0));
      ReindexColumnTask task(col, row_axis_offset, ixcols, allow_rows);
      return task.dispatch(context(), runtime());
    });

  // launch task that creates the reindexed table
  std::vector<int> iaxes = axes;
  if (allow_rows)
    iaxes.push_back(0);
  ReindexedTableTask
    task(name(), axes_uid(), iaxes, keywords_region(), reindexed);
  return task.dispatch(context(), runtime());
}

void
Table::register_tasks(Runtime* runtime) {
  IndexColumnTask::register_task(runtime);
  ReindexedTableTask::register_task(runtime);
  IndexAccumulateTasks::register_tasks(runtime);
  ComputeRectanglesTask::register_task(runtime);
  ReindexColumnTask::register_task(runtime);
  ReindexColumnCopyTask::register_task(runtime);
}

#ifdef USE_CASACORE

std::unique_ptr<Table>
Table::from_ms(
  Legion::Context ctx,
  Legion::Runtime* runtime,
  const std::experimental::filesystem::path& path,
  const std::unordered_set<std::string>& column_selections) {

  std::string table_name = path.filename();

#define FROM_MS_TABLE(N)                                                \
  do {                                                                  \
    if (table_name == MSTable<MS_##N>::name)                            \
      return                                                            \
        legms:: template from_ms<MS_##N>(ctx, runtime, path, column_selections); \
  } while (0);

  LEGMS_FOREACH_MSTABLE(FROM_MS_TABLE);

  // try to read as main table
  return
    legms:: template from_ms<MS_MAIN>(
      ctx,
      runtime,
      path,
      column_selections);

#undef FROM_MS_TABLE
}

#endif // USE_CASACORE

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
