#pragma GCC visibility push(default)
#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>
#pragma GCC visibility pop

#include <legion/legion_c_util.h>
#include <legms/legms.h>
#include <legms/Column.h>
#include <legms/Table.h>
#include <legms/TableBuilder.h>

#ifdef LEGMS_USE_HDF5
#pragma GCC visibility push(default)
# include <hdf5.h>
#pragma GCC visibility pop
#endif

using namespace legms;
using namespace Legion;

#undef HIERARCHICAL_COMPUTE_RECTANGLES
#undef WORKAROUND

#undef SAVE_LAYOUT_CONSTRAINT_IDS

Table::Table() {}

#ifdef LEGMS_USE_CASACORE

Table::Table(
  LogicalRegion metadata,
  LogicalRegion axes,
  LogicalRegion columns,
  const std::vector<MeasRef>& new_meas_refs,
  const MeasRefContainer& inherited_meas_refs,
  const Keywords& keywords)
  : MeasRefContainer(new_meas_refs, inherited_meas_refs)
  , metadata_lr(metadata)
  , axes_lr(axes)
  , columns_lr(columns)
  , keywords(keywords) {
}

Table::Table(
  LogicalRegion metadata,
  LogicalRegion axes,
  LogicalRegion columns,
  const std::vector<MeasRef>& new_meas_refs,
  const MeasRefContainer& inherited_meas_refs,
  Keywords&& keywords)
  : MeasRefContainer(new_meas_refs, inherited_meas_refs)
  , metadata_lr(metadata)
  , axes_lr(axes)
  , columns_lr(columns)
  , keywords(std::move(keywords)) {
}

#else

Table::Table(
  LogicalRegion metadata,
  LogicalRegion axes,
  LogicalRegion columns,
  const Keywords& keywords)
  : metadata_lr(metadata)
  , axes_lr(axes)
  , columns_lr(columns)
  , keywords(keywords) {
}

Table::Table(
  LogicalRegion metadata,
  LogicalRegion axes,
  LogicalRegion columns,
  Keywords&& keywords)
  : metadata_lr(metadata)
  , axes_lr(axes)
  , columns_lr(columns)
  , keywords(std::move(keywords)) {
}

#endif

std::string
Table::name(Context ctx, Runtime* rt) const {
  RegionRequirement req(metadata_lr, READ_ONLY, EXCLUSIVE, metadata_lr);
  req.add_field(METADATA_NAME_FID);
  auto pr = rt->map_region(ctx, req);
  std::string result(name(pr));
  rt->unmap_region(ctx, pr);
  return result;
}

const char*
Table::name(const PhysicalRegion& metadata) {
  const NameAccessor<READ_ONLY> name(metadata, METADATA_NAME_FID);
  return name[0].val;
}

std::string
Table::axes_uid(Context ctx, Runtime* rt) const {
  RegionRequirement req(metadata_lr, READ_ONLY, EXCLUSIVE, metadata_lr);
  req.add_field(METADATA_AXES_UID_FID);
  auto pr = rt->map_region(ctx, req);
  std::string result(axes_uid(pr));
  rt->unmap_region(ctx, pr);
  return result;
}

const char*
Table::axes_uid(const PhysicalRegion& metadata) {
  const AxesUidAccessor<READ_ONLY> axes_uid(metadata, METADATA_AXES_UID_FID);
  return axes_uid[0].val;
}

std::vector<int>
Table::index_axes(Context ctx, Runtime* rt) const {
  RegionRequirement req(axes_lr, READ_ONLY, EXCLUSIVE, axes_lr);
  req.add_field(AXES_FID);
  auto pr = rt->map_region(ctx, req);
  IndexSpaceT<1> is(axes_lr.get_index_space());
  DomainT<1> dom = rt->get_index_space_domain(is);
  std::vector<int> result(Domain(dom).hi()[0] + 1);
  const AxesAccessor<READ_ONLY> ax(pr, AXES_FID);
  for (PointInDomainIterator<1> pid(dom); pid(); pid++)
    result[pid[0]] = ax[*pid];
  rt->unmap_region(ctx, pr);
  return result;
}

#ifdef LEGMS_USE_CASACORE
MeasRefDict
Table::get_measure_references_dictionary(
  Legion::Context ctx,
  Legion::Runtime* rt) const {

  std::vector<const MeasRef*> mrps;
  std::transform(
    &meas_refs[0],
    &meas_refs[num_meas_refs],
    std::back_inserter(mrps),
    [](auto& mr) { return &std::get<1>(mr); });
  return MeasRefDict(ctx, rt, mrps);
}
#endif // LEGMS_USE_CASACORE

Table
Table::create(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const std::string& name,
  const std::string& axes_uid,
  const std::vector<int>& index_axes,
  const std::vector<Column>& columns_,
#ifdef LEGMS_USE_CASACORE
  const std::vector<MeasRef>& new_meas_refs,
  const MeasRefContainer& inherited_meas_refs,
#endif
  const Keywords::kw_desc_t& kws,
  const std::string& name_prefix) {

  std::string component_name_prefix = name;
  if (name_prefix.size() > 0)
    component_name_prefix =
      ((name_prefix.back() != '/') ? (name_prefix + "/") : name_prefix)
      + component_name_prefix;

  Legion::LogicalRegion metadata =
    create_metadata(ctx, rt, name, axes_uid, component_name_prefix);
  Legion::LogicalRegion axes =
    create_axes(ctx, rt, index_axes, component_name_prefix);
  Keywords keywords = Keywords::create(ctx, rt, kws, component_name_prefix);
  Legion::LogicalRegion columns;
  {
    Legion::Rect<1> rect(0, columns_.size() - 1);
    Legion::IndexSpace is = rt->create_index_space(ctx, rect);
    Legion::FieldSpace fs = rt->create_field_space(ctx);
    Legion::FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    fa.allocate_field(sizeof(Column), COLUMNS_FID);
    columns = rt->create_logical_region(ctx, is, fs);
    {
      std::string columns_name = component_name_prefix + "/columns";
      rt->attach_name(columns, columns_name.c_str());
    }
    Legion::RegionRequirement req(columns, WRITE_ONLY, EXCLUSIVE, columns);
    req.add_field(COLUMNS_FID);
    Legion::PhysicalRegion pr = rt->map_region(ctx, req);
    const ColumnsAccessor<WRITE_ONLY> cols(pr, COLUMNS_FID);
    Legion::PointInRectIterator<1> pir(rect);
    for (auto& col : columns_) {
      assert(pir());
      cols[*pir] = col;
      pir++;
    }
    assert(!pir());
    rt->unmap_region(ctx, pr);
  }
  return Table(
    metadata,
    axes,
    columns,
#ifdef LEGMS_USE_CASACORE
    new_meas_refs,
    inherited_meas_refs,
#endif //LEGMS_USE_CASACORE
    keywords);
}

Table
Table::create(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const std::string& name,
  const std::string& axes_uid,
  const std::vector<int>& index_axes,
  const std::vector<Column::Generator>& column_generators,
#ifdef LEGMS_USE_CASACORE
  const std::vector<MeasRef>& new_meas_refs,
  const MeasRefContainer& inherited_meas_refs,
#endif
  const Keywords::kw_desc_t& kws,
  const std::string& name_prefix) {

  std::string component_name_prefix = name;
  if (name_prefix.size() > 0)
    component_name_prefix =
      ((name_prefix.back() != '/') ? (name_prefix + "/") : name_prefix)
      + component_name_prefix;

  std::vector<Column> cols;
  for (auto& cg : column_generators)
    cols.push_back(cg(ctx, rt, component_name_prefix));

  return
    create(
      ctx,
      rt,
      name,
      axes_uid,
      index_axes,
      cols,
#ifdef LEGMS_USE_CASACORE
      new_meas_refs,
      inherited_meas_refs,
#endif
      kws,
      name_prefix);
}

LogicalRegion
Table::create_metadata(
  Context ctx,
  Runtime* rt,
  const std::string& name,
  const std::string& axes_uid,
  const std::string& name_prefix) {

  IndexSpace is = rt->create_index_space(ctx, Rect<1>(0, 0));
  FieldSpace fs = rt->create_field_space(ctx);
  FieldAllocator fa = rt->create_field_allocator(ctx, fs);
  fa.allocate_field(sizeof(legms::string), METADATA_NAME_FID);
  rt->attach_name(fs, METADATA_NAME_FID, "name");
  fa.allocate_field(sizeof(legms::string), METADATA_AXES_UID_FID);
  rt->attach_name(fs, METADATA_AXES_UID_FID, "axes_uid");
  LogicalRegion result = rt->create_logical_region(ctx, is, fs);
  {
    std::string result_name = name_prefix + "/metadata";
    rt->attach_name(result, result_name.c_str());
  }
  {
    RegionRequirement req(result, WRITE_ONLY, EXCLUSIVE, result);
    req.add_field(METADATA_NAME_FID);
    req.add_field(METADATA_AXES_UID_FID);
    PhysicalRegion pr = rt->map_region(ctx, req);
    const NameAccessor<WRITE_ONLY> nm(pr, METADATA_NAME_FID);
    const AxesUidAccessor<WRITE_ONLY> au(pr, METADATA_AXES_UID_FID);
    nm[0] = name;
    au[0] = axes_uid;
    rt->unmap_region(ctx, pr);
  }
  return result;
}

LogicalRegion
Table::create_axes(
  Context ctx,
  Runtime* rt,
  const std::vector<int>& index_axes,
  const std::string& name_prefix)  {

  Rect<1> rect(0, index_axes.size() - 1);
  IndexSpace is = rt->create_index_space(ctx, rect);
  FieldSpace fs = rt->create_field_space(ctx);
  FieldAllocator fa = rt->create_field_allocator(ctx, fs);
  fa.allocate_field(sizeof(int), AXES_FID);
  LogicalRegion result = rt->create_logical_region(ctx, is, fs);
  {
    std::string result_name = name_prefix + "/axes";
    rt->attach_name(result, result_name.c_str());
  }
  {
    RegionRequirement req(result, WRITE_ONLY, EXCLUSIVE, result);
    req.add_field(AXES_FID);
    PhysicalRegion pr = rt->map_region(ctx, req);
    const AxesAccessor<WRITE_ONLY> ax(pr, AXES_FID);
    for (PointInRectIterator<1> pir(rect); pir(); pir++)
      ax[*pir] = index_axes[pir[0]];
    rt->unmap_region(ctx, pr);
  }
  return result;
}

void
Table::destroy(Context ctx, Runtime* rt, bool destroy_columns) {

  if (destroy_columns && columns_lr != LogicalRegion::NO_REGION) {
    RegionRequirement req(columns_lr, READ_WRITE, EXCLUSIVE, columns_lr);
    req.add_field(COLUMNS_FID);
    PhysicalRegion pr = rt->map_region(ctx, req);
    const ColumnsAccessor<READ_WRITE> cols(pr, COLUMNS_FID);
    DomainT<1> dom(rt->get_index_space_domain(columns_lr.get_index_space()));
    for (PointInDomainIterator<1> pid(dom); pid(); pid++)
      cols[*pid].destroy(ctx, rt);
    rt->unmap_region(ctx, pr);
  }
  if (metadata_lr != LogicalRegion::NO_REGION) {
    assert(axes_lr != LogicalRegion::NO_REGION);
    std::vector<LogicalRegion*> lrs{&metadata_lr, &axes_lr, &columns_lr};
    for (auto lr : lrs)
      rt->destroy_field_space(ctx, lr->get_field_space());
    for (auto lr : lrs)
      rt->destroy_index_space(ctx, lr->get_index_space());
    for (auto lr : lrs) {
      rt->destroy_logical_region(ctx, *lr);
      *lr = LogicalRegion::NO_REGION;
    }
  }
  keywords.destroy(ctx, rt);

  for (auto& mr : owned_meas_ref())
    mr->destroy(ctx, rt);
}

bool
Table::is_empty(Context ctx, Runtime* rt) const {
  RegionRequirement req(columns_lr, READ_ONLY, EXCLUSIVE, columns_lr);
  req.add_field(COLUMNS_FID);
  PhysicalRegion pr = rt->map_region(ctx, req);
  auto result = is_empty(ctx, rt, pr);
  rt->unmap_region(ctx, pr);
  return result;
}

bool
Table::is_empty(Context ctx, Runtime* rt, const PhysicalRegion& columns) {

  return
    column_names(ctx, rt, columns).empty()
    || min_rank_column(ctx, rt, columns).is_empty();
}

std::vector<std::string>
Table::column_names(Context ctx, Runtime* rt) const {
  RegionRequirement req(columns_lr, READ_ONLY, EXCLUSIVE, columns_lr);
  req.add_field(COLUMNS_FID);
  PhysicalRegion pr = rt->map_region(ctx, req);
  auto result = column_names(ctx, rt, pr);
  rt->unmap_region(ctx, pr);
  return result;
}

std::vector<std::string>
Table::column_names(Context ctx, Runtime* rt, const PhysicalRegion& columns) {

  std::vector<std::string> result;
  DomainT<1> dom =
    rt->get_index_space_domain(columns.get_logical_region().get_index_space());
  const ColumnsAccessor<READ_ONLY> cols(columns, COLUMNS_FID);
  for (PointInDomainIterator<1> pid(dom); pid(); pid++)
    result.push_back(cols[*pid].name(ctx, rt));
  return result;
}

Column
Table::column(Context ctx, Runtime* rt, const std::string& name) const {

  RegionRequirement req(columns_lr, READ_ONLY, EXCLUSIVE, columns_lr);
  req.add_field(COLUMNS_FID);
  PhysicalRegion pr = rt->map_region(ctx, req);
  auto result = column(ctx, rt, pr, name);
  rt->unmap_region(ctx, pr);
  return result;
}

Column
Table::column(
  Context ctx,
  Runtime* rt,
  const PhysicalRegion& columns,
  const std::string& name) {

  Column result;
  auto names = column_names(ctx, rt, columns);
  auto p = std::find(names.begin(), names.end(), name);
  if (p != names.end()) {
    const ColumnsAccessor<READ_ONLY> cols(columns, COLUMNS_FID);
    result = cols[std::distance(names.begin(), p)];
  }
  return result;
}

Column
Table::min_rank_column(Context ctx, Runtime* rt) const {
  Legion::RegionRequirement req(columns_lr, READ_ONLY, EXCLUSIVE, columns_lr);
  req.add_field(COLUMNS_FID);
  PhysicalRegion pr = rt->map_region(ctx, req);
  auto result = min_rank_column(ctx, rt, pr);
  rt->unmap_region(ctx, pr);
  return result;
}

Column
Table::min_rank_column(
  Context ctx,
  Runtime* rt,
  const PhysicalRegion& columns) {

  Column result;
  unsigned min_rank = std::numeric_limits<unsigned>::max();
  const ColumnsAccessor<READ_ONLY> cols(columns, COLUMNS_FID);
  DomainT<1> dom =
    rt->get_index_space_domain(columns.get_logical_region().get_index_space());
  for (PointInDomainIterator<1> pid(dom); pid(); pid++) {
    auto c = cols[*pid];
    auto rank = c.rank(rt);
    if (rank < min_rank) {
      result = c;
      min_rank = rank;
    }
  }
  return result;
}

#ifdef LEGMS_USE_HDF5
std::vector<PhysicalRegion>
Table::with_columns_attached_prologue(
  Context ctx,
  Runtime* rt,
  const LEGMS_FS::path& file_path,
  const std::string& root_path,
  const std::tuple<
  Table*,
  std::unordered_set<std::string>,
  std::unordered_set<std::string>>& table_columns) {

  auto& [table, read_only, read_write] = table_columns;

  std::unordered_set<std::string> only_read_only;
  std::copy_if(
    read_only.begin(),
    read_only.end(),
    std::inserter(only_read_only, only_read_only.end()),
    [&read_write](auto& c) {
      return read_write.count(c) == 0;
    });

  std::string table_root = root_path;
  if (table_root.back() != '/')
    table_root.push_back('/');
  table_root += table->name(ctx, rt);

  std::vector<PhysicalRegion> result;
  table->foreach_column(
    ctx,
    rt,
    [&file_path, &table_root, &only_read_only, &read_write, &result]
    (Context c, Runtime* r, const Column& col) {
      auto cn = col.name(c, r);
      if (only_read_only.count(cn) > 0 || read_write.count(cn) > 0) {
        auto pr =
          hdf5::attach_column_values(
            c,
            r,
            file_path,
            table_root,
            col,
            false,
            read_write.count(cn) > 0);
        AcquireLauncher acquire(col.values_lr, col.values_lr, pr);
        acquire.add_field(Column::VALUE_FID);
        r->issue_acquire(c, acquire);
        result.push_back(pr);
      }
    });
  return result;
}

void
Table::with_columns_attached_epilogue(
  Context ctx,
  Runtime* rt,
  std::vector<PhysicalRegion>& prs) {

  for (auto& pr : prs) {
    ReleaseLauncher
      release(pr.get_logical_region(), pr.get_logical_region(), pr);
    release.add_field(Column::VALUE_FID);
    rt->issue_release(ctx, release);
    rt->detach_external_resource(ctx, pr);
  }
}
#endif // LEGMS_USE_HDF5

#ifndef NO_REINDEX
TaskID ReindexedTableTask::TASK_ID;
const char* ReindexedTableTask::TASK_NAME = "ReindexedTableTask";

ReindexedTableTask::ReindexedTableTask(
  const std::string& name,
  const std::string& axes_uid,
  const std::vector<int>& index_axes,
  LogicalRegion keywords_region,
  const std::vector<Future>& reindexed) {

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
  std::for_each(
    reindexed.begin(),
    reindexed.end(),
    [this](auto& f) { m_launcher.add_future(f); });
}

void
ReindexedTableTask::preregister_task() {
  TASK_ID = Runtime::generate_static_task_id();
  TaskVariantRegistrar registrar(TASK_ID, TASK_NAME, false);
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  registrar.set_leaf();
  // registrar.set_idempotent();
  // registrar.set_replicable();
  Runtime::preregister_task_variant<Table,base_impl>(
    registrar,
    TASK_NAME);
}

Future
ReindexedTableTask::dispatch(Context ctx, Runtime* runtime) {
  return runtime->execute_task(ctx, m_launcher);
}

Table
ReindexedTableTask::base_impl(
  const Task* task,
  const std::vector<PhysicalRegion>&,
  Context,
  Runtime *) {

  TableGenArgs result;
  result.legion_deserialize(task->args);

  std::transform(
    task->futures.begin(),
    task->futures.end(),
    std::inserter(result.col_genargs, result.col_genargs.end()),
    [](auto& f) {
      return f.template get_result<ColumnGenArgs>();
    });
  return result;
}
#endif // !NO_REINDEX

template <typename T, int DIM, bool CHECK_BOUNDS=false>
using ROAccessor =
  FieldAccessor<
  READ_ONLY,
  T,
  DIM,
  coord_t,
  AffineAccessor<T, DIM, coord_t>,
  CHECK_BOUNDS>;

template <typename T, int DIM, bool CHECK_BOUNDS=false>
using WDAccessor =
  FieldAccessor<
  WRITE_DISCARD,
  T,
  DIM,
  coord_t,
  AffineAccessor<T, DIM, coord_t>,
  CHECK_BOUNDS>;

template <typename T, int DIM, bool CHECK_BOUNDS=false>
using WOAccessor =
  FieldAccessor<
  WRITE_ONLY,
  T,
  DIM,
  coord_t,
  AffineAccessor<T, DIM, coord_t>,
  CHECK_BOUNDS>;

template <typename T>
class LEGMS_LOCAL IndexAccumulateTask {
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
    m_launcher.add_field(0, Column::VALUE_FID);
  }

  Future
  dispatch(Context ctx, Runtime* runtime) {
    return runtime->execute_index_space(ctx, m_launcher, DT::af_redop_id);
  }

  static void
  preregister_task() {
    TASK_ID = Runtime::generate_static_task_id();
    std::string tname =
      std::string("IndexAccumulateTask<") + DT::s + std::string(">");
    strncpy(TASK_NAME, tname.c_str(), sizeof(TASK_NAME));
    TASK_NAME[sizeof(TASK_NAME) - 1] = '\0';
    TaskVariantRegistrar registrar(TASK_ID, TASK_NAME, false);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    registrar.set_idempotent();
    // registrar.set_replicable();
    Runtime::preregister_task_variant<acc_field_redop_rhs<T>, base_impl>(
      registrar,
      TASK_NAME);
  }

  static acc_field_redop_rhs<T>
  base_impl(
    const Task* task,
    const std::vector<PhysicalRegion>& regions,
    Context,
    Runtime*) {

    acc_field_redop_rhs<T> result;
    switch (task->index_point.get_dim()) {
#define ACC(D)                                                          \
      case (D): {                                                       \
        const ROAccessor<T, D, true> acc(regions[0], Column::VALUE_FID); \
        Point<D, coord_t> pt(task->index_point);                        \
        result = acc_field_redop_rhs<T>{                                \
          {std::make_tuple(acc[pt],                                     \
                      std::vector<DomainPoint>{task->index_point})}};   \
        break;                                                          \
      }
      LEGMS_FOREACH_N(ACC);
#undef ACC
    default:
      assert(false);
      break;
    }
    return result;
  }

private:

  IndexTaskLauncher m_launcher;
};

template <typename T>
TaskID IndexAccumulateTask<T>::TASK_ID;

template <typename T>
char IndexAccumulateTask<T>::TASK_NAME[40];

TaskID IndexColumnTask::TASK_ID;
const char* IndexColumnTask::TASK_NAME = "IndexColumnTask";

IndexColumnTask::IndexColumnTask(const Column& column)
  : TaskLauncher(TASK_ID, TaskArgument(NULL, 0)) {
  {
    RegionRequirement
      req(column.metadata_lr, READ_ONLY, EXCLUSIVE, column.metadata_lr);
    req.add_field(Column::METADATA_DATATYPE_FID);
    add_region_requirement(req);
  }
  {
    RegionRequirement
      req(column.values_lr, READ_ONLY, EXCLUSIVE, column.values_lr);
    req.add_field(Column::VALUE_FID);
    add_region_requirement(req);
  }
}

void
IndexColumnTask::preregister_task() {
  TASK_ID = Runtime::generate_static_task_id();
  TaskVariantRegistrar registrar(TASK_ID, TASK_NAME, false);
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  registrar.set_idempotent();
  // registrar.set_replicable();
  Runtime::preregister_task_variant<LogicalRegion,base_impl>(
    registrar,
    TASK_NAME);
}

Future
IndexColumnTask::dispatch(Context ctx, Runtime* runtime) {
  return runtime->execute_task(ctx, *this);
}

template <typename T>
static LogicalRegion
index_column(
  const Task* task,
  Context ctx,
  Runtime *runtime,
  legms::TypeTag dt,
  const RegionRequirement& col_req) {

  // launch index space task on input region to compute accumulator value
  IndexAccumulateTask<T> acc_index_task(col_req);
  Future acc_future = acc_index_task.dispatch(ctx, runtime);
  // TODO: create and initialize the resulting LogicalRegion in another task, so
  // we don't have to wait on this future explicitly
  auto acc = acc_future.get_result<acc_field_redop_rhs<T>>().v;

  LogicalRegionT<1> result_lr;
  if (acc.size() > 0) {
    auto result_fs = runtime->create_field_space(ctx);
    {
      auto fa = runtime->create_field_allocator(ctx, result_fs);
      add_field(dt, fa, IndexColumnTask::VALUE_FID);
      fa.allocate_field(
        sizeof(std::vector<DomainPoint>),
        IndexColumnTask::ROWS_FID,
        OpsManager::V_DOMAIN_POINT_SID);
    }
    IndexSpaceT<1> result_is =
      runtime->create_index_space(ctx, Rect<1>(0, acc.size() - 1));
    result_lr = runtime->create_logical_region(ctx, result_is, result_fs);

    // transfer values and row numbers from acc_lr to result_lr
    RegionRequirement result_req(result_lr, WRITE_ONLY, EXCLUSIVE, result_lr);
    result_req.add_field(IndexColumnTask::VALUE_FID);
    result_req.add_field(IndexColumnTask::ROWS_FID);
    PhysicalRegion result_pr = runtime->map_region(ctx, result_req);
    const WOAccessor<T, 1, true> values(result_pr, IndexColumnTask::VALUE_FID);
    const WOAccessor<std::vector<DomainPoint>, 1, true>
      rns(result_pr, IndexColumnTask::ROWS_FID);
    for (size_t i = 0; i < acc.size(); ++i) {
      ::new (rns.ptr(i)) std::vector<DomainPoint>;
      tie(values[i], rns[i]) = acc[i];
    }
    runtime->unmap_region(ctx, result_pr);
    // TODO: keep?
    //runtime->destroy_field_space(ctx, result_fs);
    //runtime->destroy_index_space(ctx, result_is);
  }
  return result_lr;
}

LogicalRegion
IndexColumnTask::base_impl(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *runtime) {

  const Column::DatatypeAccessor<READ_ONLY>
    datatype(regions[0], Column::METADATA_DATATYPE_FID);

  LogicalRegion result;
  switch (datatype[0]) {
#define ICR(DT)                                 \
    case DT:                                    \
      result =                                  \
        index_column<DataType<DT>::ValueType>(  \
          task,                                 \
          ctx,                                  \
          runtime,                              \
          DT,                                   \
          task->regions[1]);                    \
      break;
    LEGMS_FOREACH_DATATYPE(ICR);
#undef ICR
  default:
    assert(false);
    break;
  }
  return result;
}

#ifndef NO_REINDEX
#ifdef HIERARCHICAL_COMPUTE_RECTANGLES

class LEGMS_LOCAL ComputeRectanglesTask {
public:

  static TaskID TASK_ID;
  static const char* TASK_NAME;

  ComputeRectanglesTask(
    bool allow_rows,
    IndexPartition row_partition,
    const std::vector<LogicalRegion>& ix_columns,
    LogicalRegion new_rects,
    const std::vector<PhysicalRegion>& parent_regions,
    const std::vector<coord_t>& ix0,
    const std::vector<DomainPoint>& rows) {

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
    std::for_each(
      ix0.begin(),
      ix0.end(),
      [](auto& d) { cout << d << " "; });
    std::cout << ")" << std::endl;

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
      req.add_field(IndexColumnTask::rows_fid);
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
    const std::vector<PhysicalRegion>& regions,
    Context ctx,
    Runtime *runtime) {

    TaskArgs args;
    TaskArgs::deserialize(args, static_cast<const void *>(task->args));

    const FieldAccessor<
      READ_ONLY,
      std::vector<DomainPoint>,
      1,
      coord_t,
      AffineAccessor<std::vector<DomainPoint>, 1, coord_t>,
      false> rows(regions[args.ix0.size()], IndexColumnTask::rows_fid);

    auto pt = task->index_point[0];
    args.ix0.push_back(pt);
    if (args.ix0.size() == 1)
      args.rows = rows[pt];
    else
      args.rows = intersection(args.rows, rows[pt]);
    if (args.rows.size() > 0) {
      if (args.ix0.size() < regions.size() - 1) {
        // start task at next index level
        std::vector<LogicalRegion> col_lrs;
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
  preregister_task() {
    TASK_ID = Runtime::generate_static_task_id();
    TaskVariantRegistrar registrar(TASK_ID, TASK_NAME, false);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    // registrar.set_replicable();
    Runtime::preregister_task_variant<base_impl>(registrar, TASK_NAME);
  }

private:

  struct TaskArgs {
    bool allow_rows;
    IndexPartition row_partition;
    std::vector<coord_t> ix0;
    std::vector<DomainPoint> rows;

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

#else // !HIERARCHICAL_COMPUTE_RECTANGLES

class LEGMS_LOCAL ComputeRectanglesTask {
public:

  static TaskID TASK_ID;
  static const char* TASK_NAME;

  ComputeRectanglesTask(
    bool allow_rows,
    IndexPartition row_partition,
    const std::vector<LogicalRegion>& ix_columns,
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

    std::for_each(
      m_ix_columns.begin(),
      m_ix_columns.end(),
      [&launcher](auto& lr) {
        RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
        req.add_field(IndexColumnTask::rows_fid);
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
    const std::vector<PhysicalRegion>& regions,
    Context ctx,
    Runtime *runtime) {

    TaskArgs args;
    TaskArgs::deserialize(args, static_cast<const void *>(task->args));

    typedef const FieldAccessor<
      READ_ONLY,
      std::vector<DomainPoint>,
      1,
      coord_t,
      AffineAccessor<std::vector<DomainPoint>, 1, coord_t>,
      false> rows_acc_t;

    auto ixdim = regions.size() - 1;

    std::vector<DomainPoint> common_rows;
    {
      rows_acc_t rows(regions[0], IndexColumnTask::rows_fid);
      common_rows = rows[task->index_point[0]];
    }
    for (size_t i = 1; i < ixdim; ++i) {
      rows_acc_t rows(regions[i], IndexColumnTask::rows_fid);
      common_rows = intersection(common_rows, rows[task->index_point[i]]);
    }

#define WRITE_RECTS(ROWDIM, RECTDIM)                                    \
    case (ROWDIM * LEGION_MAX_DIM + RECTDIM): {                         \
      const FieldAccessor<                                              \
        WRITE_DISCARD, \
        Rect<RECTDIM>, \
        ROWDIM> rects(regions.back(), ReindexColumnTask::row_rects_fid); \
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
  preregister_task() {
    TASK_ID = Runtime::generate_static_task_id();
    TaskVariantRegistrar registrar(TASK_ID, TASK_NAME, false);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    // registrar.set_replicable();
    Runtime::preregister_task_variant<base_impl>(registrar, TASK_NAME);
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
      val.row_partition =
        *reinterpret_cast<const IndexPartition*>(buff + result);
      result += sizeof(row_partition);
      return result;
    }
  };

  bool m_allow_rows;

  IndexPartition m_row_partition;

  std::vector<LogicalRegion> m_ix_columns;

  LogicalRegion m_new_rects_lr;

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

#endif

TaskID ComputeRectanglesTask::TASK_ID;
const char* ComputeRectanglesTask::TASK_NAME = "ComputeRectanglesTask";

class LEGMS_LOCAL ReindexColumnCopyTask {
public:

  static TaskID TASK_ID;
  static const char* TASK_NAME;

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
      const SA<DT,SRCDIM> from(src, Column::VALUE_FID); \
      const DA<DT,DSTDIM> to(dst, Column::VALUE_FID);   \
      DomainT<SRCDIM,coord_t> src_bounds(src);          \
      DomainT<DSTDIM,coord_t> dst_bounds(dst);          \
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
    src_req.add_field(Column::VALUE_FID);

    RegionRequirement dst_req(
      new_col_lp,
      0,
      WRITE_ONLY,
      EXCLUSIVE,
      m_new_col_lr);
    dst_req.add_field(Column::VALUE_FID);

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
    const std::vector<PhysicalRegion>& regions,
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
  preregister_task() {
    TASK_ID = Runtime::generate_static_task_id();
    TaskVariantRegistrar registrar(TASK_ID, TASK_NAME, false);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    // registrar.set_replicable();
    Runtime::preregister_task_variant<base_impl>(registrar, TASK_NAME);
  }

private:

  LogicalRegion m_column;

  legms::TypeTag m_column_dt;

  IndexPartition m_row_partition;

  LogicalRegion m_new_rects_lr;

  LogicalRegion m_new_col_lr;
};

TaskID ReindexColumnCopyTask::TASK_ID;
const char* ReindexColumnCopyTask::TASK_NAME = "ReindexColumnCopyTask";

TaskID ReindexColumnTask::TASK_ID;
const char* ReindexColumnTask::TASK_NAME = "ReindexColumnTask";

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
  const std::vector<shared_ptr<Column>>& ixcols,
  bool allow_rows) {

  // get column partition down to row axis
  assert(row_axis_offset >= 0);
  std::vector<int> col_part_axes;
  std::copy_n(
    col->axes().begin(),
    row_axis_offset + 1,
    std::back_inserter(col_part_axes));
  m_partition = col->partition_on_axes(col_part_axes);

  std::vector<int> index_axes =
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
  col_req.add_field(Column::VALUE_FID);
  m_launcher.add_region_requirement(col_req);
  std::for_each(
    ixcols.begin(),
    ixcols.end(),
    [this](auto& ixc) {
      auto lr = ixc->logical_region();
      assert(lr != LogicalRegion::NO_REGION);
      RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
      req.add_field(IndexColumnTask::ROWS_FID);
      m_launcher.add_region_requirement(req);
    });
}

Future
ReindexColumnTask::dispatch(Context ctx, Runtime* runtime) {
  return runtime->execute_task(ctx, m_launcher);
}

template <int OLDDIM, int NEWDIM>
static ColumnGenArgs
reindex_column(
  const ReindexColumnTask::TaskArgs& args,
  const std::vector<PhysicalRegion>& regions,
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

    LayoutConstraintRegistrar lc(new_rects_fs);
    add_row_major_order_constraint(lc, rows_is.get_dim())
      .add_constraint(MemoryConstraint(Memory::Kind::GLOBAL_MEM));
    // TODO: free LayoutConstraintID returned from following call...maybe
    // generate field spaces and constraints once at startup
    runtime->register_layout(lc);
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

  std::vector<LogicalRegion> ix_lrs;
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
      for (PointInDomainIterator<N> \
             pid(runtime->get_index_space_domain(ctx, rows_is));        \
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
  return ColumnGenArgs();
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
      add_field(args.col.datatype, fa, Column::VALUE_FID);
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
  const std::vector<PhysicalRegion>& regions,
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
ReindexColumnTask::preregister_task() {
  TASK_ID = Runtime::generate_static_task_id();
  TaskVariantRegistrar registrar(TASK_ID, TASK_NAME, false);
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  registrar.set_inner();
  registrar.set_idempotent();
  // registrar.set_replicable();
  Runtime::preregister_task_variant<ColumnGenArgs,base_impl>(
    registrar,
    TASK_NAME);
}

Future/*Table*/
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
    TableGenArgs empty;
    empty.name = name();
    empty.axes_uid = axes_uid();
    return Future::from_value(runtime(), empty);
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
      return make_pair(nm, move(ax));
    });

  // index associated columns; the Future in "index_cols" below contains a
  // ColumnGenArgs of a LogicalRegion with two fields: at Column::VALUE_FID, the
  // column values (sorted in ascending order); and at
  // IndexColumnTask::rows_fid, a sorted vector of DomainPoints in the original
  // column.
  std::unordered_map<int, Future> index_cols;
  std::for_each(
    col_reindex_axes.begin(),
    col_reindex_axes.end(),
    [this, &axis_names, &index_cols](auto& nm_ds) {
      const std::vector<int>& ds = get<1>(nm_ds);
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
  std::vector<Future> reindexed;
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
      std::vector<shared_ptr<Column>> ixcols;
      for (auto d : ds) {
        ixcols.push_back(
          index_cols.at(d)
          .template get_result<ColumnGenArgs>()
          .operator()(context(), runtime()));
      }
      auto col = column(nm);
      auto col_axes = col->axes();
      auto row_axis_offset =
        distance(col_axes.begin(), find(col_axes.begin(), col_axes.end(), 0));
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
#endif // !NO_REINDEX

std::unordered_map<int, Future>
Table::iindex_by_value(
  Context ctx,
  Runtime* rt,
  const std::vector<std::string>& axis_names,
  const std::unordered_set<int>& axes) const {

  RegionRequirement req(columns_lr, READ_ONLY, EXCLUSIVE, columns_lr);
  req.add_field(Table::COLUMNS_FID);
  auto cols = rt->map_region(ctx, req);
  std::unordered_map<int, Future> result;
  std::for_each(
    axes.begin(),
    axes.end(),
    [&ctx, rt, &cols, &axis_names, &result](auto a) {
      auto col = column(ctx, rt, cols, axis_names[a]);
      if (!col.is_empty()) {
        IndexColumnTask task(col);
        result[a] = task.dispatch(ctx, rt);
      }
    });
  rt->unmap_region(ctx, cols);
  return result;
}

class LEGMS_LOCAL ComputeRowColorsTask {
public:

  static TaskID TASK_ID;
  static const char* TASK_NAME;
  ComputeRowColorsTask(
    const std::vector<LogicalRegion>& ix_columns,
    LogicalRegion row_colors,
    LogicalRegion colors)
    : m_ix_columns(ix_columns)
    , m_row_colors(row_colors)
    , m_colors(colors) {
  };

  void
  dispatch(Context ctx, Runtime* runtime) {

    auto ixdim = m_ix_columns.size();
    switch (ixdim) {
#define FILL_COLOR(DIM)                                                 \
      case DIM: {                                                       \
        runtime->fill_field(                                            \
          ctx, m_row_colors, m_row_colors, 0, point_add_redop<DIM>::identity); \
        break;                                                          \
      }
      LEGMS_FOREACH_N(FILL_COLOR);
#undef FILL_COLOR
    default:
      assert(false);
      break;
    }
    runtime->fill_field(ctx, m_colors, m_colors, 0, (coord_t)0);

    IndexTaskLauncher
      launcher(
        TASK_ID,
        m_colors.get_index_space(),
        TaskArgument(NULL, 0),
        ArgumentMap());

    for (auto& lr : m_ix_columns) {
      RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
      req.add_field(IndexColumnTask::ROWS_FID);
      launcher.add_region_requirement(req);
    }
    {
      RegionRequirement
        req(
          m_row_colors,
          OpsManager::POINT_ADD_REDOP(ixdim),
          ATOMIC,
          m_row_colors);
      req.add_field(0);
      launcher.add_region_requirement(req);
    }
    {
      RegionRequirement
        req(m_colors, OpsManager::COORD_BOR_REDOP, ATOMIC, m_colors);
      req.add_field(0);
      launcher.add_region_requirement(req);
    }
    runtime->execute_index_space(ctx, launcher);
  }

  static void
  base_impl(
    const Task* task,
    const std::vector<PhysicalRegion>& regions,
    Context context,
    Runtime *runtime) {

    typedef const ROAccessor<std::vector<DomainPoint>, 1, true> rows_acc_t;

    auto ixdim = regions.size() - 2;

    std::vector<DomainPoint> common_rows;
    {
      rows_acc_t rows(regions[0], IndexColumnTask::ROWS_FID);
      common_rows = rows[task->index_point[0]];
    }
    for (size_t i = 1; i < ixdim; ++i) {
      rows_acc_t rows(regions[i], IndexColumnTask::ROWS_FID);
      common_rows = intersection(common_rows, rows[task->index_point[i]]);
    }

    if (common_rows.size() > 0) {
      auto rowdim = common_rows[0].get_dim();
      switch (rowdim * LEGION_MAX_DIM + ixdim) {
#define WRITE_COLORS(ROWDIM, COLORDIM)                                  \
        case (ROWDIM * LEGION_MAX_DIM + COLORDIM): {                    \
          const ReductionAccessor<point_add_redop<COLORDIM>, true, ROWDIM, coord_t, AffineAccessor<Point<COLORDIM>,ROWDIM,coord_t>> \
            colors(regions[COLORDIM], 0, OpsManager::POINT_ADD_REDOP(ixdim)); \
          const ReductionAccessor<coord_bor_redop, true, COLORDIM, coord_t, AffineAccessor<coord_t,COLORDIM,coord_t>> \
            flags(regions[COLORDIM + 1], 0, OpsManager::COORD_BOR_REDOP); \
          Point<COLORDIM,coord_t> color(task->index_point);             \
          flags[color] <<= 1;                                           \
          for (size_t i = 0; i < common_rows.size(); ++i) {             \
            Point<ROWDIM> r(common_rows[i]);                            \
            colors[r] <<= color;                                        \
          }                                                             \
          break;                                                        \
        }
        LEGMS_FOREACH_NN(WRITE_COLORS);
#undef WRITE_COLORS
      default:
        assert(false);
        break;
      }
    }
  }

  static void
  preregister_task() {
    TASK_ID = Runtime::generate_static_task_id();
    TaskVariantRegistrar registrar(TASK_ID, TASK_NAME, false);
    registrar.set_idempotent();
    registrar.set_leaf();
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<base_impl>(registrar, TASK_NAME);
  }

private:

  std::vector<LogicalRegion> m_ix_columns;

  LogicalRegion m_row_colors;

  LogicalRegion m_colors;

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

TaskID
ComputeRowColorsTask::TASK_ID;

const char*
ComputeRowColorsTask::TASK_NAME = "ComputeRowColorsTask";

// TODO: replace this macro value with a class member variable
#define PART_FID 0

class LEGMS_LOCAL InitColorsTask {
public:

  static TaskID TASK_ID;
  static const char* TASK_NAME;

  InitColorsTask(
    unsigned color_dim,
    unsigned row_dim,
    LogicalRegion colors_lr,
    LogicalRegion parts_lr)
    : m_task_args{color_dim, row_dim}
    , m_colors_lr(colors_lr)
    , m_parts_lr(parts_lr) {
  }

  void
  dispatch(Context context, Runtime* runtime) {

    IndexTaskLauncher
      launcher(
        TASK_ID,
        m_parts_lr.get_index_space(),
        TaskArgument(&m_task_args, sizeof(m_task_args)),
        ArgumentMap());
    launcher.add_region_requirement(
      RegionRequirement(m_colors_lr, READ_ONLY, EXCLUSIVE, m_colors_lr));
    launcher.add_field(0, 0);
    // FIXME: cleanup
    auto parts_ip =
      runtime->create_equal_partition(
        context,
        m_parts_lr.get_index_space(),
        m_parts_lr.get_index_space());
    auto parts_lp =
      runtime->get_logical_partition(context, m_parts_lr, parts_ip);
    launcher.add_region_requirement(
      RegionRequirement(parts_lp, 0, WRITE_ONLY, EXCLUSIVE, m_parts_lr));
    launcher.add_field(1, PART_FID);
    runtime->execute_index_space(context, launcher);
  }

  struct TaskArgs {
    unsigned color_dim;
    unsigned row_dim;
  };

  template <int COLOR_DIM>
  static void
  impl(
    Context context,
    Runtime* runtime,
    const Task* task,
    const std::vector<PhysicalRegion>& regions) {

    const TaskArgs* args = static_cast<const TaskArgs*>(task->args);

    switch (args->row_dim * LEGION_MAX_DIM
            + task->regions[1].region.get_dim()) {
#define COLOR_PARTS(ROW_DIM, COL_DIM)                                 \
      case (ROW_DIM * LEGION_MAX_DIM + COL_DIM):  {                   \
        static_assert(ROW_DIM <= COL_DIM);                            \
        const ROAccessor<Point<COLOR_DIM>, ROW_DIM, true>             \
          colors(regions[0], 0);                                      \
        const WOAccessor<Point<COLOR_DIM>, COL_DIM, true>             \
          parts(regions[1], PART_FID);                                \
        Point<ROW_DIM> pt;                                            \
        for (size_t i = 0; i < ROW_DIM; ++i)                          \
          pt[i] = task->index_point[i];                               \
        parts[task->index_point] = colors[pt];                        \
        break;                                                        \
    }
      LEGMS_FOREACH_MN(COLOR_PARTS);
#undef COLOR_PARTS
    default:
      assert(false);
      break;
    }
  }

  static void
  base_impl(
    const Task* task,
    const std::vector<PhysicalRegion>& regions,
    Context context,
    Runtime *runtime) {

    const TaskArgs* args = static_cast<const TaskArgs*>(task->args);

    switch (args->color_dim) {
#define IMPL(COLOR_DIM)                                   \
      case (COLOR_DIM):                                   \
        impl<COLOR_DIM>(context, runtime, task, regions); \
        break;
      LEGMS_FOREACH_N(IMPL);
#undef IMPL
    default:
      assert(false);
      break;
    }
  }

  static void
  preregister_task() {
    TASK_ID = Runtime::generate_static_task_id();
    TaskVariantRegistrar registrar(TASK_ID, TASK_NAME, false);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    registrar.set_leaf();
    // registrar.set_replicable();
    Runtime::preregister_task_variant<base_impl>(registrar, TASK_NAME);
  }

private:

  TaskArgs m_task_args;

  LogicalRegion m_colors_lr;

  LogicalRegion m_parts_lr;
};

TaskID InitColorsTask::TASK_ID;
const char* InitColorsTask::TASK_NAME = "InitColorsTask";

class LEGMS_LOCAL PartitionRowsTask {
public:

  static TaskID TASK_ID;
  static const char* TASK_NAME;

  PartitionRowsTask(
    const IndexSpace& column_is,
    const std::string& axes_uid,
    const std::vector<int>& axes,
    LogicalRegion colors_lr,
    IndexSpace colors_is,
    unsigned rowdim) {

    m_args.col_ispace = column_is;
    m_args.axes_uid = axes_uid;
    assert(axes.size() <= LEGMS_MAX_NUM_TABLE_COLUMNS);
    assert(axes.size() == static_cast<size_t>(colors_is.get_dim()));
    for (size_t i = 0; i < axes.size(); ++i)
      m_args.axes[i] = axes[i];
    m_args.colors_is = colors_is;
    m_args.rowdim = rowdim;
    m_launcher = TaskLauncher(TASK_ID, TaskArgument(&m_args, sizeof(m_args)));
    RegionRequirement req(colors_lr, READ_ONLY, EXCLUSIVE, colors_lr);
    req.add_field(0);
    m_launcher.add_region_requirement(req);
  }

  Future
  dispatch(Context context, Runtime* runtime) {
    return runtime->execute_task(context, m_launcher);
  }

  template <int COLOR_DIM>
  static IndexPartition
  impl(
    const Task* task,
    const std::vector<PhysicalRegion>& regions,
    Context context,
    Runtime *runtime) {

    const TaskArgs* args = static_cast<TaskArgs*>(task->args);

    FieldSpace fs = runtime->create_field_space(context);
    {
      FieldAllocator fa = runtime->create_field_allocator(context, fs);
      fa.allocate_field(sizeof(Point<COLOR_DIM>), PART_FID);
    }
    // LayoutConstraintRegistrar lc(fs);
    // add_row_major_order_constraint(lc, args->col_ispace.get_dim());
    // TODO: free LayoutConstraintID returned from following call
    // runtime->register_layout(lc);
    LogicalRegion lr =
      runtime->create_logical_region(context, args->col_ispace, fs);
    InitColorsTask ctask(
      COLOR_DIM,
      args->rowdim,
      regions[0].get_logical_region(),
      lr);
    ctask.dispatch(context, runtime);
    auto result =
      runtime->create_partition_by_field(
        context,
        lr,
        lr,
        PART_FID,
        args->colors_is);
    // runtime->destroy_logical_region(context, lr);
    // runtime->destroy_field_space(context, fs);
    return result;
  }

  static ColumnPartition
  base_impl(
    const Task* task,
    const std::vector<PhysicalRegion>& regions,
    Context ctx,
    Runtime *rt) {

    const TaskArgs* args = static_cast<TaskArgs*>(task->args);
    IndexPartition ip;
    switch(args->colors_is.get_dim()) {
#define IMPL(CDIM)                                                      \
      case (CDIM): ip = impl<CDIM>(task, regions, ctx, rt); break;
      LEGMS_FOREACH_N(IMPL);
#undef IMPL
    default:
      assert(false);
      break;
    }
    std::string axes_uid(args->axes_uid.val);
    std::vector<int> axes(args->colors_is.get_dim());
    for (size_t i = 0; i < axes.size(); ++i)
      axes[i] = args->axes[i];
    return ColumnPartition::create(ctx, rt, args->axes_uid, axes, ip);
  }

  static void
  preregister_task() {
    TASK_ID = Runtime::generate_static_task_id();
    TaskVariantRegistrar registrar(TASK_ID, TASK_NAME, false);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    // registrar.set_replicable();
    Runtime::preregister_task_variant<ColumnPartition, base_impl>(
      registrar,
      TASK_NAME);
  }

private:

  struct TaskArgs {
    IndexSpace col_ispace;
    legms::string axes_uid;
    int axes[LEGMS_MAX_NUM_TABLE_COLUMNS];
    IndexSpace colors_is;
    unsigned rowdim;
  };

  TaskArgs m_args;

  TaskLauncher m_launcher;
};

TaskID PartitionRowsTask::TASK_ID;
const char* PartitionRowsTask::TASK_NAME = "PartitionRowsTask";

std::unordered_map<std::string, Future>
Table::ipartition_by_value(
  Context ctx,
  Runtime* rt,
  const std::vector<std::string>& axis_names,
  const std::vector<int>& axes) const {

  RegionRequirement req(columns_lr, READ_ONLY, EXCLUSIVE, columns_lr);
  req.add_field(Table::COLUMNS_FID);
  auto cols = rt->map_region(ctx, req);

  assert(
    std::all_of(
      axes.begin(),
      axes.end(),
      [&ctx, rt, &cols, &axis_names](const auto& a) {
        return !column(ctx, rt, cols, axis_names[a]).is_empty();
      }));

  auto tbl_axes = index_axes(ctx, rt);
  auto ax_uid = axes_uid(ctx, rt);

  // TODO: Support partitioning on columns that are in the index_axes set of the
  // table -- the current implementation doesn't support this case. A possible
  // approach: remove those columns in the index_axes and proceed with the
  // currently implemented algorithm; create a partition on the columns in the
  // index_axes set; use Legion::Runtime::create_cross_product_partitions() to
  // create the requested partition; then perhaps use
  // Legion::Runtime::create_pending_partition() to return unified partitions
  // rather than the deconstructed versions provided by
  // create_cross_product_partitions().
  assert(
    std::none_of(
      axes.begin(),
      axes.end(),
      [&tbl_axes](const auto& a) {
        return std::find(tbl_axes.begin(), tbl_axes.end(), a) != tbl_axes.end();
      }));

  // For now we only allow partitioning on columns that have the same axes as
  // the table; this restriction allows for a simplification in implementation
  // since every "row" of every column can then have only one color (i.e, every
  // column has a disjoint partition). I'm fairly certain that the
  // implementation could be extended so that aliased partitions are supported,
  // but I'm leaving that undone since the utility of that case isn't clear to
  // me at the moment.
  assert(
    std::all_of(
      axes.begin(),
      axes.end(),
      [&ctx, rt, &cols, &axis_names, &tbl_axes](const auto& a) {
        return column(ctx, rt, cols, axis_names[a]).axes(ctx, rt) == tbl_axes;
      }));

  std::unordered_set<int> axesset(axes.begin(), axes.end());
  auto f_ixcols = iindex_by_value(ctx, rt, axis_names, axesset);
  std::vector<LogicalRegion> ixcols(f_ixcols.size());
  for (auto& a_f : f_ixcols) {
    auto& [a, f] = a_f;
    auto d = find(axes.begin(), axes.end(), a);
    assert(d != axes.end());
    ixcols[distance(axes.begin(), d)] = f.template get_result<LogicalRegion>();
  }

  // first we create the bounding row color index space (product space of all
  // index column colors)
  IndexSpace colors_is;
  FieldSpace row_color_fs = rt->create_field_space(ctx);
  {
    FieldAllocator fa = rt->create_field_allocator(ctx, row_color_fs);
    switch (ixcols.size()) {
#define ROW_COLOR_INIT(DIM)                                           \
      case DIM: {                                                     \
        Rect<DIM> bounds;                                             \
        for (size_t i = 0; i < DIM; ++i){                             \
          Rect<1> r =                                                 \
            rt->get_index_space_domain(ixcols[i].get_index_space());  \
          bounds.lo[i] = r.lo[0];                                     \
          bounds.hi[i] = r.hi[0];                                     \
        }                                                             \
        colors_is = rt->create_index_space(ctx, bounds);              \
        fa.allocate_field(sizeof(Point<DIM>), 0);                     \
        break;                                                        \
      }
      LEGMS_FOREACH_N(ROW_COLOR_INIT);
#undef ROW_COLOR_INIT
    default:
      assert(false);
      break;
    }
  }
  IndexSpace rows_is =
    column(ctx, rt, cols, axis_names[axes[0]]).values_lr.get_index_space();
  // row_colors_lr for color of each row
  LogicalRegion row_colors_lr =
    rt->create_logical_region(ctx, rows_is, row_color_fs);

  FieldSpace colors_fs = rt->create_field_space(ctx);
  {
    FieldAllocator fa = rt->create_field_allocator(ctx, colors_fs);
    fa.allocate_field(sizeof(coord_t), 0);
  }
  // colors_lr for tracking row color usage
  LogicalRegion colors_lr =
    rt->create_logical_region(ctx, colors_is, colors_fs);

  ComputeRowColorsTask task(ixcols, row_colors_lr, colors_lr);
  task.dispatch(ctx, rt);

  // we require the color space of the partition, but, in order to have an
  // accurate color space, we should not assume that there is at least one row
  // for all colors in the product of the index column colors; to do that we
  // rely on the colors_lr field

  IndexSpace flags_cs = rt->create_index_space(ctx, Rect<1>(0, 1));
  IndexPartition color_flag_ip =
    rt->create_partition_by_field(
      ctx,
      colors_lr,
      colors_lr,
      0,
      flags_cs);
  IndexSpace color_set_is = rt->get_index_subspace(color_flag_ip, 1);
  // rt->destroy_index_space(ctx, cset_is);
  // rt->destroy_index_partition(ctx, color_bounds_ip);
  // rt->destroy_logical_partition(ctx, color_flag_lp);
  // rt->destroy_index_partition(ctx, color_flag_ip);
  // rt->destroy_index_space(ctx, flags_cs);

  std::unordered_map<std::string, Future> result;
  auto colnames = column_names(ctx, rt, cols);
  auto rowdim = tbl_axes.size();
  for (auto& nm : colnames) {
    auto col = column(ctx, rt, cols, nm);
    PartitionRowsTask
      pt(
        col.values_lr.get_index_space(),
        ax_uid,
        axes,
        row_colors_lr,
        color_set_is,
        rowdim);
    result.emplace(nm, pt.dispatch(ctx, rt));
  }

  // FIXME: clean up ixcols
  // rt->destroy_logical_region(ctx, colors_lr); FIXME
  rt->unmap_region(ctx, cols);
  return result;
}

void
Table::register_tasks(Context context, Runtime* runtime) {
}

void
Table::preregister_tasks() {
  ComputeRowColorsTask::preregister_task();
  PartitionRowsTask::preregister_task();
#define PREREG_IDX_ACCUMULATE(DT)                                   \
  IndexAccumulateTask<DataType<DT>::ValueType>::preregister_task();
  LEGMS_FOREACH_DATATYPE(PREREG_IDX_ACCUMULATE);
#undef PREREG_IDX_ACCUMULATE
  IndexColumnTask::preregister_task();
  InitColorsTask::preregister_task();
#ifndef NO_REINDEX
  ComputeRectanglesTask::preregister_task();
  ReindexColumnTask::preregister_task();
  ReindexColumnCopyTask::preregister_task();
  ReindexedTableTask::preregister_task();
#endif // !NO_REINDEX
}

#ifdef LEGMS_USE_CASACORE

Table
Table::from_ms(
  Context ctx,
  Runtime* runtime,
  const LEGMS_FS::path& path,
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
    legms:: template from_ms<MS_MAIN>(ctx, runtime, path, column_selections);

#undef FROM_MS_TABLE
}

#endif // LEGMS_USE_CASACORE

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
