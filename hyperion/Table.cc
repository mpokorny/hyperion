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
#include <hyperion/hyperion.h>
#include <hyperion/utility.h>
#include <hyperion/Column.h>
#include <hyperion/Table.h>
#ifdef HYPERION_USE_CASACORE
# include <hyperion/TableBuilder.h>
# include <hyperion/MeasRef.h>
#endif

#pragma GCC visibility push(default)
# include <legion/legion_c_util.h>
# include <algorithm>
# include <array>
# include <limits>
# include <numeric>
# include <tuple>
# include <vector>

# ifdef HYPERION_USE_HDF5
#  include <hdf5.h>
# endif
#pragma GCC visibility pop

using namespace hyperion;
using namespace Legion;

#undef HIERARCHICAL_COMPUTE_RECTANGLES

#undef SAVE_LAYOUT_CONSTRAINT_IDS

Table::Table() {}

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

Table
Table::create(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const std::string& name,
  const std::string& axes_uid,
  const std::vector<int>& index_axes,
  const std::vector<Column>& columns_,
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
  return Table(metadata, axes, columns, keywords);
}

Table
Table::create(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const std::string& name,
  const std::string& axes_uid,
  const std::vector<int>& index_axes,
  const std::vector<Column>& columns_,
  const Keywords& keywords) {

  std::string component_name_prefix = name;

  Legion::LogicalRegion metadata =
    create_metadata(ctx, rt, name, axes_uid, component_name_prefix);
  Legion::LogicalRegion axes =
    create_axes(ctx, rt, index_axes, component_name_prefix);
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
  return Table(metadata, axes, columns, keywords);
}

Table
Table::create(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const std::string& name,
  const std::string& axes_uid,
  const std::vector<int>& index_axes,
  const std::vector<Column::Generator>& column_generators,
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
  fa.allocate_field(sizeof(hyperion::string), METADATA_NAME_FID);
  rt->attach_name(fs, METADATA_NAME_FID, "name");
  fa.allocate_field(sizeof(hyperion::string), METADATA_AXES_UID_FID);
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
    for (PointInDomainIterator<1>
           pid(rt->get_index_space_domain(columns_lr.get_index_space()));
         pid();
         pid++)
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

#ifdef HYPERION_USE_HDF5
std::vector<PhysicalRegion>
Table::with_columns_attached_prologue(
  Context ctx,
  Runtime* rt,
  const CXX_FILESYSTEM_NAMESPACE::path& file_path,
  const std::string& root_path,
  const std::tuple<
  Table*,
  std::unordered_set<std::string>,
  std::unordered_set<std::string>>& table_columns) {

  Table* table;
  std::unordered_set<std::string> read_only;
  std::unordered_set<std::string> read_write;
  std::tie(table, read_only, read_write) = table_columns;

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
#endif // HYPERION_USE_HDF5

TaskID ReindexedTableTask::TASK_ID;
const char* ReindexedTableTask::TASK_NAME = "ReindexedTableTask";

ReindexedTableTask::ReindexedTableTask(
  const Table& table,
  const std::vector<int>& index_axes,
  const std::vector<Future>& reindexed)
  : m_table(table)
  , m_reindexed(reindexed) {

  m_args.num_index_axes = index_axes.size();
  for (size_t i = 0; i < m_args.num_index_axes; ++i)
    m_args.index_axes[i] = index_axes[i];
  m_args.kws_region_offset = -1;
}

void
ReindexedTableTask::preregister_task() {
  TASK_ID = Runtime::generate_static_task_id();
  TaskVariantRegistrar registrar(TASK_ID, TASK_NAME, false);
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  registrar.set_idempotent();
  // registrar.set_replicable();
  Runtime::preregister_task_variant<Table,base_impl>(registrar, TASK_NAME);
}

Future
ReindexedTableTask::dispatch(Context ctx, Runtime* rt) {

  std::vector<RegionRequirement> reqs;
  {
    RegionRequirement
      req(m_table.metadata_lr, READ_ONLY, EXCLUSIVE, m_table.metadata_lr);
    req.add_field(Table::METADATA_NAME_FID);
    req.add_field(Table::METADATA_AXES_UID_FID);
    reqs.push_back(req);
  }
  if (!m_table.keywords.is_empty()) {
    m_args.kws_region_offset = reqs.size();
    auto kwsz = m_table.keywords.size(rt);
    std::vector<FieldID> fids(kwsz);
    std::iota(fids.begin(), fids.end(), 0);
    auto kwreqs = m_table.keywords.requirements(rt, fids, READ_ONLY).value();
    reqs.push_back(kwreqs.type_tags);
    reqs.push_back(kwreqs.values);
  }

  TaskLauncher
    launcher(
      ReindexedTableTask::TASK_ID,
      TaskArgument(&m_args, sizeof(m_args)));
  for (auto& r : reqs)
    launcher.add_region_requirement(r);
  for (auto& f : m_reindexed)
    launcher.add_future(f);

  return rt->execute_task(ctx, launcher);
}

Table
ReindexedTableTask::base_impl(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *rt) {

  const TaskArgs* args = static_cast<TaskArgs*>(task->args);

  Keywords kws;
  if (args->kws_region_offset != -1)
    kws = Keywords::clone(
      ctx,
      rt,
      Keywords::pair<PhysicalRegion>{
        regions[args->kws_region_offset],
        regions[args->kws_region_offset + 1]});

  std::vector<int> index_axes;
  std::copy(
    args->index_axes,
    args->index_axes + args->num_index_axes,
    std::back_inserter(index_axes));

  std::vector<Column> cols;
  std::transform(
    task->futures.begin(),
    task->futures.end(),
    std::back_inserter(cols),
    [](auto& f) {
      return f.template get_result<Column>();
    });

  return
    Table::create(
      ctx,
      rt,
      Table::name(regions[0]),
      Table::axes_uid(regions[0]),
      index_axes,
      cols,
      kws);
}

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
class HYPERION_LOCAL IndexAccumulateTask {
public:

  typedef DataType<ValueType<T>::DataType> DT;

  static TaskID TASK_ID;
  static char TASK_NAME[40];
  static const constexpr size_t min_block_size = 10000;

  IndexAccumulateTask(const RegionRequirement& col_req)
    : m_col_req(col_req) {
  }

  Future
  dispatch(Context ctx, Runtime* rt) {

    IndexPartition ip =
      partition_over_all_cpus(
        ctx,
        rt,
        m_col_req.region.get_index_space(),
        min_block_size);
    IndexSpace cs = rt->get_index_partition_color_space_name(ctx, ip);
    LogicalPartition col_lp =
      rt->get_logical_partition(ctx, m_col_req.region, ip);

    IndexTaskLauncher
      launcher(TASK_ID, cs, TaskArgument(NULL, 0), ArgumentMap());
    launcher.add_region_requirement(
      RegionRequirement(
        col_lp,
        0,
        READ_ONLY,
        EXCLUSIVE,
        m_col_req.region));
    launcher.add_field(0, Column::VALUE_FID);
    auto result =
      rt->execute_index_space(
        ctx,
        launcher,
        OpsManager::reduction_id(DT::af_redop_id));
    rt->destroy_logical_partition(ctx, col_lp);
    rt->destroy_index_space(ctx, cs);
    rt->destroy_index_partition(ctx, ip);
    return result;
  }

  static acc_field_redop_rhs<T>
  base_impl(
    const Task* task,
    const std::vector<PhysicalRegion>& regions,
    Context,
    Runtime* rt) {

    std::map<T, std::vector<DomainPoint>> pts;
    switch (task->regions[0].region.get_index_space().get_dim()) {
#define ACC(D)                                                          \
      case (D): {                                                       \
        const ROAccessor<T, D> acc(regions[0], Column::VALUE_FID);      \
        for (PointInDomainIterator<D> pid(                              \
               rt->get_index_space_domain(                              \
                 task->regions[0].region.get_index_space()));           \
             pid();                                                     \
             pid++) {                                                   \
          if (pts.count(acc[*pid]) == 0)                                \
            pts[acc[*pid]] = std::vector<DomainPoint>();                \
          pts[acc[*pid]].push_back(*pid);                               \
        }                                                               \
        break;                                                          \
      }
      HYPERION_FOREACH_N(ACC);
#undef ACC
    default:
      assert(false);
      break;
    }
    acc_field_redop_rhs<T> result;
    result.v.reserve(pts.size());
    std::copy(pts.begin(), pts.end(), std::back_inserter(result.v));
    return result;
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

private:

  RegionRequirement m_col_req;
};

template <typename T>
TaskID IndexAccumulateTask<T>::TASK_ID;

template <typename T>
char IndexAccumulateTask<T>::TASK_NAME[40];

TaskID IndexColumnTask::TASK_ID;

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
  hyperion::TypeTag dt,
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
        OpsManager::serdez_id(OpsManager::V_DOMAIN_POINT_SID));
    }
    IndexSpaceT<1> result_is =
      runtime->create_index_space(ctx, Rect<1>(0, acc.size() - 1));
    result_lr = runtime->create_logical_region(ctx, result_is, result_fs);

    // transfer values and row numbers from acc_lr to result_lr
    RegionRequirement result_req(result_lr, WRITE_ONLY, EXCLUSIVE, result_lr);
    result_req.add_field(IndexColumnTask::VALUE_FID);
    result_req.add_field(IndexColumnTask::ROWS_FID);
    PhysicalRegion result_pr = runtime->map_region(ctx, result_req);
    const WOAccessor<T, 1> values(result_pr, IndexColumnTask::VALUE_FID);
    const WOAccessor<std::vector<DomainPoint>, 1>
      rns(result_pr, IndexColumnTask::ROWS_FID);
    for (size_t i = 0; i < acc.size(); ++i) {
      ::new (rns.ptr(i)) std::vector<DomainPoint>;
      tie(values[i], rns[i]) = acc[i];
    }
    runtime->unmap_region(ctx, result_pr);
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
    HYPERION_FOREACH_DATATYPE(ICR);
#undef ICR
  default:
    assert(false);
    break;
  }
  return result;
}

#ifdef HIERARCHICAL_COMPUTE_RECTANGLES

class HYPERION_LOCAL ComputeRectanglesTask {
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
    req.add_field(ReindexColumnTask::ROW_RECTS_FID);
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
              false> rects(regions.back(), ReindexColumnTask::ROW_RECTS_FID); \
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
            HYPERION_FOREACH_MN(WRITE_RECTS);
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

class HYPERION_LOCAL ComputeRectanglesTask {
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
  dispatch(Context ctx, Runtime* rt) {

    IndexTaskLauncher launcher;
    std::unique_ptr<char[]> args_buffer;
    TaskArgs args{m_allow_rows, m_row_partition};
    args_buffer = std::make_unique<char[]>(args.serialized_size());
    args.serialize(args_buffer.get());

    //LogicalRegion new_rects_lr = m_new_rects.get_logical_region();
    // AcquireLauncher acquire(new_rects_lr, new_rects_lr, m_new_rects);
    // acquire.add_field(ReindexColumnTask::ROW_RECTS_FID);
    // PhaseBarrier acquired = rt->create_phase_barrier(ctx, 1);
    // acquire.add_arrival_barrier(acquired);
    // rt->issue_acquire(ctx, acquire);

    //PhaseBarrier released;
    Domain bounds;

    switch (m_ix_columns.size()) {
#define INIT_LAUNCHER(DIM)                                        \
      case DIM: {                                                 \
        Rect<DIM> rect;                                           \
        for (size_t i = 0; i < DIM; ++i) {                        \
          IndexSpaceT<1> cis(m_ix_columns[i].get_index_space());  \
          Rect<1> dom = rt->get_index_space_domain(cis).bounds;   \
          rect.lo[i] = dom.lo[0];                                 \
          rect.hi[i] = dom.hi[0];                                 \
        }                                                         \
        bounds = rect;                                            \
        break;                                                    \
      }
      HYPERION_FOREACH_N(INIT_LAUNCHER);
#undef INIT_LAUNCHER
      default:
        assert(false);
      break;
    }

    /*released = rt->create_phase_barrier(ctx, rect.volume());*/
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
        req.add_field(IndexColumnTask::ROWS_FID);
        launcher.add_region_requirement(req);
      });

    RegionRequirement
      req(m_new_rects_lr, WRITE_DISCARD, SIMULTANEOUS, m_new_rects_lr);
    req.add_field(ReindexColumnTask::ROW_RECTS_FID);
    launcher.add_region_requirement(req);

    rt->execute_index_space(ctx, launcher);

    // PhaseBarrier complete = rt->advance_phase_barrier(ctx, released);
    // ReleaseLauncher release(new_rects_lr, new_rects_lr, m_new_rects);
    // release.add_field(ReindexColumnTask::ROW_RECTS_FID);
    // release.add_wait_barrier(complete);
    // rt->issue_release(ctx, release);
  }

  static void
  base_impl(
    const Task* task,
    const std::vector<PhysicalRegion>& regions,
    Context ctx,
    Runtime *rt) {

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
      rows_acc_t rows(regions[0], IndexColumnTask::ROWS_FID);
      common_rows = rows[task->index_point[0]];
    }
    for (size_t i = 1; i < ixdim; ++i) {
      rows_acc_t rows(regions[i], IndexColumnTask::ROWS_FID);
      common_rows = intersection(common_rows, rows[task->index_point[i]]);
    }

    if (common_rows.size() > 0
        && (args.allow_rows || common_rows.size() == 1)) {
      auto rowdim = common_rows[0].get_dim();
      auto rectdim =
        ixdim + (args.allow_rows ? 1 : 0)
        + args.row_partition.get_dim() - rowdim;
      switch (rowdim * LEGION_MAX_DIM + rectdim) {
#define WRITE_RECTS(ROWDIM, RECTDIM)                                    \
        case (ROWDIM * LEGION_MAX_DIM + RECTDIM): {                     \
          const FieldAccessor<                                          \
            WRITE_DISCARD, \
            Rect<RECTDIM>, \
            ROWDIM> rects(regions.back(), ReindexColumnTask::ROW_RECTS_FID); \
                                                                        \
          for (size_t i = 0; i < common_rows.size(); ++i) {             \
            Domain row_d =                                              \
              rt->get_index_space_domain(                               \
                ctx,                                                    \
                rt->get_index_subspace(                                 \
                  ctx,                                                  \
                  args.row_partition,                                   \
                  common_rows[i]));                                     \
            Rect<RECTDIM> row_rect;                                     \
            size_t j = 0;                                               \
            for (; j < ixdim; ++j) {                                    \
              row_rect.lo[j] = task->index_point[j];                    \
              row_rect.hi[j] = task->index_point[j];                    \
            }                                                           \
            if (args.allow_rows) {                                      \
              row_rect.lo[j] = i;                                       \
              row_rect.hi[j] = i;                                       \
              ++j;                                                      \
            }                                                           \
            size_t k = j;                                               \
            for (; j < RECTDIM; ++j) {                                  \
              row_rect.lo[j] = row_d.lo()[j - k + ROWDIM];              \
              row_rect.hi[j] = row_d.hi()[j - k + ROWDIM];              \
            }                                                           \
            rects[common_rows[i]] = row_rect;                           \
          }                                                             \
          break;                                                        \
        }
        HYPERION_FOREACH_MN(WRITE_RECTS);
#undef WRITE_RECTS
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

class HYPERION_LOCAL ReindexColumnCopyTask {
public:

  static TaskID TASK_ID;
  static const char* TASK_NAME;
  static const unsigned min_block_size = 10000;

  ReindexColumnCopyTask(
    const Column& column,
    hyperion::TypeTag column_dt,
    ColumnPartition row_partition,
    LogicalRegion new_rects_lr,
    LogicalRegion new_col_lr)
    : m_column(column)
    , m_column_dt(column_dt)
    , m_row_partition(row_partition)
    , m_new_rects_lr(new_rects_lr)
    , m_new_col_lr(new_col_lr) {
  }

  template <hyperion::TypeTag DT, int DIM>
  using SA = FieldAccessor<
    READ_ONLY,
    typename DataType<DT>::ValueType,
    DIM,
    coord_t,
    AffineAccessor<typename DataType<DT>::ValueType, DIM, coord_t>,
    true>;

  template <hyperion::TypeTag DT, int DIM>
  using DA = FieldAccessor<
    WRITE_ONLY,
    typename DataType<DT>::ValueType,
    DIM,
    coord_t,
    AffineAccessor<typename DataType<DT>::ValueType, DIM, coord_t>,
    true>;

  template <int DDIM, int RDIM>
  using RA = FieldAccessor<
    READ_ONLY,
    Rect<DDIM>,
    RDIM,
    coord_t,
    AffineAccessor<Rect<DDIM>, RDIM, coord_t>,
    true>;

  template <hyperion::TypeTag DT>
  static void
  copy(
    const Task* task,
    const std::vector<PhysicalRegion>& regions,
    Runtime* rt) {

    const RegionRequirement& rect_req = task->regions[1];
    const PhysicalRegion& src = regions[0];
    const PhysicalRegion& rect = regions[1];
    const PhysicalRegion& dst = regions[2];
    int rowdim = rect.get_logical_region().get_dim();
    int srcdim = src.get_logical_region().get_dim();
    int dstdim = dst.get_logical_region().get_dim();

    switch ((rowdim * LEGION_MAX_DIM + srcdim) * LEGION_MAX_DIM + dstdim) {
#define CPY(ROWDIM,SRCDIM,DSTDIM)                                       \
      case ((ROWDIM * LEGION_MAX_DIM + SRCDIM) * LEGION_MAX_DIM + DSTDIM): { \
        const SA<DT,SRCDIM> from(src, Column::VALUE_FID);               \
        const RA<DSTDIM,ROWDIM> rct(rect, ReindexColumnTask::ROW_RECTS_FID); \
        const DA<DT,DSTDIM> to(dst, Column::VALUE_FID);                 \
        for (PointInDomainIterator<ROWDIM> row(                         \
               rt->get_index_space_domain(rect_req.region.get_index_space()), \
               false);                                                  \
             row();                                                     \
             ++row) {                                                   \
          Point<SRCDIM> ps;                                             \
          for (size_t i = 0; i < ROWDIM; ++i)                           \
            ps[i] = row[i];                                             \
          for (PointInRectIterator<DSTDIM> pd(rct[*row], false); pd(); pd++) { \
            size_t i = SRCDIM - 1;                                      \
            size_t j = DSTDIM - 1;                                      \
            while (i >= ROWDIM)                                         \
              ps[i--] = pd[j--];                                        \
            to[*pd] = from[ps];                                         \
          }                                                             \
        }                                                               \
        break;                                                          \
      }
      HYPERION_FOREACH_LMN(CPY)
#undef CPY
      default:
        assert(false);
        break;
    }
  }

  void
  dispatch(Context ctx, Runtime* rt) {

    // use partition of m_new_rects_lr by m_row_partition to get partition of
    // m_new_col_lr index space

    IndexSpace old_rows_is = m_new_rects_lr.get_index_space();
    IndexPartition old_rows_ip =
      partition_over_all_cpus(ctx, rt, old_rows_is, min_block_size);
    IndexSpace cs = rt->get_index_partition_color_space_name(ctx, old_rows_ip);
    LogicalPartition new_rects_lp =
      rt->get_logical_partition(ctx, m_new_rects_lr, old_rows_ip);

    // we now have partitions over the same color space on both m_new_rects_lr
    // and m_new_col_lr
    IndexTaskLauncher
      copier(
        TASK_ID,
        cs,
        TaskArgument(&m_column_dt, sizeof(m_column_dt)),
        ArgumentMap());
    {
      auto cp =
        m_column.projected_column_partition(
          ctx,
          rt,
          ColumnPartition(
            m_row_partition.axes_uid_lr,
            m_row_partition.axes_lr,
            old_rows_ip));
      LogicalPartition lp =
        rt->get_logical_partition(ctx, m_column.values_lr, cp.index_partition);
      RegionRequirement req(lp, 0, READ_ONLY, EXCLUSIVE, m_column.values_lr);
      req.add_field(Column::VALUE_FID);
      copier.add_region_requirement(req);
      // FIXME: clean up cp, lp
    }
    {
      RegionRequirement
        req(new_rects_lp, 0, READ_ONLY, EXCLUSIVE, m_new_rects_lr);
      req.add_field(ReindexColumnTask::ROW_RECTS_FID);
      copier.add_region_requirement(req);
    }
    {
      IndexPartition ip =
        rt->create_partition_by_image_range(
          ctx,
          m_new_col_lr.get_index_space(),
          new_rects_lp,
          m_new_rects_lr,
          ReindexColumnTask::ROW_RECTS_FID,
          cs,
          DISJOINT_COMPLETE_KIND);
      LogicalPartition lp =
        rt->get_logical_partition(ctx, m_new_col_lr, ip);
      RegionRequirement req(lp, 0, WRITE_ONLY, EXCLUSIVE, m_new_col_lr);
      req.add_field(Column::VALUE_FID);
      copier.add_region_requirement(req);
      // FIXME: clean up ip, lp
    }
    // FIXME: clean up
    rt->execute_index_space(ctx, copier);
  }

  static void
  base_impl(
    const Task* task,
    const std::vector<PhysicalRegion>& regions,
    Context,
    Runtime* rt) {

    hyperion::TypeTag dt = *static_cast<hyperion::TypeTag*>(task->args);

    switch (dt) {
#define CPYDT(DT)                               \
      case DT:                                  \
        copy<DT>(task, regions, rt);            \
        break;
      HYPERION_FOREACH_DATATYPE(CPYDT)
#undef CPYDT
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
    // registrar.set_replicable();
    Runtime::preregister_task_variant<base_impl>(registrar, TASK_NAME);
  }

private:

  Column m_column;

  hyperion::TypeTag m_column_dt;

  ColumnPartition m_row_partition;

  LogicalRegion m_new_rects_lr;

  LogicalRegion m_new_col_lr;
};

TaskID ReindexColumnCopyTask::TASK_ID;
const char* ReindexColumnCopyTask::TASK_NAME = "ReindexColumnCopyTask";

TaskID ReindexColumnTask::TASK_ID;
const char* ReindexColumnTask::TASK_NAME = "ReindexColumnTask";

ReindexColumnTask::ReindexColumnTask(
  const Column& col,
  bool is_index,
  const std::vector<int>& col_axes,
  ssize_t row_axis_offset,
  const std::vector<std::tuple<int, LogicalRegion>>& ixcols,
  bool allow_rows)
  : m_is_index(is_index)
  , m_col_axes(col_axes)
  , m_row_axis_offset(row_axis_offset) {

  assert(row_axis_offset >= 0);
  assert(!m_is_index || ixcols.size() == 1);

  m_args.allow_rows = allow_rows;
  m_args.col = col;

  unsigned i = 0;
  for (auto& [ix, lr] : ixcols) {
    m_args.index_axes[i++] = ix;
    m_ixlrs.push_back(lr);
  }
  m_args.num_index_axes = i;
  m_args.values_region_offset = -1;
  m_args.kws_region_offset = -1;
  m_args.mr_region_offset = -1;
}

enum ReindexColumnRegionIndexes {
  METADATA,
  AXES,
  PART_AXES,
  PART_AUID,
  INDEX_COLS
};

Future
ReindexColumnTask::dispatch(Context ctx, Runtime* rt) {

  // get column partition down to row axis
  std::vector<int> col_part_axes;
  std::copy_n(
    m_col_axes.begin(),
    m_row_axis_offset + 1,
    std::back_inserter(col_part_axes));
  m_args.row_partition = m_args.col.partition_on_axes(ctx, rt, col_part_axes);

  std::vector<RegionRequirement> reqs;
  {
    static_assert(ReindexColumnRegionIndexes::METADATA == 0);
    RegionRequirement
      req(m_args.col.metadata_lr, READ_ONLY, EXCLUSIVE, m_args.col.metadata_lr);
    req.add_field(Column::METADATA_NAME_FID);
    req.add_field(Column::METADATA_AXES_UID_FID);
    req.add_field(Column::METADATA_DATATYPE_FID);
    req.add_field(Column::METADATA_REF_COL_FID);
    reqs.push_back(req);
  }
  {
    static_assert(ReindexColumnRegionIndexes::AXES == 1);
    RegionRequirement
      req(m_args.col.axes_lr, READ_ONLY, EXCLUSIVE, m_args.col.axes_lr);
    req.add_field(Column::AXES_FID);
    reqs.push_back(req);
  }
  {
    static_assert(ReindexColumnRegionIndexes::PART_AXES == 2);
    RegionRequirement
      req(m_args.row_partition.axes_lr,
          READ_ONLY,
          EXCLUSIVE,
          m_args.row_partition.axes_lr);
    req.add_field(ColumnPartition::AXES_FID);
    reqs.push_back(req);
  }
  {
    static_assert(ReindexColumnRegionIndexes::PART_AUID == 3);
    RegionRequirement
      req(m_args.row_partition.axes_uid_lr,
          READ_ONLY,
          EXCLUSIVE,
          m_args.row_partition.axes_uid_lr);
    req.add_field(ColumnPartition::AXES_UID_FID);
    reqs.push_back(req);
  }
  static_assert(ReindexColumnRegionIndexes::INDEX_COLS == 4);
  FieldID ix_fid =
    m_is_index ? IndexColumnTask::VALUE_FID : IndexColumnTask::ROWS_FID;
  for (auto& lr : m_ixlrs) {
    assert(lr != LogicalRegion::NO_REGION);
    RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
    req.add_field(ix_fid);
    reqs.push_back(req);
  }
  if (!m_is_index) {
    m_args.values_region_offset = reqs.size();
    RegionRequirement
      req(m_args.col.values_lr, READ_ONLY, EXCLUSIVE, m_args.col.values_lr);
    req.add_field(Column::VALUE_FID);
    reqs.push_back(req);  
  }
  if (!m_args.col.keywords.is_empty()) {
    m_args.kws_region_offset = reqs.size();
    auto kwsz = m_args.col.keywords.size(rt);
    std::vector<FieldID> fids(kwsz);
    std::iota(fids.begin(), fids.end(), 0);
    auto kwreqs = m_args.col.keywords.requirements(rt, fids, READ_ONLY).value();
    reqs.push_back(kwreqs.type_tags);
    reqs.push_back(kwreqs.values);
  }
  if (!m_args.col.meas_ref.is_empty()) {
    m_args.mr_region_offset = reqs.size();
    auto [mrq, vrq, oirq] = m_args.col.meas_ref.requirements(READ_ONLY);
    reqs.push_back(mrq);
    reqs.push_back(vrq);
    if (oirq)
      reqs.push_back(oirq.value());
  }

  TaskLauncher launcher(TASK_ID, TaskArgument(&m_args, sizeof(m_args)));
  for (auto& req : reqs)
    launcher.add_region_requirement(req);
  return rt->execute_task(ctx, launcher);
}

template <int OLDDIM, int NEWDIM>
static Column
reindex_column(
  const Task* task,
  const ReindexColumnTask::TaskArgs& args,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *rt) {

  Rect<OLDDIM> col_domain =
    rt->get_index_space_domain(
      ctx,
      task->regions[args.values_region_offset].region.get_index_space());
  // we use the name "rows_is" for the index space at or above the "ROW" axis
  IndexSpace rows_is =
    rt->get_index_partition_color_space_name(
      ctx,
      args.row_partition.index_partition);
  // logical region over rows_is with a field for the rectangle in the new
  // column index space for each value in row_is
  auto new_rects_fs = rt->create_field_space(ctx);
  {
    auto fa = rt->create_field_allocator(ctx, new_rects_fs);
    fa.allocate_field(sizeof(Rect<NEWDIM>), ReindexColumnTask::ROW_RECTS_FID);

    LayoutConstraintRegistrar lc(new_rects_fs);
    add_row_major_order_constraint(lc, rows_is.get_dim())
      .add_constraint(MemoryConstraint(Memory::Kind::GLOBAL_MEM));
    // TODO: free LayoutConstraintID returned from following call...maybe
    // generate field spaces and constraints once at startup
    rt->register_layout(lc);
  }
  // new_rects_lr is a mapping from current column row index to a rectangle in
  // the new (reindexed) column index space
  auto new_rects_lr = rt->create_logical_region(ctx, rows_is, new_rects_fs);

  // initialize new_rects_lr values to empty rectangles
  Rect<NEWDIM> empty;
  empty.lo[0] = 0;
  empty.hi[0] = -1;
  assert(empty.empty());
  rt->fill_field(
    ctx,
    new_rects_lr,
    new_rects_lr,
    ReindexColumnTask::ROW_RECTS_FID,
    empty);

  std::vector<LogicalRegion> ix_lrs;
  ix_lrs.reserve(args.num_index_axes);
  for (size_t i = 0; i < args.num_index_axes; ++i)
    ix_lrs.push_back(
      regions[i + ReindexColumnRegionIndexes::INDEX_COLS].get_logical_region());

  // task to compute new index space rectangle for each row in column
#ifdef HIERARCHICAL_COMPUTE_RECTANGLES
  ComputeRectanglesTask
    new_rects_task(
      args.allow_rows,
      args.row_partition.index_partition,
      ix_lrs,
      new_rects_lr,
      {},
      {},
      {});
#else
  ComputeRectanglesTask
    new_rects_task(
      args.allow_rows,
      args.row_partition.index_partition,
      ix_lrs,
      new_rects_lr);
#endif
  new_rects_task.dispatch(ctx, rt);

  const Column::AxesAccessor<READ_ONLY, true>
    col_axes(regions[ReindexColumnRegionIndexes::AXES], Column::AXES_FID);
  // create the new index space via create_partition_by_image_range based on
  // new_rects_lr; this index space should be exact (i.e, appropriately sparse
  // or dense), but we start with the bounding index space first
  Rect<NEWDIM> new_bounds;
  std::vector<int> new_axes(NEWDIM);
  {
    // start with axes above original row axis
    auto d =
      rt
      ->get_index_partition_color_space(args.row_partition.index_partition)
      .get_dim()
      - 1;
    int i = 0; // index in new_bounds
    int j = 1; // index in col arrays
    while (i < d) {
      new_bounds.lo[i] = col_domain.lo[j];
      new_bounds.hi[i] = col_domain.hi[j];
      new_axes[i] = col_axes[j];
      ++i; ++j;
    }
    // append new index axes
    for (size_t k = 0; k < ix_lrs.size(); ++k) {
      Rect<1> ix_domain =
        rt->get_index_space_domain(ix_lrs[k].get_index_space());
      new_bounds.lo[i] = ix_domain.lo[0];
      new_bounds.hi[i] = ix_domain.hi[0];
      new_axes[i] = args.index_axes[k];
      ++i;
    }
    // append row axis, if allowed
    if (args.allow_rows) {
      new_bounds.lo[i] = col_domain.lo[j];
      new_bounds.hi[i] = col_domain.hi[j];
      assert(col_axes[j] == 0);
      new_axes[i] = 0;
      ++i; ++j;
    }
    // append remaining (ctds element-level) axes
    while (i < NEWDIM) {
      assert(j < OLDDIM);
      new_bounds.lo[i] = col_domain.lo[j];
      new_bounds.hi[i] = col_domain.hi[j];
      new_axes[i] = col_axes[j];
      ++i; ++j;
    }
  }
  auto new_bounds_is = rt->create_index_space(ctx, new_bounds);

  // now reduce the bounding index space to the exact, possibly sparse index
  // space of the reindexed column

  // to do this, we need a logical partition of new_rects_lr, which will
  // comprise a single index subspace
  IndexSpaceT<1> all_rows_cs = rt->create_index_space(ctx, Rect<1>(0, 0));
  auto all_rows_ip =
    rt->create_equal_partition(ctx, rows_is, all_rows_cs);
  auto all_rows_new_rects_lp =
    rt->get_logical_partition(ctx, new_rects_lr, all_rows_ip);
  // those rows in rows_is that are mapped to an empty rectangle correspond to
  // indexes in rows_is that are not present in the exact cross-product index
  // space, and the create_partition_by_image_range function will leave those
  // indexes out of the resulting partition, leaving the index space we're
  // looking for
  IndexPartitionT<NEWDIM> new_bounds_ip(
    rt->create_partition_by_image_range(
      ctx,
      new_bounds_is,
      all_rows_new_rects_lp,
      new_rects_lr,
      ReindexColumnTask::ROW_RECTS_FID,
      all_rows_cs));
  // new_col_is is the exact index space of the reindexed column
  IndexSpaceT<NEWDIM> new_col_is(rt->get_index_subspace(new_bounds_ip, 0));

  LogicalRegion new_col_lr;

  const Column::DatatypeAccessor<READ_ONLY> col_datatype(
    regions[ReindexColumnRegionIndexes::METADATA],
    Column::METADATA_DATATYPE_FID);
  // if reindexing failed, new_col_is should be empty
  if (!rt->get_index_space_domain(ctx, new_col_is).empty()) {
    // finally, we create the new column logical region
    auto new_col_fs = rt->create_field_space(ctx);
    {
      auto fa = rt->create_field_allocator(ctx, new_col_fs);
      add_field(col_datatype[0], fa, Column::VALUE_FID);
    }
    new_col_lr = rt->create_logical_region(ctx, new_col_is, new_col_fs);

    // copy values from the column logical region to new_col_lr
    ReindexColumnCopyTask
      copy_task(
        args.col,
        col_datatype[0],
        args.row_partition,
        new_rects_lr,
        new_col_lr);
    copy_task.dispatch(ctx, rt);
  }

  rt->destroy_field_space(ctx, new_rects_fs);
  rt->destroy_logical_region(ctx, new_rects_lr);

  rt->destroy_index_space(ctx, all_rows_cs);
  rt->destroy_index_partition(ctx, all_rows_ip);

  rt->destroy_index_space(ctx, new_bounds_is);
  rt->destroy_index_partition(ctx, new_bounds_ip);

  Keywords kws;
  MeasRef mr;
  if (args.kws_region_offset != -1)
    kws = Keywords::clone(
      ctx,
      rt,
      Keywords::pair<PhysicalRegion>{
        regions[args.kws_region_offset],
        regions[args.kws_region_offset + 1]});
  if (args.mr_region_offset != -1) {
    MeasRef::DataRegions dr;
    dr.metadata = regions[args.mr_region_offset];
    dr.values = regions[args.mr_region_offset + 1];
    if (args.mr_region_offset + 2 < (int)regions.size())
      dr.index = regions[args.mr_region_offset + 2];
    mr = MeasRef::clone(ctx, rt, dr);
  }

  const Column::NameAccessor<READ_ONLY> col_name(
    regions[ReindexColumnRegionIndexes::METADATA],
    Column::METADATA_NAME_FID);
  const Column::AxesUidAccessor<READ_ONLY> col_axes_uid(
    regions[ReindexColumnRegionIndexes::METADATA],
    Column::METADATA_AXES_UID_FID);
  const Column::RefColAccessor<READ_ONLY> ref_col(
    regions[ReindexColumnRegionIndexes::METADATA],
    Column::METADATA_REF_COL_FID);
  return
    Column::create(
      ctx,
      rt,
      col_name[0],
      col_axes_uid[0],
      new_axes,
      col_datatype[0],
      new_col_lr,
#ifdef HYPERION_USE_CASACORE
      mr,
      ((ref_col[0].size() > 0)
       ? std::make_optional<std::string>(ref_col[0])
       : std::nullopt),
#endif
      kws);
}

Column
ReindexColumnTask::base_impl(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *rt) {

  const TaskArgs* args = static_cast<const TaskArgs*>(task->args);

  if (args->values_region_offset != -1) {
    // reindex this column using index column values
    auto ip = args->row_partition.index_partition;
    auto olddim = ip.get_dim();
    auto eltdim =
      olddim
      - rt->get_index_partition_color_space(ctx, ip).get_dim();
    auto newdim =
      args->num_index_axes + eltdim + (args->allow_rows ? 1 : 0);
    switch (olddim * LEGION_MAX_DIM + newdim) {
#define REINDEX_COLUMN(OLDDIM, NEWDIM)            \
      case (OLDDIM * LEGION_MAX_DIM + NEWDIM): {  \
        return                                    \
          reindex_column<OLDDIM, NEWDIM>(         \
            task,                                 \
            *args,                                \
            regions,                              \
            ctx,                                  \
            rt);                                  \
        break;                                    \
      }
      HYPERION_FOREACH_MN(REINDEX_COLUMN);
#undef REINDEX_COLUMN
      default:
        assert(false);
        return Column(); // keep compiler happy
        break;
    }
  } else {
    // this column is an index column, copy values out of provided index column
    // into a new logical region from which to construct a Column
    LogicalRegion new_col_lr;
    {
      auto ixlr = task->regions[ReindexColumnRegionIndexes::INDEX_COLS].region;
      auto sz = rt->get_index_space_domain(ixlr.get_index_space()).get_volume();
      IndexSpace is = rt->create_index_space<1>(ctx, Rect<1>(0, sz - 1));
      FieldSpace fs = rt->create_field_space(ctx);
      {
        FieldAllocator fa = rt->create_field_allocator(ctx, fs);
        // NB: the following only works for fields without serdez, which ought
        // to be the case
        fa.allocate_field(
          rt->get_field_size(
            ixlr.get_field_space(),
            IndexColumnTask::VALUE_FID),
          Column::VALUE_FID);
      }
      new_col_lr = rt->create_logical_region(ctx, is, fs);
      {
        CopyLauncher copier;
        RegionRequirement
          src_req(ixlr, READ_ONLY, EXCLUSIVE, ixlr);
        RegionRequirement
          dst_req(new_col_lr, WRITE_ONLY, EXCLUSIVE, new_col_lr);
        copier.add_copy_requirements(src_req, dst_req);
        copier.add_src_field(0, IndexColumnTask::VALUE_FID);
        copier.add_dst_field(0, Column::VALUE_FID);
        rt->issue_copy_operation(ctx, copier);
      }
    }

    Keywords kws;
    MeasRef mr;
    if (args->kws_region_offset != -1)
      kws = Keywords::clone(
        ctx,
        rt,
        Keywords::pair<PhysicalRegion>{
          regions[args->kws_region_offset],
          regions[args->kws_region_offset + 1]});
    if (args->mr_region_offset != -1) {
      MeasRef::DataRegions dr;
      dr.metadata = regions[args->mr_region_offset];
      dr.values = regions[args->mr_region_offset + 1];
      if (args->mr_region_offset + 2 < (int)regions.size())
        dr.index = regions[args->mr_region_offset + 2];
      mr = MeasRef::clone(ctx, rt, dr);
    }

    std::vector<int> new_axes{args->index_axes[0]};

    const Column::DatatypeAccessor<READ_ONLY> col_datatype(
      regions[ReindexColumnRegionIndexes::METADATA],
      Column::METADATA_DATATYPE_FID);
    const Column::NameAccessor<READ_ONLY> col_name(
      regions[ReindexColumnRegionIndexes::METADATA],
      Column::METADATA_NAME_FID);
    const Column::AxesUidAccessor<READ_ONLY> col_axes_uid(
      regions[ReindexColumnRegionIndexes::METADATA],
      Column::METADATA_AXES_UID_FID);
    const Column::RefColAccessor<READ_ONLY> ref_col(
      regions[ReindexColumnRegionIndexes::METADATA],
      Column::METADATA_REF_COL_FID);

    return
      Column::create(
        ctx,
        rt,
        col_name[0],
        col_axes_uid[0],
        new_axes,
        col_datatype[0],
        new_col_lr,
#ifdef HYPERION_USE_CASACORE
        mr,
        ((ref_col[0].size() > 0)
         ? std::make_optional<std::string>(ref_col[0])
         : std::nullopt),
#endif
        kws);
  }
}

void
ReindexColumnTask::preregister_task() {
  TASK_ID = Runtime::generate_static_task_id();
  TaskVariantRegistrar registrar(TASK_ID, TASK_NAME, false);
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  registrar.set_idempotent();
  // registrar.set_replicable();
  Runtime::preregister_task_variant<Column,base_impl>(
    registrar,
    TASK_NAME);
}

Future/*Table*/
Table::ireindexed(
  Context ctx,
  Runtime* rt,
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
  std::vector<int> ixax = index_axes(ctx, rt);
  if ((ixax.size() > 1) || (ixax.back() != 0)) {
    Table empty;
    return Future::from_value(rt, empty);
  }

  // map columns_lr
  RegionRequirement cols_req(columns_lr, READ_ONLY, EXCLUSIVE, columns_lr);
  cols_req.add_field(Table::COLUMNS_FID);
  auto cols_pr = rt->map_region(ctx, cols_req);

  auto col_names = Table::column_names(ctx, rt, cols_pr);
  std::unordered_map<std::string, Column> cols;
  std::unordered_map<std::string, std::vector<int>> col_axes;
  for (auto& nm : col_names) {
    cols[nm] = Table::column(ctx, rt, cols_pr, nm);
    col_axes[nm] = cols[nm].axes(ctx, rt);
  }

  // for every column in table, determine which axes need indexing
  std::unordered_map<std::string, std::vector<int>> col_reindex_axes;
  for (auto& nm : col_names) {
    std::vector<int> ax;
    auto cax = col_axes[nm];
    // skip the column if it does not have a "row" axis
    if (cax.front() == 0) { // TODO: needs adjustment if row dimension > 1
      // if column is a reindexing axis, reindexing depends only on itself
      auto myaxis = column_is_axis(axis_names, nm, axes);
      if (myaxis) {
        ax.push_back(myaxis.value());
      } else {
        // select those axes in "axes" that are not already an axis of the
        // column
        for (auto& d : axes)
          if (std::find(cax.begin(), cax.end(), d) == cax.end())
            ax.push_back(d);
      }
      col_reindex_axes[nm] = std::move(ax);
    }
  }

  // the returned Futures contain a LogicalRegion with two fields: at
  // IndexColumnTask::VALUE_FID, the column values (sorted in ascending order);
  // and at IndexColumnTask::ROWS_FID, a sorted vector of DomainPoints in the
  // original column. The LogicalRegions, along with their IndexSpaces and
  // FieldSpaces, should eventually be reclaimed.
  std::unordered_map<int, Future> index_cols;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  for (auto& [nm, ds] : col_reindex_axes)
#pragma GCC diagnostic pop
    for (auto& d : ds)
      if (index_cols.count(d) == 0) {
        IndexColumnTask task(cols[axis_names[d]]);
        index_cols[d] = task.dispatch(ctx, rt);
      }

  // do reindexing of columns
  std::vector<Future> reindexed;
  for (auto& [nm, ds] : col_reindex_axes) {
    // if this column is an index column, we've already launched a task to
    // create its logical region, so we can use that
    bool col_is_index = ds.size() == 1 && index_cols.count(ds[0]) > 0;
    // create reindexing task launcher
    // TODO: start intermediary task dependent on Futures of index columns
    std::vector<std::tuple<int, LogicalRegion>> ixcols;
    for (auto d : ds)
      ixcols.emplace_back(
        d,
        index_cols.at(d).template get_result<LogicalRegion>());
    auto cax = col_axes[nm];
    auto row_axis_offset =
      std::distance(cax.begin(), std::find(cax.begin(), cax.end(), 0));
    ReindexColumnTask
      task(cols[nm], col_is_index, cax, row_axis_offset, ixcols, allow_rows);
    reindexed.push_back(task.dispatch(ctx, rt));
  }

  rt->unmap_region(ctx, cols_pr);

  // launch task that creates the reindexed table
  std::vector<int> iaxes = axes;
  if (allow_rows)
    iaxes.push_back(0);
  ReindexedTableTask task(*this, iaxes, reindexed);
  auto result = task.dispatch(ctx, rt);

  // free logical regions in index_cols (the call to get_result should not cause
  // any delay since the Futures were already waited upon, above)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  for (auto& [d, f] : index_cols) {
#pragma GCC diagnostic pop
    auto lr = f.template get_result<LogicalRegion>();
    rt->destroy_field_space(ctx, lr.get_field_space());
    rt->destroy_index_space(ctx, lr.get_index_space());
  }
  return result;
}

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
    [&ctx, rt, this, &cols, &axis_names, &result](auto a) {
      auto col = column(ctx, rt, cols, axis_names[a]);
      if (!col.is_empty()) {
        IndexColumnTask task(col);
        result[a] = task.dispatch(ctx, rt);
      }
    });
  rt->unmap_region(ctx, cols);
  return result;
}

class HYPERION_LOCAL ComputeRowColorsTask {
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
      HYPERION_FOREACH_N(FILL_COLOR);
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
          OpsManager::reduction_id(OpsManager::POINT_ADD_REDOP(ixdim)),
          ATOMIC,
          m_row_colors);
      req.add_field(0);
      launcher.add_region_requirement(req);
    }
    {
      RegionRequirement
        req(
          m_colors,
          OpsManager::reduction_id(OpsManager::COORD_BOR_REDOP),
          ATOMIC,
          m_colors);
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

    typedef const ROAccessor<std::vector<DomainPoint>, 1> rows_acc_t;

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
          const ReductionAccessor<                                      \
            point_add_redop<COLORDIM>, \
            true, \
            ROWDIM, \
            coord_t, \
            AffineAccessor<Point<COLORDIM>,ROWDIM,coord_t>> \
            colors(                                                     \
              regions[COLORDIM],                                        \
              0,                                                        \
              OpsManager::reduction_id(                                 \
                OpsManager::POINT_ADD_REDOP(ixdim)));                   \
          const ReductionAccessor<                                      \
            coord_bor_redop, \
            true, \
            COLORDIM, \
            coord_t, \
            AffineAccessor<coord_t,COLORDIM,coord_t>> \
            flags(                                                      \
              regions[COLORDIM + 1],                                    \
              0,                                                        \
              OpsManager::reduction_id(OpsManager::COORD_BOR_REDOP));   \
          Point<COLORDIM,coord_t> color(task->index_point);             \
          flags[color] <<= 1;                                           \
          for (size_t i = 0; i < common_rows.size(); ++i) {             \
            Point<ROWDIM> r(common_rows[i]);                            \
            colors[r] <<= color;                                        \
          }                                                             \
          break;                                                        \
        }
        HYPERION_FOREACH_NN(WRITE_COLORS);
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

class HYPERION_LOCAL InitColorsTask {
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
#define COLOR_PARTS(ROW_DIM, COL_DIM)               \
      case (ROW_DIM * LEGION_MAX_DIM + COL_DIM):  { \
        static_assert(ROW_DIM <= COL_DIM);          \
        const ROAccessor<Point<COLOR_DIM>, ROW_DIM> \
          colors(regions[0], 0);                    \
        const WOAccessor<Point<COLOR_DIM>, COL_DIM> \
          parts(regions[1], PART_FID);              \
        Point<ROW_DIM> pt;                          \
        for (size_t i = 0; i < ROW_DIM; ++i)        \
          pt[i] = task->index_point[i];             \
        parts[task->index_point] = colors[pt];      \
        break;                                      \
      }
      HYPERION_FOREACH_MN(COLOR_PARTS);
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
      HYPERION_FOREACH_N(IMPL);
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

class HYPERION_LOCAL PartitionRowsTask {
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
    assert(axes.size() <= HYPERION_MAX_NUM_TABLE_COLUMNS);
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
    runtime->destroy_logical_region(context, lr);
    runtime->destroy_field_space(context, fs);
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
      HYPERION_FOREACH_N(IMPL);
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
    hyperion::string axes_uid;
    int axes[HYPERION_MAX_NUM_TABLE_COLUMNS];
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
      [&ctx, rt, this, &cols, &axis_names](const auto& a) {
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
      [&ctx, rt, this, &cols, &axis_names, &tbl_axes](const auto& a) {
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
      HYPERION_FOREACH_N(ROW_COLOR_INIT);
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

  // TODO: I'm not sure that destroying color_flag_ip is OK, as one of its
  // sub-spaces is used by PartitionRowsTask, which would then be the color
  // space of the row partition; alternatively might need caller to manage
  // its deletion somehow
  rt->destroy_index_partition(ctx, color_flag_ip);
  rt->destroy_index_space(ctx, flags_cs);

  rt->destroy_field_space(ctx, row_color_fs);
  // don't destroy row_colors_lr.get_index_space() (== rows_is)
  rt->destroy_logical_region(ctx, row_colors_lr);

  rt->destroy_field_space(ctx, colors_fs);
  rt->destroy_index_space(ctx, colors_is);
  rt->destroy_logical_region(ctx, colors_lr);

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
  HYPERION_FOREACH_DATATYPE(PREREG_IDX_ACCUMULATE);
#undef PREREG_IDX_ACCUMULATE
  IndexColumnTask::preregister_task();
  InitColorsTask::preregister_task();
  ComputeRectanglesTask::preregister_task();
  ReindexColumnTask::preregister_task();
  ReindexColumnCopyTask::preregister_task();
  ReindexedTableTask::preregister_task();
}

#ifdef HYPERION_USE_CASACORE

Table
Table::from_ms(
  Context ctx,
  Runtime* runtime,
  const CXX_FILESYSTEM_NAMESPACE::path& path,
  const std::unordered_set<std::string>& column_selections) {

  std::string table_name = path.filename();

#define FROM_MS_TABLE(N)                                                \
  do {                                                                  \
    if (table_name == MSTable<MS_##N>::name)                            \
      return                                                            \
        hyperion:: template from_ms<MS_##N>(ctx, runtime, path, column_selections); \
  } while (0);

  HYPERION_FOREACH_MS_TABLE(FROM_MS_TABLE);

  // try to read as main table
  return
    hyperion:: template from_ms<MS_MAIN>(ctx, runtime, path, column_selections);

#undef FROM_MS_TABLE
}

#endif // HYPERION_USE_CASACORE

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
