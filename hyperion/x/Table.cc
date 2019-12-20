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
#include <hyperion/x/Table.h>

#include <unordered_set>

using namespace hyperion::x;

using namespace Legion;

struct InitTaskColArgs {
  hyperion::string nm;
  hyperion::TypeTag dt;
  hyperion::Keywords kw;
  hyperion::MeasRef mr;
};

struct InitTaskArgs {
  std::array<InitTaskColArgs, Table::MAX_COLUMNS> column_args;
};

TaskID Table::init_task_id;

const char* Table::init_task_name = "x::Table::init_task";

void
Table::init_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *rt) {

  const InitTaskArgs* args = static_cast<const InitTaskArgs*>(task->args);
  assert(regions.size() == 1);
  const ColumnsNameAccessor<WRITE_ONLY> nms(regions[0], COLUMNS_NM_FID);
  const ColumnsDatatypeAccessor<WRITE_ONLY> dts(regions[0], COLUMNS_DT_FID);
  const ColumnsKeywordsAccessor<WRITE_ONLY> kws(regions[0], COLUMNS_KW_FID);
  const ColumnsMeasRefAccessor<WRITE_ONLY> mrs(regions[0], COLUMNS_MR_FID);
  for (PointInDomainIterator<1> pid(
         rt->get_index_space_domain(task->regions[0].region.get_index_space()));
       pid();
       pid++) {
    const InitTaskColArgs& ca = args->column_args[pid[0]];
    nms[*pid] = ca.nm;
    dts[*pid] = ca.dt;
    kws[*pid] = ca.kw;
    mrs[*pid] = ca.mr;
  }
}

template <TableFid F>
static Legion::FieldID
allocate_field(Legion::FieldAllocator& fa) {
  return fa.allocate_field(sizeof(typename TableFieldType<F>::type), F);
}

Table
Table::create(
  Context ctx,
  Runtime* rt,
  const std::vector<std::pair<std::string, Column>>& columns,
  const ColumnSpace& column_space) {

  LogicalRegion columns_lr;
  {
    IndexSpace is = rt->create_index_space(ctx, Rect<1>(0, columns.size() - 1));
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    allocate_field<COLUMNS_NM_FID>(fa);
    allocate_field<COLUMNS_DT_FID>(fa);
    allocate_field<COLUMNS_KW_FID>(fa);
    allocate_field<COLUMNS_MR_FID>(fa);
    columns_lr = rt->create_logical_region(ctx, is, fs);
  }
  LogicalRegion values_lr;
  {
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    for (auto& nm_col : columns) {
      const Column& col = std::get<1>(nm_col);
      switch(col.dt) {
#define ALLOC_FLD(DT)                                           \
      case DT:                                                  \
        fa.allocate_field(hyperion::DataType<DT>::serdez_size, col.fid); \
        break;
      HYPERION_FOREACH_DATATYPE(ALLOC_FLD)
#undef ALLOC_FLD
      default:
        assert(false);
        break;
      }
    }
    values_lr = rt->create_logical_region(ctx, column_space.column_is, fs);
  }
  {
    {
      std::unordered_set<std::string> cnames;
      for (auto& nm_col : columns)
        cnames.insert(std::get<0>(nm_col));
      assert(cnames.size() == columns.size());
      assert(cnames.count("") == 0);
    }

    InitTaskArgs args;
    for (size_t i = 0; i < columns.size(); ++i) {
      auto& [nm, col] = columns[i];
      InitTaskColArgs& ca = args.column_args[i];
      ca.nm = nm;
      ca.dt = col.dt;
      ca.kw = col.kw;
      ca.mr = col.mr;
    }
    RegionRequirement req(columns_lr, WRITE_ONLY, EXCLUSIVE, columns_lr);
    req.add_field(COLUMNS_NM_FID);
    req.add_field(COLUMNS_DT_FID);
    req.add_field(COLUMNS_KW_FID);
    req.add_field(COLUMNS_MR_FID);
    TaskLauncher init(init_task_id, TaskArgument(&args, sizeof(args)));
    init.add_region_requirement(req);
    init.enable_inlining = true;
    rt->execute_task(ctx, init);
  }
  return Table(columns_lr, values_lr, column_space);
}

void
Table::destroy(
  Context ctx,
  Runtime* rt,
  bool destroy_column_space,
  bool destroy_column_index_space) {

  if (destroy_column_space)
    column_space.destroy(ctx, rt, destroy_column_index_space);
  for (auto& lr : {&columns_lr, &values_lr}) {
    if (*lr != LogicalRegion::NO_REGION) {
      rt->destroy_index_space(ctx, lr->get_index_space());
      rt->destroy_field_space(ctx, lr->get_field_space());
      rt->destroy_logical_region(ctx, *lr);
      *lr = LogicalRegion::NO_REGION;
    }
  }
}

TaskID Table::column_names_task_id;

const char* Table::column_names_task_name = "x::Table::column_names_task";

Table::column_names_result_t
Table::column_names_task(
  const Task*,
  const std::vector<PhysicalRegion>& regions,
  Context,
  Runtime *rt) {

  assert(regions.size() == 0);
  return column_names(rt, regions[0]);
}

Future /* column_names_result_t */
Table::column_names(Context ctx, Runtime *rt) const {
  RegionRequirement req(columns_lr, READ_ONLY, EXCLUSIVE, columns_lr);
  req.add_field(COLUMNS_NM_FID);
  TaskLauncher task(column_names_task_id, TaskArgument(NULL, 0));
  task.add_region_requirement(req);
  task.enable_inlining = true;
  return rt->execute_task(ctx, task);
}

std::array<hyperion::string, Table::MAX_COLUMNS>
Table::column_names(
  Runtime *rt,
  const PhysicalRegion& columns_pr) {

  std::array<hyperion::string, MAX_COLUMNS> result;
  coord_t i = 0;
  const ColumnsNameAccessor<READ_ONLY> names(columns_pr, COLUMNS_NM_FID);
  for (PointInDomainIterator<1> pid(
         rt->get_index_space_domain(
           columns_pr.get_logical_region().get_index_space()));
       pid();
       pid++) {
    i = pid[0];
    result[i] = names[*pid];
  }
  static const hyperion::string empty;
  for (++i; i < (coord_t)MAX_COLUMNS; ++i)
    result[i] = empty;
  return result;
}

struct ColumnTaskArgs {
  hyperion::string name;
  FieldSpace values_fs;
};

TaskID Table::column_task_id;

const char* Table::column_task_name = "x::Table::column_task";

Table::column_result_t
Table::column_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context,
  Runtime *rt) {

  const ColumnTaskArgs *args = static_cast<const ColumnTaskArgs*>(task->args);
  assert(regions.size() == 1);
  return column(rt, regions[0], args->values_fs, args->name);
}

Future /* column_result_t */
Table::column(Context ctx, Runtime* rt, const std::string& name) const {
  RegionRequirement req(columns_lr, READ_ONLY, EXCLUSIVE, columns_lr);
  req.add_field(COLUMNS_NM_FID);
  req.add_field(COLUMNS_DT_FID);
  req.add_field(COLUMNS_KW_FID);
  req.add_field(COLUMNS_MR_FID);
  ColumnTaskArgs args;
  args.name = name;
  args.values_fs = values_lr.get_field_space();
  TaskLauncher task(column_task_id, TaskArgument(&args, sizeof(args)));
  task.add_region_requirement(req);
  task.enable_inlining = true;
  return rt->execute_task(ctx, task);
}

Column
Table::column(
  Runtime* rt,
  const PhysicalRegion& columns_pr,
  const FieldSpace& values_fs,
  const hyperion::string& name) {

  const ColumnsNameAccessor<READ_ONLY> nms(columns_pr, COLUMNS_NM_FID);
  const ColumnsDatatypeAccessor<READ_ONLY> dts(columns_pr, COLUMNS_DT_FID);
  const ColumnsKeywordsAccessor<READ_ONLY> kws(columns_pr, COLUMNS_KW_FID);
  const ColumnsMeasRefAccessor<READ_ONLY> mrs(columns_pr, COLUMNS_MR_FID);
  std::optional<Point<1>> match;
  for (PointInDomainIterator<1> pid(
         rt->get_index_space_domain(
           columns_pr.get_logical_region().get_index_space()));
       !match && pid();
       pid++)
    if (nms[*pid] == name)
      match = *pid;
  Column result;
  if (match) {
    auto m = match.value();
    std::vector<FieldID> fids;
    rt->get_field_space_fields(values_fs, fids);
    assert((coord_t)fids.size() > m[0]);
    result = Column(dts[m], fids[m[0]], mrs[m], kws[m]);
  }
  return result;
}

TaskID Table::convert_task_id;

const char* Table::convert_task_name = "x::Table::convert_task";

struct ConvertTaskArgs {
  size_t num_cols;
  std::array<IndexSpace, Table::MAX_COLUMNS> values_iss;
  std::array<std::pair<hyperion::string, FieldID>, Table::MAX_COLUMNS>
  fids;
  std::array<unsigned, Table::MAX_COLUMNS> mr_sz;
  std::array<unsigned, Table::MAX_COLUMNS> kw_sz;
};

Table::convert_result_t
Table::convert_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *rt) {

  ConvertTaskArgs* args = static_cast<ConvertTaskArgs*>(task->args);
  assert(regions.size() ==
         std::accumulate(
           args->mr_sz.begin(),
           args->mr_sz.begin() + args->num_cols,
           0)
         + std::accumulate(
           args->kw_sz.begin(),
           args->kw_sz.begin() + args->num_cols,
           0)
         + 2 * args->num_cols);

  std::unordered_map<std::string, FieldID> fids;
  std::vector<IndexSpace> values_iss;
  values_iss.reserve(args->num_cols);
  std::vector<
    std::tuple<
      PhysicalRegion,
      PhysicalRegion,
      std::optional<hyperion::MeasRef::DataRegions>,
      std::optional<hyperion::Keywords::pair<PhysicalRegion>>>> col_prs;
  col_prs.reserve(args->num_cols);
  size_t rg = 0;
  for (size_t i = 0; i < args->num_cols; ++i) {
    auto& [nm, fid] = args->fids[i];
    fids[nm] = fid;
    values_iss.push_back(args->values_iss[i]);
    PhysicalRegion md = regions[rg++];
    PhysicalRegion ax = regions[rg++];
    std::optional<hyperion::MeasRef::DataRegions> mr;
    std::optional<hyperion::Keywords::pair<PhysicalRegion>> kw;
    if (args->mr_sz[i] > 0) {
      assert(args->mr_sz[i] <= 2);
      hyperion::MeasRef::DataRegions drs;
      drs.metadata = regions[rg++];
      if (args->mr_sz[i] > 1)
        drs.values = regions[rg++];
      mr = drs;
    }
    if (args->kw_sz[i] > 0) {
      assert(args->kw_sz[i] == 2);
      PhysicalRegion tt = regions[rg++];
      PhysicalRegion vals = regions[rg++];
      kw = hyperion::Keywords::pair<PhysicalRegion>{tt, vals};
    }
    col_prs.emplace_back(md, ax, mr, kw);
  }
  return convert(ctx, rt, fids, values_iss, col_prs);
}

Future /* convert_result_t */
Table::convert(
  Context ctx,
  Runtime* rt,
  const hyperion::Table& table,
  const std::unordered_map<std::string, FieldID> fids) {

  PhysicalRegion cols_pr;
  {
    RegionRequirement
      req(table.columns_lr, READ_ONLY, EXCLUSIVE, table.columns_lr);
    req.add_field(hyperion::Table::COLUMNS_FID);
    cols_pr = rt->map_region(ctx, req);
  }
  const hyperion::Table::ColumnsAccessor<READ_ONLY>
    cols(cols_pr, hyperion::Table::COLUMNS_FID);
  bool missing_fid = false;
  ConvertTaskArgs args;
  args.num_cols = 0;
  std::vector<RegionRequirement> reqs;
  for (PointInDomainIterator<1> pid(
         rt->get_index_space_domain(
           cols_pr.get_logical_region().get_index_space()));
       !missing_fid && pid();
       pid++) {
    const hyperion::Column& col = cols[*pid];
    RegionRequirement
      req(col.metadata_lr, READ_ONLY, EXCLUSIVE, col.metadata_lr);
    req.add_field(hyperion::Column::METADATA_NAME_FID);
    auto md_pr = rt->map_region(ctx, req);
    const hyperion::Column::NameAccessor<READ_ONLY>
      nm(md_pr, hyperion::Column::METADATA_NAME_FID);
    missing_fid = fids.count(nm[0]) == 0;
    if (!missing_fid) {
      args.fids[args.num_cols] = std::make_pair(nm[0], fids.at(nm[0]));
      args.values_iss[args.num_cols] = col.values_lr.get_index_space();
      reqs.emplace_back(col.metadata_lr, READ_ONLY, EXCLUSIVE, col.metadata_lr);
      reqs.emplace_back(col.axes_lr, READ_ONLY, EXCLUSIVE, col.axes_lr);
      if (!col.meas_ref.is_empty()) {
        auto [mreq, o_vreq] = col.meas_ref.requirements(READ_ONLY);
        reqs.push_back(mreq);
        args.mr_sz[args.num_cols] = 1;
        if (o_vreq) {
          reqs.push_back(o_vreq.value());
          args.mr_sz[args.num_cols] = 2;
        }
      } else {
        args.mr_sz[args.num_cols] = 0;
      }
      if (!col.keywords.is_empty()) {
        reqs.emplace_back(
          col.keywords.type_tags_lr,
          READ_ONLY,
          EXCLUSIVE,
          col.keywords.type_tags_lr);
        reqs.emplace_back(
          col.keywords.values_lr,
          READ_ONLY,
          EXCLUSIVE,
          col.keywords.values_lr);
        args.kw_sz[args.num_cols] = 2;
      } else {
        args.kw_sz[args.num_cols] = 0;
      }
    }
    rt->unmap_region(ctx, md_pr);
    ++args.num_cols;
  }
  rt->unmap_region(ctx, cols_pr);

  if (args.num_cols == 0 || missing_fid)
    return Future::from_value(rt, convert_result_t());

  TaskLauncher task(convert_task_id, TaskArgument(&args, sizeof(args)));
  for (auto& r : reqs)
    task.add_region_requirement(r);
  return rt->execute_task(ctx, task);
}

Table::convert_result_t
Table::convert(
  Context ctx,
  Runtime* rt,
  const std::unordered_map<std::string, FieldID>& fids,
  const std::vector<IndexSpace>& col_values_iss,
  const std::vector<
    std::tuple<
      PhysicalRegion,
      PhysicalRegion,
      std::optional<hyperion::MeasRef::DataRegions>,
      std::optional<hyperion::Keywords::pair<PhysicalRegion>>>>& col_prs) {

  // Collect columns that share the same axis vectors. This ought to be
  // sufficient for the creation of common ColumnSpaces -- since all of the
  // given columns are associated with a single Table to begin with, a fixed
  // axis order ensures that the shapes of the index spaces are the same for all
  // the columns with that axis order.
  std::map<
    std::vector<unsigned>,
    std::tuple<
      std::vector<std::pair<std::string, Column>>,
      IndexSpace>> tbls;
  std::optional<std::string> axes_uid;
  for (size_t i = 0; i < col_prs.size(); ++i) {
    auto& [md_pr, ax_pr, o_mr_drs, o_kw_pair] = col_prs[i];
    IndexSpaceT<1> is(ax_pr.get_logical_region().get_index_space());
    DomainT<1> dom = rt->get_index_space_domain(is);
    std::vector<unsigned> axes(Domain(dom).hi()[0] + 1);
    const hyperion::Column::AxesAccessor<READ_ONLY>
      ax(ax_pr, hyperion::Column::AXES_FID);
    for (PointInDomainIterator<1> pid(dom); pid(); pid++)
      axes[pid[0]] = ax[*pid];
    const hyperion::Column::NameAccessor<READ_ONLY>
      nm(md_pr, hyperion::Column::METADATA_NAME_FID);
    const hyperion::Column::DatatypeAccessor<READ_ONLY>
      dt(md_pr, hyperion::Column::METADATA_DATATYPE_FID);
    if (!axes_uid) {
      const hyperion::Column::AxesUidAccessor<READ_ONLY>
        uid(md_pr, hyperion::Column::METADATA_AXES_UID_FID);
      axes_uid = uid[0];
    }
    std::string name(nm[0]);
    auto nm_col =
      std::make_pair(
        name,
        Column(
          dt[0],
          fids.at(name),
          (o_mr_drs
           ? hyperion::MeasRef::clone(ctx, rt, o_mr_drs.value())
           : hyperion::MeasRef()),
          (o_kw_pair
           ? hyperion::Keywords::clone(ctx, rt, o_kw_pair.value())
           : hyperion::Keywords())));
    if (tbls.count(axes) == 0){
      // use a cloned copy of the column values IndexSpace to avoid ambiguity of
      // responsibility for cleanup
      tbls[axes] =
        std::make_tuple(
          std::vector{nm_col},
          rt->create_index_space(
            ctx,
            rt->get_index_space_domain(
              ctx,
              col_values_iss[i])));
    } else {
      std::get<0>(tbls[axes]).push_back(nm_col);
    }
  }
  convert_result_t result;
  size_t i = 0;
  for (auto& [ax, t] : tbls) {
    auto& [cols, is] = t;
    auto csp = ColumnSpace::create(ctx, rt, ax, axes_uid.value(), is);
    result[i++] = Table::create(ctx, rt, cols, csp);
  }
  return result;
}

TaskID copy_values_from_task_id;

const char* copy_values_from_task_name = "x::Table::copy_values_from_task";

void
Table::copy_values_from_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *rt) {

  assert((regions.size() >= 2) && (regions.size() % 2 == 0));
  auto columns_pr = regions[0];
  auto values_pr = regions[1];
  std::vector<std::tuple<PhysicalRegion, PhysicalRegion>> src_col_prs;
  for (size_t i = 2; i < regions.size(); ++i)
    src_col_prs.emplace_back(regions[i], regions[i + 1]);
  copy_values_from(ctx, rt, values_pr, columns_pr, src_col_prs);
}

void
Table::copy_values_from(
  Context ctx,
  Runtime* rt,
  const hyperion::Table& table) const {

  std::vector<RegionRequirement> reqs;
  {
    std::vector<FieldID> fids;
    rt->get_field_space_fields(ctx, columns_lr.get_field_space(), fids);
    RegionRequirement req(columns_lr, READ_ONLY, EXCLUSIVE, columns_lr);
    req.add_fields(fids);
    reqs.push_back(req);
  }
  {
    std::vector<FieldID> fids;
    rt->get_field_space_fields(ctx, values_lr.get_field_space(), fids);
    RegionRequirement req(values_lr, WRITE_ONLY, EXCLUSIVE, values_lr);
    req.add_fields(fids, false);
    reqs.push_back(req);
  }
  {
    RegionRequirement
      cols_req(table.columns_lr, READ_ONLY, EXCLUSIVE, table.columns_lr);
    cols_req.add_field(hyperion::Table::COLUMNS_FID);
    auto pr = rt->map_region(ctx, cols_req);
    const hyperion::Table::ColumnsAccessor<READ_ONLY>
      cols_pr(pr, hyperion::Table::COLUMNS_FID);
    for (PointInDomainIterator<1> pid(
           rt->get_index_space_domain(table.columns_lr.get_index_space()));
         pid();
         pid++) {
      auto col = cols_pr[*pid];
      reqs.emplace_back(
        col.metadata_lr,
        std::set{hyperion::Column::METADATA_NAME_FID},
        std::vector{hyperion::Column::METADATA_NAME_FID},
        READ_ONLY,
        EXCLUSIVE,
        col.metadata_lr);
      reqs.emplace_back(
        col.values_lr,
        std::set{hyperion::Column::VALUE_FID},
        std::vector{hyperion::Column::VALUE_FID},
        READ_ONLY,
        EXCLUSIVE,
        col.values_lr);
    }
    rt->unmap_region(ctx, pr);
  }
  TaskLauncher task(copy_values_from_task_id, TaskArgument(NULL, 0));
  rt->execute_task(ctx, task);
}

void
Table::copy_values_from(
  Context ctx,
  Runtime* rt,
  const PhysicalRegion& columns_pr,
  const PhysicalRegion& values_pr,
  const std::vector<std::tuple<PhysicalRegion, PhysicalRegion>>& src_col_prs) {

  LogicalRegion values_lr = values_pr.get_logical_region();
  RegionRequirement dst_req(values_lr, WRITE_ONLY, EXCLUSIVE, values_lr);
  RegionRequirement src_req(
    LogicalRegion::NO_REGION,
    {hyperion::Column::VALUE_FID},
    {hyperion::Column::VALUE_FID},
    READ_ONLY,
    EXCLUSIVE,
    LogicalRegion::NO_REGION);
  CopyLauncher copy;
  unsigned n = 0;
  for (size_t i = 0; i < src_col_prs.size(); ++i) {
    auto& [md_pr, src_vals_pr] = src_col_prs[i];
    const hyperion::Column::NameAccessor<READ_ONLY>
      nm(md_pr, hyperion::Column::METADATA_NAME_FID);
    auto dst_col =
      Table::column(rt, columns_pr, values_lr.get_field_space(), nm[0]);
    if (dst_col.is_valid()) {
      ++n;
      dst_req.privilege_fields.clear();
      dst_req.instance_fields.clear();
      dst_req.add_field(dst_col.fid);
      src_req.region = src_req.parent = src_vals_pr.get_logical_region();
      copy.add_copy_requirements(src_req, dst_req);
    }
  }
  if (n > 0)
    rt->issue_copy_operation(ctx, copy);
}

void
Table::preregister_tasks() {
  {
    // init_task
    init_task_id = Runtime::generate_static_task_id();
    TaskVariantRegistrar registrar(init_task_id, init_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    registrar.set_leaf();
    Runtime::preregister_task_variant<init_task>(registrar, init_task_name);
  }
  {
    // column_task
    column_task_id = Runtime::generate_static_task_id();
    TaskVariantRegistrar registrar(column_task_id, column_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    registrar.set_leaf();
    Runtime::preregister_task_variant<column_result_t, column_task>(
      registrar,
      column_task_name);
  }
  {
    // column_names_task
    column_names_task_id = Runtime::generate_static_task_id();
    TaskVariantRegistrar
      registrar(column_names_task_id, column_names_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    registrar.set_leaf();
    Runtime::preregister_task_variant<column_names_result_t, column_names_task>(
      registrar,
      column_names_task_name);
  }
  {
    // convert_task
    convert_task_id = Runtime::generate_static_task_id();
    TaskVariantRegistrar
      registrar(convert_task_id, convert_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    registrar.set_leaf();
    Runtime::preregister_task_variant<convert_result_t, convert_task>(
      registrar,
      convert_task_name);
  }
  {
    // copy_values_from_task
    copy_values_from_task_id = Runtime::generate_static_task_id();
    TaskVariantRegistrar
      registrar(copy_values_from_task_id, copy_values_from_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    Runtime::preregister_task_variant<copy_values_from_task>(
      registrar,
      copy_values_from_task_name);
  }
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
