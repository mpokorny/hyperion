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

size_t
Table::columns_result_tt::legion_buffer_size(void) const {
  size_t result = sizeof(unsigned);
  for (size_t i = 0; i < fields.size(); ++i)
    result +=
      sizeof(ColumnSpace)
      + sizeof(Legion::LogicalRegion)
      + sizeof(unsigned)
      + std::get<2>(fields[i]).size() * sizeof(tbl_fld_t);
  return result;
}

size_t
Table::columns_result_tt::legion_serialize(void* buffer) const {
  char* b = static_cast<char*>(buffer);
  *reinterpret_cast<unsigned*>(b) = (unsigned)fields.size();
  b += sizeof(unsigned);
  for (size_t i = 0; i < fields.size(); ++i) {
    auto& [csp, lr, fs] = fields[i];
    *reinterpret_cast<ColumnSpace*>(b) = csp;
    b += sizeof(csp);
    *reinterpret_cast<Legion::LogicalRegion*>(b) = lr;
    b += sizeof(lr);
    *reinterpret_cast<unsigned*>(b) = (unsigned)fs.size();
    b += sizeof(unsigned);
    for (auto& f : fs) {
      *reinterpret_cast<tbl_fld_t*>(b) = f;
      b += sizeof(f);
    }
  }
  return b - static_cast<char*>(buffer);
}

size_t
Table::columns_result_tt::legion_deserialize(const void* buffer) {
  const char* b = static_cast<const char*>(buffer);
  unsigned n = *reinterpret_cast<const unsigned*>(b);
  b += sizeof(n);
  fields.resize(n);
  for (size_t i = 0; i < n; ++i) {
    auto& [csp, lr, fs] = fields[i];
    csp = *reinterpret_cast<const ColumnSpace*>(b);
    b += sizeof(csp);
    lr = *reinterpret_cast<const Legion::LogicalRegion*>(b);
    b += sizeof(lr);
    unsigned nn = *reinterpret_cast<const unsigned*>(b);
    b += sizeof(nn);
    fs.resize(nn);
    for (auto& f : fs) {
      f = *reinterpret_cast<const tbl_fld_t*>(b);
      b += sizeof(f);
    }
  }
  return b - static_cast<const char*>(buffer);
}

std::unordered_map<std::string, Column>
Table::column_map(const columns_result_t& columns_result) {

  std::unordered_map<std::string, Column> result;
  for (auto& [nm, c] : columns_result)
    if (c.is_valid())
      result[nm] = c;
    else
      break;
  return result;
}

std::unordered_map<std::string, Column>
Table::column_map(const columns_result_tt& columns_result) {
  std::unordered_map<std::string, Column> result;
  for (auto& [csp, lr, tfs] : columns_result.fields) {
    for (auto& [nm, tf] : tfs)
      result[nm] = Column(tf.dt, tf.fid, tf.mr, tf.kw, csp, lr);
  }
  return result;
}

struct InitTaskTableFieldArgs {
  hyperion::string nm;
  hyperion::TypeTag dt;
  hyperion::Keywords kw;
  hyperion::MeasRef mr;
  LogicalRegion md;
  LogicalRegion vs;
};

struct InitTaskArgs {
  size_t index_offset;
  size_t num_fields;
  std::array<InitTaskTableFieldArgs, Table::MAX_COLUMNS> field_args;
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
  const NameAccessor<WRITE_ONLY>
    nms(regions[0], static_cast<FieldID>(TableFieldsFid::NM));
  const DatatypeAccessor<WRITE_ONLY>
    dts(regions[0], static_cast<FieldID>(TableFieldsFid::DT));
  const KeywordsAccessor<WRITE_ONLY>
    kws(regions[0], static_cast<FieldID>(TableFieldsFid::KW));
  const MeasRefAccessor<WRITE_ONLY>
    mrs(regions[0], static_cast<FieldID>(TableFieldsFid::MR));
  const MetadataAccessor<WRITE_ONLY>
    mds(regions[0], static_cast<FieldID>(TableFieldsFid::MD));
  const ValuesAccessor<WRITE_ONLY>
    vss(regions[0], static_cast<FieldID>(TableFieldsFid::VS));
  PointInDomainIterator<1> pid(
    rt->get_index_space_domain(task->regions[0].region.get_index_space()));
  for (size_t i = 0; i < args->index_offset; ++i)
    pid++;
  for (size_t i = 0; i < args->num_fields; pid++, ++i) {
    const InitTaskTableFieldArgs& tfa = args->field_args[i];
    nms[*pid] = tfa.nm;
    dts[*pid] = tfa.dt;
    kws[*pid] = tfa.kw;
    mrs[*pid] = tfa.mr;
    mds[*pid] = tfa.md;
    vss[*pid] = tfa.vs;
  }
}

template <TableFieldsFid F>
static Legion::FieldID
allocate_field(Legion::FieldAllocator& fa) {
  return
    fa.allocate_field(
      sizeof(typename TableFieldsType<F>::type),
      static_cast<FieldID>(F));
}

void
Table::create_columns(
  Context ctx,
  Runtime* rt,
  const ColumnSpace& column_space,
  const std::vector<std::pair<std::string, TableField>>& tbl_fields,
  Legion::LogicalRegion& fields_lr,
  size_t tbl_field_offset) {

  LogicalRegion values_lr;
  {
    std::unordered_set<FieldID> fids;
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    for (auto& nm_tf : tbl_fields) {
      const TableField& tf = std::get<1>(nm_tf);
      assert(fids.count(tf.fid) == 0);
      switch(tf.dt) {
#define ALLOC_FLD(DT)                                           \
      case DT:                                                  \
        fa.allocate_field(hyperion::DataType<DT>::serdez_size, tf.fid); \
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
    InitTaskArgs args;
    args.index_offset = tbl_field_offset;
    args.num_fields = tbl_fields.size();
    for (size_t i = 0; i < tbl_fields.size(); ++i) {
      auto& [nm, col] = tbl_fields[i];
      InitTaskTableFieldArgs& tfa = args.field_args[i];
      tfa.nm = nm;
      tfa.dt = col.dt;
      tfa.kw = col.kw;
      tfa.mr = col.mr;
      tfa.md = column_space.metadata_lr;
      tfa.vs = values_lr;
    }
    RegionRequirement req(fields_lr, WRITE_ONLY, EXCLUSIVE, fields_lr);
    req.add_field(static_cast<FieldID>(TableFieldsFid::NM));
    req.add_field(static_cast<FieldID>(TableFieldsFid::DT));
    req.add_field(static_cast<FieldID>(TableFieldsFid::KW));
    req.add_field(static_cast<FieldID>(TableFieldsFid::MR));
    req.add_field(static_cast<FieldID>(TableFieldsFid::MD));
    req.add_field(static_cast<FieldID>(TableFieldsFid::VS));
    TaskLauncher init(init_task_id, TaskArgument(&args, sizeof(args)));
    init.add_region_requirement(req);
    init.enable_inlining = true;
    rt->execute_task(ctx, init);
  }
}

Table
Table::create(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const std::vector<
    std::tuple<
      ColumnSpace,
      std::vector<std::pair<std::string, TableField>>>>& columns) {

  size_t num_cols = 0;
  for (auto& csp_cs : columns)
    num_cols += std::get<1>(csp_cs).size();
  {
    std::unordered_set<std::string> cnames;
    for (auto& csp_cs : columns)
      for (auto& nm_c : std::get<1>(csp_cs))
        cnames.insert(std::get<0>(nm_c));
    assert(cnames.count("") == 0);
    assert(cnames.size() == num_cols);
  }

  LogicalRegion fields_lr;
  {
    IndexSpace is = rt->create_index_space(ctx, Rect<1>(0, num_cols - 1));
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    allocate_field<TableFieldsFid::NM>(fa);
    allocate_field<TableFieldsFid::DT>(fa);
    allocate_field<TableFieldsFid::KW>(fa);
    allocate_field<TableFieldsFid::MR>(fa);
    allocate_field<TableFieldsFid::MD>(fa);
    allocate_field<TableFieldsFid::VS>(fa);
    fields_lr = rt->create_logical_region(ctx, is, fs);
  }
  size_t field_offset = 0;
  for (auto& [csp, tfs] : columns) {
    create_columns(ctx, rt, csp, tfs, fields_lr, field_offset);
    field_offset += tfs.size();
  }
  return Table(fields_lr);
}

void
Table::destroy(
  Context ctx,
  Runtime* rt,
  bool destroy_column_space_components) {

  auto cols = column_map(columns(ctx, rt).get<columns_result_t>());
  std::vector<Column> csp_cols;
  for (auto& nm_col : cols) {
    Column& col = std::get<1>(nm_col);
    auto csp_c =
      std::find_if(
        csp_cols.begin(),
        csp_cols.end(),
        [vlr=col.vlr](auto& c) { return c.vlr == vlr; });
    if (csp_c == csp_cols.end())
      csp_cols.push_back(col);
  }
  for (auto& col : csp_cols) {
    if (col.vlr != LogicalRegion::NO_REGION) {
      rt->destroy_field_space(ctx, col.vlr.get_field_space());
      rt->destroy_logical_region(ctx, col.vlr);
    }
    if (destroy_column_space_components)
      col.csp.destroy(ctx, rt, true);
  }
  if (fields_lr != LogicalRegion::NO_REGION) {
    rt->destroy_index_space(ctx, fields_lr.get_index_space());
    rt->destroy_field_space(ctx, fields_lr.get_field_space());
    rt->destroy_logical_region(ctx, fields_lr);
    fields_lr = LogicalRegion::NO_REGION;
  }
}

TaskID Table::columns_task_id;

const char* Table::columns_task_name = "x::Table::columns_task";

Table::columns_result_t
Table::columns_task(
  const Task*,
  const std::vector<PhysicalRegion>& regions,
  Context,
  Runtime *rt) {

  assert(regions.size() == 0);
  return columns(rt, regions[0]);
}

Future /* columns_result_t */
Table::columns(Context ctx, Runtime *rt) const {
  RegionRequirement req(fields_lr, READ_ONLY, EXCLUSIVE, fields_lr);
  req.add_field(static_cast<FieldID>(TableFieldsFid::NM));
  req.add_field(static_cast<FieldID>(TableFieldsFid::DT));
  req.add_field(static_cast<FieldID>(TableFieldsFid::KW));
  req.add_field(static_cast<FieldID>(TableFieldsFid::MR));
  req.add_field(static_cast<FieldID>(TableFieldsFid::MD));
  req.add_field(static_cast<FieldID>(TableFieldsFid::VS));
  TaskLauncher task(columns_task_id, TaskArgument(NULL, 0));
  task.add_region_requirement(req);
  task.enable_inlining = true;
  return rt->execute_task(ctx, task);
}

Table::columns_result_t
Table::columns(Runtime *rt, const PhysicalRegion& fields_pr) {

  columns_result_t result;

  const NameAccessor<READ_ONLY>
    nms(fields_pr, static_cast<FieldID>(TableFieldsFid::NM));
  const DatatypeAccessor<READ_ONLY>
    dts(fields_pr, static_cast<FieldID>(TableFieldsFid::DT));
  const KeywordsAccessor<READ_ONLY>
    kws(fields_pr, static_cast<FieldID>(TableFieldsFid::KW));
  const MeasRefAccessor<READ_ONLY>
    mrs(fields_pr, static_cast<FieldID>(TableFieldsFid::MR));
  const MetadataAccessor<READ_ONLY>
    mds(fields_pr, static_cast<FieldID>(TableFieldsFid::MD));
  const ValuesAccessor<READ_ONLY>
    vss(fields_pr, static_cast<FieldID>(TableFieldsFid::VS));

  PointInDomainIterator<1> pid(
    rt->get_index_space_domain(
      fields_pr.get_logical_region().get_index_space()));
  std::vector<FieldID> fids;
  size_t i = 0;
  if (pid()) {
    Point<1> v0 = *pid;
    rt->get_field_space_fields(vss[v0].get_field_space(), fids);
    while (pid()) {
      if (vss[v0] != vss[*pid]) {
        v0 = *pid;
        rt->get_field_space_fields(vss[v0].get_field_space(), fids);
      }
      result[i++] =
        std::make_tuple(
          nms[*pid],
          Column(
            dts[*pid],
            fids[pid[0] - v0[0]],
            mrs[*pid],
            kws[*pid],
            ColumnSpace(vss[*pid].get_index_space(), mds[*pid]),
            vss[*pid]));
      pid++;
    }
  }
  static const std::tuple<hyperion::string, Column> empty;
  for (; i < (coord_t)MAX_COLUMNS; ++i)
    result[i] = empty;
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
      IndexSpace,
      std::vector<std::pair<std::string, TableField>>>> tbl_flds;
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
    auto nm_tf =
      std::make_pair(
        name,
        TableField(
          dt[0],
          fids.at(name),
          (o_mr_drs
           ? hyperion::MeasRef::clone(ctx, rt, o_mr_drs.value())
           : hyperion::MeasRef()),
          (o_kw_pair
           ? hyperion::Keywords::clone(ctx, rt, o_kw_pair.value())
           : hyperion::Keywords())));
    if (tbl_flds.count(axes) == 0) {
      // use a cloned copy of the column values IndexSpace to avoid ambiguity of
      // responsibility for cleanup
      tbl_flds[axes] =
        std::make_tuple(
          rt->create_index_space(
            ctx,
            rt->get_index_space_domain(
              ctx,
              col_values_iss[i])),
          std::vector{nm_tf});
    } else {
      std::get<1>(tbl_flds[axes]).push_back(nm_tf);
    }
  }
  std::vector<
    std::tuple<
      ColumnSpace,
      std::vector<std::pair<std::string, TableField>>>> columns;
  columns.reserve(tbl_flds.size());
  for (auto& [ax, is_tfs] : tbl_flds) {
    auto& [is, tfs] = is_tfs;
    columns.emplace_back(
      ColumnSpace::create(ctx, rt, ax, axes_uid.value(), is),
      tfs);
  }
  return Table::create(ctx, rt, columns);
}

struct CopyValuesFromTaskArgs {
  size_t num_value_prs;
};

TaskID Table::copy_values_from_task_id;

const char* Table::copy_values_from_task_name
= "x::Table::copy_values_from_task";

void
Table::copy_values_from_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *rt) {

  const CopyValuesFromTaskArgs *args =
    static_cast<const CopyValuesFromTaskArgs*>(task->args);
  size_t num_dst_regions = 1 + args->num_value_prs;
  assert((regions.size() >= num_dst_regions)
         && ((regions.size() - num_dst_regions) % 2 == 0));
  auto fields_pr = regions[0];
  // note that we're receiving unmapped PhysicalRegions for the value regions,
  // so write permissions are set, and the writing occurs in a child (copy) task
  std::vector<std::tuple<PhysicalRegion, PhysicalRegion>> src_col_prs;
  for (size_t i = num_dst_regions; i < regions.size(); ++i)
    src_col_prs.emplace_back(regions[i], regions[i + 1]);
  copy_values_from(ctx, rt, fields_pr, src_col_prs);
}

void
Table::copy_values_from(
  Context ctx,
  Runtime* rt,
  const hyperion::Table& table) const {

  CopyValuesFromTaskArgs args;
  std::vector<RegionRequirement> reqs;
  std::vector<LogicalRegion> vlrs;
  {
    RegionRequirement req(fields_lr, READ_ONLY, EXCLUSIVE, fields_lr);
    req.add_field(static_cast<FieldID>(TableFieldsFid::VS));
    {
      auto pr = rt->map_region(ctx, req);
      const ValuesAccessor<READ_ONLY>
        vss(pr, static_cast<FieldID>(TableFieldsFid::VS));
      for (PointInDomainIterator<1> pid(
             rt->get_index_space_domain(fields_lr.get_index_space()));
           pid();
        pid++) {
        auto lr = std::find(vlrs.begin(), vlrs.end(), vss[*pid]);
        if (lr == vlrs.end())
          vlrs.push_back(vss[*pid]);
      }
      rt->unmap_region(ctx, pr);
    }
    req.add_field(static_cast<FieldID>(TableFieldsFid::NM));
    req.add_field(static_cast<FieldID>(TableFieldsFid::DT));
    req.add_field(static_cast<FieldID>(TableFieldsFid::KW));
    req.add_field(static_cast<FieldID>(TableFieldsFid::MR));
    req.add_field(static_cast<FieldID>(TableFieldsFid::MD));
    reqs.push_back(req);
  }
  args.num_value_prs = vlrs.size();
  for (auto& vlr : vlrs) {
    std::vector<FieldID> fids;
    rt->get_field_space_fields(ctx, vlr.get_field_space(), fids);
    RegionRequirement req(vlr, WRITE_ONLY, EXCLUSIVE, vlr);
    // this region requires permissions, but can remain unmapped
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
  TaskLauncher task(
    copy_values_from_task_id,
    TaskArgument(&args, sizeof(args)));
  rt->execute_task(ctx, task);
}

void
Table::copy_values_from(
  Context ctx,
  Runtime* rt,
  const PhysicalRegion& fields_pr,
  const std::vector<std::tuple<PhysicalRegion, PhysicalRegion>>& src_col_prs) {

  std::unordered_map<std::string, Column> dst_cols =
    column_map(columns(rt, fields_pr));
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
    if (dst_cols.count(nm[0]) > 0) {
      ++n;
      auto& dst_col = dst_cols.at(nm[0]);
      RegionRequirement
        dst_req(dst_col.vlr, WRITE_ONLY, EXCLUSIVE, dst_col.vlr);
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
    // columns_task
    columns_task_id = Runtime::generate_static_task_id();
    TaskVariantRegistrar registrar(columns_task_id, columns_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    registrar.set_leaf();
    Runtime::preregister_task_variant<columns_result_t, columns_task>(
      registrar,
      columns_task_name);
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
