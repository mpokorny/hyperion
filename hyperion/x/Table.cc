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

#include <map>
#include <unordered_set>

using namespace hyperion::x;

using namespace Legion;

#define FOREACH_TABLE_FIELD_FID(__FUNC__)       \
  __FUNC__(TableFieldsFid::NM)                  \
  __FUNC__(TableFieldsFid::DT)                  \
  __FUNC__(TableFieldsFid::KW)                  \
  __FUNC__(TableFieldsFid::MR)                  \
  __FUNC__(TableFieldsFid::MD)                  \
  __FUNC__(TableFieldsFid::VF)                  \
  __FUNC__(TableFieldsFid::VS)

size_t
Table::partition_rows_result_t::legion_buffer_size(void) const {
  size_t result = sizeof(unsigned);
  for (size_t i = 0; i < partitions.size(); ++i) {
    result += sizeof(ColumnSpace) + sizeof(IndexPartition) + sizeof(unsigned);
    auto& p = partitions[i].partition;
    result +=
      std::distance(
        p.begin(),
        std::find_if(
          p.begin(),
          p.end(),
          [](auto& ap){ return ap.stride == 0; }))
      * sizeof(AxisPartition);
  }
  return result;
}

size_t
Table::partition_rows_result_t::legion_serialize(void* buffer) const {
  char* b = static_cast<char*>(buffer);
  *reinterpret_cast<unsigned*>(b) = (unsigned)partitions.size();
  b += sizeof(unsigned);
  for (size_t i = 0; i < partitions.size(); ++i) {
    auto& p = partitions[i];
    *reinterpret_cast<ColumnSpace*>(b) = p.column_space;
    b += sizeof(ColumnSpace);
    *reinterpret_cast<IndexPartition*>(b) = p.column_ip;
    b += sizeof(IndexPartition);
    *reinterpret_cast<unsigned*>(b) = (unsigned)p.partition.size();
    b += sizeof(unsigned);
    for (size_t i = 0;
         i < ColumnSpace::MAX_DIM && p.partition[i].stride != 0;
         ++i) {
      *reinterpret_cast<AxisPartition*>(b) = p.partition[i];
      b += sizeof(AxisPartition);
    }
  }
  return b - static_cast<char*>(buffer);
}

size_t
Table::partition_rows_result_t::legion_deserialize(const void* buffer) {
  const char* b = static_cast<const char*>(buffer);
  unsigned n = *reinterpret_cast<const unsigned*>(b);
  b += sizeof(n);
  partitions.resize(n);
  for (size_t i = 0; i < n; ++i) {
    auto& p = partitions[i];
    p.column_space = *reinterpret_cast<const ColumnSpace*>(b);
    b += sizeof(ColumnSpace);
    p.column_ip= *reinterpret_cast<const IndexPartition*>(b);
    b += sizeof(IndexPartition);
    unsigned nn = *reinterpret_cast<const unsigned*>(b);
    b += sizeof(nn);
    for (size_t j = 0; j < nn; ++ j) {
      p.partition[j] = *reinterpret_cast<const AxisPartition*>(b);
      b += sizeof(AxisPartition);
    }
    while (nn < ColumnSpace::MAX_DIM)
      p.partition[nn++].stride = 0;
  }
  return b - static_cast<const char*>(buffer);
}

size_t
Table::columns_result_t::legion_buffer_size(void) const {
  size_t result = sizeof(unsigned);
  for (size_t i = 0; i < fields.size(); ++i)
    result +=
      sizeof(ColumnSpace)
      + sizeof(LogicalRegion)
      + sizeof(unsigned)
      + std::get<2>(fields[i]).size() * sizeof(tbl_fld_t);
  return result;
}

size_t
Table::columns_result_t::legion_serialize(void* buffer) const {
  char* b = static_cast<char*>(buffer);
  *reinterpret_cast<unsigned*>(b) = (unsigned)fields.size();
  b += sizeof(unsigned);
  for (size_t i = 0; i < fields.size(); ++i) {
    auto& [csp, lr, fs] = fields[i];
    *reinterpret_cast<ColumnSpace*>(b) = csp;
    b += sizeof(csp);
    *reinterpret_cast<LogicalRegion*>(b) = lr;
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
Table::columns_result_t::legion_deserialize(const void* buffer) {
  const char* b = static_cast<const char*>(buffer);
  unsigned n = *reinterpret_cast<const unsigned*>(b);
  b += sizeof(n);
  fields.resize(n);
  for (size_t i = 0; i < n; ++i) {
    auto& [csp, lr, fs] = fields[i];
    csp = *reinterpret_cast<const ColumnSpace*>(b);
    b += sizeof(csp);
    lr = *reinterpret_cast<const LogicalRegion*>(b);
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
Table::column_map(
  const columns_result_t& columns_result,
  legion_privilege_mode_t mode) {

  std::unordered_map<std::string, Column> result;
  for (auto& [csp, lr, tfs] : columns_result.fields) {
    for (auto& [nm, tf] : tfs) {
      RegionRequirement vreq(lr, mode, EXCLUSIVE, lr);
      vreq.add_field(tf.fid);
      result[nm] = Column(tf.dt, tf.fid, tf.mr, tf.kw, csp, vreq);
    }
  }
  return result;
}

template <TableFieldsFid F>
static FieldID
allocate_field(FieldAllocator& fa) {
  return
    fa.allocate_field(
      sizeof(typename TableFieldsType<F>::type),
      static_cast<FieldID>(F));
}

static void
allocate_table_fields(FieldAllocator& fa) {
#define ALLOC_F(F) allocate_field<F>(fa);
  FOREACH_TABLE_FIELD_FID(ALLOC_F);
#undef ALLOC_F
}

static RegionRequirement
table_fields_requirement(LogicalRegion lr, legion_privilege_mode_t mode) {

  RegionRequirement result(lr, mode, EXCLUSIVE, lr);
#define ADD_F(F) result.add_field(static_cast<FieldID>(F));
  FOREACH_TABLE_FIELD_FID(ADD_F);
#undef ADD_F
  return result;
}

Table
Table::create(
  Context ctx,
  Runtime* rt,
  const std::map<
    ColumnSpace,
    std::vector<std::pair<std::string, TableField>>>& columns) {

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
    IndexSpace is = rt->create_index_space(ctx, Rect<1>(0, MAX_COLUMNS - 1));
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    allocate_table_fields(fa);
    fields_lr = rt->create_logical_region(ctx, is, fs);
    {
      PhysicalRegion fields_pr =
        rt->map_region(ctx, table_fields_requirement(fields_lr, WRITE_ONLY));

      static const hyperion::string nm;
      static const Keywords kw;
      static const MeasRef mr;

      const NameAccessor<WRITE_ONLY>
        nms(fields_pr, static_cast<FieldID>(TableFieldsFid::NM));
      const DatatypeAccessor<WRITE_ONLY>
        dts(fields_pr, static_cast<FieldID>(TableFieldsFid::DT));
      const KeywordsAccessor<WRITE_ONLY>
        kws(fields_pr, static_cast<FieldID>(TableFieldsFid::KW));
      const MeasRefAccessor<WRITE_ONLY>
        mrs(fields_pr, static_cast<FieldID>(TableFieldsFid::MR));
      const MetadataAccessor<WRITE_ONLY>
        mds(fields_pr, static_cast<FieldID>(TableFieldsFid::MD));
      const ValueFidAccessor<WRITE_ONLY>
        vfs(fields_pr, static_cast<FieldID>(TableFieldsFid::VF));
      const ValuesAccessor<WRITE_ONLY>
        vss(fields_pr, static_cast<FieldID>(TableFieldsFid::VS));
      for (PointInDomainIterator<1> pid(
             rt->get_index_space_domain(fields_lr.get_index_space()));
           pid();
           pid++) {
        nms[*pid] = nm;
        dts[*pid] = (type_tag_t)0;
        kws[*pid] = kw;
        mrs[*pid] = mr;
        mds[*pid] = LogicalRegion::NO_REGION;
        vfs[*pid] = 0;
        vss[*pid] = LogicalRegion::NO_REGION;
      }
      rt->unmap_region(ctx, fields_pr);
    }
    PhysicalRegion fields_pr =
      rt->map_region(ctx, table_fields_requirement(fields_lr, READ_WRITE));
    add_columns(ctx, rt, columns, std::nullopt, fields_pr);
    rt->unmap_region(ctx, fields_pr);
  }
  return Table(fields_lr);
}

TaskID Table::index_axes_task_id;

const char* Table::index_axes_task_name = "x::Table::index_axes_task";

Table::index_axes_result_t
Table::index_axes_task(
  const Task*,
  const std::vector<PhysicalRegion>& regions,
  Context,
  Runtime*) {
  return index_axes(regions);
}

Future
Table::index_axes(Context ctx, Runtime* rt) const {
  auto cols = columns(ctx, rt).get_result<columns_result_t>();
  TaskLauncher task(index_axes_task_id, TaskArgument(NULL, 0));
  for (auto& csp_vlr_tfs : cols.fields) {
    auto& md = std::get<0>(csp_vlr_tfs).metadata_lr;
    RegionRequirement req(md, READ_ONLY, EXCLUSIVE, md);
    req.add_field(ColumnSpace::AXIS_VECTOR_FID);
    req.add_field(ColumnSpace::INDEX_FLAG_FID);
    task.add_region_requirement(req);
  }
  return rt->execute_task(ctx, task);
}

Table::index_axes_result_t
Table::index_axes(const std::vector<PhysicalRegion>& csp_metadata_prs) {

  Table::index_axes_result_t result;
  if (csp_metadata_prs.size() > 0) {
    Table::index_axes_result_t::iterator result_end = result.begin();
    size_t i = 0;
    for (;
         result_end == result.begin() && i < csp_metadata_prs.size();
         ++i) {
      const ColumnSpace::IndexFlagAccessor<READ_ONLY>
        ifl(csp_metadata_prs[i], ColumnSpace::INDEX_FLAG_FID);
      if (!ifl[0]) {
        const ColumnSpace::AxisVectorAccessor<READ_ONLY>
          ax(csp_metadata_prs[i], ColumnSpace::AXIS_VECTOR_FID);
        result = ax[0];
        result_end = result.begin() + ColumnSpace::size(ax[0]);
      }
    }
    for (;
         result_end != result.begin() && i < csp_metadata_prs.size();
         ++i) {
      const ColumnSpace::IndexFlagAccessor<READ_ONLY>
        ifl(csp_metadata_prs[i], ColumnSpace::INDEX_FLAG_FID);
      if (!ifl[0]) {
        const ColumnSpace::AxisVectorAccessor<READ_ONLY>
          ax(csp_metadata_prs[i], ColumnSpace::AXIS_VECTOR_FID);
        auto axes_sz = ColumnSpace::size(ax[0]);
        auto resultp = result.begin();
        auto axesp = ax[0].begin();
        while (resultp != result_end
               && std::distance(axesp, ax[0].begin()) != axes_sz
               && *resultp == *axesp) {
          ++resultp;
          ++axesp;
        }
        result_end = resultp;
      }
    }
    std::fill(result_end, result.end(), -1);
  }
  return result;
}

void
Table::add_columns(
  Context ctx,
  Runtime* rt,
  const std::map<
    ColumnSpace,
    std::vector<std::pair<std::string, TableField>>>& columns) {

  PhysicalRegion fields_pr =
    rt->map_region(ctx, table_fields_requirement(fields_lr, READ_WRITE));
  std::optional<PhysicalRegion> csp_md_pr;
  {
    auto cols = Table::columns(rt, fields_pr);
    if (cols.fields.size() > 0) {
      auto md = std::get<0>(cols.fields[0]).metadata_lr;
      RegionRequirement req(md, READ_ONLY, EXCLUSIVE, md);
      req.add_field(ColumnSpace::AXIS_SET_UID_FID);
      csp_md_pr = rt->map_region(ctx, req);
    }
  }
  add_columns(ctx, rt, columns, csp_md_pr, fields_pr);
  rt->unmap_region(ctx, fields_pr);
  if (csp_md_pr)
    rt->unmap_region(ctx, csp_md_pr.value());
}

void
Table::add_columns(
  Context ctx,
  Runtime* rt,
  const std::map<
    ColumnSpace,
    std::vector<std::pair<std::string, TableField>>>& columns,
  const std::optional<PhysicalRegion>& csp_md_pr,
  const PhysicalRegion& fields_pr) {

  std::optional<ColumnSpace::AXIS_SET_UID_TYPE> auid;
  if (csp_md_pr) {
    const ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
      au(csp_md_pr.value(), ColumnSpace::AXIS_SET_UID_FID);
    auid = au[0];
  }

  const NameAccessor<READ_WRITE>
    nms(fields_pr, static_cast<FieldID>(TableFieldsFid::NM));
  const DatatypeAccessor<READ_WRITE>
    dts(fields_pr, static_cast<FieldID>(TableFieldsFid::DT));
  const KeywordsAccessor<READ_WRITE>
    kws(fields_pr, static_cast<FieldID>(TableFieldsFid::KW));
  const MeasRefAccessor<READ_WRITE>
    mrs(fields_pr, static_cast<FieldID>(TableFieldsFid::MR));
  const MetadataAccessor<READ_WRITE>
    mds(fields_pr, static_cast<FieldID>(TableFieldsFid::MD));
  const ValueFidAccessor<READ_WRITE>
    vfs(fields_pr, static_cast<FieldID>(TableFieldsFid::VF));
  const ValuesAccessor<READ_WRITE>
    vss(fields_pr, static_cast<FieldID>(TableFieldsFid::VS));

  std::map<ColumnSpace, LogicalRegion> csp_vlrs;

  PointInDomainIterator<1> fields_pid(
    rt->get_index_space_domain(
      fields_pr.get_logical_region().get_index_space()));

  while (fields_pid() && mds[*fields_pid] != LogicalRegion::NO_REGION) {
    // No support for adding columns after complete indexing of Table
    assert(vss[*fields_pid] != LogicalRegion::NO_REGION);
    auto csp =
      ColumnSpace(vss[*fields_pid].get_index_space(), mds[*fields_pid]);
    if (csp_vlrs.count(csp) == 0)
      csp_vlrs.emplace(csp, vss[*fields_pid]);
    fields_pid++;
  }

  assert(fields_pid());

  for (auto& [csp, tfs] : columns) {
    {
      RegionRequirement
        req(csp.metadata_lr, READ_ONLY, EXCLUSIVE, csp.metadata_lr);
      req.add_field(ColumnSpace::AXIS_SET_UID_FID);
      auto md_pr = rt->map_region(ctx, req);
      const ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
        au(md_pr, ColumnSpace::AXIS_SET_UID_FID);
      if (!auid)
        auid = au[0];
      assert(auid.value() == au[0]);
      rt->unmap_region(ctx, md_pr);
    }
    if (csp_vlrs.count(csp) == 0) {
      FieldSpace fs = rt->create_field_space(ctx);
      csp_vlrs.emplace(
        csp,
        rt->create_logical_region(ctx, csp.column_is, fs));
    }
    LogicalRegion& values_lr = csp_vlrs[csp];
    std::set<FieldID> fids;
    FieldSpace fs = values_lr.get_field_space();
    rt->get_field_space_fields(fs, fids);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    for (auto& nm_tf : tfs) {
      auto& [nm, tf] = nm_tf;
      assert(fids.count(tf.fid) == 0);
      switch(tf.dt) {
#define ALLOC_FLD(DT)                                                   \
        case DT:                                                        \
          fa.allocate_field(hyperion::DataType<DT>::serdez_size, tf.fid); \
          break;
        HYPERION_FOREACH_DATATYPE(ALLOC_FLD)
#undef ALLOC_FLD
      default:
          assert(false);
        break;
      }
      nms[*fields_pid] = nm;
      dts[*fields_pid] = tf.dt;
      kws[*fields_pid] = tf.kw;
      mrs[*fields_pid] = tf.mr;
      mds[*fields_pid] = csp.metadata_lr;
      vfs[*fields_pid] = tf.fid;
      vss[*fields_pid] = values_lr;
      fids.insert(tf.fid);
      fields_pid++;
    }
  }
}

void
Table::remove_columns(
  Context ctx,
  Runtime* rt,
  const std::unordered_set<std::string>& columns,
  bool destroy_orphan_column_spaces) {

  auto fields_pr =
    rt->map_region(ctx, table_fields_requirement(fields_lr, READ_WRITE));
  remove_columns(ctx, rt, columns, destroy_orphan_column_spaces, fields_pr);
  rt->unmap_region(ctx, fields_pr);
}

void
Table::remove_columns(
  Context ctx,
  Runtime* rt,
  const std::unordered_set<std::string>& columns,
  bool destroy_orphan_column_spaces,
  const PhysicalRegion& fields_pr) {

  std::map<
    ColumnSpace,
    std::tuple<LogicalRegion, LogicalRegion, FieldAllocator>>
    csp_lrs_fa;

  {
    const NameAccessor<READ_WRITE>
      nms(fields_pr, static_cast<FieldID>(TableFieldsFid::NM));
    const DatatypeAccessor<READ_WRITE>
      dts(fields_pr, static_cast<FieldID>(TableFieldsFid::DT));
    const KeywordsAccessor<READ_WRITE>
      kws(fields_pr, static_cast<FieldID>(TableFieldsFid::KW));
    const MeasRefAccessor<READ_WRITE>
      mrs(fields_pr, static_cast<FieldID>(TableFieldsFid::MR));
    const MetadataAccessor<READ_WRITE>
      mds(fields_pr, static_cast<FieldID>(TableFieldsFid::MD));
    const ValueFidAccessor<READ_WRITE>
      vfs(fields_pr, static_cast<FieldID>(TableFieldsFid::VF));
    const ValuesAccessor<READ_WRITE>
      vss(fields_pr, static_cast<FieldID>(TableFieldsFid::VS));

    PointInDomainIterator<1> src_pid(
      rt->get_index_space_domain(
        fields_pr.get_logical_region().get_index_space()));
    PointInDomainIterator<1> dst_pid = src_pid;
    while (src_pid()) {
      bool remove = columns.count(nms[*src_pid]) > 0;
      if (remove) {
        auto csp =
          ColumnSpace(vss[*src_pid].get_index_space(), mds[*src_pid]);
        if (csp_lrs_fa.count(csp) == 0)
          csp_lrs_fa[csp] =
            std::make_tuple(
              mds[*src_pid],
              vss[*src_pid],
              rt->create_field_allocator(ctx, vss[*src_pid].get_field_space()));
        std::get<2>(csp_lrs_fa[csp]).free_field(vfs[*src_pid]);
        mrs[*src_pid].destroy(ctx, rt);
        kws[*src_pid].destroy(ctx, rt);
      } else if (src_pid[0] != dst_pid[0]) {
        nms[*dst_pid] = nms[*src_pid];
        dts[*dst_pid] = dts[*src_pid];
        kws[*dst_pid] = kws[*src_pid];
        mrs[*dst_pid] = mrs[*src_pid];
        mds[*dst_pid] = mds[*src_pid];
        vfs[*dst_pid] = vfs[*src_pid];
        vss[*dst_pid] = vss[*src_pid];
      }
      src_pid++;
      if (!remove)
        dst_pid++;
    }
    {
      static const hyperion::string nm;
      static const Keywords kw;
      static const MeasRef mr;
      while (dst_pid()) {
        nms[*dst_pid] = nm;
        dts[*dst_pid] = (type_tag_t)0;
        kws[*dst_pid] = kw;
        mrs[*dst_pid] = mr;
        mds[*dst_pid] = LogicalRegion::NO_REGION;
        vfs[*dst_pid] = 0;
        vss[*dst_pid] = LogicalRegion::NO_REGION;
        dst_pid++;
      }
    }
  }
  for (auto& clf : csp_lrs_fa) {
    ColumnSpace csp = std::get<0>(clf);
    LogicalRegion md, v;
    std::tie(md, v, std::ignore) = std::get<1>(clf);
    std::vector<FieldID> fids;
    rt->get_field_space_fields(v.get_field_space(), fids);
    if (fids.size() == 0) {
      if (destroy_orphan_column_spaces)
        csp.destroy(ctx, rt, true);
      rt->destroy_field_space(ctx, v.get_field_space());
      rt->destroy_logical_region(ctx, v);
    }
  }
}

void
Table::destroy(
  Context ctx,
  Runtime* rt,
  bool destroy_column_space_components) {

  auto cols = columns(ctx, rt).get_result<columns_result_t>();
  for (auto& [csp, vlr, tfs] : cols.fields) {
    if (vlr != LogicalRegion::NO_REGION) {
      rt->destroy_field_space(ctx, vlr.get_field_space());
      rt->destroy_logical_region(ctx, vlr);
    }
    if (destroy_column_space_components)
      csp.destroy(ctx, rt, true);
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

  assert(regions.size() == 1);
  return columns(rt, regions[0]);
}

Future /* columns_result_t */
Table::columns(Context ctx, Runtime *rt) const {
  TaskLauncher task(columns_task_id, TaskArgument(NULL, 0));
  task.add_region_requirement(table_fields_requirement(fields_lr, READ_ONLY));
  task.enable_inlining = true;
  return rt->execute_task(ctx, task);
}

Table::columns_result_t
Table::columns(Runtime *rt, const PhysicalRegion& fields_pr) {

  std::map<
    ColumnSpace,
    std::tuple<
      LogicalRegion,
      std::vector<columns_result_t::tbl_fld_t>>> cols;

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
  const ValueFidAccessor<READ_ONLY>
    vfs(fields_pr, static_cast<FieldID>(TableFieldsFid::VF));
  const ValuesAccessor<READ_ONLY>
    vss(fields_pr, static_cast<FieldID>(TableFieldsFid::VS));

  bool has_empty = false;
  for (PointInDomainIterator<1> pid(
         rt->get_index_space_domain(
           fields_pr.get_logical_region().get_index_space()));
       pid() && mds[*pid] != LogicalRegion::NO_REGION;
       pid++) {
    auto csp =
      ColumnSpace(
        ((vss[*pid] != LogicalRegion::NO_REGION)
         ? vss[*pid].get_index_space()
         : IndexSpace::NO_SPACE),
        mds[*pid]);
    if (!csp.is_empty()) {
      columns_result_t::tbl_fld_t tf =
        std::make_tuple(
          nms[*pid],
          TableField(dts[*pid], vfs[*pid], mrs[*pid], kws[*pid]));
      if (cols.count(csp) == 0)
        cols[csp] = std::make_tuple(vss[*pid], std::vector{tf});
      else
        std::get<1>(cols[csp]).push_back(tf);
    } else {
      assert(!has_empty);
      cols[csp] =
        std::make_tuple(
          LogicalRegion::NO_REGION,
          std::vector<columns_result_t::tbl_fld_t>{});
      has_empty = true;
    }
  }
  columns_result_t result;
  result.fields.reserve(cols.size());
  for (auto& [csp, lr_tfs] : cols) {
    auto& [lr, tfs] = lr_tfs;
    result.fields.emplace_back(csp, lr, tfs);
  }
  return result;
}

struct PartitionRowsTaskArgs {
  std::array<std::pair<bool, size_t>, Table::MAX_COLUMNS> block_sizes;
  std::array<IndexSpace, Table::MAX_COLUMNS> csp_iss;
};

TaskID Table::partition_rows_task_id;

const char* Table::partition_rows_task_name = "x::Table::partition_rows_task";

Table::partition_rows_result_t
Table::partition_rows_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *rt) {

  const PartitionRowsTaskArgs* args =
    static_cast<const PartitionRowsTaskArgs*>(task->args);

  std::vector<IndexSpace> csp_iss;
  for (size_t i = 0; i < MAX_COLUMNS; ++i) {
    if (args->csp_iss[i] == IndexSpace::NO_SPACE)
      break;
    csp_iss.push_back(args->csp_iss[i]);
  }
  std::vector<std::optional<size_t>> block_sizes;
  for (size_t i = 0; i < MAX_COLUMNS; ++i) {
    auto& [has_value, value] = args->block_sizes[i];
    if (has_value && value == 0)
      break;
    block_sizes.push_back(has_value ? value : std::optional<size_t>());
  }

  std::vector<PhysicalRegion> metadata_prs;
  metadata_prs.reserve(regions.size());
  std::copy(regions.begin(), regions.end(), std::back_inserter(metadata_prs));
  return partition_rows(ctx, rt, block_sizes, csp_iss, metadata_prs);
}

Future /* partition_rows_result_t */
Table::partition_rows(
  Context ctx,
  Runtime* rt,
  const std::vector<std::optional<size_t>>& block_sizes) const {

  PartitionRowsTaskArgs args;
  for (size_t i = 0; i < block_sizes.size(); ++i) {
    assert(block_sizes[i].value_or(1) > 0);
    args.block_sizes[i] =
      std::make_pair(block_sizes[i].has_value(), block_sizes[i].value_or(0));
  }
  args.block_sizes[block_sizes.size()] = std::make_pair(true, 0);

  auto cols = columns(ctx, rt).get_result<columns_result_t>();
  size_t csp_idx = 0;
  TaskLauncher task(partition_rows_task_id, TaskArgument(&args, sizeof(args)));
  for (auto& csp_vlr_tfs : cols.fields) {
    auto& csp = std::get<0>(csp_vlr_tfs);
    args.csp_iss[csp_idx++] = csp.column_is;
    auto md = csp.metadata_lr;
    RegionRequirement req(md, READ_ONLY, EXCLUSIVE, md);
    req.add_field(ColumnSpace::AXIS_VECTOR_FID);
    req.add_field(ColumnSpace::AXIS_SET_UID_FID);
    task.add_region_requirement(req);
  }
  return rt->execute_task(ctx, task);
}

Table::partition_rows_result_t
Table::partition_rows(
  Context ctx,
  Runtime* rt,
  const std::vector<std::optional<size_t>>& block_sizes,
  const std::vector<Legion::IndexSpace>& csp_iss,
  const std::vector<PhysicalRegion>& csp_metadata_prs) {

  assert(csp_iss.size() == csp_metadata_prs.size());

  partition_rows_result_t result;
  auto ixax = Table::index_axes(csp_metadata_prs);
  auto ixax_sz = ColumnSpace::size(ixax);
  if (block_sizes.size() > ixax_sz)
    return result;

  // copy block_sizes, extended to size of ixax with std::nullopt
  std::vector<std::optional<size_t>> blkszs(ixax_sz);
  {
    auto e = std::copy(block_sizes.begin(), block_sizes.end(), blkszs.begin());
    std::fill(e, blkszs.end(), std::nullopt);
  }

  std::vector<std::pair<int, Legion::coord_t>> parts;
  for (size_t i = 0; i < ixax_sz; ++i)
    if (blkszs[i].has_value())
      parts.emplace_back(ixax[i], blkszs[i].value());

  for (size_t i = 0; i < csp_metadata_prs.size(); ++i) {
    auto& pr = csp_metadata_prs[i];
    const ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
      auids(pr, ColumnSpace::AXIS_SET_UID_FID);
    result.partitions[i] =
      ColumnSpacePartition::create(ctx, rt, csp_iss[i], auids[0], parts, pr);
  }
  return result;
}

TaskID Table::convert_task_id;

const char* Table::convert_task_name = "x::Table::convert_task";

struct ConvertTaskArgs {
  size_t num_cols;
  std::array<IndexSpace, Table::MAX_COLUMNS> values_iss;
  std::array<std::pair<hyperion::string, FieldID>, Table::MAX_COLUMNS> fids;
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
    std::vector<int>,
    std::tuple<
      IndexSpace,
      std::vector<std::pair<std::string, TableField>>>> tbl_flds;
  std::optional<std::string> axes_uid;
  for (size_t i = 0; i < col_prs.size(); ++i) {
    auto& [md_pr, ax_pr, o_mr_drs, o_kw_pair] = col_prs[i];
    IndexSpaceT<1> is(ax_pr.get_logical_region().get_index_space());
    DomainT<1> dom = rt->get_index_space_domain(is);
    std::vector<int> axes(Domain(dom).hi()[0] + 1);
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
  std::map<
    ColumnSpace,
    std::vector<std::pair<std::string, TableField>>> columns;
  for (auto& [ax, is_tfs] : tbl_flds) {
    auto& [is, tfs] = is_tfs;
    // TODO: it's assumed here that the columns are not index columns...may want
    // to change this
    columns.emplace(
      ColumnSpace::create(ctx, rt, ax, axes_uid.value(), is, false),
      tfs);
  }
  return Table::create(ctx, rt, columns);
}

struct CopyValuesFromTaskArgs {
  size_t num_value_prs;
};

TaskID Table::copy_values_from_task_id;

const char* Table::copy_values_from_task_name =
  "x::Table::copy_values_from_task";

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
    RegionRequirement req = table_fields_requirement(fields_lr, READ_ONLY);
    {
      auto pr = rt->map_region(ctx, req);
      const ValuesAccessor<READ_ONLY>
        vss(pr, static_cast<FieldID>(TableFieldsFid::VS));
      for (PointInDomainIterator<1> pid(
             rt->get_index_space_domain(fields_lr.get_index_space()));
           pid();
        pid++) {
        if (vss[*pid] != LogicalRegion::NO_REGION) {
          auto lr = std::find(vlrs.begin(), vlrs.end(), vss[*pid]);
          if (lr == vlrs.end())
            vlrs.push_back(vss[*pid]);
        }
      }
      rt->unmap_region(ctx, pr);
    }
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
  for (auto& r : reqs)
    task.add_region_requirement(r);
  rt->execute_task(ctx, task);
}

void
Table::copy_values_from(
  Context ctx,
  Runtime* rt,
  const PhysicalRegion& fields_pr,
  const std::vector<std::tuple<PhysicalRegion, PhysicalRegion>>& src_col_prs) {

  std::unordered_map<std::string, Column> dst_cols =
    column_map(columns(rt, fields_pr), WRITE_ONLY);
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
      src_req.region = src_req.parent = src_vals_pr.get_logical_region();
      copy.add_copy_requirements(src_req, dst_cols[nm[0]].vreq);
    }
  }
  if (n > 0)
    rt->issue_copy_operation(ctx, copy);
}

TaskID Table::reindexed_task_id;

const char* Table::reindexed_task_name = "x::Table::reindexed_task";

struct ReindexedTaskArgs {
  std::array<std::pair<int, hyperion::string>, Table::MAX_COLUMNS> index_axes;
  bool allow_rows;
  char columns_buffer[]; // serialized Table::columns_result_t
};

Table::reindexed_result_t
Table::reindexed_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *rt) {

  static const ReindexedTaskArgs* args =
    static_cast<const ReindexedTaskArgs*>(task->args);
  Table::columns_result_t columns;
  columns.legion_deserialize(args->columns_buffer);

  std::vector<std::pair<int, std::string>> index_axes;
  for (size_t i = 0;
       i < MAX_COLUMNS && std::get<0>(args->index_axes[i]) >= 0;
       ++i) {
    auto& [d, nm] = args->index_axes[i];
    index_axes.emplace_back(d, nm);
  }

  const ValuesAccessor<READ_ONLY>
    vss(regions[0], static_cast<FieldID>(TableFieldsFid::VS));
  const ValueFidAccessor<READ_ONLY>
    vfs(regions[0], static_cast<FieldID>(TableFieldsFid::VF));
  std::vector<std::tuple<coord_t, ColumnRegions>> cregions;
  size_t rg = 1;
  for (auto& [csp, vlr, tfs] : columns.fields) {
    // Cannot reindex fully indexed Table -- catch this case before calling this
    // method
    assert(!csp.is_empty());
    RegionRequirement values_req = task->regions[rg];
    PhysicalRegion values = regions[rg++];
    PhysicalRegion metadata = regions[rg++];
    for (auto& nm_tf : tfs) {
      ColumnRegions cr;
      cr.values = std::make_tuple(values_req, values);
      cr.metadata = metadata;
      const TableField& tf = std::get<1>(nm_tf);
      if (!tf.mr.is_empty()) {
        cr.mr_metadata = regions[rg++];
        auto ovreq = std::get<1>(tf.mr.requirements(READ_ONLY));
        if (ovreq)
          cr.mr_values = regions[rg++];
      }
      if (!tf.kw.is_empty()) {
        cr.kw_type_tags = regions[rg++];
        cr.kw_values = regions[rg++];
      }
      std::optional<unsigned> offset;
      for (unsigned i = 0; !offset && i < MAX_COLUMNS; ++i) {
        if (vss[i] == vlr && vfs[i] == tf.fid)
          offset = i;
      }
      cregions.emplace_back(offset.value(), cr);
    }
  }
  return reindexed(ctx, rt, index_axes, args->allow_rows, regions[0], cregions);
}

Future /* reindexed_result_t */
Table::reindexed(
  Context ctx,
  Runtime *rt,
  const std::vector<std::pair<int, std::string>>& index_axes,
  bool allow_rows) const {

  size_t args_buffer_sz = 0;
  std::unique_ptr<char[]> args_buffer;
  ReindexedTaskArgs* args = NULL;
  std::vector<RegionRequirement> reqs;
  bool can_reindex;
  {
    RegionRequirement req = table_fields_requirement(fields_lr, READ_ONLY);
    reqs.push_back(req);
    {
      std::vector<LogicalRegion> vlrs;
      auto pr = rt->map_region(ctx, req);
      Table::columns_result_t columns = Table::columns(rt, pr);
      can_reindex =
        std::none_of(
          columns.fields.begin(),
          columns.fields.end(),
          [](auto& csp_vlr_tfs) {
            return std::get<0>(csp_vlr_tfs).is_empty();
          });
      if (can_reindex) {
        args_buffer_sz =
          sizeof(ReindexedTaskArgs) + columns.legion_buffer_size();
        args_buffer = std::move(std::make_unique<char[]>(args_buffer_sz));
        args = reinterpret_cast<ReindexedTaskArgs*>(args_buffer.get());
        columns.legion_serialize(args->columns_buffer);
        rt->unmap_region(ctx, pr);
        for (auto& [csp, vlr, tfs] : columns.fields) {
          {
            vlrs.push_back(vlr);
            std::vector<FieldID> fids;
            rt->get_field_space_fields(ctx, vlr.get_field_space(), fids);
            RegionRequirement req(vlr, READ_ONLY, EXCLUSIVE, vlr);
            // this region requires permissions, but can remain unmapped
            req.add_fields(fids, false);
            reqs.push_back(req);
          }
          {
            RegionRequirement
              req(csp.metadata_lr, READ_ONLY, EXCLUSIVE, csp.metadata_lr);
            req.add_field(ColumnSpace::AXIS_VECTOR_FID);
            req.add_field(ColumnSpace::AXIS_SET_UID_FID);
            req.add_field(ColumnSpace::INDEX_FLAG_FID);
            reqs.push_back(req);
          }
          for (auto& nm_tf : tfs) {
            const TableField& tf = std::get<1>(nm_tf);
            if (!tf.mr.is_empty()) {
              auto [mreq, ovreq] = tf.mr.requirements(READ_ONLY);
              reqs.push_back(mreq);
              if (ovreq)
                reqs.push_back(ovreq.value());
            }
            if (!tf.kw.is_empty()) {
              std::vector<FieldID> fids(tf.kw.size(rt));
              std::iota(fids.begin(), fids.end(), 0);
              auto kwr = tf.kw.requirements(rt, fids, READ_ONLY).value();
              reqs.push_back(kwr.type_tags);
              reqs.push_back(kwr.values);
            }
          }
        }
      }
      rt->unmap_region(ctx, pr);
    }
  }
  if (can_reindex) {
    {
      auto e =
        std::copy(index_axes.begin(), index_axes.end(), args->index_axes.begin());
      std::fill(
        e,
        args->index_axes.end(),
        std::make_pair(-1, hyperion::string()));
    }
    args->allow_rows = allow_rows;

    TaskLauncher task(
      reindexed_task_id,
      TaskArgument(args_buffer.get(), args_buffer_sz));
    for (auto& r : reqs)
      task.add_region_requirement(r);
    return rt->execute_task(ctx, task);
  } else {
    // TODO: log an error message: cannot be reindex a totally indexed Table
    return Future::from_value(rt, Table());
  }
}

struct ReindexCopyValuesTaskArgs {
  hyperion::TypeTag dt;
  FieldID fid;
};

Table::reindexed_result_t
Table::reindexed(
  Context ctx,
  Runtime *rt,
  const std::vector<std::pair<int, std::string>>& index_axes,
  bool allow_rows,
  const PhysicalRegion& fields_pr,
  const std::vector<std::tuple<coord_t, ColumnRegions>>& column_regions) {

  std::set<PhysicalRegion> md_prs;
  for (auto& crg : column_regions)
    md_prs.insert(std::get<1>(crg).metadata);
  std::vector<int> ixax =
    ColumnSpace::from_axis_vector(
      Table::index_axes(
        std::vector<PhysicalRegion>(md_prs.begin(), md_prs.end())));

  auto index_axes_extension = index_axes.begin();
  {
    auto ixaxp = ixax.begin();
    while (ixaxp != ixax.end()
           && index_axes_extension != index_axes.end()
           && *ixaxp != 0
           && *ixaxp == index_axes_extension->first) {
      ++ixaxp;
      ++index_axes_extension;
    }
    // for index_axes to extend the current table index axes, at this point
    // ixaxp should point to the row index value (0), and index_axes_extension
    // should not be index_axes.end(); anything else, and either index_axes is
    // not a proper index extension, or the table index axes are already
    // index_axes
    if (!(ixaxp != ixax.end() && *ixaxp == 0
          && index_axes_extension != index_axes.end())) {
      // TODO: log an error message: index_axes does not extend current Table
      // index axes
      return Table();
    }
  }

  // every index axis must be present in all ColumnSpaces, except those that are
  // already associated with index columns (only check from index_axes_extension
  // onward, since it can be assumed that the current set of index axes
  // satisfies this constraint)
  for (auto& pr : md_prs) {
    const ColumnSpace::AxisVectorAccessor<READ_ONLY>
      avs(pr, ColumnSpace::AXIS_VECTOR_FID);
    const ColumnSpace::IndexFlagAccessor<READ_ONLY>
      ifl(pr, ColumnSpace::INDEX_FLAG_FID);
    if (!ifl[0]) {
      auto av = ColumnSpace::from_axis_vector(avs[0]);
      auto valid_index_axes =
        std::all_of(
          index_axes_extension,
          index_axes.end(),
          [&av](auto& d_nm) {
            return
              std::find(av.begin(), av.end(), std::get<0>(d_nm)) != av.end();
          });
      if (!valid_index_axes) {
        // TODO: log an error message: all requested index_axes are not present
        // in all columns
        return Table();
      }
    }
  }

  // construct map to associate column name with various Column fields provided
  // by the Table instance and the column_regions
  std::unordered_map<
    std::string,
    std::tuple<Column, ColumnRegions, std::optional<int>>> named_columns;
  {
    auto cols = Table::column_map(Table::columns(rt, fields_pr));
    const NameAccessor<READ_ONLY>
      nms(fields_pr, static_cast<FieldID>(TableFieldsFid::NM));
    for (auto& [i, cr] : column_regions) {
      auto& col = cols[nms[i]];
      Column
        col1(col.dt, col.fid, col.mr, col.kw, col.csp, std::get<0>(cr.values));
      named_columns[nms[i]] = std::make_tuple(col1, cr, std::nullopt);
    }
  }

  // can only reindex on an axis if table has a column with the associated name
  {
    std::set<std::string> missing;
    std::for_each(
      index_axes_extension,
      index_axes.end(),
      [&named_columns, &missing](auto& d_nm) {
        auto& nm = std::get<1>(d_nm);
        if (named_columns.count(nm) == 0)
          missing.insert(nm);
      });
    if (missing.size() > 0) {
      // TODO: log an error message: requested indexing column does not exist in
      // Table
      return Table();
    }
  }

  // ColumnSpaces are shared by Columns, so a map from ColumnSpaces to Column
  // names is useful
  std::map<ColumnSpace, std::vector<std::string>> csp_cols;
  for (auto& [nm, col_crg_ix] : named_columns) {
    auto& csp = std::get<0>(col_crg_ix).csp;
    if (csp_cols.count(csp) == 0)
      csp_cols[csp] = {nm};
    else
      csp_cols[csp].push_back(nm);
  }

  // compute new index column indexes, and map the index regions
  std::unordered_map<int, std::pair<LogicalRegion, PhysicalRegion>> index_cols;
  std::for_each(
    index_axes_extension,
    index_axes.end(),
    [&index_cols, &named_columns, &ctx, rt](auto& d_nm) {
      auto& [d, nm] = d_nm;
      auto lr = std::get<0>(named_columns[nm]).create_index(ctx, rt);
      RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
      req.add_field(Column::COLUMN_INDEX_ROWS_FID);
      auto pr = rt->map_region(ctx, req);
      index_cols[d] = std::make_pair(lr, pr);
      std::get<2>(named_columns[nm]) = d;
    });

  // do reindexing of ColumnSpaces
  std::map<ColumnSpace, ColumnSpace::reindexed_result_t> reindexed;
  {
    std::vector<std::pair<int, LogicalRegion>> ixcols;
    ixcols.reserve(index_cols.size());
    std::transform(
      index_axes_extension,
      index_axes.end(),
      std::back_inserter(ixcols),
      [&index_cols](auto& d_nm) {
        auto& d = std::get<0>(d_nm);
        return std::make_pair(d, std::get<0>(index_cols[d]));
      });

    for (auto& [csp, nms] : csp_cols) {
      auto& [col, crg, ix] = named_columns[nms[0]];
      const ColumnSpace::IndexFlagAccessor<READ_ONLY>
        ifl(crg.metadata, ColumnSpace::INDEX_FLAG_FID);
      if (!ifl[0]) {
        assert((unsigned)col.csp.column_is.get_dim() >= ixax.size());
        unsigned element_rank =
          (unsigned)col.csp.column_is.get_dim() - ixax.size();
        reindexed[csp] =
          ColumnSpace::reindexed(
            ctx,
            rt,
            element_rank,
            ixcols,
            allow_rows,
            col.csp.column_is,
            crg.metadata);
      }
    }
  }

  // create the reindexed table
  Table result;
  {
    std::map<ColumnSpace, std::vector<std::pair<std::string, TableField>>>
      nmtfs;
    bool index_cols_only = true;
    std::string axuid;
    for (auto& [csp, nms] : csp_cols) {
      std::vector<std::pair<std::string, TableField>> tfs;
      for (auto& nm : nms) {
        auto& [col, crg, ix] = named_columns.at(nm);
        const ColumnSpace::IndexFlagAccessor<READ_ONLY>
          ifl(crg.metadata, ColumnSpace::INDEX_FLAG_FID);
        const ColumnSpace::AxisVectorAccessor<READ_ONLY>
          av(crg.metadata, ColumnSpace::AXIS_VECTOR_FID);
        const ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
          auid(crg.metadata, ColumnSpace::AXIS_SET_UID_FID);
        axuid = auid[0];
        TableField
          tf(col.dt, col.fid, col.mr.clone(ctx, rt), col.kw.clone(ctx, rt));
        if (ix || ifl[0]) {
          ColumnSpace icsp;
          if (ix)
            icsp =
              ColumnSpace::create(
                ctx,
                rt,
                {ix.value()},
                auid[0],
                col.csp.column_is, // NB: assume ownership
                true);
          else
            icsp =
              ColumnSpace::create(
                ctx,
                rt,
                ColumnSpace::from_axis_vector(av[0]),
                auid[0],
                rt->create_index_space(
                  ctx,
                  rt->get_index_space_domain(col.csp.column_is)),
                true);
          nmtfs[icsp] = {{nm, tf}};
        } else {
          index_cols_only = false;
          tfs.emplace_back(nm, tf);
        }
      }
      if (tfs.size() > 0 && reindexed.count(csp) > 0)
        nmtfs[std::get<0>(reindexed[csp])] = tfs;
    }
    if (index_cols_only) {
      // Table now consists of only index columns. We maintain a phantom column
      // with an empty ColumnSpace to record the index axes (we do this for
      // consistency, even though any order of index axes could be supported in
      // this case)
      ColumnSpace icsp =
        ColumnSpace::create(
          ctx,
          rt,
          ixax,
          axuid,
          IndexSpace::NO_SPACE,
          false);
      nmtfs[icsp] = std::vector<std::pair<std::string, TableField>>{};
    }
    result = Table::create(ctx, rt, nmtfs);
  }

  // copy values from old table to new
  {
    const unsigned min_block_size = 1000000;
    CopyLauncher index_column_copier;
    auto dflds = result.columns(ctx, rt).get_result<columns_result_t>();
    for (auto& [dcsp, dvlr, dtfs] : dflds.fields) {
      RegionRequirement
        md_req(dcsp.metadata_lr, READ_ONLY, EXCLUSIVE, dcsp.metadata_lr);
      auto dcsp_md_pr = rt->map_region(ctx, md_req);
      const ColumnSpace::AxisVectorAccessor<READ_ONLY>
        dav(dcsp_md_pr, ColumnSpace::AXIS_VECTOR_FID);
      const ColumnSpace::IndexFlagAccessor<READ_ONLY>
        difl(dcsp_md_pr, ColumnSpace::INDEX_FLAG_FID);
      if (difl[0]) {
        // an index column in result Table
        LogicalRegion slr;
        FieldID sfid;
        if (index_cols.count(dav[0][0]) > 0) {
          // a new index column
          slr = std::get<0>(index_cols[dav[0][0]]);
          sfid = Column::COLUMN_INDEX_VALUE_FID;
        } else {
          // an old index column
          auto& [col, crg, ix] = named_columns[std::get<0>(dtfs[0])];
          slr = std::get<0>(crg.values).region;
          sfid = col.fid;
        }
        RegionRequirement src(slr, {sfid}, {sfid}, READ_ONLY, EXCLUSIVE, slr);
        auto& dfid = std::get<1>(dtfs[0]).fid;
        RegionRequirement
          dst(dvlr, {dfid}, {dfid}, WRITE_ONLY, EXCLUSIVE, dvlr);
        index_column_copier.add_copy_requirements(src, dst);
      } else {
        // a reindexed column
        IndexSpace sis;
        IndexSpace cs;
        LogicalRegion slr;
        LogicalPartition slp;
        LogicalRegion rctlr;
        LogicalPartition rctlp;
        LogicalPartition dlp;
        {
          // all table fields in rtfs share an IndexSpace and LogicalRegion
          auto& [col, crg, ix] = named_columns[std::get<0>(dtfs[0])];
          sis = col.csp.column_is;
          IndexPartition sip =
            hyperion::partition_over_all_cpus(ctx, rt, sis, min_block_size);
          cs = rt->get_index_partition_color_space_name(ctx, sip);
          slr = col.vreq.region;
          slp = rt->get_logical_partition(ctx, slr, sip);
          rctlr = std::get<1>(reindexed[col.csp]);
          rctlp = rt->get_logical_partition(ctx, rctlr, sip);
          IndexPartition dip =
            rt->create_partition_by_image_range(
              ctx,
              dvlr.get_index_space(),
              rctlp,
              rctlr,
              ColumnSpace::REINDEXED_ROW_RECTS_FID,
              cs,
              DISJOINT_COMPLETE_KIND);
          dlp = rt->get_logical_partition(ctx, dvlr, dip);
        }
        ReindexCopyValuesTaskArgs args;
        IndexTaskLauncher task(
          reindex_copy_values_task_id,
          cs,
          TaskArgument(&args, sizeof(args)),
          ArgumentMap());
        task.add_region_requirement(
          RegionRequirement(
            rctlp,
            0,
            {ColumnSpace::REINDEXED_ROW_RECTS_FID},
            {ColumnSpace::REINDEXED_ROW_RECTS_FID},
            READ_ONLY,
            EXCLUSIVE,
            rctlr));
        for (auto& [nm, tf] : dtfs) {
          auto& [col, crg, ix] = named_columns[nm];
          args.dt = col.dt;
          args.fid = col.fid;
          task.region_requirements.erase(task.region_requirements.begin() + 1);
          task.add_region_requirement(
            RegionRequirement(slp, 0, READ_ONLY, EXCLUSIVE, slr));
          task.add_field(1, col.fid);
          task.add_region_requirement(
            RegionRequirement(dlp, 0, WRITE_ONLY, EXCLUSIVE, dvlr));
          task.add_field(2, tf.fid);
          rt->execute_index_space(ctx, task);
        }
        rt->destroy_index_partition(ctx, dlp.get_index_partition());
        rt->destroy_logical_partition(ctx, dlp);
        rt->destroy_logical_partition(ctx, rctlp);
        rt->destroy_index_partition(ctx, slp.get_index_partition());
        rt->destroy_logical_partition(ctx, slp);
      }
      rt->unmap_region(ctx, dcsp_md_pr);
    }
    rt->issue_copy_operation(ctx, index_column_copier);
  }

  for (auto& [d, lr_pr] : index_cols) {
    auto& [lr, pr] = lr_pr;
    rt->unmap_region(ctx, pr);
    rt->destroy_field_space(ctx, lr.get_field_space());
    // DON'T do this: rt->destroy_index_space(ctx, lr.get_index_space());
    rt->destroy_logical_region(ctx, lr);
  }
  for (auto& [csp, rcsp_rlr] : reindexed) {
    auto& [rcsp, rlr] = rcsp_rlr;
    rt->destroy_field_space(ctx, rlr.get_field_space());
    // DON'T destroy index space
    rt->destroy_logical_region(ctx, rlr);
  }
  return result;
}

TaskID Table::reindex_copy_values_task_id;

const char* Table::reindex_copy_values_task_name =
  "x::Table::reindex_copy_values_task";

template <hyperion::TypeTag DT, int DIM>
using SA = FieldAccessor<
  READ_ONLY,
  typename hyperion::DataType<DT>::ValueType,
  DIM,
  coord_t,
  AffineAccessor<typename hyperion::DataType<DT>::ValueType, DIM, coord_t>,
  true>;

template <hyperion::TypeTag DT, int DIM>
using DA = FieldAccessor<
  WRITE_ONLY,
  typename hyperion::DataType<DT>::ValueType,
  DIM,
  coord_t,
  AffineAccessor<typename hyperion::DataType<DT>::ValueType, DIM, coord_t>,
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
reindex_copy_values(
  Runtime *rt,
  FieldID val_fid,
  const RegionRequirement& rect_req,
  const PhysicalRegion& rect_pr,
  const PhysicalRegion& src_pr,
  const PhysicalRegion& dst_pr) {

  int rowdim = rect_pr.get_logical_region().get_dim();
  int srcdim = src_pr.get_logical_region().get_dim();
  int dstdim = dst_pr.get_logical_region().get_dim();

  switch ((rowdim * LEGION_MAX_DIM + srcdim) * LEGION_MAX_DIM + dstdim) {
#define CPY(ROWDIM,SRCDIM,DSTDIM)                                       \
    case ((ROWDIM * LEGION_MAX_DIM + SRCDIM) * LEGION_MAX_DIM + DSTDIM): { \
      const SA<DT,SRCDIM> from(src_pr, val_fid);                        \
      const RA<DSTDIM,ROWDIM> rct(rect_pr, Column::COLUMN_INDEX_ROWS_FID); \
      const DA<DT,DSTDIM> to(dst_pr, val_fid);                          \
      for (PointInDomainIterator<ROWDIM> row(                           \
             rt->get_index_space_domain(rect_req.region.get_index_space()), \
             false);                                                    \
           row();                                                       \
           ++row) {                                                     \
        Point<SRCDIM> ps;                                               \
        for (size_t i = 0; i < ROWDIM; ++i)                             \
          ps[i] = row[i];                                               \
        for (PointInRectIterator<DSTDIM> pd(rct[*row], false); pd(); pd++) { \
          size_t i = SRCDIM - 1;                                        \
          size_t j = DSTDIM - 1;                                        \
          while (i >= ROWDIM)                                           \
            ps[i--] = pd[j--];                                          \
          to[*pd] = from[ps];                                           \
        }                                                               \
      }                                                                 \
      break;                                                            \
    }
    HYPERION_FOREACH_LMN(CPY)
#undef CPY
    default:
      assert(false);
      break;
  }
}

void
Table::reindex_copy_values_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *rt) {

  const ReindexCopyValuesTaskArgs* args =
    static_cast<const ReindexCopyValuesTaskArgs*>(task->args);

  switch (args->dt) {
#define CPYDT(DT)                               \
    case DT:                                    \
      reindex_copy_values<DT>(                  \
        rt,                                     \
        args->fid,                              \
        task->regions[0],                       \
        regions[0],                             \
        regions[1],                             \
        regions[2]);                            \
      break;
    HYPERION_FOREACH_DATATYPE(CPYDT)
#undef CPYDT
    default:
      assert(false);
      break;
  }
}

void
Table::preregister_tasks() {
  {
    // index_axes_task
    index_axes_task_id = Runtime::generate_static_task_id();
    TaskVariantRegistrar registrar(index_axes_task_id, index_axes_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    registrar.set_leaf();
    Runtime::preregister_task_variant<index_axes_result_t, index_axes_task>(
      registrar,
      index_axes_task_name);
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
    // partition_rows_task
    partition_rows_task_id = Runtime::generate_static_task_id();
    TaskVariantRegistrar
      registrar(partition_rows_task_id, partition_rows_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    registrar.set_leaf();
    Runtime::preregister_task_variant<
      partition_rows_result_t,
      partition_rows_task>(
      registrar,
      partition_rows_task_name);
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
  {
    // reindexed_task
    reindexed_task_id = Runtime::generate_static_task_id();
    TaskVariantRegistrar
      registrar(reindexed_task_id, reindexed_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    Runtime::preregister_task_variant<reindexed_result_t, reindexed_task>(
      registrar,
      reindexed_task_name);
  }
  {
    // reindex_copy_values_task
    reindex_copy_values_task_id = Runtime::generate_static_task_id();
    TaskVariantRegistrar
      registrar(reindex_copy_values_task_id, reindex_copy_values_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    Runtime::preregister_task_variant<reindex_copy_values_task>(
        registrar,
        reindex_copy_values_task_name);
  }
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
