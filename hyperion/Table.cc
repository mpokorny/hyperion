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
#include <hyperion/Table.h>

#include <mappers/default_mapper.h>

#include <map>
#include <unordered_set>

using namespace hyperion;

using namespace Legion;

#ifdef HYPERION_USE_CASACORE
# define FOREACH_TABLE_FIELD_FID(__FUNC__)      \
  __FUNC__(TableFieldsFid::NM)                  \
  __FUNC__(TableFieldsFid::DT)                  \
  __FUNC__(TableFieldsFid::KW)                  \
  __FUNC__(TableFieldsFid::MR)                  \
  __FUNC__(TableFieldsFid::RC)                  \
  __FUNC__(TableFieldsFid::MD)                  \
  __FUNC__(TableFieldsFid::VF)                  \
  __FUNC__(TableFieldsFid::VS)
#else
# define FOREACH_TABLE_FIELD_FID(__FUNC__)      \
  __FUNC__(TableFieldsFid::NM)                  \
  __FUNC__(TableFieldsFid::DT)                  \
  __FUNC__(TableFieldsFid::KW)                  \
  __FUNC__(TableFieldsFid::MD)                  \
  __FUNC__(TableFieldsFid::VF)                  \
  __FUNC__(TableFieldsFid::VS)
#endif

static const hyperion::string empty_nm;
static const Keywords empty_kw;
#ifdef HYPERION_USE_CASACORE
static const MeasRef empty_mr;
static const hyperion::string empty_rc;
#endif
static const type_tag_t empty_dt = (type_tag_t) 0;
static const FieldID empty_vf = 0;
static const LogicalRegion empty_md;
static const LogicalRegion empty_vs;

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
    unsigned* npart = reinterpret_cast<unsigned*>(b);
    b += sizeof(unsigned);
    for (*npart = 0;
         *npart < p.partition.size() && p.partition[*npart].stride != 0;
         ++(*npart)) {
      *reinterpret_cast<AxisPartition*>(b) = p.partition[*npart];
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

std::optional<ColumnSpacePartition>
Table::partition_rows_result_t::find(const ColumnSpace& cs) const {
  auto p =
    std::find_if(
      partitions.begin(),
      partitions.end(),
      [&cs](auto& csp) { return csp.column_space == cs; });
  if (p != partitions.end())
    return *p;
  return std::nullopt;
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
      result[nm] = Column(
        tf.dt,
        tf.fid,
#ifdef HYPERION_USE_CASACORE
        tf.mr,
        tf.rc,
#endif
        tf.kw,
        csp,
        vreq);
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
  const std::vector<
    std::pair<
      ColumnSpace,
      std::vector<std::pair<std::string, TableField>>>>& columns) {

  size_t num_cols = 0;
  for (auto& csp_cs : columns) {
    // phantom column not allowed in "columns"; one will be generated by this
    // method if it is needed
    assert(!csp_cs.first.is_empty());
    num_cols += csp_cs.second.size();
  }
  {
    std::unordered_set<std::string> cnames;
    for (auto& csp_cs : columns)
      for (auto& nm_c : csp_cs.second)
        cnames.insert(nm_c.first);
    assert(cnames.count("") == 0);
    assert(cnames.size() == num_cols);
  }

  std::vector<PhysicalRegion> csp_md_prs;
  std::vector<int> ixcol_axes;
  std::string axuid;
  // index columns must be at the head of "columns" vector
  {
    size_t i = 0;
    {
      bool ixcol_prefix = true;
      while (i < columns.size() && ixcol_prefix) {
        auto& lr = columns[i].first.metadata_lr;
        RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
        req.add_field(ColumnSpace::INDEX_FLAG_FID);
        req.add_field(ColumnSpace::AXIS_VECTOR_FID);
        req.add_field(ColumnSpace::AXIS_SET_UID_FID);
        csp_md_prs.push_back(rt->map_region(ctx, req));
        const ColumnSpace::AxisVectorAccessor<READ_ONLY>
          ax(csp_md_prs.back(), ColumnSpace::AXIS_VECTOR_FID);
        const ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
          au(csp_md_prs.back(), ColumnSpace::AXIS_SET_UID_FID);
        const ColumnSpace::IndexFlagAccessor<READ_ONLY>
          ifl(csp_md_prs.back(), ColumnSpace::INDEX_FLAG_FID);
        ixcol_prefix = ifl[0];
        if (ixcol_prefix) {
          ixcol_axes.push_back(ax[0][0]);
          axuid = au[0];
          ++i;
        } else {
          ixcol_axes.clear();
        }
      }
    }
    while (i < columns.size()) {
      auto& lr = columns[i++].first.metadata_lr;
      RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
      req.add_field(ColumnSpace::INDEX_FLAG_FID);
      req.add_field(ColumnSpace::AXIS_VECTOR_FID);
      req.add_field(ColumnSpace::AXIS_SET_UID_FID);
      csp_md_prs.push_back(rt->map_region(ctx, req));
      const ColumnSpace::IndexFlagAccessor<READ_ONLY>
        ifl(csp_md_prs.back(), ColumnSpace::INDEX_FLAG_FID);
      bool is_index = ifl[0];
      if (is_index) {
        // FIXME: log error: Index columns must be at the head of new table
        // columns
        assert(!is_index);
      }
    }
  }

  bool added;
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

      const NameAccessor<WRITE_ONLY>
        nms(fields_pr, static_cast<FieldID>(TableFieldsFid::NM));
      const DatatypeAccessor<WRITE_ONLY>
        dts(fields_pr, static_cast<FieldID>(TableFieldsFid::DT));
      const KeywordsAccessor<WRITE_ONLY>
        kws(fields_pr, static_cast<FieldID>(TableFieldsFid::KW));
#ifdef HYPERION_USE_CASACORE
      const MeasRefAccessor<WRITE_ONLY>
        mrs(fields_pr, static_cast<FieldID>(TableFieldsFid::MR));
      const RefColumnAccessor<WRITE_ONLY>
        rcs(fields_pr, static_cast<FieldID>(TableFieldsFid::RC));
#endif
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
        nms[*pid] = empty_nm;
        dts[*pid] = empty_dt;
        kws[*pid] = empty_kw;
#ifdef HYPERION_USE_CASACORE
        mrs[*pid] = empty_mr;
        rcs[*pid] = empty_rc;
#endif
        mds[*pid] = empty_md;
        vfs[*pid] = empty_vf;
        vss[*pid] = empty_vs;
      }
      rt->unmap_region(ctx, fields_pr);
    }
    PhysicalRegion fields_pr =
      rt->map_region(ctx, table_fields_requirement(fields_lr, READ_WRITE));
    std::vector<
      std::pair<
        ColumnSpace,
        std::pair<
          ssize_t,
          std::vector<std::pair<hyperion::string, TableField>>>>>
      hcols;
    for (size_t i = 0; i < columns.size(); ++i) {
      auto& [csp, nm_tfs] = columns[i];
      std::vector<std::pair<hyperion::string, TableField>> htfs;
      for (auto& [nm, tf]: nm_tfs)
        htfs.emplace_back(nm, tf);
      hcols.emplace_back(csp, std::make_pair(i, htfs));
    }
    // add a phantom column, if needed
    if (ixcol_axes.size() > 0) {
      auto csp =
        ColumnSpace::create(
          ctx,
          rt,
          ixcol_axes,
          axuid,
          IndexSpace::NO_SPACE,
          false);
      RegionRequirement
        req(csp.metadata_lr, READ_ONLY, EXCLUSIVE, csp.metadata_lr);
      req.add_field(ColumnSpace::INDEX_FLAG_FID);
      req.add_field(ColumnSpace::AXIS_VECTOR_FID);
      req.add_field(ColumnSpace::AXIS_SET_UID_FID);
      csp_md_prs.push_back(rt->map_region(ctx, req));
      hcols.emplace_back(
        csp,
        std::make_pair(
          csp_md_prs.size() - 1,
          std::vector<std::pair<hyperion::string, TableField>>()));
    }

    added =
      add_columns(
        ctx,
        rt,
        hcols,
        std::vector<LogicalRegion>(),
        fields_pr,
        csp_md_prs);
    for (auto& pr : csp_md_prs)
      rt->unmap_region(ctx, pr);
    rt->unmap_region(ctx, fields_pr);
  }
  Table result(fields_lr);
  if (!added)
    result.destroy(ctx, rt);
  return result;
}

bool
Table::is_empty() const {
  return fields_lr == LogicalRegion::NO_REGION;
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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  for (auto& [csp, vlr, tfs] : cols.fields) {
#pragma GCC diagnostic pop
    RegionRequirement
      req(csp.metadata_lr, READ_ONLY, EXCLUSIVE, csp.metadata_lr);
    req.add_field(ColumnSpace::AXIS_VECTOR_FID);
    req.add_field(ColumnSpace::INDEX_FLAG_FID);
    task.add_region_requirement(req);
  }
  return rt->execute_task(ctx, task);
}

Table::index_axes_result_t
Table::index_axes(const std::vector<PhysicalRegion>& csp_metadata_prs) {

  Table::index_axes_result_t result;
  Table::index_axes_result_t::iterator result_end = result.begin();
  std::unordered_set<int> indexes;
  if (csp_metadata_prs.size() > 0) {
    size_t i = 0;
    for (;
         result_end == result.begin() && i < csp_metadata_prs.size();
         ++i) {
      const ColumnSpace::IndexFlagAccessor<READ_ONLY>
        ifl(csp_metadata_prs[i], ColumnSpace::INDEX_FLAG_FID);
      const ColumnSpace::AxisVectorAccessor<READ_ONLY>
        ax(csp_metadata_prs[i], ColumnSpace::AXIS_VECTOR_FID);
      if (!ifl[0]) {
        result = ax[0];
        result_end = result.begin() + ColumnSpace::size(ax[0]);
      } else {
        indexes.insert(ax[0][0]);
      }
    }
    for (;
         result_end != result.begin() && i < csp_metadata_prs.size();
         ++i) {
      const ColumnSpace::IndexFlagAccessor<READ_ONLY>
        ifl(csp_metadata_prs[i], ColumnSpace::INDEX_FLAG_FID);
      const ColumnSpace::AxisVectorAccessor<READ_ONLY>
        ax(csp_metadata_prs[i], ColumnSpace::AXIS_VECTOR_FID);
      if (!ifl[0]) {
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
      } else {
        indexes.insert(ax[0][0]);
      }
    }
  }
  // all values in result must be either 0 (ROW) or correspond to one of the
  // index columns
  if (std::any_of(
        result.begin(),
        result_end,
        [&indexes](int& r) { return r > 0 && indexes.count(r) == 0; }))
    result_end = result.begin();
  // all values in indexes must appear somewhere in result
  else if (std::any_of(
        indexes.begin(),
        indexes.end(),
        [result_begin=result.begin(), &result_end](const int& i) {
          return std::find(result_begin, result_end, i) == result_end;
        }))
    result_end = result.begin();
  std::fill(result_end, result.end(), -1);
  return result;
}

TaskID Table::add_columns_task_id;

const char* Table::add_columns_task_name = "x::Table::add_columns_task";

struct AddColumnsTaskArgs {
  std::array<
    std::tuple<ColumnSpace, size_t, hyperion::string, TableField>,
    Table::MAX_COLUMNS> columns;
  std::array<LogicalRegion, Table::MAX_COLUMNS> vlrs;
};

Table::add_columns_result_t
Table::add_columns_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *rt) {

  const AddColumnsTaskArgs* args =
    static_cast<const AddColumnsTaskArgs*>(task->args);
  std::map<
    ColumnSpace,
    std::pair<ssize_t, std::vector<std::pair<hyperion::string, TableField>>>>
    columns;
  {
    size_t i = 0;
    while (i < MAX_COLUMNS) {
      auto& [csp, idx, nm, tf] = args->columns[i++];
      if (!csp.is_empty()) {
        if (columns.count(csp) == 0)
          columns[csp] =
            std::make_pair(
              idx,
              std::vector<std::pair<hyperion::string, TableField>>());
        columns[csp].second.emplace_back(nm, tf);
      } else {
        break;
      }
    }
  }
  std::vector<LogicalRegion> vlrs;
  {
    size_t i = 0;
    while (i < MAX_COLUMNS) {
      auto& vlr = args->vlrs[i++];
      if (vlr != LogicalRegion::NO_REGION)
        vlrs.push_back(vlr);
      else
        break;
    }
  }
  const PhysicalRegion& fields_pr = regions[0];
  std::vector<PhysicalRegion> csp_md_prs(regions.begin() + 1, regions.end());
  std::vector<
    std::pair<
      ColumnSpace,
      std::pair<ssize_t, std::vector<std::pair<hyperion::string, TableField>>>>>
    colv(columns.begin(), columns.end());
  return add_columns(ctx, rt, colv, vlrs, fields_pr, csp_md_prs);
}

Future /* add_columns_result_t */
Table::add_columns(
  Context ctx,
  Runtime* rt,
  const std::vector<
    std::pair<
      ColumnSpace,
      std::vector<std::pair<std::string, TableField>>>>& new_columns) const {

  if (new_columns.size() == 0)
    return Future::from_value(rt, true);

  AddColumnsTaskArgs args;
  TaskLauncher task(add_columns_task_id, TaskArgument(&args, sizeof(args)));
  std::vector<RegionRequirement> reqs;
  std::vector<ColumnSpace> new_csps;
  reqs.push_back(table_fields_requirement(fields_lr, READ_WRITE));
  auto current_columns = columns(ctx, rt).get_result<columns_result_t>();
  std::map<ColumnSpace, size_t> current_csp_idxs;
  {
    size_t i = 0;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
    for (auto& [csp, vlr, tfs] : current_columns.fields) {
#pragma GCC diagnostic pop
      current_csp_idxs[csp] = i;
      assert(vlr != LogicalRegion::NO_REGION);
      args.vlrs[i++] = vlr;
      RegionRequirement
        req(csp.metadata_lr, READ_ONLY, EXCLUSIVE, csp.metadata_lr);
      req.add_field(ColumnSpace::AXIS_VECTOR_FID);
      req.add_field(ColumnSpace::AXIS_SET_UID_FID);
      req.add_field(ColumnSpace::INDEX_FLAG_FID);
      reqs.push_back(req);
    }
    if (i < MAX_COLUMNS)
      args.vlrs[i] = LogicalRegion::NO_REGION;
  }
  {
    size_t i = 0;
    for (auto& [csp, nm_tfs]: new_columns) {
      if (current_csp_idxs.count(csp) == 0) {
        current_csp_idxs[csp] = reqs.size() - 1;
        RegionRequirement
          req(csp.metadata_lr, READ_ONLY, EXCLUSIVE, csp.metadata_lr);
        req.add_field(ColumnSpace::AXIS_VECTOR_FID);
        req.add_field(ColumnSpace::AXIS_SET_UID_FID);
        req.add_field(ColumnSpace::INDEX_FLAG_FID);
        reqs.push_back(req);
      }
      auto idx = current_csp_idxs[csp];
      for (auto& [nm, tf]: nm_tfs)
        args.columns[i++] = std::make_tuple(csp, idx, string(nm), tf);
    }
    if (i < MAX_COLUMNS)
      args.columns[i] =
        std::make_tuple(ColumnSpace(), -1, string(), TableField());
  }
  for (auto& req : reqs)
    task.add_region_requirement(req);
  return rt->execute_task(ctx, task);
}

Table::add_columns_result_t
Table::add_columns(
  Context ctx,
  Runtime* rt,
  const std::vector<
    std::pair<
      ColumnSpace,
      std::pair<
        ssize_t,
        std::vector<std::pair<hyperion::string, TableField>>>>>& new_columns,
  const std::vector<LogicalRegion>& vlrs,
  const PhysicalRegion& fields_pr,
  const std::vector<PhysicalRegion>& csp_md_prs) {

  if (new_columns.size() == 0)
    return true;

  // all columns must have a common axes uid
  assert(csp_md_prs.size() > 0);
  const ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
    au(csp_md_prs[0], ColumnSpace::AXIS_SET_UID_FID);
  ColumnSpace::AXIS_SET_UID_TYPE auid = au[0];
  for (size_t i = 1; i < csp_md_prs.size(); ++i) {
    const ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
      au(csp_md_prs[i], ColumnSpace::AXIS_SET_UID_FID);
    if (auid != au[0])
      return false;
  }

  auto current_columns = columns(rt, fields_pr);
  auto current_column_map = column_map(current_columns);
  // column names must be unique
  {
    std::set<std::string> new_column_names;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
    for (auto& [csp, idx_nmtfs]: new_columns)
      for (auto& [hnm, tf]: idx_nmtfs.second) {
#pragma GCC diagnostic pop
        std::string nm = hnm;
        if (current_column_map.count(nm) > 0
            || new_column_names.count(nm) > 0)
          return false;
        new_column_names.insert(nm);
      }
  }
  // get ColumnSpace regions for current columns only
  std::vector<PhysicalRegion> current_csp_md_prs;
  for (auto& pr : csp_md_prs) {
    auto c =
      std::find_if(
        current_columns.fields.begin(),
        current_columns.fields.end(),
        [lr=pr.get_logical_region()](auto& csp_vlr_tfs) {
          return lr == std::get<0>(csp_vlr_tfs).metadata_lr;
        });
    if (c != current_columns.fields.end())
      current_csp_md_prs.push_back(pr);
  }
  // compute index axes for set of all (current+new) columns
  auto new_ixax =
    ColumnSpace::from_axis_vector(index_axes(csp_md_prs));
  // if size of new_ixax is zero, index axes could not be determined, meaning
  // that the new table cannot be created
  if (new_ixax.size() == 0)
    return false;

  // new index columns can be added only if the current table has no index
  // columns (there may be special cases where such an operation would be
  // feasible (starting from a totally indexed table?), but there is always the
  // alternative of adding non-index columns first, and then reindexing the
  // table)
  auto current_ixax =
    ColumnSpace::from_axis_vector(index_axes(current_csp_md_prs));
  if (current_ixax.size() > 0
      && !std::equal(
        current_ixax.begin(),
        current_ixax.end(),
        new_ixax.begin(),
        new_ixax.end()))
    return false;

  const NameAccessor<READ_WRITE>
    nms(fields_pr, static_cast<FieldID>(TableFieldsFid::NM));
  const DatatypeAccessor<READ_WRITE>
    dts(fields_pr, static_cast<FieldID>(TableFieldsFid::DT));
  const KeywordsAccessor<READ_WRITE>
    kws(fields_pr, static_cast<FieldID>(TableFieldsFid::KW));
#ifdef HYPERION_USE_CASACORE
  const MeasRefAccessor<READ_WRITE>
    mrs(fields_pr, static_cast<FieldID>(TableFieldsFid::MR));
  const RefColumnAccessor<READ_WRITE>
    rcs(fields_pr, static_cast<FieldID>(TableFieldsFid::RC));
#endif
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

  {
    size_t num_csp = 0;
    // gather up all ColumnSpaces, not including any associated with a phantom
    // column (which is always the last field in the table)
    while (fields_pid()
           && mds[*fields_pid] != empty_md
           && vss[*fields_pid] != empty_vs) {
      auto csp =
        ColumnSpace(vss[*fields_pid].get_index_space(), mds[*fields_pid]);
      if (csp_vlrs.count(csp) == 0) {
        csp_vlrs[csp] = vss[*fields_pid];
        ++num_csp;
      }
      fields_pid++;
    }
  }
  if (!fields_pid()) {
    // FIXME: log error: cannot add further columns to Table with maximum
    // allowed number of columns
    assert(fields_pid());
  }
  // a phantom column is never reused, and so we can reclaim its resources
  if (mds[*fields_pid] != empty_md && vss[*fields_pid] == empty_vs) {
    rt->destroy_field_space(ctx, mds[*fields_pid].get_field_space());
    rt->destroy_index_space(ctx, mds[*fields_pid].get_index_space());
    rt->destroy_logical_region(ctx, mds[*fields_pid]);
    mds[*fields_pid] = empty_md;
  }

  std::optional<ColumnSpace> new_phantom_csp;
  std::set<FieldID> all_fids;
  for (auto& [csp, idx_nmtfs] : new_columns) {
    auto& [idx, nm_tfs] = idx_nmtfs;
    if (csp.is_empty()) {
      assert(!new_phantom_csp.has_value());
      new_phantom_csp = csp;
    } else {
      assert(0 <= idx && (size_t)idx < csp_md_prs.size());
      if (csp_vlrs.count(csp) == 0) {
        auto& md = csp_md_prs[idx];
        auto current_mdp =
          std::find(current_csp_md_prs.begin(), current_csp_md_prs.end(), md);
        if (current_mdp == current_csp_md_prs.end()) {
          FieldSpace fs = rt->create_field_space(ctx);
          csp_vlrs.emplace(
            csp,
            rt->create_logical_region(ctx, csp.column_is, fs));
        } else {
          assert((size_t)idx < vlrs.size());
          csp_vlrs[csp] = vlrs[idx];
        }
      }
      LogicalRegion& values_lr = csp_vlrs[csp];
      std::set<FieldID> fids;
      FieldSpace fs = values_lr.get_field_space();
      rt->get_field_space_fields(fs, fids);
      all_fids.merge(fids);
      FieldAllocator fa = rt->create_field_allocator(ctx, fs);
      for (auto& [nm, tf] : nm_tfs) {
        assert(all_fids.count(tf.fid) == 0);
        switch(tf.dt) {
#define ALLOC_FLD(DT)                                                   \
          case DT:                                                      \
            fa.allocate_field(DataType<DT>::serdez_size, tf.fid); \
            break;
          HYPERION_FOREACH_DATATYPE(ALLOC_FLD)
#undef ALLOC_FLD
        default:
            assert(false);
          break;
        }
        assert(fields_pid());
        nms[*fields_pid] = nm;
        dts[*fields_pid] = tf.dt;
        kws[*fields_pid] = tf.kw;
#ifdef HYPERION_USE_CASACORE
        mrs[*fields_pid] = tf.mr;
        rcs[*fields_pid] = tf.rc.value_or(empty_rc);
#endif
        mds[*fields_pid] = csp.metadata_lr;
        vfs[*fields_pid] = tf.fid;
        vss[*fields_pid] = values_lr;
        all_fids.insert(tf.fid);
        fields_pid++;
      }
    }
  }
  if (new_phantom_csp) {
    assert(fields_pid());
    nms[*fields_pid] = empty_nm;
    dts[*fields_pid] = empty_dt;
    kws[*fields_pid] = empty_kw;
#ifdef HYPERION_USE_CASACORE
    mrs[*fields_pid] = empty_mr;
    rcs[*fields_pid] = empty_rc;
#endif
    mds[*fields_pid] = new_phantom_csp.value().metadata_lr;
    vfs[*fields_pid] = empty_vf;
    vss[*fields_pid] = empty_vs;
    fields_pid++;
  }
  return true;
}

void
Table::remove_columns(
  Context ctx,
  Runtime* rt,
  const std::unordered_set<std::string>& columns,
  bool destroy_orphan_column_spaces) const {

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
#ifdef HYPERION_USE_CASACORE
    const MeasRefAccessor<READ_WRITE>
      mrs(fields_pr, static_cast<FieldID>(TableFieldsFid::MR));
    const RefColumnAccessor<READ_WRITE>
      rcs(fields_pr, static_cast<FieldID>(TableFieldsFid::RC));
#endif
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
#ifdef HYPERION_USE_CASACORE
        mrs[*src_pid].destroy(ctx, rt);
#endif
        kws[*src_pid].destroy(ctx, rt);
      } else if (src_pid[0] != dst_pid[0]) {
        nms[*dst_pid] = nms[*src_pid];
        dts[*dst_pid] = dts[*src_pid];
        kws[*dst_pid] = kws[*src_pid];
#ifdef HYPERION_USE_CASACORE
        mrs[*dst_pid] = mrs[*src_pid];
        rcs[*dst_pid] = rcs[*src_pid];
#endif
        mds[*dst_pid] = mds[*src_pid];
        vfs[*dst_pid] = vfs[*src_pid];
        vss[*dst_pid] = vss[*src_pid];
      }
      src_pid++;
      if (!remove)
        dst_pid++;
    }
    while (dst_pid()) {
      nms[*dst_pid] = empty_nm;
      dts[*dst_pid] = empty_dt;
      kws[*dst_pid] = empty_kw;
#ifdef HYPERION_USE_CASACORE
      mrs[*dst_pid] = empty_mr;
      rcs[*dst_pid] = empty_rc;
#endif
      mds[*dst_pid] = empty_md;
      vfs[*dst_pid] = empty_vf;
      vss[*dst_pid] = empty_vs;
      dst_pid++;
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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  for (auto& [csp, vlr, tfs] : cols.fields) {
#pragma GCC diagnostic pop
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
#ifdef HYPERION_USE_CASACORE
  const MeasRefAccessor<READ_ONLY>
    mrs(fields_pr, static_cast<FieldID>(TableFieldsFid::MR));
  const RefColumnAccessor<READ_ONLY>
    rcs(fields_pr, static_cast<FieldID>(TableFieldsFid::RC));
#endif
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
       pid() && mds[*pid] != empty_md;
       pid++) {
    auto csp =
      ColumnSpace(
        ((vss[*pid] != empty_vs)
         ? vss[*pid].get_index_space()
         : IndexSpace::NO_SPACE),
        mds[*pid]);
    if (!csp.is_empty()) {
      columns_result_t::tbl_fld_t tf =
        std::make_tuple(
          nms[*pid],
          TableField(
            dts[*pid],
            vfs[*pid],
#ifdef HYPERION_USE_CASACORE
            mrs[*pid],
            ((rcs[*pid].size() > 0)
             ? std::make_optional<hyperion::string>(rcs[*pid])
             : std::nullopt),
#endif
            kws[*pid]));
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
    req.add_field(ColumnSpace::INDEX_FLAG_FID);
    task.add_region_requirement(req);
  }
  return rt->execute_task(ctx, task);
}

Table::partition_rows_result_t
Table::partition_rows(
  Context ctx,
  Runtime* rt,
  const std::vector<std::optional<size_t>>& block_sizes,
  const std::vector<IndexSpace>& csp_iss,
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

  std::vector<std::pair<int, coord_t>> parts;
  for (size_t i = 0; i < ixax_sz; ++i)
    if (blkszs[i].has_value())
      parts.emplace_back(ixax[i], blkszs[i].value());

  for (size_t i = 0; i < csp_metadata_prs.size(); ++i) {
    auto& pr = csp_metadata_prs[i];
    const ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
      auids(pr, ColumnSpace::AXIS_SET_UID_FID);
    result.partitions.push_back(
      ColumnSpacePartition::create(ctx, rt, csp_iss[i], auids[0], parts, pr));
  }
  return result;
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

  const ReindexedTaskArgs* args =
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
#ifdef HYPERION_USE_CASACORE
      if (!tf.mr.is_empty()) {
        cr.mr_metadata = regions[rg++];
        cr.mr_values = regions[rg++];
        if (std::get<2>(tf.mr.requirements(READ_ONLY)))
          cr.mr_index = regions[rg++];
      }
#endif
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
#ifdef HYPERION_USE_CASACORE
            if (!tf.mr.is_empty()) {
              auto [mrq, vrq, oirq] = tf.mr.requirements(READ_ONLY);
              reqs.push_back(mrq);
              reqs.push_back(vrq);
              if (oirq)
                reqs.push_back(oirq.value());
            }
#endif
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
        std::make_pair(-1, string()));
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
      return reindexed_result_t();
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
      Column col1(
        col.dt,
        col.fid,
#ifdef HYPERION_USE_CASACORE
        col.mr,
        col.rc,
#endif
        col.kw,
        col.csp,
        std::get<0>(cr.values));
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
      return reindexed_result_t();
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
      for (auto& nm : nms) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
        auto& [col, crg, ix] = named_columns[nm];
#pragma GCC diagnostic pop
        const ColumnSpace::IndexFlagAccessor<READ_ONLY>
          ifl(crg.metadata, ColumnSpace::INDEX_FLAG_FID);
        if (!ifl[0] && !ix && reindexed.count(csp) == 0) {
          assert((unsigned)csp.column_is.get_dim() >= ixax.size());
          unsigned element_rank =
            (unsigned)csp.column_is.get_dim() - ixax.size();
          reindexed[csp] =
            ColumnSpace::reindexed(
              ctx,
              rt,
              element_rank,
              ixcols,
              allow_rows,
              csp.column_is,
              crg.metadata);
        }
      }
    }
  }

  // create the reindexed table
  reindexed_result_t result;
  {
    std::map<
      // first element exists to order the map in index axes order
      std::pair<size_t, ColumnSpace>,
      std::vector<std::pair<std::string, TableField>>>
      nmtfs;
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
        std::optional<MeasRef::DataRegions> odrs;
        if (crg.mr_metadata) {
          MeasRef::DataRegions drs;
          drs.metadata = crg.mr_metadata.value();
          drs.values = crg.mr_values.value();
          drs.index = crg.mr_index;
          odrs = drs;
        }
        std::optional<Keywords::pair<PhysicalRegion>> okwrs;
        if (crg.kw_values) {
          Keywords::pair<PhysicalRegion> kwrs;
          kwrs.values = crg.kw_values.value();
          kwrs.type_tags = crg.kw_type_tags.value();
          okwrs = kwrs;
        }
        TableField tf(
          col.dt,
          col.fid,
#ifdef HYPERION_USE_CASACORE
          (odrs ? MeasRef::clone(ctx, rt, odrs.value()) : MeasRef()),
          col.rc,
#endif
          (okwrs ? Keywords::clone(ctx, rt, okwrs.value()) : Keywords()));
        if (ix || ifl[0]) {
          ColumnSpace icsp;
          int d;
          if (ix) {
            d = ix.value();
            icsp =
              ColumnSpace::create(
                ctx,
                rt,
                {d},
                auid[0],
                // NB: take ownership of index space
                index_cols[ix.value()].first.get_index_space(),
                true);
          } else {
            d = av[0][0];
            icsp =
              ColumnSpace::create(
                ctx,
                rt,
                {d},
                auid[0],
                rt->create_index_space(
                  ctx,
                  rt->get_index_space_domain(col.csp.column_is)),
                true);
          }
          size_t i =
            std::distance(
              index_axes.begin(),
              std::find_if(
                index_axes.begin(),
                index_axes.end(),
                [&d](auto& d_nm) {
                  return std::get<0>(d_nm) == d;
                }));
          assert(i < index_axes.size());
          nmtfs[std::make_pair(i, icsp)] = {{nm, tf}};
        } else {
          tfs.emplace_back(nm, tf);
        }
      }
      if (tfs.size() > 0 && reindexed.count(csp) > 0)
        nmtfs[std::make_pair(ixax.size(), std::get<0>(reindexed[csp]))] = tfs;
    }
    std::vector<
      std::pair<
      ColumnSpace,
        std::vector<std::pair<std::string, TableField>>>> cols;
    for (auto& [i_csp, tfs] : nmtfs)
      cols.emplace_back(std::get<1>(i_csp), tfs);
    result = Table::create(ctx, rt, cols);
  }

  // copy values from old table to new
  {
    const unsigned min_block_size = 1000000;
    CopyLauncher index_column_copier;
    auto dflds = result.columns(ctx, rt).get_result<columns_result_t>();
    for (auto& [dcsp, dvlr, dtfs] : dflds.fields) {
      RegionRequirement
        md_req(dcsp.metadata_lr, READ_ONLY, EXCLUSIVE, dcsp.metadata_lr);
      md_req.add_field(ColumnSpace::AXIS_VECTOR_FID);
      md_req.add_field(ColumnSpace::INDEX_FLAG_FID);
      auto dcsp_md_pr = rt->map_region(ctx, md_req);
      const ColumnSpace::AxisVectorAccessor<READ_ONLY>
        dav(dcsp_md_pr, ColumnSpace::AXIS_VECTOR_FID);
      const ColumnSpace::IndexFlagAccessor<READ_ONLY>
        difl(dcsp_md_pr, ColumnSpace::INDEX_FLAG_FID);
      if (difl[0]) {
        assert(dtfs.size() == 1);
        // an index column in result Table
        LogicalRegion slr;
        FieldID sfid;
        if (index_cols.count(dav[0][0]) > 0) {
          // a new index column
          slr = std::get<0>(index_cols[dav[0][0]]);
          sfid = Column::COLUMN_INDEX_VALUE_FID;
        } else {
          // an old index column
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
          auto& [col, crg, ix] = named_columns[std::get<0>(dtfs[0])];
#pragma GCC diagnostic pop
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
        IndexSpace cs;
        LogicalRegion slr;
        LogicalRegion rctlr;
        LogicalPartition rctlp;
        LogicalPartition dlp;
        {
          // all table fields in rtfs share an IndexSpace and LogicalRegion
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
          auto& [col, crg, ix] = named_columns[std::get<0>(dtfs[0])];
#pragma GCC diagnostic pop
          rctlr = std::get<1>(reindexed[col.csp]);
          IndexSpace ris = rctlr.get_index_space();
          IndexPartition rip =
            partition_over_default_tunable(
              ctx,
              rt,
              ris,
              min_block_size,
              Mapping::DefaultMapper::DefaultTunables::DEFAULT_TUNABLE_GLOBAL_CPUS);
          cs = rt->get_index_partition_color_space_name(ctx, rip);
          rctlp = rt->get_logical_partition(ctx, rctlr, rip);
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
          slr = col.vreq.region;
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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
          auto& [col, crg, ix] = named_columns[nm];
#pragma GCC diagnostic pop
          args.dt = col.dt;
          args.fid = col.fid;
          assert(tf.fid == col.fid);
          task.region_requirements.resize(1);
          task.add_region_requirement(
            RegionRequirement(slr, READ_ONLY, EXCLUSIVE, slr));
          task.add_field(1, col.fid);
          task.add_region_requirement(
            RegionRequirement(dlp, 0, WRITE_ONLY, EXCLUSIVE, dvlr));
          task.add_field(2, tf.fid);
          rt->execute_index_space(ctx, task);
        }
        rt->destroy_index_partition(ctx, dlp.get_index_partition());
        rt->destroy_logical_partition(ctx, dlp);
        rt->destroy_index_partition(ctx, rctlp.get_index_partition());
        rt->destroy_logical_partition(ctx, rctlp);
      }
      rt->unmap_region(ctx, dcsp_md_pr);
    }
    rt->issue_copy_operation(ctx, index_column_copier);
  }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  for (auto& [d, lr_pr] : index_cols) {
#pragma GCC diagnostic pop
    auto& [lr, pr] = lr_pr;
    rt->unmap_region(ctx, pr);
    rt->destroy_field_space(ctx, lr.get_field_space());
    // DON'T do this: rt->destroy_index_space(ctx, lr.get_index_space());
    rt->destroy_logical_region(ctx, lr);
  }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  for (auto& [csp, rcsp_rlr] : reindexed) {
    auto& [rcsp, rlr] = rcsp_rlr;
#pragma GCC diagnostic pop
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
      const RA<DSTDIM,ROWDIM> rct(rect_pr, ColumnSpace::REINDEXED_ROW_RECTS_FID); \
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
    // add_columns_task
    add_columns_task_id = Runtime::generate_static_task_id();
    TaskVariantRegistrar registrar(add_columns_task_id, add_columns_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    registrar.set_leaf();
    Runtime::preregister_task_variant<add_columns_result_t, add_columns_task>(
      registrar,
      add_columns_task_name);
  }
  {
    // partition_rows_task
    partition_rows_task_id = Runtime::generate_static_task_id();
    TaskVariantRegistrar
      registrar(partition_rows_task_id, partition_rows_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    Runtime::preregister_task_variant<
      partition_rows_result_t,
      partition_rows_task>(
      registrar,
      partition_rows_task_name);
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
