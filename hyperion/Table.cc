/*
 * Copyright 2020 Associated Universities, Inc. Washington DC, USA.
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
#include <hyperion/PhysicalTable.h>
#include <hyperion/TableMapper.h>

#include <mappers/default_mapper.h>

#include <map>
#include <unordered_set>

using namespace hyperion;

using namespace Legion;

template <size_t N>
std::array<std::tuple<hyperion::string, Column>, N>
to_columns_array(const std::unordered_map<std::string, Column>& cols) {

  std::array<std::tuple<hyperion::string, Column>, N> result;
  assert(cols.size() < N);
  size_t i = 0;
  for (auto& [nm, col] : cols)
    result[i++] = {nm, col};
  if (i < N)
    std::get<0>(result[i]) = "";
  return result;
}

template <size_t N>
std::unordered_map<std::string, Column>
from_columns_array(
  const std::array<std::tuple<hyperion::string, Column>, N>& ary) {

  std::unordered_map<std::string, Column> result;
  for (size_t i = 0; i < N && std::get<0>(ary[i]).size() > 0; ++i) {
    auto& [nm, col] = ary[i];
    result[nm] = col;
  }
  return result;
}

size_t
Table::add_columns_result_t::legion_buffer_size(void) const {
  size_t result = sizeof(unsigned);
  for (size_t i = 0; i < cols.size(); ++i) {
    auto& [nm, col] = cols[i];
    result += (nm.size() + 1) * sizeof(char) + sizeof(col);
  }
  return result;
}

size_t
Table::add_columns_result_t::legion_serialize(void* buffer) const {
  char* b = static_cast<char*>(buffer);
  *reinterpret_cast<unsigned*>(b) = (unsigned)cols.size();
  b += sizeof(unsigned);
  for (size_t i = 0; i < cols.size(); ++i) {
    auto& [nm, col] = cols[i];
    std::strcpy(b, nm.c_str());
    b += (nm.size() + 1) * sizeof(char);
    *reinterpret_cast<Column*>(b) = col;
    b += sizeof(col);
  }
  return b - static_cast<char*>(buffer);
}

size_t
Table::add_columns_result_t::legion_deserialize(const void* buffer) {
  const char* b = static_cast<const char*>(buffer);
  unsigned n = *reinterpret_cast<const unsigned*>(b);
  b += sizeof(n);
  cols.resize(n);
  for (size_t i = 0; i < n; ++i) {
    auto& [nm, col] = cols[i];
    nm = std::string(b);
    b += (nm.size() + 1) * sizeof(char);
    col = *reinterpret_cast<const Column*>(b);
    b += sizeof(col);
  }
  return b - static_cast<const char*>(buffer);
}

const Table::cgroup_t Table::cgroup_none = Legion::LogicalRegion::NO_REGION;

Table::Table(
  Runtime* rt,
  ColumnSpace&& index_col_cs,
  const LogicalRegion& index_col_region,
  const LogicalRegion& fields_lr_,
  const std::unordered_map<std::string, Column>& columns)
  : m_index_col_cs(index_col_cs)
  , m_index_col_parent(index_col_region)
  , m_index_col_region(index_col_region)
  , m_fields_lr(fields_lr_)
  , m_fixed_fields_lr(LogicalRegion::NO_REGION)
  , m_free_fields_lr(fields_lr_)
  , fields_partition(LogicalPartition::NO_PART)
  , m_columns(columns) {}

PhysicalTable
Table::attach_columns(
  Context ctx,
  Runtime* rt,
  Legion::PrivilegeMode table_privilege,
  const CXX_FILESYSTEM_NAMESPACE::path& file_path,
  const std::unordered_map<std::string, std::string>& column_paths,
  const std::unordered_map<std::string, std::tuple<bool, bool, bool>>&
  column_modes) const {

  std::unordered_set<std::string> colnames;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  for (auto& [nm, pth] : column_paths) {
#pragma GCC diagnostic pop
    if (column_modes.count(nm) > 0)
      colnames.insert(nm);
  }

  std::map<std::string, std::optional<Column::Requirements>> omitted;
  for (auto& [nm, col] : m_columns)
    if (colnames.count(nm) == 0)
      omitted[nm] = std::nullopt;
  auto [table_reqs, table_parts] =
    requirements(ctx, rt, ColumnSpacePartition(), table_privilege, omitted);
  PhysicalRegion index_col_md = rt->map_region(ctx, table_reqs[0]);
  unsigned idx_rank = ColumnSpace::size(ColumnSpace::axes(index_col_md));
  std::tuple<LogicalRegion, PhysicalRegion> index_col =
    {table_reqs[1].region, rt->map_region(ctx, table_reqs[1])};

  std::unordered_map<std::string, std::shared_ptr<PhysicalColumn>> pcols;
  for (auto& [nm, col] : m_columns) {
    std::optional<PhysicalRegion> metadata;
    if (colnames.count(nm) > 0) {
      if (!metadata) {
        auto req = col.cs.requirements(READ_ONLY, EXCLUSIVE);
        metadata = rt->map_region(ctx, req);
      }
      std::optional<Keywords::pair<Legion::PhysicalRegion>> kws;
      if (!col.kw.is_empty()) {
        auto nkw = col.kw.size(rt);
        std::vector<FieldID> fids(nkw);
        std::iota(fids.begin(), fids.end(), 0);
        auto rqs = col.kw.requirements(rt, fids, READ_ONLY, true).value();
        Keywords::pair<Legion::PhysicalRegion> kwprs;
        kwprs.type_tags = rt->map_region(ctx, rqs.type_tags);
        kwprs.values = rt->map_region(ctx, rqs.values);
        kws = kwprs;
      }
#ifdef HYPERION_USE_CASACORE
      std::optional<MeasRef::DataRegions> mr_drs;
      if (!col.mr.is_empty()) {
        auto [mrq, vrq, oirq] = col.mr.requirements(READ_ONLY, true);
        MeasRef::DataRegions prs;
        prs.metadata = rt->map_region(ctx, mrq);
        prs.values = rt->map_region(ctx, vrq);
        if (oirq)
          prs.index = rt->map_region(ctx, oirq.value());
        mr_drs = prs;
      }
#endif
      pcols[nm] =
        std::make_shared<PhysicalColumn>(
          rt,
          col.dt,
          col.fid,
          idx_rank,
          metadata.value(),
          col.region,
          col.region,
          kws
#ifdef HYPERION_USE_CASACORE
          , mr_drs
          , map(
            col.rc,
            [](const auto& n) {
              return
                std::make_tuple(
                  std::string(n),
                  std::shared_ptr<PhysicalColumn>());
            })
#endif
          );
    }
  }
#ifdef HYPERION_USE_CASACORE
  // Add pointers to reference columns. This should fail if the reference
  // column was left out of the arguments. FIXME!
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  for (auto& [nm, pc] : pcols) {
#pragma GCC diagnostic pop
    if (pc->refcol()) {
      auto& rcnm = std::get<0>(pc->refcol().value());
      pc->set_refcol(rcnm, pcols.at(rcnm));
    }
  }
#endif

  std::optional<std::tuple<Legion::LogicalRegion, Legion::PhysicalRegion>>
    free_fields;
  size_t rq = 2;
  if (table_reqs[rq].privilege != READ_ONLY) {
    free_fields = {table_reqs[rq].region, rt->map_region(ctx, table_reqs[rq])};
    ++rq;
  }
  std::tuple<Legion::LogicalRegion, Legion::PhysicalRegion> fixed_fields =
    {table_reqs[rq].region, rt->map_region(ctx, table_reqs[rq])};

  PhysicalTable result(
    index_col_md,
    table_reqs[1].parent,
    index_col,
    m_fields_lr, // FIXME
    fixed_fields,
    free_fields,
    pcols);
  result.attach_columns(ctx, rt, file_path, column_paths, column_modes);
  for (auto& p : table_parts)
    rt->destroy_logical_partition(ctx, p);
  return result;
}

RegionRequirement
Table::table_fields_requirement(
  LogicalRegion lr,
  LogicalRegion parent,
  PrivilegeMode mode) {

  RegionRequirement result(
    lr,
    mode,
    EXCLUSIVE,
    parent,
    TableMapper::to_mapping_tag(
      TableMapper::table_layout_tag));
  result.add_field(Table::cgroup_fid);
  result.add_field(Table::column_desc_fid);
  return result;
}

RegionRequirement
Table::table_fields_requirement(
  LogicalPartition lp,
  ProjectionID proj,
  LogicalRegion parent,
  PrivilegeMode mode) {

  RegionRequirement result(
    lp,
    proj,
    mode,
    EXCLUSIVE,
    parent,
    TableMapper::to_mapping_tag(
      TableMapper::table_layout_tag));
  result.add_field(Table::cgroup_fid);
  result.add_field(Table::column_desc_fid);
  return result;
}

Table
Table::create(
  Context ctx,
  Runtime* rt,
  ColumnSpace&& index_col_cs,
  fields_t&& fields) {

  size_t num_cols = 0;
  for (auto& [cs, tfs] : fields) {
    assert(!cs.is_empty());
    assert(cs.is_valid());
    num_cols += tfs.size();
  }
  {
    std::unordered_set<std::string> cnames;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
    for (auto& [cs, nm_tfs] : fields)
      for (auto& [nm, tf] : nm_tfs)
#pragma GCC diagnostic pop
        cnames.insert(nm);
    assert(cnames.count("") == 0);
    assert(cnames.size() == num_cols);
  }

  std::vector<PhysicalRegion> cs_md_prs;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  for (auto& [cs, tfs] : fields) {
#pragma GCC diagnostic pop
    cs_md_prs.push_back(
      rt->map_region(ctx, cs.requirements(READ_ONLY, EXCLUSIVE)));
  }

  // Create the table index column
  LogicalRegion index_col_region;
  {
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    fa.allocate_field(
      sizeof(DataType<Table::m_index_col_dt>::ValueType),
      Table::m_index_col_fid);
    index_col_region =
      rt->create_logical_region(ctx, index_col_cs.column_is, fs);
  }

  std::unordered_map<std::string, Column> added;
  LogicalRegion fields_lr;
  {
    IndexSpace is = rt->create_index_space(ctx, Rect<1>(0, MAX_COLUMNS - 1));
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    fa.allocate_field(sizeof(cgroup_t), cgroup_fid);
    fa.allocate_field(sizeof(ColumnDesc), column_desc_fid);
    fields_lr = rt->create_logical_region(ctx, is, fs);
    {
      PhysicalRegion fields_pr =
        rt->map_region(
          ctx,
          table_fields_requirement(fields_lr, fields_lr, WRITE_ONLY));
      const CGroupAccessor<WRITE_ONLY> cgroups(fields_pr, cgroup_fid);
      for (PointInDomainIterator<1> pid(
             rt->get_index_space_domain(fields_lr.get_index_space()));
           pid();
           pid++)
        cgroups.write(*pid, cgroup_none);
      rt->unmap_region(ctx, fields_pr);
    }

    PhysicalRegion free_fields_pr =
      rt->map_region(
        ctx,
        table_fields_requirement(fields_lr, fields_lr, READ_WRITE));
    std::vector<
      std::tuple<
        ColumnSpace,
        size_t,
        std::vector<std::pair<hyperion::string, TableField>>>>
      hcols;
    for (size_t i = 0; i < fields.size(); ++i) {
      auto& [cs, nm_tfs] = fields[i];
      std::vector<std::pair<hyperion::string, TableField>> htfs;
      for (auto& [nm, tf]: nm_tfs)
        htfs.emplace_back(nm, tf);
      hcols.emplace_back(cs, i, htfs);
    }

    auto index_col_md =
      rt->map_region(ctx, index_col_cs.requirements(READ_ONLY, EXCLUSIVE));

    auto ixax =
      ColumnSpace::from_axis_vector(ColumnSpace::axes(index_col_md));
    for (auto& pr : cs_md_prs) {
      const ColumnSpace::IndexFlagAccessor<READ_ONLY>
        ifl(pr, ColumnSpace::INDEX_FLAG_FID);
      if (ifl[0]) {
        const ColumnSpace::AxisVectorAccessor<READ_ONLY>
          av(pr, ColumnSpace::AXIS_VECTOR_FID);
        if (ColumnSpace::size(av[0]) != 1) {
          // FIXME: log warning: index columns must have exactly one axis
          assert(false);
          // FIXME: clean up
          return Table();
        }
        auto i = std::find(ixax.begin(), ixax.end(), av[0][0]);
        if (i == ixax.end()) {
          // FIXME: log warning: index columns must appear in table index
          assert(false);
          // FIXME: clean up
          return Table();
        }
        ixax.erase(i);
      }
    }
    if (ixax.size() > 1 || (ixax.size() == 1 && ixax[0] != 0)) {
      // FIXME: log warning: table index names missing index column
      assert(false);
      // FIXME: clean up
      return Table();
    }

    added =
      add_columns(
        ctx,
        rt,
        std::move(hcols),
        {free_fields_pr.get_logical_region(), free_fields_pr},
        {{}},
        cs_md_prs,
        {index_col_cs.column_is, index_col_md});
    for (auto& pr : cs_md_prs)
      rt->unmap_region(ctx, pr);
    rt->unmap_region(ctx, free_fields_pr);
  }
  return
    Table(rt, std::move(index_col_cs), index_col_region, fields_lr, added);
}

// std::vector<int>
// Table::index_axes(Context ctx, Runtime* rt) const {
//   ColumnSpace last_cs;
//   {
//     RegionRequirement req(fields_lr, READ_ONLY, EXCLUSIVE, fields_lr);
//     req.add_field(static_cast<FieldID>(TableFieldsFid::CS));
//     auto fields_pr = rt->map_region(ctx, req);
//     const ColumnSpaceAccessor<READ_ONLY>
//       css(fields_pr, static_cast<FieldID>(TableFieldsFid::CS));
//     for (PointInDomainIterator<1> pid(
//            rt->get_index_space_domain(fields_lr.get_index_space()));
//          pid() && css[*pid].is_valid();
//          pid++)
//       last_cs = css[*pid];
//     rt->unmap_region(ctx, fields_pr);
//   }
//   assert(last_cs.is_valid());
//   return last_cs.axes(ctx, rt);
// }

ColumnSpace
Table::index_column_space(Context ctx, Runtime* rt) const {
  // don't return ColumnSpace of index_col -- we don't want external copies,
  // especially in (real) Columns
  return m_index_col_cs.clone(ctx, rt);
}

bool
Table::is_empty() const {
  return m_index_col_cs.is_empty();
}

std::tuple<std::vector<RegionRequirement>, std::vector<LogicalPartition>>
Table::requirements(
  Context ctx,
  Runtime* rt,
  const ColumnSpacePartition& table_partition,
  PrivilegeMode table_privilege,
  const std::map<std::string, std::optional<Column::Requirements>>&
    column_requirements,
  const std::optional<Column::Requirements>&
    default_column_requirements) const {

  std::optional<std::tuple<LogicalRegion, PhysicalRegion>> fixed_fields;
  if (m_fixed_fields_lr != LogicalRegion::NO_REGION)
    fixed_fields = {
      m_fixed_fields_lr,
      rt->map_region(
        ctx,
        table_fields_requirement(m_fixed_fields_lr, m_fields_lr, READ_ONLY))
    };

  std::optional<std::tuple<LogicalRegion, PhysicalRegion>> free_fields;
  if (m_free_fields_lr != LogicalRegion::NO_REGION)
    free_fields = {
      m_free_fields_lr,
      rt->map_region(
        ctx,
        table_fields_requirement(m_free_fields_lr, m_fields_lr, READ_ONLY))
    };

  auto result =
    Table::requirements(
      ctx,
      rt,
      m_index_col_cs,
      m_index_col_parent,
      m_index_col_region,
      m_fields_lr,
      fixed_fields,
      free_fields,
      m_columns,
      table_partition,
      table_privilege,
      column_requirements,
      default_column_requirements);

  if (free_fields)
    rt->unmap_region(ctx, std::get<1>(free_fields.value()));
  if (fixed_fields)
    rt->unmap_region(ctx, std::get<1>(fixed_fields.value()));
  return result;
}

std::tuple<std::vector<RegionRequirement>, std::vector<LogicalPartition>>
Table::requirements(
  Context ctx,
  Runtime* rt,
  const ColumnSpace& index_col_cs,
  const LogicalRegion& index_col_parent,
  const LogicalRegion& index_col_region,
  const LogicalRegion& fields_lr,
  const std::optional<std::tuple<LogicalRegion, PhysicalRegion>>& fixed_fields,
  const std::optional<std::tuple<LogicalRegion, PhysicalRegion>>& free_fields,
  const std::unordered_map<std::string, Column>& columns,
  const ColumnSpacePartition& table_partition,
  PrivilegeMode table_privilege,
  const std::map<std::string, std::optional<Column::Requirements>>&
    column_requirements,
  const std::optional<Column::Requirements>&
    default_column_requirements) {

  // find which fields are in use
  std::map<std::string, Point<1>> used_fields;
  std::vector<Point<1>> unused_fields;
  for (auto& olrpr : {fixed_fields, free_fields}) {
    if (olrpr) {
      auto& [lr, pr] = olrpr.value();
      const CGroupAccessor<READ_ONLY> cgroups(pr, cgroup_fid);
      const ColumnDescAccessor<READ_ONLY> cdescs(pr, column_desc_fid);
      for (PointInDomainIterator<1> pid(
             rt->get_index_space_domain(lr.get_index_space()));
           pid();
           pid++) {
        if (cgroups.read(*pid) == cgroup_none)
          unused_fields.push_back(*pid);
        else
          used_fields[cdescs.read(*pid).name] = *pid;
      }
    }
  }

  // collect requirement parameters for each column
  std::map<std::string, Column::Requirements> column_reqs;
#ifdef HYPERION_USE_CASACORE
  std::map<std::string, Column::Requirements> mrc_reqs;
#endif
  std::map<cgroup_t, Column::Req> cg_mdreqs;
  for (auto& [nm, col] : columns) {
    cgroup_t cg = col.region;
    if ((default_column_requirements
         && (column_requirements.count(nm) == 0
             || column_requirements.at(nm)))
        || (!default_column_requirements
            && (column_requirements.count(nm) > 0
                && column_requirements.at(nm)))) {
      Column::Requirements colreqs =
        default_column_requirements.value_or(Column::default_requirements);
      if (column_requirements.count(nm) > 0)
        colreqs = column_requirements.at(nm).value();
      column_reqs[nm] = colreqs;
      if (cg_mdreqs.count(cg) == 0) {
        cg_mdreqs[cg] = colreqs.column_space;
      } else {
        // FIXME: log a warning, and return empty result;
        // warning: inconsistent requirements on shared Column metadata
        // regions
        assert(cg_mdreqs[cg] == colreqs.column_space);
      }
#ifdef HYPERION_USE_CASACORE
      if (col.rc)
        mrc_reqs[col.rc.value()] = colreqs;
#endif
    }
  }

#ifdef HYPERION_USE_CASACORE
  // apply mode of value column to its measure reference column
  for (auto& [nm, rq] : mrc_reqs)
    column_reqs.at(nm) = rq;
#endif

  // record points in fields region on which privileges are requested
  std::vector<Point<1>> fixed_pts;
  std::vector<Point<1>> free_pts;
  std::vector<Point<1>> unprivileged_pts;
  for (auto& [nm, col] : columns) {
    if (column_reqs.count(nm) > 0)
      fixed_pts.push_back(used_fields.at(nm));
    else
      unprivileged_pts.push_back(used_fields.at(nm));
  }
  std::copy(
    unused_fields.begin(),
    unused_fields.end(),
    std::back_inserter(
      (table_privilege == READ_ONLY) ? unprivileged_pts : free_pts));

  // create the partition of the table fields by privilege vs no privilege
  LogicalPartition use_lp;
  if (fixed_pts.size() > 0) { // some columns are selected
    auto nparts =
      std::min(fixed_pts.size(), (size_t)1)
      + std::min(free_pts.size(), (size_t)1);
    IndexSpace use_colors = rt->create_index_space(ctx, Rect<1>(0, nparts - 1));
    IndexPartition use_ip =
      rt->create_pending_partition(
        ctx,
        fields_lr.get_index_space(),
        use_colors,
        (unprivileged_pts.size() > 0) ? DISJOINT_KIND : DISJOINT_COMPLETE_KIND);

    IndexSpace fixed_is = rt->create_index_space(ctx, fixed_pts);
    rt->create_index_space_union(ctx, use_ip, fixed_fields_color, {fixed_is});

    if (free_pts.size() > 0) {
      IndexSpace free_is = rt->create_index_space(ctx, free_pts);
      rt->create_index_space_union(ctx, use_ip, free_fields_color, {free_is});
    }

    use_lp = rt->get_logical_partition(ctx, fields_lr, use_ip);
  }

  // create requirements, applying table_partition as needed
  std::map<ColumnSpace, LogicalPartition> partitions;
  if (table_partition.is_valid()) {
    auto csp =
      table_partition.project_onto(ctx, rt, index_col_cs)
      .get_result<ColumnSpacePartition>();
    auto lp =
      rt->get_logical_partition(ctx, index_col_region, csp.column_ip);
    csp.destroy(ctx, rt);
    partitions[index_col_cs] = lp;
  }

  // boolean elements in value of following maps is used to track whether the
  // requirement has already been added when iterating through columns
  std::map<cgroup_t, std::tuple<bool, RegionRequirement>> md_reqs;
  std::map<
    std::tuple<cgroup_t, PrivilegeMode, CoherenceProperty, MappingTagID>,
    std::tuple<bool, RegionRequirement>> val_reqs;
  for (auto& [nm, col] : columns) {
    if (column_reqs.count(nm) > 0) {
      auto& reqs = column_reqs.at(nm);
      cgroup_t cg = col.region;
      if (md_reqs.count(cg) == 0)
        md_reqs[cg] =
          {false,
           col.cs.requirements(
             reqs.column_space.privilege,
             reqs.column_space.coherence)};
      decltype(val_reqs)::key_type rg_rq =
        {cg, reqs.values.privilege, reqs.values.coherence, reqs.tag};
      if (val_reqs.count(rg_rq) == 0) {
        if (!table_partition.is_valid()) {
          val_reqs[rg_rq] =
            {false,
             RegionRequirement(
               col.region,
               reqs.values.privilege,
               reqs.values.coherence,
               col.region,
               reqs.tag)};
        } else {
          LogicalPartition lp;
          if (partitions.count(col.cs) == 0) {
            auto csp =
              table_partition.project_onto(ctx, rt, col.cs)
              .get_result<ColumnSpacePartition>();
            assert(csp.column_space == col.cs);
            lp = rt->get_logical_partition(ctx, col.region, csp.column_ip);
            csp.destroy(ctx, rt);
            partitions[col.cs] = lp;
          } else {
            lp = partitions[col.cs];
          }
          val_reqs[rg_rq] =
            {false,
             RegionRequirement(
               lp,
               0,
               reqs.values.privilege,
               reqs.values.coherence,
               col.region,
               reqs.tag)};
        }
      }
      std::get<1>(val_reqs[rg_rq]).add_field(col.fid, reqs.values.mapped);
    }
  }
  std::vector<LogicalPartition> lps_result;
  if (use_lp != LogicalPartition::NO_PART)
    lps_result.push_back(use_lp);
  for (auto& [csp, lp] : partitions)
    lps_result.push_back(lp);

  // gather all requirements, in order set by this traversal of fields
  std::vector<RegionRequirement> reqs_result;

  // start with index_col ColumnSpace metadata
  reqs_result.push_back(index_col_cs.requirements(READ_ONLY, EXCLUSIVE));
  // next, index_col index space partition
  if (table_partition.is_valid()) {
    RegionRequirement req(
      partitions[index_col_cs],
      0,
      {Table::m_index_col_fid},
      {}, // always remains unmapped!
      WRITE_ONLY,
      EXCLUSIVE,
      index_col_region);
    reqs_result.push_back(req);
  } else {
    RegionRequirement req(
      index_col_region,
      {Table::m_index_col_fid},
      {}, // always remains unmapped!
      WRITE_ONLY,
      EXCLUSIVE,
      index_col_region);
    reqs_result.push_back(req);
  }

  // next, table fields_lr, partitioned by use_lp; put requirement for
  // free_fields first, so that upon construction of PhysicalTable, it can be
  // determined whether there is a free field region by reading the region
  // privilege (which cannot be READ_ONLY for the free fields region)
  if (free_pts.size() > 0) {
    assert(table_privilege != READ_ONLY);
    auto free_lr =
      rt->get_logical_subregion_by_color(ctx, use_lp, free_fields_color);
    if (!table_partition.is_valid()) {
      reqs_result
        .push_back(
          table_fields_requirement(free_lr, fields_lr, table_privilege));
    } else {
      // partition free fields evenly across table_partition
      auto free_ip =
        rt->create_equal_partition(
          ctx,
          free_lr.get_index_space(),
          rt->get_index_partition_color_space_name(
            ctx,
            table_partition.column_ip));
      // TODO: is it OK to use fields_lr for the logical region of the
      // partition?
      LogicalPartition free_lp =
        rt->get_logical_partition(ctx, fields_lr, free_ip);
      lps_result.push_back(free_lp);
      reqs_result
        .push_back(
          table_fields_requirement(free_lp, 0, fields_lr, table_privilege));
    }
  }
  if (fixed_pts.size() > 0)
    reqs_result
      .push_back(
        table_fields_requirement(
          rt->get_logical_subregion_by_color(ctx, use_lp, fixed_fields_color),
          fields_lr,
          READ_ONLY));

  // add requirements for all logical regions in all selected columns
  for (auto& olrpr : {fixed_fields, free_fields}) {
    if (olrpr) {
      auto& [lr, pr] = olrpr.value();
      const CGroupAccessor<READ_ONLY> cgroups(pr, cgroup_fid);
      const ColumnDescAccessor<READ_ONLY> cdescs(pr, column_desc_fid);
      for (PointInDomainIterator<1> pid(
             rt->get_index_space_domain(lr.get_index_space()));
           pid();
           pid++) {
        auto cdesc = cdescs.read(*pid);
        if (cgroups.read(*pid) != cgroup_none
            && column_reqs.count(cdesc.name) > 0) {
          assert(columns.count(cdesc.name) > 0);
          auto& col = columns.at(cdesc.name);
          cgroup_t cg = col.region;
          auto& reqs = column_reqs.at(cdesc.name);
          {
            auto& [added, rq] = md_reqs.at(cg);
            if (!added) {
              reqs_result.push_back(rq);
              added = true;
            }
          }
          decltype(val_reqs)::key_type rg_rq =
            {cg, reqs.values.privilege, reqs.values.coherence, reqs.tag};
          auto& [added, rq] = val_reqs.at(rg_rq);
          if (!added) {
            reqs_result.push_back(rq);
            added = true;
          }
          if (cdesc.n_kw > 0) {
            auto& kw = col.kw;
            auto nkw = kw.size(rt);
            std::vector<FieldID> fids(nkw);
            std::iota(fids.begin(), fids.end(), 0);
            auto rqs =
              kw.requirements(
                rt,
                fids,
                reqs.keywords.privilege,
                reqs.keywords.mapped)
              .value();
            assert(cdesc.n_kw == 2);
            reqs_result.push_back(rqs.type_tags);
            reqs_result.push_back(rqs.values);
          }

#ifdef HYPERION_USE_CASACORE
          if (cdesc.n_mr > 0) {
            auto& mr = col.mr;
            auto [mrq, vrq, oirq] =
              mr.requirements(reqs.measref.privilege, reqs.measref.mapped);
            assert(cdesc.n_mr == 2 || cdesc.n_mr == 3);
            reqs_result.push_back(mrq);
            reqs_result.push_back(vrq);
            if (oirq) {
              assert(cdesc.n_mr == 3);
              reqs_result.push_back(oirq.value());
            }
          }
#endif
        }
      }
    }
  }
  return {reqs_result, lps_result};
}

TaskID Table::is_conformant_task_id;

const char* Table::is_conformant_task_name = "Table::is_conformant_task";

struct IsConformantArgs {
  std::array<std::tuple<hyperion::string, Column>, Table::MAX_COLUMNS> columns;
  IndexSpace cs_is;
  IndexSpace index_cs_is;
};

bool
Table::is_conformant_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *rt) {

  const IsConformantArgs* args =
    static_cast<const IsConformantArgs*>(task->args);

  auto columns = from_columns_array(args->columns);
  std::tuple<IndexSpace, PhysicalRegion> index_cs =
    {args->index_cs_is, regions[0]};
  return Table::is_conformant(rt, columns, index_cs, args->cs_is, regions[0]);
}

Future /* bool */
Table::is_conformant(Context ctx, Runtime* rt, const ColumnSpace& cs) const {

  if (m_index_col_cs.is_empty())
    return Future::from_value(rt, true);

  IsConformantArgs args;
  args.cs_is = cs.column_is;
  args.index_cs_is = m_index_col_cs.column_is;
  TaskLauncher task(is_conformant_task_id, TaskArgument(&args, sizeof(args)));
  args.columns = to_columns_array<args.columns.size()>(m_columns);
  task.add_region_requirement(
    m_index_col_cs.requirements(READ_ONLY, EXCLUSIVE));
  task.add_region_requirement(cs.requirements(READ_ONLY, EXCLUSIVE));
  return rt->execute_task(ctx, task);
}

template <int OBJECT_RANK, int SUBJECT_RANK>
static bool
do_domains_conform(
  const DomainT<OBJECT_RANK>& object,
  const DomainT<SUBJECT_RANK>& subject) {

  // does "subject" conform to "object"?

  static_assert(OBJECT_RANK <= SUBJECT_RANK);

  bool result = true;
  PointInDomainIterator<OBJECT_RANK> opid(object, false);
  PointInDomainIterator<SUBJECT_RANK> spid(subject, false);
  while (result && spid() && opid()) {
    Point<OBJECT_RANK> pt(0);
    while (result && spid()) {
      for (size_t i = 0; i < OBJECT_RANK; ++i)
        pt[i] = spid[i];
      result = pt == *opid;
      spid++;
    }
    opid++;
    if (!result)
      result = opid() && pt == *opid;
    else
      result = !opid();
  }
  return result;
}

bool
Table::is_conformant(
  Runtime* rt,
  const std::unordered_map<std::string, Column>& columns,
  const std::tuple<IndexSpace, PhysicalRegion>& index_cs,
  const IndexSpace& cs_is,
  const PhysicalRegion& cs_md_pr) {

  // if this ColumnSpace already exists in the Table, conformance must hold
  ColumnSpace cs(cs_is, cs_md_pr.get_logical_region());
  assert(!cs.is_empty());
  for (auto& [nm, col] : columns) {
    if (cs == col.cs)
      return true;
  }

  auto& [index_cs_is, index_cs_md_pr] = index_cs;
  const ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
    index_cs_au(index_cs_md_pr, ColumnSpace::AXIS_SET_UID_FID);
  const ColumnSpace::AxisVectorAccessor<READ_ONLY>
    index_cs_av(index_cs_md_pr, ColumnSpace::AXIS_VECTOR_FID);
  const ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
    cs_au(cs_md_pr, ColumnSpace::AXIS_SET_UID_FID);
  const ColumnSpace::AxisVectorAccessor<READ_ONLY>
    cs_av(cs_md_pr, ColumnSpace::AXIS_VECTOR_FID);
  const ColumnSpace::IndexFlagAccessor<READ_ONLY>
    cs_if(cs_md_pr, ColumnSpace::INDEX_FLAG_FID);
  bool result = false;
  // for conformance the axis uid must be that of the index column space
  if (index_cs_au[0] == cs_au[0]) {
    const auto index_ax = ColumnSpace::from_axis_vector(index_cs_av[0]);
    const auto cs_ax = ColumnSpace::from_axis_vector(cs_av[0]);
    if (!cs_if[0]) {
      // for conformance, the cs axis vector must have a prefix equal to the
      // axis vector of the index column space
      auto p =
        std::get<0>(
          std::mismatch(
            index_ax.begin(),
            index_ax.end(),
            cs_ax.begin(),
            cs_ax.end()));
      if (p == index_ax.end()) {
        const auto index_cs_d = rt->get_index_space_domain(index_cs_is);
        const auto cs_d = rt->get_index_space_domain(cs.column_is);
        if (index_cs_d.dense() && cs_d.dense()) {
          // when both index_cs and cs IndexSpaces are dense, it's sufficient to
          // compare their bounds within the rank of index_cs
          const auto index_cs_lo = index_cs_d.lo();
          const auto index_cs_hi = index_cs_d.hi();
          const auto cs_lo = cs_d.lo();
          const auto cs_hi = cs_d.hi();
          result = true;
          for (int i = 0; result && i < index_cs_d.get_dim(); ++i)
            result = index_cs_lo[i] == cs_lo[i] && index_cs_hi[i] == cs_hi[i];
        } else {
          switch (index_cs_d.get_dim() * LEGION_MAX_DIM + cs_d.get_dim()) {
#define CONFORM(IRANK, CRANK)                                       \
            case (IRANK * LEGION_MAX_DIM + CRANK):                  \
              result =                                              \
                do_domains_conform<IRANK, CRANK>(index_cs_d, cs_d); \
              break;
            HYPERION_FOREACH_MN(CONFORM)
#undef CONFORM
          default:
              assert(false);
            break;
          }
        }
      }
    } else {
      result =
        cs_ax.size() == 1
        && std::find(
          index_ax.begin(),
          index_ax.end(),
          cs_ax[0]) != index_ax.end();
    }
  }
  return result;
}

TaskID Table::add_columns_task_id;

const char* Table::add_columns_task_name = "Table::add_columns_task";

struct AddColumnsTaskArgs {
  std::array<
    std::tuple<ColumnSpace, size_t, hyperion::string, TableField>,
    Table::MAX_COLUMNS> new_columns;
  std::array<std::tuple<hyperion::string, Column>, Table::MAX_COLUMNS> columns;
  IndexSpace index_cs_is;
};

Table::add_columns_result_t
Table::add_columns_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *rt) {

  const AddColumnsTaskArgs* args =
    static_cast<const AddColumnsTaskArgs*>(task->args);
  auto columns = from_columns_array(args->columns);
  std::vector<
    std::tuple<
      ColumnSpace,
      size_t,
      std::vector<std::pair<hyperion::string, TableField>>>>
    new_columns;
  ColumnSpace last_cs;
  size_t last_idx;
  std::vector<std::pair<hyperion::string, TableField>> nm_tfs;
  for (size_t i = 0;
       i < args->new_columns.size()
         && std::get<2>(args->new_columns[i]).size() > 0;
       ++i) {
    auto& [cs, idx, nm, tf] = args->new_columns[i];
    if (last_cs != cs) {
      if (last_cs.is_valid())
        new_columns.emplace_back(last_cs, last_idx, nm_tfs);
      last_cs = cs;
      last_idx = idx;
      nm_tfs.clear();
    }
    nm_tfs.emplace_back(nm, tf);
  }
  if (last_cs.is_valid())
    new_columns.emplace_back(last_cs, last_idx, nm_tfs);

  std::tuple<IndexSpace, PhysicalRegion> index_cs =
    {args->index_cs_is, regions[0]};

  std::tuple<LogicalRegion, PhysicalRegion> free_fields =
    {task->regions.back().region, regions.back()};
  std::vector<PhysicalRegion>
    cs_md_prs(regions.begin() + 1, regions.end() - 1);
  auto added =
    add_columns(
      ctx,
      rt,
      std::move(new_columns),
      free_fields,
      columns,
      cs_md_prs,
      index_cs);
  add_columns_result_t result;
  for (auto& [nm, col] : added)
    result.cols.emplace_back(nm, col);
  return result;
}

bool
Table::add_columns(Context ctx, Runtime* rt, fields_t&& new_columns)  {

  if (new_columns.size() == 0)
    return true;

  if (m_free_fields_lr == LogicalRegion::NO_REGION) {
    // FIXME: log warning: no space remaining to add columns
    assert(false);
    return false;
  }

  AddColumnsTaskArgs args;
  TaskLauncher task(add_columns_task_id, TaskArgument(&args, sizeof(args)));
  args.columns = to_columns_array<args.columns.size()>(m_columns);

  std::vector<RegionRequirement> reqs;
  args.index_cs_is = m_index_col_cs.column_is;
  reqs.push_back(m_index_col_cs.requirements(READ_ONLY, EXCLUSIVE));

  std::map<ColumnSpace, size_t> cs_indexes;
  for (auto& [nm, col] : m_columns) {
    if (cs_indexes.count(col.cs) == 0) {
      cs_indexes[col.cs] = cs_indexes.size();
      reqs.push_back(col.cs.requirements(READ_ONLY, EXCLUSIVE));
    }
  }
  std::set<std::string> new_cnames;
  {
    size_t i = 0;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
    for (auto& [cs, nm_tfs]: new_columns) {
#pragma GCC diagnostic pop
      if (cs_indexes.count(cs) == 0) {
        cs_indexes[cs] = cs_indexes.size();
        reqs.push_back(cs.requirements(READ_ONLY, EXCLUSIVE));
      }
      auto idx = cs_indexes[cs];
      for (auto& [nm, tf]: nm_tfs) {
        new_cnames.insert(nm);
        assert(i <= args.new_columns.size());
        args.new_columns[i++] = {cs, idx, string(nm), tf};
      }
    }
    if (i < args.new_columns.size())
      args.new_columns[i] = {ColumnSpace(), 0, string(), TableField()};
  }

  reqs.push_back(
    table_fields_requirement(m_free_fields_lr, m_fields_lr, READ_WRITE));

  for (auto& req : reqs)
    task.add_region_requirement(req);

  auto added = rt->execute_task(ctx, task).get_result<add_columns_result_t>();

  for (auto& [nm, col] : added.cols) {
    new_cnames.erase(nm);
    m_columns[nm] = col;
  }
  return new_cnames.size() == 0;
}

std::unordered_map<std::string, Column>
Table::add_columns(
  Context ctx,
  Runtime* rt,
  std::vector<
    std::tuple<
      ColumnSpace,
      size_t,
      std::vector<std::pair<hyperion::string, TableField>>>>&& new_columns,
  const std::tuple<LogicalRegion, PhysicalRegion>& free_fields,
  const std::unordered_map<std::string, Column>& columns,
  const std::vector<PhysicalRegion>& cs_md_prs,
  const std::tuple<IndexSpace, PhysicalRegion>& index_cs) {

  if (new_columns.size() == 0)
    return {};

  // check conformance of all ColumnSpaces in new_columns
  //
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  for (auto& [cs, idx, nmtfs] : new_columns) {
#pragma GCC diagnostic pop
    if (!Table::is_conformant(
          rt,
          columns,
          index_cs,
          cs.column_is,
          cs_md_prs[idx])) {
      // FIXME: log warning: cannot add non-conforming Columns to Table
      assert(false);
      return {};
    }
  }

  // All ColumnSpaces must have unique axis vectors.
  {
    std::set<std::vector<int>> axes;
    for (auto& pr : cs_md_prs) {
      const ColumnSpace::AxisVectorAccessor<READ_ONLY>
        ax(pr, ColumnSpace::AXIS_VECTOR_FID);
      auto axv = ColumnSpace::from_axis_vector(ax[0]);
      if (axes.count(axv) > 0) {
        // FIXME: log warning: ColumnSpaces added to Table do not have unique
        // axis vectors
        assert(false);
        return {};
      }
      axes.insert(axv);
    }
  }

  // column names must be unique
  {
    std::set<std::string> new_column_names;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
    for (auto& [csp, idx, nmtfs]: new_columns) {
      for (auto& [hnm, tf]: nmtfs) {
#pragma GCC diagnostic pop
        std::string nm = hnm;
        if (columns.count(nm) > 0 || new_column_names.count(nm) > 0) {
          assert(false);
          return {};
        }
        new_column_names.insert(nm);
      }
    }
  }
  // get ColumnSpace metadata regions for current columns only
  std::vector<PhysicalRegion> current_cs_md_prs;
  for (auto& pr : cs_md_prs) {
    auto c =
      std::find_if(
        columns.begin(),
        columns.end(),
        [lr=pr.get_logical_region()](auto& nm_col) {
          return lr == std::get<1>(nm_col).cs.metadata_lr;
        });
    if (c != columns.end())
      current_cs_md_prs.push_back(pr);
  }

  // Create a map from ColumnSpaces to cgroups
  std::map<ColumnSpace, cgroup_t> cgs;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  for (auto& [nm, col] : columns) {
#pragma GCC diagnostic pop
    if (cgs.count(col.cs) == 0)
      cgs[col.cs] = col.region;
  }

  // add new columns to free_fields_pr
  std::unordered_map<std::string, Column> result;
  auto& [free_fields_lr, free_fields_pr] = free_fields;
  PointInDomainIterator<1> pid(
    rt->get_index_space_domain(free_fields_lr.get_index_space()));
  const ColumnDescAccessor<READ_WRITE> cdescs(free_fields_pr, column_desc_fid);
  const CGroupAccessor<READ_WRITE> cgroups(free_fields_pr, cgroup_fid);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  for (auto& [cs, idx, nm_tfs] : new_columns) {
#pragma GCC diagnostic pop
    if (cgs.count(cs) == 0) {
      FieldSpace fs = rt->create_field_space(ctx);
      auto lr = rt->create_logical_region(ctx, cs.column_is, fs);
      cgs[cs] = lr;
    }
    auto& region = cgs[cs];
    std::set<FieldID> fids;
    FieldSpace fs = region.get_field_space();
    rt->get_field_space_fields(fs, fids);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    for (auto& [nm, tf] : nm_tfs) {
      // add field to logical region
      assert(fids.count(tf.fid) == 0);
      switch(tf.dt) {
#define ALLOC_FLD(DT)                                           \
        case DT:                                                \
          fa.allocate_field(DataType<DT>::serdez_size, tf.fid); \
          break;
        HYPERION_FOREACH_DATATYPE(ALLOC_FLD)
#undef ALLOC_FLD
      default:
          assert(false);
        break;
      }
      // advance to the next empty field slot
      assert(pid());
      while (cgroups.read(*pid) != cgroup_none) {
        pid++;
        assert(pid());
      };
      // write column to field slot
      ColumnDesc cdesc{
        nm,
        tf.dt,
        tf.fid,
        tf.kw.num_regions()
#ifdef HYPERION_USE_CASACORE
        , tf.rc.value_or("")
        , tf.mr.num_regions()
#endif
      };
      cgroups.write(*pid, region);
      cdescs.write(*pid, cdesc);
      fids.insert(tf.fid);
      // add Column to result
      result[nm] =
        Column(
          tf.dt,
          tf.fid,
          cs,
          region,
          tf.kw
#ifdef HYPERION_USE_CASACORE
          , tf.mr
          , map(tf.rc, [](const auto& hs){ return std::string(hs); })
#endif
          );
      pid++;
    }
  }
  return result;
}

bool
Table::remove_columns(
  Context ctx,
  Runtime* rt,
  const std::unordered_set<std::string>& columns) {

  if (m_free_fields_lr == LogicalRegion::NO_REGION)
    return columns.size() == 0;

  std::vector<ColumnSpace> css;
  std::vector<PhysicalRegion> cs_md_prs;
  for (auto& [nm, col] : m_columns) {
    if (columns.count(nm) > 0) {
      if (std::find(css.begin(), css.end(), col.cs) == css.end()) {
        cs_md_prs.push_back(
          rt->map_region(ctx, col.cs.requirements(READ_ONLY, EXCLUSIVE)));
        css.push_back(col.cs);
      }
    }
  }
  std::set<hyperion::string> rm_cols;
  for (auto& c : columns)
    rm_cols.insert(c);

  std::tuple<LogicalRegion, PhysicalRegion> free_fields = {
    m_free_fields_lr,
    rt->map_region(
      ctx,
      table_fields_requirement(m_free_fields_lr, m_fields_lr, READ_WRITE))
  };

  auto result =
    Table::remove_columns(
      ctx,
      rt,
      rm_cols,
      m_fixed_fields_lr != LogicalRegion::NO_REGION,
      free_fields,
      m_columns,
      css,
      cs_md_prs);
  for (auto& pr : cs_md_prs)
    rt->unmap_region(ctx, pr);
  rt->unmap_region(ctx, std::get<1>(free_fields));
  if (result)
    for (auto& nm : columns)
      m_columns.erase(nm);
  return result;
}

bool
Table::remove_columns(
  Context ctx,
  Runtime* rt,
  const std::set<hyperion::string>& rm_columns,
  bool has_fixed_fields,
  const std::tuple<LogicalRegion, PhysicalRegion>& free_fields,
  const std::unordered_map<std::string, Column>& columns,
  const std::vector<ColumnSpace>& css,
  const std::vector<PhysicalRegion>& cs_md_prs) {

  if (rm_columns.size() == 0)
    return true;

  auto& [free_fields_lr, free_fields_pr] = free_fields;

  const CGroupAccessor<READ_WRITE> cgroups(free_fields_pr, cgroup_fid);
  const ColumnDescAccessor<READ_ONLY> cdescs(free_fields_pr, column_desc_fid);
  // check whether all columns are being removed, which is necessary if index
  // columns or the index column space are to be removed, and also that all
  // requested columns to remove are in free_fields_pr
  bool remove_all = !has_fixed_fields;
  {
    std::set<hyperion::string> fixed_columns = rm_columns;
    for (PointInDomainIterator<1> pid(
           rt->get_index_space_domain(free_fields_lr.get_index_space()));
         pid();
         pid++) {
      if (cgroups.read(*pid) != cgroup_none) {
        std::string nm = cdescs.read(*pid).name;
        fixed_columns.erase(nm);
        remove_all = remove_all && (rm_columns.count(nm) > 0);
      }
    }
    if (fixed_columns.size() > 0) {
      // FIXME: log warning: cannot remove column not added previously in same
      // scope
      return false;
    }
  }
  std::map<ColumnSpace, std::tuple<LogicalRegion, FieldAllocator>> vlr_fa;
  for (PointInDomainIterator<1> pid(
         rt->get_index_space_domain(free_fields_lr.get_index_space()));
       pid();
       pid++) {
    auto cgroup = cgroups.read(*pid);
    auto cdesc = cdescs.read(*pid);
    if (cgroup != cgroup_none
        && (remove_all || rm_columns.count(cdesc.name) > 0)) {
      auto col = columns.at(cdesc.name);
      if (!remove_all) {
        auto idx =
          std::distance(
            css.begin(),
            std::find(css.begin(), css.end(), col.cs));
        assert(idx < (ssize_t)cs_md_prs.size());
        const ColumnSpace::IndexFlagAccessor<READ_ONLY>
          ixfl(cs_md_prs[idx], ColumnSpace::INDEX_FLAG_FID);
        if (ixfl[0]) {
          // FIXME: log warning: cannot remove a table index column
          return false;
        }
      }
      if (vlr_fa.count(col.cs) == 0)
        vlr_fa[col.cs] =
          {col.region,
           rt->create_field_allocator(ctx, col.region.get_field_space())};
      std::get<1>(vlr_fa[col.cs]).free_field(col.fid);
      col.kw.destroy(ctx, rt);
#ifdef HYPERION_USE_CASACORE
      col.mr.destroy(ctx, rt);
#endif
      cgroups.write(*pid, cgroup_none);
    }
  }
  for (auto& [cs, vlr_fa] : vlr_fa) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
    auto& [vlr, fa] = vlr_fa;
#pragma GCC diagnostic pop
    std::vector<FieldID> fids;
    rt->get_field_space_fields(vlr.get_field_space(), fids);
    if (fids.size() == 0) {
      const_cast<ColumnSpace&>(cs).destroy(ctx, rt, true);
      auto fs = vlr.get_field_space();
      rt->destroy_logical_region(ctx, vlr);
      rt->destroy_field_space(ctx, fs);
    }
  }
  return true;
}

void
Table::destroy(Context ctx, Runtime* rt) {

  if (m_fields_lr != LogicalRegion::NO_REGION) {
    if (m_free_fields_lr != LogicalRegion::NO_REGION) {
      std::set<hyperion::string> free_columns;
      std::vector<ColumnSpace> css;
      std::vector<PhysicalRegion> cs_md_prs;
      for (auto& [nm, col] : m_columns) {
        free_columns.insert(nm);
        if (std::find(css.begin(), css.end(), col.cs) == css.end()) {
          css.push_back(col.cs);
          cs_md_prs.push_back(
            rt->map_region(
              ctx,
              col.cs.requirements(READ_ONLY, EXCLUSIVE)));
        }
      }
      auto free_fields_pr =
        rt->map_region(
          ctx,
          table_fields_requirement(m_free_fields_lr, m_fields_lr, READ_WRITE));
      remove_columns(
        ctx,
        rt,
        free_columns,
        m_fixed_fields_lr != LogicalRegion::NO_REGION,
        {m_free_fields_lr, free_fields_pr},
        m_columns,
        css,
        cs_md_prs);

      for (auto& pr : cs_md_prs)
        rt->unmap_region(ctx, pr);
      rt->unmap_region(ctx, free_fields_pr);
    }

    bool destroy_all = fields_partition == LogicalPartition::NO_PART;
    if (destroy_all) {
      {
        m_index_col_cs.destroy(ctx, rt, false);
        auto is = m_index_col_region.get_index_space();
        auto fs = m_index_col_region.get_field_space();
        rt->destroy_logical_region(ctx, m_index_col_region);
        rt->destroy_field_space(ctx, fs);
        rt->destroy_index_space(ctx, is);
        m_index_col_region = LogicalRegion::NO_REGION;
      }
      {
        auto is = m_fields_lr.get_index_space();
        auto fs = m_fields_lr.get_field_space();
        rt->destroy_logical_region(ctx, m_fields_lr);
        rt->destroy_field_space(ctx, fs);
        rt->destroy_index_space(ctx, is);
        m_fields_lr = LogicalRegion::NO_REGION;
      }
    } else {
      auto ip = fields_partition.get_index_partition();
      rt->destroy_logical_partition(ctx, fields_partition);
      auto cs = rt->get_index_partition_color_space_name(ctx, ip);
      rt->destroy_index_partition(ctx, ip);
      rt->destroy_index_space(ctx, cs);
    }
  }
}

struct PartitionRowsTaskArgs {
  IndexSpace ics_is;
  std::array<std::pair<bool, size_t>, Table::MAX_COLUMNS> block_sizes;
};

TaskID Table::partition_rows_task_id;

const char* Table::partition_rows_task_name = "Table::partition_rows_task";

ColumnSpacePartition
Table::partition_rows_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *rt) {

  const PartitionRowsTaskArgs* args =
    static_cast<const PartitionRowsTaskArgs*>(task->args);

  std::vector<std::optional<size_t>> block_sizes;
  for (size_t i = 0; i < MAX_COLUMNS; ++i) {
    auto& [has_value, value] = args->block_sizes[i];
    if (has_value && value == 0)
      break;
    block_sizes.push_back(has_value ? value : std::optional<size_t>());
  }

  return partition_rows(ctx, rt, block_sizes, args->ics_is, regions[0]);
}

Future /* ColumnSpacePartition */
Table::partition_rows(
  Context ctx,
  Runtime* rt,
  const std::vector<std::optional<size_t>>& block_sizes) const {

  PartitionRowsTaskArgs args;
  for (size_t i = 0; i < block_sizes.size(); ++i) {
    assert(block_sizes[i].value_or(1) > 0);
    args.block_sizes[i] =
      {block_sizes[i].has_value(), block_sizes[i].value_or(0)};
  }
  args.block_sizes[block_sizes.size()] = {true, 0};

  args.ics_is = m_index_col_cs.column_is;
  TaskLauncher task(partition_rows_task_id, TaskArgument(&args, sizeof(args)));
  task.add_region_requirement(
    m_index_col_cs.requirements(READ_ONLY, EXCLUSIVE));
  return rt->execute_task(ctx, task);
}

ColumnSpacePartition
Table::partition_rows(
  Context ctx,
  Runtime* rt,
  const std::vector<std::optional<size_t>>& block_sizes,
  const IndexSpace& ics_is,
  const PhysicalRegion& ics_md_pr) {

  ColumnSpacePartition result;
  const ColumnSpace::AxisVectorAccessor<READ_ONLY>
    ax(ics_md_pr, ColumnSpace::AXIS_VECTOR_FID);
  const ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
    au(ics_md_pr, ColumnSpace::AXIS_SET_UID_FID);
  auto ixax = ax[0];
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

  return ColumnSpacePartition::create(ctx, rt, ics_is, au[0], parts, ics_md_pr);
}

TaskID Table::reindexed_task_id;

const char* Table::reindexed_task_name = "Table::reindexed_task";

struct ReindexedTaskArgs {
  std::array<std::pair<int, hyperion::string>, LEGION_MAX_DIM> index_axes;
  bool allow_rows;
};

Table
Table::reindexed_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *rt) {

  const ReindexedTaskArgs* args =
    static_cast<const ReindexedTaskArgs*>(task->args);

  auto [ptable, rit, pit] =
    PhysicalTable::create(
      rt,
      task->regions.begin(),
      task->regions.end(),
      regions.begin(),
      regions.end())
    .value();
  assert(rit == task->regions.end());
  assert(pit == regions.end());

  std::vector<std::pair<int, std::string>> index_axes;
  auto iaxp = args->index_axes.begin();
  while (iaxp < args->index_axes.end() && std::get<0>(*iaxp) >= 0) {
    auto& [d, nm] = *iaxp;
    index_axes.emplace_back(d, nm);
    ++iaxp;
  }

  return ptable.reindexed(ctx, rt, index_axes, args->allow_rows);
}

Future /* Table */
Table::reindexed(
  Context ctx,
  Runtime *rt,
  const std::vector<std::pair<int, std::string>>& index_axes,
  bool allow_rows) const {

  ReindexedTaskArgs args;
  args.allow_rows = allow_rows;
  auto e =
    std::copy(index_axes.begin(), index_axes.end(), args.index_axes.begin());
  std::fill(e, args.index_axes.end(), std::make_pair(-1, string()));

  auto reqs = std::get<0>(requirements(ctx, rt));
  TaskLauncher task(reindexed_task_id, TaskArgument(&args, sizeof(args)));
  for (auto& r : reqs)
    task.add_region_requirement(r);
  return rt->execute_task(ctx, task);
}

void
Table::preregister_tasks() {
  {
    // is_conformant_task
    is_conformant_task_id = Runtime::generate_static_task_id();
    TaskVariantRegistrar registrar(
      is_conformant_task_id,
      is_conformant_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    registrar.set_leaf();
    Runtime::preregister_task_variant<bool, is_conformant_task>(
      registrar,
      is_conformant_task_name);
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
      ColumnSpacePartition,
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
    Runtime::preregister_task_variant<Table, reindexed_task>(
      registrar,
      reindexed_task_name);
  }
}

size_t
Table::legion_buffer_size(void) const {
  size_t result =
    sizeof(m_index_col_cs)
    + sizeof(m_index_col_parent)
    + sizeof(m_index_col_region)
    + 3 * sizeof(m_fields_lr)
    + sizeof(fields_partition)
    + sizeof(size_t);
  for (auto& [nm, col] : m_columns)
    result += nm.size() + 1 + sizeof(Column);
  return result;
}

size_t
Table::legion_serialize(void* buffer) const {
  char *b = static_cast<char*>(buffer);
  *reinterpret_cast<decltype(m_index_col_cs)*>(b) = m_index_col_cs;
  b += sizeof(m_index_col_cs);
  *reinterpret_cast<decltype(m_index_col_parent)*>(b) = m_index_col_parent;
  b += sizeof(m_index_col_parent);
  *reinterpret_cast<decltype(m_index_col_region)*>(b) = m_index_col_region;
  b += sizeof(m_index_col_region);
  *reinterpret_cast<decltype(m_fields_lr)*>(b) = m_fields_lr;
  b += sizeof(m_fields_lr);
  *reinterpret_cast<decltype(m_fixed_fields_lr)*>(b) = m_fixed_fields_lr;
  b += sizeof(m_fixed_fields_lr);
  *reinterpret_cast<decltype(m_free_fields_lr)*>(b) = m_free_fields_lr;
  b += sizeof(m_free_fields_lr);
  *reinterpret_cast<decltype(fields_partition)*>(b) = fields_partition;
  b += sizeof(fields_partition);
  *reinterpret_cast<size_t*>(b) = m_columns.size();
  b += sizeof(size_t);
  for (auto& [nm, col] : m_columns) {
    std::strcpy(b, nm.c_str());
    b += nm.size() + 1;
    *reinterpret_cast<Column*>(b) = col;
    b += sizeof(col);
  }
  return b - static_cast<char*>(buffer);
}

size_t
Table::legion_deserialize(const void* buffer) {
  const char *b = static_cast<const char*>(buffer);
  m_index_col_cs = *reinterpret_cast<const decltype(m_index_col_cs)*>(b);
  b += sizeof(m_index_col_cs);
  m_index_col_parent =
    *reinterpret_cast<const decltype(m_index_col_parent)*>(b);
  b += sizeof(m_index_col_parent);
  m_index_col_region =
    *reinterpret_cast<const decltype(m_index_col_region)*>(b);
  b += sizeof(m_index_col_region);
  m_fields_lr = *reinterpret_cast<const decltype(m_fields_lr)*>(b);
  b += sizeof(m_fields_lr);
  m_fixed_fields_lr = *reinterpret_cast<const decltype(m_fixed_fields_lr)*>(b);
  b += sizeof(m_fixed_fields_lr);
  m_free_fields_lr = *reinterpret_cast<const decltype(m_free_fields_lr)*>(b);
  b += sizeof(m_free_fields_lr);
  fields_partition = *reinterpret_cast<const decltype(fields_partition)*>(b);
  b += sizeof(fields_partition);
  size_t ncols = *reinterpret_cast<const size_t*>(b);
  b += sizeof(ncols);
  for (size_t i = 0; i < ncols; ++i) {
    std::string nm(b);
    b += nm.size() + 1;
    Column col = *reinterpret_cast<const Column*>(b);
    b += sizeof(col);
    m_columns[nm] = col;
  }
  return b - static_cast<const char*>(buffer);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
