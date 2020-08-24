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
  for (auto& nm_col : cols)
    result[i++] = nm_col;
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
#if HAVE_CXX17
    auto& [nm, col] = ary[i];
#else // !HAVE_CXX17
    auto& nm = std::get<0>(ary[i]);
    auto& col = std::get<1>(ary[i]);
#endif // HAVE_CXX17
    result[nm] = col;
  }
  return result;
}

size_t
Table::add_columns_result_t::legion_buffer_size(void) const {
  size_t result = sizeof(unsigned);
  for (size_t i = 0; i < cols.size(); ++i) {
#if HAVE_CXX17
    auto& [nm, col] = cols[i];
#else // !HAVE_CXX17
    auto& nm = std::get<0>(cols[i]);
    auto& col = std::get<1>(cols[i]);
#endif // HAVE_CXX17
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
#if HAVE_CXX17
    auto& [nm, col] = cols[i];
#else // !HAVE_CXX17
    auto& nm = std::get<0>(cols[i]);
    auto& col = std::get<1>(cols[i]);
#endif // HAVE_CXX17
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
#if HAVE_CXX17
    auto& [nm, col] = cols[i];
#else // !HAVE_CXX17
    auto& nm = std::get<0>(cols[i]);
    auto& col = std::get<1>(cols[i]);
#endif // HAVE_CXX17
    nm = std::string(b);
    b += (nm.size() + 1) * sizeof(char);
    col = *reinterpret_cast<const Column*>(b);
    b += sizeof(col);
  }
  return b - static_cast<const char*>(buffer);
}

Table::Table(
  Runtime* rt,
  ColumnSpace&& index_col_cs,
  const LogicalRegion& index_col_region,
  const std::unordered_map<std::string, Column>& columns)
  : m_index_col_cs(index_col_cs)
  , m_index_col_region(index_col_region)
  , m_index_col_parent(index_col_region)
  , m_columns(columns) {
  assert(m_index_col_cs.column_is == m_index_col_region.get_index_space());
}

Table::Table(const Table& other)
  : m_index_col_cs(other.m_index_col_cs)
  , m_index_col_region(other.m_index_col_region)
  , m_index_col_parent(other.m_index_col_parent)
  , m_columns(other.m_columns) {}

Table::Table(Table&& other)
  : m_index_col_cs(other.m_index_col_cs)
  , m_index_col_region(other.m_index_col_region)
  , m_index_col_parent(other.m_index_col_parent)
  , m_columns(std::move(other).m_columns) {}

Table&
Table::operator=(const Table& rhs) {
  Table tmp(rhs);
  m_index_col_cs = tmp.m_index_col_cs;
  m_index_col_region = tmp.m_index_col_region;
  m_index_col_parent = tmp.m_index_col_parent;
  m_columns = tmp.m_columns;
  return *this;
}

Table&
Table::operator=(Table&& rhs) {
  m_index_col_cs = std::move(rhs).m_index_col_cs;
  m_index_col_region = std::move(rhs).m_index_col_region;
  m_index_col_parent = std::move(rhs).m_index_col_parent;
  m_columns = std::move(rhs).m_columns;
  return *this;
}

PhysicalTable
Table::attach_columns(
  Context ctx,
  Runtime* rt,
  const CXX_FILESYSTEM_NAMESPACE::path& file_path,
  const std::unordered_map<std::string, std::string>& column_paths,
  const std::unordered_map<std::string, std::tuple<bool, bool, bool>>&
    column_modes) const {

  std::unordered_set<std::string> colnames;
  for (auto& nm_pth : column_paths) {
    auto& nm = std::get<0>(nm_pth);
    if (column_modes.count(nm) > 0)
      colnames.insert(nm);
  }

  std::map<std::string, CXX_OPTIONAL_NAMESPACE::optional<Column::Requirements>>
    omitted;
  for (auto& nm_col : m_columns) {
    auto& nm = std::get<0>(nm_col);
    if (colnames.count(nm) == 0)
      omitted[nm] = CXX_OPTIONAL_NAMESPACE::nullopt;
  }
#if HAVE_CXX17
  auto [table_reqs, table_parts, table_desc] =
    requirements(ctx, rt, ColumnSpacePartition(), omitted);
#else // !HAVE_CXX17
  auto reqs = requirements(ctx, rt, ColumnSpacePartition(), omitted);
  auto& table_reqs = std::get<0>(reqs);
  auto& table_parts = std::get<1>(reqs);
#endif // HAVE_CXX17
  PhysicalRegion index_col_md = rt->map_region(ctx, table_reqs[0]);
  std::string axes_uid = ColumnSpace::axes_uid(index_col_md);
  std::vector<int> index_axes =
    ColumnSpace::from_axis_vector(ColumnSpace::axes(index_col_md));
  unsigned idx_rank = static_cast<unsigned>(index_axes.size());

  std::tuple<LogicalRegion, PhysicalRegion> index_col =
    {table_reqs[1].region, rt->map_region(ctx, table_reqs[1])};

  std::unordered_map<std::string, std::shared_ptr<PhysicalColumn>> pcols;
  for (auto& nm_col : m_columns) {
#if HAVE_CXX17
    auto& [nm, col] = nm_col;
#else // !HAVE_CXX17
    auto& nm = std::get<0>(nm_col);
    auto& col = std::get<1>(nm_col);
#endif // HAVE_CXX17
    CXX_OPTIONAL_NAMESPACE::optional<PhysicalRegion> metadata;
    if (colnames.count(nm) > 0) {
      if (!metadata) {
        auto req = col.cs.requirements(READ_ONLY, EXCLUSIVE);
        metadata = rt->map_region(ctx, req);
      }
      CXX_OPTIONAL_NAMESPACE::optional<Keywords::pair<Legion::PhysicalRegion>>
        kws;
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
      CXX_OPTIONAL_NAMESPACE::optional<MeasRef::DataRegions> mr_drs;
      if (!col.mr.is_empty()) {
        auto rqs = col.mr.requirements(READ_ONLY, true);
        MeasRef::DataRegions prs;
        prs.metadata = rt->map_region(ctx, std::get<0>(rqs));
        prs.values = rt->map_region(ctx, std::get<1>(rqs));
        if (std::get<2>(rqs))
          prs.index = rt->map_region(ctx, std::get<2>(rqs).value());
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
          CXX_OPTIONAL_NAMESPACE::nullopt,
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
  for (auto& nm_pc : pcols) {
    auto& pc = std::get<1>(nm_pc);
    if (pc->refcol()) {
      auto& rcnm = std::get<0>(pc->refcol().value());
      pc->set_refcol(rcnm, pcols.at(rcnm));
    }
  }
#endif

  PhysicalTable result(
    axes_uid,
    index_axes,
    index_col_md,
    index_col,
    table_reqs[1].parent,
    pcols);
  result.attach_columns(ctx, rt, file_path, column_paths, column_modes);
  for (auto& p : table_parts)
    p.destroy(ctx, rt);
  return result;
}

PhysicalTable
Table::map_inline(
  Context ctx,
  Runtime* rt,
  const std::map<
    std::string,
    CXX_OPTIONAL_NAMESPACE::optional<Column::Requirements>>&
    column_requirements,
  const CXX_OPTIONAL_NAMESPACE::optional<Column::Requirements>&
    default_column_requirements) const {

  auto reqs =
    requirements(
      ctx,
      rt,
      ColumnSpacePartition(),
      column_requirements,
      default_column_requirements);
#if HAVE_CXX17
  auto& [treqs, tparts, tdesc] = reqs;
#else
  auto& treqs = std::get<0>(reqs);
  auto& tdesc = std::get<2>(reqs);
#endif
  std::vector<PhysicalRegion> tprs;
  for (auto& tr : treqs)
    tprs.push_back(rt->map_region(ctx, tr));

  return
    std::get<0>(
      PhysicalTable::create(
        rt,
        tdesc,
        treqs.begin(),
        treqs.end(),
        tprs.begin(),
        tprs.end())
      .value());
}

Table
Table::create(
  Context ctx,
  Runtime* rt,
  ColumnSpace&& index_col_cs,
  fields_t&& fields) {

  size_t num_cols = 0;
  for (auto& cs_tfs : fields) {
#if HAVE_CXX17
    auto& [cs, tfs] = cs_tfs;
#else // !HAVE_CXX17
    auto& cs = std::get<0>(cs_tfs);
    auto& tfs = std::get<1>(cs_tfs);
#endif // HAVE_CXX17
    assert(!cs.is_empty());
    assert(cs.is_valid());
    num_cols += tfs.size();
  }
  {
    std::unordered_set<std::string> cnames;
    for (auto& cs_nm_tfs : fields)
      for (auto& nm_tf : std::get<1>(cs_nm_tfs))
        cnames.insert(std::get<0>(nm_tf));
    assert(cnames.count("") == 0);
    assert(cnames.size() == num_cols);
  }

  std::vector<PhysicalRegion> cs_md_prs;
  for (auto& cs_tfs : fields) {
    cs_md_prs.push_back(
      rt->map_region(
        ctx,
        std::get<0>(cs_tfs).requirements(READ_ONLY, EXCLUSIVE)));
  }

  // Create the table index column
  LogicalRegion index_col_region;
  {
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    fa.allocate_field(
      sizeof(DataType<Table::m_index_col_dt>::ValueType),
      Table::m_index_col_fid);
    rt->attach_name(fs, Table::m_index_col_fid, "Table::index_column_flag");
    index_col_region =
      rt->create_logical_region(ctx, index_col_cs.column_is, fs);
  }

  std::unordered_map<std::string, Column> added;
  {
    std::vector<
      std::tuple<
        ColumnSpace,
        size_t,
        std::vector<std::pair<hyperion::string, TableField>>>>
      hcols;
    for (size_t i = 0; i < fields.size(); ++i) {
#if HAVE_CXX17
      auto& [cs, nm_tfs] = fields[i];
#else // !HAVE_CXX17
      auto& cs = std::get<0>(fields[i]);
      auto& nm_tfs = std::get<1>(fields[i]);
#endif // HAVE_CXX17
      std::vector<std::pair<hyperion::string, TableField>> htfs;
      for (auto& nm_tf: nm_tfs)
        htfs.emplace_back(std::get<0>(nm_tf), std::get<1>(nm_tf));
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
        {{}},
        cs_md_prs,
        {index_col_cs.column_is, index_col_md});
    for (auto& pr : cs_md_prs)
      rt->unmap_region(ctx, pr);
  }
  return Table(rt, std::move(index_col_cs), index_col_region, added);
}

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

std::tuple<
  std::vector<RegionRequirement>,
  std::vector<ColumnSpacePartition>,
  Table::Desc>
Table::requirements(
  Context ctx,
  Runtime* rt,
  const ColumnSpacePartition& table_partition,
  const std::map<
    std::string,
    CXX_OPTIONAL_NAMESPACE::optional<Column::Requirements>>&
    column_requirements,
  const CXX_OPTIONAL_NAMESPACE::optional<Column::Requirements>&
    default_column_requirements) const {

  return
    Table::requirements(
      ctx,
      rt,
      m_index_col_cs,
      m_index_col_region,
      m_index_col_parent,
      m_columns,
      table_partition,
      column_requirements,
      default_column_requirements);
}

std::tuple<std::vector<Legion::RegionRequirement>, Table::Desc>
Table::requirements() const {

  auto reqs =
    Table::requirements(
      CXX_OPTIONAL_NAMESPACE::nullopt,
      CXX_OPTIONAL_NAMESPACE::nullopt,
      m_index_col_cs,
      m_index_col_region,
      m_index_col_parent,
      m_columns,
      ColumnSpacePartition(),
      {},
      Column::default_requirements);
#if HAVE_CXX17
  auto& [treqs, tparts, tdesc] = reqs;
#else // !HAVE_CXX17
  auto& treqs = std::get<0>(reqs);
  auto& tdesc = std::get<2>(reqs);
#endif // HAVE_CXX17
  return {treqs, tdesc};
}

std::tuple<
  std::vector<RegionRequirement>,
  std::vector<ColumnSpacePartition>,
  Table::Desc>
Table::requirements(
  CXX_OPTIONAL_NAMESPACE::optional<Context> ctx,
  CXX_OPTIONAL_NAMESPACE::optional<Runtime*> rt,
  const ColumnSpace& index_col_cs,
  const LogicalRegion& index_col_region,
  const LogicalRegion& index_col_parent,
  const std::unordered_map<std::string, Column>& columns,
  const ColumnSpacePartition& table_partition,
  const std::map<
    std::string,
    CXX_OPTIONAL_NAMESPACE::optional<Column::Requirements>>&
    column_requirements,
  const CXX_OPTIONAL_NAMESPACE::optional<Column::Requirements>&
    default_column_requirements) {

  assert(!table_partition.is_valid() || (ctx && rt));

  // collect requirement parameters for each column
  std::map<std::string, Column::Requirements> column_reqs;
  {
#ifdef HYPERION_USE_CASACORE
    std::map<std::string, Column::Requirements> mrc_reqs;
#endif
    std::map<LogicalRegion, Column::Req> lr_mdreqs;
    for (auto& nm_col : columns) {
#if HAVE_CXX17
      auto& [nm, col] = nm_col;
#else // !HAVE_CXX17
      auto& nm = std::get<0>(nm_col);
      auto& col = std::get<1>(nm_col);
#endif // HAVE_CXX17
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
        if (lr_mdreqs.count(col.region) == 0) {
          lr_mdreqs[col.region] = colreqs.column_space;
        } else {
          // FIXME: log a warning, and return empty result;
          // warning: inconsistent requirements on shared Column metadata
          // regions
          assert(lr_mdreqs[col.region] == colreqs.column_space);
        }
#ifdef HYPERION_USE_CASACORE
        if (col.rc)
          mrc_reqs[col.rc.value()] = colreqs;
#endif
      }
    }

#ifdef HYPERION_USE_CASACORE
    // apply mode of value column to its measure reference column
#if HAVE_CXX17
    for (auto& [nm, rq] : mrc_reqs)
      column_reqs.at(nm) = rq;
#else // !HAVE_CXX17
    for (auto& nm_rq : mrc_reqs)
      column_reqs.at(std::get<0>(nm_rq)) = std::get<1>(nm_rq);
#endif // HAVE_CXX17
#endif
  }
  // create requirements, applying table_partition as needed
  std::multimap<
    ColumnSpace,
    std::tuple<ColumnSpacePartition, ProjectionID, LogicalPartition, bool>>
    partitions;
  if (table_partition.is_valid()) {
    if (table_partition.column_space.column_is
        != index_col_region.get_index_space()) {
      auto csp =
        table_partition.project_onto(ctx.value(), rt.value(), index_col_cs)
        .get_result<ColumnSpacePartition>();
      auto lp =
        rt.value()
        ->get_logical_partition(ctx.value(), index_col_region, csp.column_ip);
      partitions.insert({index_col_cs, {csp, 0, lp, true}});
    } else {
      auto lp =
        rt.value()
        ->get_logical_partition(
          ctx.value(),
          index_col_region,
          table_partition.column_ip);
      // set boolean flag to false to indicate that this partition should not be
      // returned in list of ColumnSpacePartitions
      partitions.insert({index_col_cs, {table_partition, 0, lp, false}});
    }
  }

  // boolean elements in value of following maps is used to track whether the
  // requirement has already been added when iterating through columns
  std::map<LogicalRegion, std::tuple<bool, RegionRequirement>> md_reqs;
  std::map<
    std::tuple<
      LogicalRegion,
      PrivilegeMode,
      CoherenceProperty,
      MappingTagID,
      LogicalPartition>,
    std::tuple<bool, RegionRequirement>> val_reqs;
  for (auto& nm_col : columns) {
#if HAVE_CXX17
    auto& [nm, col] = nm_col;
#else // !HAVE_CXX17
    auto& nm = std::get<0>(nm_col);
    auto& col = std::get<1>(nm_col);
#endif // HAVE_CXX17
    if (column_reqs.count(nm) > 0) {
      auto& reqs = column_reqs.at(nm);
      if (md_reqs.count(col.region) == 0)
        md_reqs[col.region] =
          {false,
           col.cs.requirements(
             reqs.column_space.privilege,
             reqs.column_space.coherence)};
      decltype(val_reqs)::key_type rg_rq =
        {col.region,
         reqs.values.privilege,
         reqs.values.coherence,
         reqs.tag,
         reqs.partition};
      if (val_reqs.count(rg_rq) == 0) {
        if (!table_partition.is_valid()
            && reqs.partition == LogicalPartition::NO_PART) {
          // no column partition case
          val_reqs[rg_rq] =
            {false,
             RegionRequirement(
               col.region,
               reqs.values.privilege,
               reqs.values.coherence,
               col.region,
               reqs.tag)};
        } else {
          // Need a column partition; either it's provided or it's derived from
          // table_partition. First look for existing record of this partition
          LogicalPartition lp = LogicalPartition::NO_PART;
          ProjectionID pjid;
          for (auto p = partitions.lower_bound(col.cs);
               (lp == LogicalPartition::NO_PART
                && p != partitions.upper_bound(col.cs));
               ++p) {
#if HAVE_CXX17
            auto& [pcsp, pid, plp, pnew] = std::get<1>(*p);
#else
            auto& pid = std::get<1>(std::get<1>(*p));
            auto& plp = std::get<2>(std::get<1>(*p));
#endif
            if (reqs.partition != LogicalPartition::NO_PART) {
              // when a partition is provided, we use it
              if (plp == reqs.partition && pid == reqs.projection) {
                lp = plp;
                pjid = pid;
              }
            } else if (pid == reqs.projection) {
              // when a partition is not provided, use the induced table
              // partition
              assert(table_partition.is_valid());
              lp = plp;
              pjid = pid;
            }
          }
          if (lp == LogicalPartition::NO_PART) {
            // no record of this partition exists, create one (and if needed,
            // create the partition as well)
            ColumnSpacePartition csp;
            bool new_csp;
            if (reqs.partition != LogicalPartition::NO_PART) {
              lp = reqs.partition;
              csp = ColumnSpacePartition();
              new_csp = false;
              pjid = reqs.projection;
            } else {
              assert(table_partition.is_valid());
              // use an induced (projected) table partition
              csp =
                table_partition.project_onto(ctx.value(), rt.value(), col.cs)
                .get_result<ColumnSpacePartition>();
              lp =
                rt.value()
                ->get_logical_partition(ctx.value(), col.region, csp.column_ip);
              new_csp = true;
              pjid = reqs.projection;
            }
            // record this partition
            partitions.insert({col.cs, {csp, pjid, lp, new_csp}});
          }
          // record the requirement for this column
          val_reqs[rg_rq] =
            {false,
             RegionRequirement(
               lp,
               pjid,
               reqs.values.privilege,
               reqs.values.coherence,
               col.region,
               reqs.tag)};
        }
      }
      std::get<1>(val_reqs[rg_rq]).add_field(col.fid, reqs.values.mapped);
    }
  }
  std::vector<ColumnSpacePartition> csps_result;
  for (auto& cs_part : partitions) {
#if HAVE_CXX17
    auto& [csp, pjid, lp, isnew] = std::get<1>(cs_part);
#else // !HAVE_CXX17
    auto& part = std::get<1>(cs_part);
    auto& csp = std::get<0>(part);
    auto& isnew = std::get<3>(part);
#endif
    if (isnew)
      csps_result.push_back(csp);
  }

  // gather all requirements, in order set by this traversal of fields
  std::vector<RegionRequirement> reqs_result;

  // start with index_col ColumnSpace metadata
  reqs_result.push_back(index_col_cs.requirements(READ_ONLY, EXCLUSIVE));
  // next, index_col index space partition
  if (table_partition.is_valid()) {
    RegionRequirement req(
      std::get<2>(partitions.find(index_col_cs)->second),
      0,
      {Table::m_index_col_fid},
      {}, // always remains unmapped!
      WRITE_ONLY,
      SIMULTANEOUS,
      index_col_region);
    reqs_result.push_back(req);
  } else {
    RegionRequirement req(
      index_col_region,
      {Table::m_index_col_fid},
      {}, // always remains unmapped!
      WRITE_ONLY,
      SIMULTANEOUS,
      index_col_region);
    reqs_result.push_back(req);
  }

  Desc desc_result;
  if (ctx && rt) {
    auto pr =
      rt.value()->map_region(
        ctx.value(),
        index_col_cs.requirements(LEGION_READ_ONLY, EXCLUSIVE));
    desc_result.axes_uid = ColumnSpace::axes_uid(pr);
    desc_result.index_axes = ColumnSpace::axes(pr);
    rt.value()->unmap_region(ctx.value(), pr);
  }
  desc_result.num_columns = column_reqs.size();
  assert(desc_result.num_columns <= desc_result.columns.size());

  // add requirements for all logical regions in all selected columns
  size_t desc_idx = 0;
  for (auto& nm_col : columns) {
#if HAVE_CXX17
    auto& [nm, col] = nm_col;
#else // !HAVE_CXX17
    auto& nm = std::get<0>(nm_col);
    auto& col = std::get<1>(nm_col);
#endif // HAVE_CXX17
    if (column_reqs.count(nm) > 0) {
      auto cdesc = col.desc(nm);
      auto& reqs = column_reqs.at(nm);
      {
        auto& added_rq = md_reqs.at(col.region);
#if HAVE_CXX17
        auto& [added, rq] = added_rq;
#else // !HAVE_CXX17
        auto& added = std::get<0>(added_rq);
        auto& rq = std::get<1>(added_rq);
#endif // HAVE_CXX17
        if (!added) {
          reqs_result.push_back(rq);
          added = true;
        }
      }
      decltype(val_reqs)::key_type rg_rq =
        {col.region,
         reqs.values.privilege,
         reqs.values.coherence,
         reqs.tag,
         reqs.partition};
      auto& added_rq = val_reqs.at(rg_rq);
#if HAVE_CXX17
      auto& [added, rq] = added_rq;
#else // !HAVE_CXX17
      auto& added = std::get<0>(added_rq);
      auto& rq = std::get<1>(added_rq);
#endif // HAVE_CXX17
      cdesc.region = rq.parent;
      if (!added) {
        reqs_result.push_back(rq);
        added = true;
      }
      if (cdesc.n_kw > 0) {
        auto& kw = col.kw;
        assert(cdesc.n_kw == 2);
        if (rt) {
          auto nkw = kw.size(rt.value());
          std::vector<FieldID> fids(nkw);
          std::iota(fids.begin(), fids.end(), 0);
          auto rqs =
            kw.requirements(
              rt.value(),
              fids,
              reqs.keywords.privilege,
              reqs.keywords.mapped)
            .value();
          reqs_result.push_back(rqs.type_tags);
          reqs_result.push_back(rqs.values);
        } else {
          // This is a corner case, which should only be reached when having
          // originally called Table::requirements() without arguments, which
          // should only be called in Table serialization. In that case, the
          // requirement is only used to identify a LogicalRegion, so the FieldIDs
          // are basically insignificant -- TODO: do something a bit more
          // explicit, which should probably wait until a Keyword dictionary is
          // simply a value in a region
          LogicalRegion ttlr = kw.type_tags_lr;
          LogicalRegion vlr = kw.values_lr;
          RegionRequirement tt(ttlr, reqs.keywords.privilege, EXCLUSIVE, ttlr);
          tt.add_field(0, reqs.keywords.mapped);
          reqs_result.push_back(tt);
          RegionRequirement v(vlr, reqs.keywords.privilege, EXCLUSIVE, vlr);
          v.add_field(0, reqs.keywords.mapped);
          reqs_result.push_back(v);
        }
      }

#ifdef HYPERION_USE_CASACORE
      if (cdesc.n_mr > 0) {
        auto& mr = col.mr;
        auto rqs =
          mr.requirements(reqs.measref.privilege, reqs.measref.mapped);
        assert(cdesc.n_mr == 2 || cdesc.n_mr == 3);
        reqs_result.push_back(std::get<0>(rqs));
        reqs_result.push_back(std::get<1>(rqs));
        if (std::get<2>(rqs)) {
          assert(cdesc.n_mr == 3);
          reqs_result.push_back(std::get<2>(rqs).value());
        }
      }
#endif
      desc_result.columns[desc_idx++] = cdesc;
    }
  }
  return {reqs_result, csps_result, desc_result};
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
  for (auto& nm_col : columns) {
    if (cs == std::get<1>(nm_col).cs)
      return true;
  }

#if HAVE_CXX17
  auto& [index_cs_is, index_cs_md_pr] = index_cs;
#else // !HAVE_CXX17
  auto& index_cs_is = std::get<0>(index_cs);
  auto& index_cs_md_pr = std::get<1>(index_cs);
#endif // HAVE_CXX17
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
    auto& cs_idx_nm_tf = args->new_columns[i];
#if HAVE_CXX17
    auto& [cs, idx, nm, tf] = cs_idx_nm_tf;
#else // !HAVE_CXX17
    auto& cs = std::get<0>(cs_idx_nm_tf);
    auto& idx = std::get<1>(cs_idx_nm_tf);
    auto& nm = std::get<2>(cs_idx_nm_tf);
    auto& tf = std::get<3>(cs_idx_nm_tf);
#endif // HAVE_CXX17
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

  std::vector<PhysicalRegion> cs_md_prs(regions.begin() + 1, regions.end());
  auto added =
    add_columns(
      ctx,
      rt,
      std::move(new_columns),
      columns,
      cs_md_prs,
      index_cs);
  add_columns_result_t result;
  for (auto& nm_col : added) {
#if HAVE_CXX17
    auto& [nm, col] = nm_col;
#else //!HAVE_CXX17
    auto& nm = std::get<0>(nm_col);
    auto& col = std::get<1>(nm_col);
#endif
    result.cols.emplace_back(nm, col);
  }
  return result;
}

bool
Table::add_columns(Context ctx, Runtime* rt, fields_t&& new_columns)  {

  if (new_columns.size() == 0)
    return true;

  AddColumnsTaskArgs args;
  TaskLauncher task(add_columns_task_id, TaskArgument(&args, sizeof(args)));
  args.columns = to_columns_array<args.columns.size()>(m_columns);

  std::vector<RegionRequirement> reqs;
  args.index_cs_is = m_index_col_cs.column_is;
  reqs.push_back(m_index_col_cs.requirements(READ_ONLY, EXCLUSIVE));

  std::map<ColumnSpace, size_t> cs_indexes;
  for (auto& nm_col : m_columns) {
    auto& col = std::get<1>(nm_col);
    if (cs_indexes.count(col.cs) == 0) {
      size_t i = cs_indexes.size();
      cs_indexes[col.cs] = i;
      reqs.push_back(col.cs.requirements(READ_ONLY, EXCLUSIVE));
    }
  }
  std::set<std::string> new_cnames;
  {
    size_t i = 0;
    for (auto& cs_nm_tfs: new_columns) {
#if HAVE_CXX17
      auto& [cs, nm_tfs] = cs_nm_tfs;
#else // !HAVE_CXX17
      auto& cs = std::get<0>(cs_nm_tfs);
      auto& nm_tfs = std::get<1>(cs_nm_tfs);
#endif // HAVE_CXX17
      if (cs_indexes.count(cs) == 0) {
        size_t i = cs_indexes.size();
        cs_indexes[cs] = i;
        reqs.push_back(cs.requirements(READ_ONLY, EXCLUSIVE));
      }
      size_t idx = cs_indexes[cs];
      for (auto& nm_tf: nm_tfs) {
#if HAVE_CXX17
        auto& [nm, tf] = nm_tf;
#else // !HAVE_CXX17
        auto& nm = std::get<0>(nm_tf);
        auto& tf = std::get<1>(nm_tf);
#endif // HAVE_CXX17
        new_cnames.insert(nm);
        assert(i <= args.new_columns.size());
        args.new_columns[i++] = {cs, idx, string(nm), tf};
      }
    }
    if (i < args.new_columns.size())
      args.new_columns[i] = {ColumnSpace(), 0, string(), TableField()};
  }

  for (auto& req : reqs)
    task.add_region_requirement(req);

  auto added = rt->execute_task(ctx, task).get_result<add_columns_result_t>();

  for (auto& nm_col : added.cols) {
#if HAVE_CXX17
    auto& [nm, col] = nm_col;
#else // !HAVE_CXX17
    auto& nm = std::get<0>(nm_col);
    auto& col = std::get<1>(nm_col);
#endif // HAVE_CXX17
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
  const std::unordered_map<std::string, Column>& columns,
  const std::vector<PhysicalRegion>& cs_md_prs,
  const std::tuple<IndexSpace, PhysicalRegion>& index_cs) {

  if (new_columns.size() == 0)
    return {};

  // check conformance of all ColumnSpaces in new_columns
  //
  for (auto& cs_idx_nmtfs : new_columns) {
    auto& cs = std::get<0>(cs_idx_nmtfs);
    auto& idx = std::get<1>(cs_idx_nmtfs);
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
    for (auto& cs_idx_nmtfs: new_columns) {
      for (auto& hnm_tf: std::get<2>(cs_idx_nmtfs)) {
        std::string nm = std::get<0>(hnm_tf);
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

  // Create a map from ColumnSpaces to LogicalRegions
  std::map<ColumnSpace, LogicalRegion> lrs;
  for (auto& nm_col : columns) {
#if HAVE_CXX17
    auto& [nm, col] = nm_col;
#else // !HAVE_CXX17
    auto& col = std::get<1>(nm_col);
#endif // HAVE_CXX17
    if (lrs.count(col.cs) == 0)
      lrs[col.cs] = col.region;
  }

  // add new columns to free_fields_pr
  std::unordered_map<std::string, Column> result;

  for (auto& cs_idx_nmtfs : new_columns) {
#if HAVE_CXX17
    auto& [cs, idx, nm_tfs] = cs_idx_nmtfs;
#else // !HAVE_CXX17
    auto& cs = std::get<0>(cs_idx_nmtfs);
    auto& nm_tfs = std::get<2>(cs_idx_nmtfs);
#endif // HAVE_CXX17
    if (lrs.count(cs) == 0) {
      FieldSpace fs = rt->create_field_space(ctx);
      auto lr = rt->create_logical_region(ctx, cs.column_is, fs);
      lrs[cs] = lr;
    }
    auto& region = lrs[cs];
    std::set<FieldID> fids;
    FieldSpace fs = region.get_field_space();
    rt->get_field_space_fields(fs, fids);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    for (auto& nm_tf : nm_tfs) {
      auto& nm = std::get<0>(nm_tf);
      auto& tf = std::get<1>(nm_tf);
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
      rt->attach_name(fs, tf.fid, nm.val);
      fids.insert(tf.fid);
      // add Column to result
      result[nm] =
        Column(
          tf.dt,
          tf.fid,
          cs,
          region,
          region,
          tf.kw
#ifdef HYPERION_USE_CASACORE
          , tf.mr
          , tf.rc
#endif
          );
    }
  }
  return result;
}

bool
Table::remove_columns(
  Context ctx,
  Runtime* rt,
  const std::unordered_set<std::string>& columns) {

  std::vector<ColumnSpace> css;
  std::vector<PhysicalRegion> cs_md_prs;
  for (auto& nm_col : m_columns) {
#if HAVE_CXX17
    auto& [nm, col] = nm_col;
#else // !HAVE_CXX17
    auto& nm = std::get<0>(nm_col);
    auto& col = std::get<1>(nm_col);
#endif // HAVE_CXX17
    if (columns.count(nm) > 0) {
      if (std::find(css.begin(), css.end(), col.cs) == css.end()) {
        cs_md_prs.push_back(
          rt->map_region(ctx, col.cs.requirements(READ_ONLY, EXCLUSIVE)));
        css.push_back(col.cs);
      }
    }
  }

  auto result =
    Table::remove_columns(ctx, rt, columns, m_columns, css, cs_md_prs);
  for (auto& pr : cs_md_prs)
    rt->unmap_region(ctx, pr);
  if (result) {
    std::map<LogicalRegion, ColumnSpace> lrcss;
    for (auto& nm : columns) {
      auto& col = m_columns.at(nm);
      if (lrcss.count(col.region) == 0)
        lrcss[col.region] = col.cs;
      col.kw.destroy(ctx, rt);
#ifdef HYPERION_USE_CASACORE
      col.mr.destroy(ctx, rt);
#endif
      m_columns.erase(nm);
    }
    for (auto& lr_cs : lrcss) {
      auto& lr = std::get<0>(lr_cs);
      auto& cs = std::get<1>(lr_cs);
      std::vector<FieldID> fids;
      rt->get_field_space_fields(lr.get_field_space(), fids);
      if (fids.size() == 0) {
        cs.destroy(ctx, rt, true);
        auto fs = lr.get_field_space();
        rt->destroy_logical_region(ctx, lr);
        rt->destroy_field_space(ctx, fs);
      }
    }
  }
  return result;
}

bool
Table::remove_columns(
  Context ctx,
  Runtime* rt,
  const std::unordered_set<std::string>& rm_columns,
  const std::unordered_map<std::string, Column>& columns,
  const std::vector<ColumnSpace>& css,
  const std::vector<PhysicalRegion>& cs_md_prs) {

  if (rm_columns.size() == 0)
    return true;

  // check whether all columns are being removed, which is necessary if index
  // columns are to be removed
  bool remove_all;
  {
    std::unordered_set<std::string> all_columns = rm_columns;
    for (auto& c : rm_columns)
      all_columns.erase(c);
    remove_all = all_columns.size() == 0;
  }
  std::map<ColumnSpace, FieldAllocator> fas;
  for (auto& nm : rm_columns) {
    auto& col = columns.at(nm);
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
    if (fas.count(col.cs) == 0)
      fas[col.cs] =
        rt->create_field_allocator(ctx, col.region.get_field_space());
    fas[col.cs].free_field(col.fid);
  }
  return true;
}

void
Table::destroy(Context ctx, Runtime* rt) {

  std::unordered_set<std::string> colnames;
  for (auto& nm_col : m_columns)
    colnames.insert(std::get<0>(nm_col));
  remove_columns(ctx, rt, colnames);

  m_index_col_cs.destroy(ctx, rt, false);
  auto is = m_index_col_region.get_index_space();
  auto fs = m_index_col_region.get_field_space();
  rt->destroy_logical_region(ctx, m_index_col_region);
  rt->destroy_field_space(ctx, fs);
  rt->destroy_index_space(ctx, is);
  m_index_col_region = LogicalRegion::NO_REGION;
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

  std::vector<CXX_OPTIONAL_NAMESPACE::optional<size_t>> block_sizes;
  for (size_t i = 0; i < MAX_COLUMNS; ++i) {
#if HAVE_CXX17
    auto& [has_value, value] = args->block_sizes[i];
#else // !HAVE_CXX17
    auto& has_value = std::get<0>(args->block_sizes[i]);
    auto& value = std::get<1>(args->block_sizes[i]);
#endif // HAVE_CXX17
    if (has_value && value == 0)
      break;
    block_sizes.push_back(
      has_value ? value : CXX_OPTIONAL_NAMESPACE::optional<size_t>());
  }

  return partition_rows(ctx, rt, block_sizes, args->ics_is, regions[0]);
}

Future /* ColumnSpacePartition */
Table::partition_rows(
  Context ctx,
  Runtime* rt,
  const std::vector<CXX_OPTIONAL_NAMESPACE::optional<size_t>>& block_sizes)
  const {

  PartitionRowsTaskArgs args;
  for (size_t i = 0; i < block_sizes.size(); ++i) {
    assert(block_sizes[i].value_or(1) > 0);
    args.block_sizes[i] =
      {(bool)block_sizes[i], block_sizes[i].value_or(0)};
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
  const std::vector<CXX_OPTIONAL_NAMESPACE::optional<size_t>>& block_sizes,
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
  std::vector<CXX_OPTIONAL_NAMESPACE::optional<size_t>> blkszs(ixax_sz);
  {
    auto e = std::copy(block_sizes.begin(), block_sizes.end(), blkszs.begin());
    std::fill(e, blkszs.end(), CXX_OPTIONAL_NAMESPACE::nullopt);
  }

  std::vector<std::pair<int, coord_t>> parts;
  for (size_t i = 0; i < ixax_sz; ++i)
    if (blkszs[i])
      parts.emplace_back(ixax[i], blkszs[i].value());

  return ColumnSpacePartition::create(ctx, rt, ics_is, au[0], parts, ics_md_pr);
}

TaskID Table::reindexed_task_id;

const char* Table::reindexed_task_name = "Table::reindexed_task";

struct ReindexedTaskArgs {
  Table::Desc desc;
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

  auto ptcr =
    PhysicalTable::create(
      rt,
      args->desc,
      task->regions.begin(),
      task->regions.end(),
      regions.begin(),
      regions.end())
    .value();
#if HAVE_CXX17
  auto& [ptable, rit, pit] = ptcr;
#else // !HAVE_CXX17
  auto& ptable = std::get<0>(ptcr);
  auto& rit = std::get<1>(ptcr);
  auto& pit = std::get<2>(ptcr);
#endif // HAVE_CXX17
  assert(rit == task->regions.end());
  assert(pit == regions.end());

  std::vector<std::pair<int, std::string>> index_axes;
  auto iaxp = args->index_axes.begin();
  while (iaxp < args->index_axes.end() && std::get<0>(*iaxp) >= 0) {
    index_axes.emplace_back(std::get<0>(*iaxp), std::get<1>(*iaxp));
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

  auto reqs = requirements(ctx, rt);
#if HAVE_CXX17
  auto& [treqs, tparts, tdesc] = reqs;
#else // !HAVE_CXX17
  auto& treqs = std::get<0>(reqs);
  auto& tdesc = std::get<2>(reqs);
#endif // HAVE_CXX17
  args.desc = tdesc;
  TaskLauncher task(reindexed_task_id, TaskArgument(&args, sizeof(args)));
  for (auto& r : treqs)
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
#if HAVE_CXX17
  auto [treqs, tdesc] = requirements();
#else // !HAVE_CXX17
  auto reqs = requirements();
  auto& treqs = std::get<0>(reqs);
  auto& tdesc = std::get<1>(reqs);
#endif // HAVE_CXX17
  return
    sizeof(unsigned) // number of columns
    + tdesc.num_columns * sizeof(Column::Desc)
    + sizeof(unsigned) // number of LogicalRegions
    + treqs.size() * sizeof(LogicalRegion);
}

size_t
Table::legion_serialize(void* buffer) const {
#if HAVE_CXX17
  auto [treqs, tdesc] = requirements();
#else // !HAVE_CXX17
  auto reqs = requirements();
  auto& treqs = std::get<0>(reqs);
  auto& tdesc = std::get<1>(reqs);
#endif // HAVE_CXX17

  char *b = static_cast<char*>(buffer);

  *reinterpret_cast<decltype(tdesc.num_columns)*>(b) = tdesc.num_columns;
  b += sizeof(tdesc.num_columns);
  for (size_t i = 0; i < tdesc.num_columns; ++i) {
    *reinterpret_cast<decltype(tdesc.columns)::value_type*>(b) =
      tdesc.columns[i];
    b += sizeof(decltype(tdesc.columns)::value_type);
  }

  *reinterpret_cast<unsigned*>(b) = static_cast<unsigned>(treqs.size());
  b += sizeof(unsigned);
  for (auto& req : treqs) {
    *reinterpret_cast<LogicalRegion*>(b) = req.region;
    b += sizeof(LogicalRegion);
  }

  return b - static_cast<char*>(buffer);
}

size_t
Table::legion_deserialize(const void* buffer) {
  const char *b = static_cast<const char*>(buffer);

  Desc desc;
  desc.num_columns = *reinterpret_cast<const decltype(desc.num_columns)*>(b);
  b += sizeof(desc.num_columns);
  for (size_t i = 0; i < desc.num_columns; ++i) {
    desc.columns[i] =
      *reinterpret_cast<const decltype(desc.columns)::value_type*>(b);
    b += sizeof(decltype(desc.columns)::value_type);
  }

  std::vector<LogicalRegion> lrs;
  unsigned n_lr = *reinterpret_cast<const unsigned*>(b);
  b += sizeof(unsigned);
  lrs.reserve(n_lr);
  for (unsigned i = 0; i < n_lr; ++i) {
    lrs.push_back(*reinterpret_cast<const LogicalRegion*>(b));
    b += sizeof(LogicalRegion);
  }

  auto lrp = lrs.begin();
  assert(lrp != lrs.end());
  m_index_col_cs.metadata_lr = *lrp++;
  assert(lrp != lrs.end());
  m_index_col_cs.column_is = lrp->get_index_space();
  m_index_col_region = *lrp;
  m_index_col_parent = *lrp++;

  std::map<LogicalRegion, ColumnSpace> css;
  for (size_t i = 0; i < desc.num_columns; ++i) {
    auto& cdesc = desc.columns[i];
    if (css.count(cdesc.region) == 0) {
      ColumnSpace cs;
      assert(lrp != lrs.end());
      cs.metadata_lr = *lrp++;
      assert(lrp != lrs.end());
      cs.column_is = lrp++->get_index_space();
      css[cdesc.region] = cs;
    }
    Keywords kw;
    if (cdesc.n_kw > 0) {
      assert(cdesc.n_kw == 2);
      assert(lrp != lrs.end());
      LogicalRegion tt = *lrp++;
      assert(lrp != lrs.end());
      LogicalRegion vl = *lrp++;
      kw = Keywords(Keywords::pair<LogicalRegion>{tt, vl});
    }
#ifdef HYPERION_USE_CASACORE
    MeasRef mr;
    if (cdesc.n_mr > 0) {
      assert(cdesc.n_mr >= 2);
      assert(lrp != lrs.end());
      LogicalRegion md = *lrp++;
      assert(lrp != lrs.end());
      LogicalRegion vl = *lrp++;
      LogicalRegion ix;
      if (cdesc.n_mr > 2) {
        assert(cdesc.n_mr == 3);
        assert(lrp != lrs.end());
        ix = *lrp++;
      }
      mr = MeasRef(md, vl, ix);
    }
    CXX_OPTIONAL_NAMESPACE::optional<hyperion::string> rc;
    if (cdesc.refcol.size() > 0)
      rc = cdesc.refcol;
#endif
    m_columns[cdesc.name] =
      Column(
        cdesc.dt,
        cdesc.fid,
        css.at(cdesc.region),
        cdesc.region,
        cdesc.region,
        kw
#ifdef HYPERION_USE_CASACORE
        , mr
        , rc
#endif
        );
  }
  return b - static_cast<const char*>(buffer);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
