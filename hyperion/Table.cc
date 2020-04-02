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
#include <hyperion/PhysicalTable.h>

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
  __FUNC__(TableFieldsFid::CS)                  \
  __FUNC__(TableFieldsFid::VF)                  \
  __FUNC__(TableFieldsFid::VS)
#else
# define FOREACH_TABLE_FIELD_FID(__FUNC__)      \
  __FUNC__(TableFieldsFid::NM)                  \
  __FUNC__(TableFieldsFid::DT)                  \
  __FUNC__(TableFieldsFid::KW)                  \
  __FUNC__(TableFieldsFid::CS)                  \
  __FUNC__(TableFieldsFid::VF)                  \
  __FUNC__(TableFieldsFid::VS)
#endif

static const hyperion::string empty_nm;
static const Keywords empty_kw;
#ifdef HYPERION_USE_CASACORE
static const MeasRef empty_mr;
static const hyperion::string empty_rc;
#endif
static const hyperion::TypeTag empty_dt = (hyperion::TypeTag)0;
static const FieldID empty_vf = 0;
static const ColumnSpace empty_cs;
static const bool empty_ix = false;
static const LogicalRegion empty_vs;

size_t
Table::columns_result_t::legion_buffer_size(void) const {
  size_t result = sizeof(unsigned);
  for (size_t i = 0; i < fields.size(); ++i)
    result +=
      sizeof(ColumnSpace)
      + sizeof(bool)
      + sizeof(LogicalRegion)
      + sizeof(unsigned)
      + std::get<3>(fields[i]).size() * sizeof(tbl_fld_t);
  return result;
}

size_t
Table::columns_result_t::legion_serialize(void* buffer) const {
  char* b = static_cast<char*>(buffer);
  *reinterpret_cast<unsigned*>(b) = (unsigned)fields.size();
  b += sizeof(unsigned);
  for (size_t i = 0; i < fields.size(); ++i) {
    auto& [csp, ixcs, lr, fs] = fields[i];
    *reinterpret_cast<ColumnSpace*>(b) = csp;
    b += sizeof(csp);
    *reinterpret_cast<bool*>(b) = ixcs;
    b += sizeof(ixcs);
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
    auto& [csp, ixcs, lr, fs] = fields[i];
    csp = *reinterpret_cast<const ColumnSpace*>(b);
    b += sizeof(csp);
    ixcs = *reinterpret_cast<const bool*>(b);
    b += sizeof(ixcs);
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
Table::column_map(const columns_result_t& columns_result) {

  std::unordered_map<std::string, Column> result;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  for (auto& [csp, ixcs, lr, tfs] : columns_result.fields) {
#pragma GCC diagnostic pop
    for (auto& [nm, tf] : tfs) {
      result[nm] = Column(
        tf.dt,
        tf.fid,
#ifdef HYPERION_USE_CASACORE
        tf.mr,
        tf.rc,
#endif
        tf.kw,
        csp,
        lr);
    }
  }
  return result;
}

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
  auto all_columns = columns(ctx, rt).get_result<columns_result_t>();
  auto all_columns_map = column_map(all_columns);
  std::map<std::string, std::optional<Column::Requirements>> omitted_columns;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  for (auto& [nm, col] : all_columns_map) {
#pragma GCC diagnostic pop
    if (colnames.count(nm) == 0)
      omitted_columns[nm] = std::nullopt;
  }
  // TODO: implement a more direct way to compute the table region requirements
  // for the set of attached columns
  RegionRequirement table_req =
    std::get<0>(
      requirements(
        ctx,
        rt,
        ColumnSpacePartition(),
        table_privilege,
        omitted_columns
        /*, column requirements are not be used other than to declare omitted
         *  columns */))[0];
  unsigned index_rank = 0;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  for (auto& [cs, ixcs, vlr, nm_tfs] : all_columns.fields) {
#pragma GCC diagnostic pop
    if (ixcs) {
      auto pr = rt->map_region(ctx, cs.requirements(READ_ONLY, EXCLUSIVE));
      index_rank = ColumnSpace::size(ColumnSpace::axes(pr));
      rt->unmap_region(ctx, pr);
      break;
    }
  }
  assert(index_rank != 0);

  std::unordered_map<std::string, std::shared_ptr<PhysicalColumn>> pcols;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  for (auto& [cs, ixcs, vlr, nm_tfs] : all_columns.fields) {
#pragma GCC diagnostic pop
    std::optional<PhysicalRegion> metadata;
    for (auto& [nm, tf]: nm_tfs) {
      if (colnames.count(nm) > 0 || tf.fid == no_column) {
        if (!metadata) {
          auto req = cs.requirements(READ_ONLY, EXCLUSIVE);
          metadata = rt->map_region(ctx, req);
        }
        std::optional<Keywords::pair<Legion::PhysicalRegion>> kws;
        if (!tf.kw.is_empty()) {
          auto nkw = tf.kw.size(rt);
          std::vector<FieldID> fids(nkw);
          std::iota(fids.begin(), fids.end(), 0);
          auto rqs = tf.kw.requirements(rt, fids, READ_ONLY, true).value();
          Keywords::pair<Legion::PhysicalRegion> kwprs;
          kwprs.type_tags = rt->map_region(ctx, rqs.type_tags);
          kwprs.values = rt->map_region(ctx, rqs.values);
          kws = kwprs;
        }
#ifdef HYPERION_USE_CASACORE
        std::optional<MeasRef::DataRegions> mr_drs;
        if (!tf.mr.is_empty()) {
          auto [mrq, vrq, oirq] = tf.mr.requirements(READ_ONLY, true);
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
            tf.dt,
            tf.fid,
            index_rank,
            metadata.value(),
            ((column_parents.count(nm) > 0) ? column_parents.at(nm) : vlr),
            vlr,
            kws
#ifdef HYPERION_USE_CASACORE
            , mr_drs
            , map(
              tf.rc,
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

  PhysicalTable result(
    table_req.parent,
    rt->map_region(ctx, table_req),
    pcols);
  result.attach_columns(ctx, rt, file_path, column_paths, column_modes);
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
table_fields_requirement(
  LogicalRegion lr,
  LogicalRegion parent,
  PrivilegeMode mode) {

  RegionRequirement result(lr, mode, EXCLUSIVE, parent);
#define ADD_F(F) result.add_field(static_cast<FieldID>(F));
  FOREACH_TABLE_FIELD_FID(ADD_F);
#undef ADD_F
  return result;
}

Table
Table::create(Context ctx, Runtime* rt, const fields_t& fields) {

  size_t num_cols = 0;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  for (auto& [csp, ixcs, tfs] : fields) {
#pragma GCC diagnostic pop
    assert(!csp.is_empty());
    assert(csp.is_valid());
    num_cols += tfs.size();
  }
  {
    std::unordered_set<std::string> cnames;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
    for (auto& [csp, ixcs, cs] : fields)
      for (auto& [nm, c] : cs)
#pragma GCC diagnostic pop
        cnames.insert(nm);
    assert(cnames.count("") == 0);
    assert(cnames.size() == num_cols);
  }

  std::vector<PhysicalRegion> csp_md_prs;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  for (auto& [csp, ixcs, tfs] : fields) {
#pragma GCC diagnostic pop
    RegionRequirement
      req(csp.metadata_lr, READ_ONLY, EXCLUSIVE, csp.metadata_lr);
    req.add_field(ColumnSpace::INDEX_FLAG_FID);
    req.add_field(ColumnSpace::AXIS_VECTOR_FID);
    req.add_field(ColumnSpace::AXIS_SET_UID_FID);
    csp_md_prs.push_back(rt->map_region(ctx, req));
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
        rt->map_region(
          ctx,
          table_fields_requirement(fields_lr, fields_lr, WRITE_ONLY));

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
      const ColumnSpaceAccessor<WRITE_ONLY>
        css(fields_pr, static_cast<FieldID>(TableFieldsFid::CS));
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
        css[*pid] = empty_cs;
        vfs[*pid] = empty_vf;
        vss[*pid] = empty_vs;
      }
      rt->unmap_region(ctx, fields_pr);
    }
    PhysicalRegion fields_pr =
      rt->map_region(
        ctx,
        table_fields_requirement(fields_lr, fields_lr, READ_WRITE));
    std::vector<
      std::tuple<
        ColumnSpace,
        bool,
        size_t,
        std::vector<std::pair<hyperion::string, TableField>>>>
      hcols;
    for (size_t i = 0; i < fields.size(); ++i) {
      auto& [csp, ixcs, nm_tfs] = fields[i];
      std::vector<std::pair<hyperion::string, TableField>> htfs;
      for (auto& [nm, tf]: nm_tfs)
        htfs.emplace_back(nm, tf);
      hcols.emplace_back(csp, ixcs, i, htfs);
    }

    added =
      add_columns(
        ctx,
        rt,
        hcols,
        std::vector<LogicalRegion>(),
        fields_lr,
        fields_pr,
        std::nullopt,
        csp_md_prs);
    for (auto& pr : csp_md_prs)
      rt->unmap_region(ctx, pr);
    rt->unmap_region(ctx, fields_pr);
  }
  Table result(fields_lr);
  if (!added) {
    // FIXME: log a warning: Failed to create non-empty table
    result.destroy(ctx, rt);
    assert(false);
  }
  return result;
}

TaskID Table::index_column_space_task_id;

const char* Table::index_column_space_task_name =
  "Table::index_column_space_task";

Table::index_column_space_result_t
Table::index_column_space_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context,
  Runtime* rt) {

  return index_column_space(rt, task->regions[0].region, regions[0]);
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

static std::tuple<
  std::vector<int>,
  std::optional<size_t>>
index_axes(const std::vector<PhysicalRegion>& csp_metadata_prs) {

  std::vector<int> axes;
  std::unordered_set<int> indexes;
  if (csp_metadata_prs.size() > 0) {
    size_t i = 0;
    for (; axes.size() == 0 && i < csp_metadata_prs.size(); ++i) {
      const ColumnSpace::IndexFlagAccessor<READ_ONLY>
        ifl(csp_metadata_prs[i], ColumnSpace::INDEX_FLAG_FID);
      const ColumnSpace::AxisVectorAccessor<READ_ONLY>
        ax(csp_metadata_prs[i], ColumnSpace::AXIS_VECTOR_FID);
      if (!ifl[0])
        axes = ColumnSpace::from_axis_vector(ax[0]);
      else
        indexes.insert(ax[0][0]);
    }
    for (; axes.size() && i < csp_metadata_prs.size(); ++i) {
      const ColumnSpace::IndexFlagAccessor<READ_ONLY>
        ifl(csp_metadata_prs[i], ColumnSpace::INDEX_FLAG_FID);
      const ColumnSpace::AxisVectorAccessor<READ_ONLY>
        ax(csp_metadata_prs[i], ColumnSpace::AXIS_VECTOR_FID);
      if (!ifl[0]) {
        auto axi = ColumnSpace::from_axis_vector(ax[0]);
        auto mm =
          std::get<0>(
            std::mismatch(axes.begin(), axes.end(), axi.begin(), axi.end()));
        axes.erase(mm, axes.end());
      } else {
        indexes.insert(ax[0][0]);
      }
    }
  }
  // all values in axes must be either 0 (ROW) or correspond to one of the
  // index columns
  if (std::any_of(
        axes.begin(),
        axes.end(),
        [&indexes](int& r) { return r > 0 && indexes.count(r) == 0; }))
    axes.clear();
  // all values in indexes must appear somewhere in axes
  else if (std::any_of(
        indexes.begin(),
        indexes.end(),
        [&axes](const int& i) {
          return std::find(axes.begin(), axes.end(), i) == axes.end();
        }))
    axes.clear();

  std::optional<size_t> index_col_candidate;
  for (size_t i = 0; !index_col_candidate && i < csp_metadata_prs.size(); ++i) {
    const ColumnSpace::AxisVectorAccessor<READ_ONLY>
      ax(csp_metadata_prs[i], ColumnSpace::AXIS_VECTOR_FID);
    auto axi = ColumnSpace::from_axis_vector(ax[0]);
    if (axes == axi)
      index_col_candidate = i;
  }
  return {axes, index_col_candidate};
}

Future /* Table::index_column_space_result_t */
Table::index_column_space(Context ctx, Runtime* rt) const {
  RegionRequirement req(
    fields_lr,
    READ_ONLY,
    EXCLUSIVE,
    fields_parent.value_or(fields_lr));
  req.add_field(static_cast<FieldID>(TableFieldsFid::CS));
  req.add_field(static_cast<FieldID>(TableFieldsFid::VF));
  TaskLauncher task(index_column_space_task_id, TaskArgument(NULL, 0));
  task.add_region_requirement(req);
  return rt->execute_task(ctx, task);
}

Table::index_column_space_result_t
Table::index_column_space(
  Runtime* rt,
  const LogicalRegion& fields_parent,
  const PhysicalRegion& fields_pr) {

  std::optional<ColumnSpace> result;
  const ColumnSpaceAccessor<READ_ONLY>
    css(fields_pr, static_cast<FieldID>(TableFieldsFid::CS));
  const ValueFidAccessor<READ_ONLY>
    vfs(fields_pr, static_cast<FieldID>(TableFieldsFid::VF));
  for (PointInDomainIterator<1> pid(
         rt->get_index_space_domain(fields_parent.get_index_space()));
       pid() && !result;
       pid++) {
    auto css_pid = css.read(*pid);
    if (css_pid.is_empty())
      break;
    if (vfs.read(*pid) == no_column)
      result = css_pid;
  }
  return result;
}

bool
Table::is_empty(const Table::index_column_space_result_t& index_cs) {
  return !index_cs.has_value();
}

std::tuple<std::vector<RegionRequirement>, std::vector<LogicalPartition>>
Table::requirements(
  Context ctx,
  Runtime* rt,
  const ColumnSpacePartition& table_partition,
  PrivilegeMode table_privilege,
  const std::map<std::string, std::optional<Column::Requirements>>&
    column_requirements,
  const Column::Requirements& default_column_requirements) const {

  auto fields_pr =
    rt->map_region(
      ctx,
      table_fields_requirement(
        fields_lr,
        fields_parent.value_or(fields_lr),
        READ_ONLY));

  auto result =
    Table::requirements(
      ctx,
      rt,
      fields_parent.value_or(fields_lr),
      fields_pr,
      column_parents,
      table_partition,
      table_privilege,
      column_requirements,
      default_column_requirements);

  rt->unmap_region(ctx, fields_pr);
  return result;
}

std::tuple<std::vector<RegionRequirement>, std::vector<LogicalPartition>>
Table::requirements(
  Context ctx,
  Runtime* rt,
  const LogicalRegion& fields_parent,
  const PhysicalRegion& fields_pr,
  const std::unordered_map<std::string, LogicalRegion>& column_parents,
  const ColumnSpacePartition& table_partition,
  PrivilegeMode table_privilege,
  const std::map<std::string, std::optional<Column::Requirements>>&
      column_requirements,
  const Column::Requirements& default_column_requirements) {

  const NameAccessor<READ_ONLY>
    nms(fields_pr, static_cast<FieldID>(TableFieldsFid::NM));
  const KeywordsAccessor<READ_ONLY>
    kws(fields_pr, static_cast<FieldID>(TableFieldsFid::KW));
#ifdef HYPERION_USE_CASACORE
  const MeasRefAccessor<READ_ONLY>
    mrs(fields_pr, static_cast<FieldID>(TableFieldsFid::MR));
  const RefColumnAccessor<READ_ONLY>
    rcs(fields_pr, static_cast<FieldID>(TableFieldsFid::RC));
#endif
  const ColumnSpaceAccessor<READ_ONLY>
    css(fields_pr, static_cast<FieldID>(TableFieldsFid::CS));
  const ValueFidAccessor<READ_ONLY>
    vfs(fields_pr, static_cast<FieldID>(TableFieldsFid::VF));
  const ValuesAccessor<READ_ONLY>
    vss(fields_pr, static_cast<FieldID>(TableFieldsFid::VS));

  // add a flag field as the basis of a partition by column selection
  FieldID col_select_fid;
  {
    auto fs = fields_pr.get_logical_region().get_field_space();
    auto fa = rt->create_field_allocator(ctx, fs);
    // TODO: can col_select_fid be a local field?
    col_select_fid = fa.allocate_field(sizeof(int));
  }
  auto col_select_pr =
    rt->map_region(
      ctx,
      RegionRequirement(
        fields_pr.get_logical_region(),
        {col_select_fid},
        {col_select_fid},
        WRITE_ONLY,
        EXCLUSIVE,
        fields_parent));

  const FieldAccessor<
    WRITE_ONLY,
    int,
    1,
    coord_t,
    AffineAccessor<int, 1, coord_t>>
    sel_flags(col_select_pr, col_select_fid);
  bool all_cols_selected = true;
  bool some_cols_selected = false;

  DomainT<1> tdom = rt->get_index_space_domain(fields_parent.get_index_space());

  // collect requirement parameters for each column
  std::map<
    hyperion::string,
    std::tuple<
      LogicalRegion, // ColumnSpace metadata
      LogicalRegion, // Column values -- can be NO_REGION!
      Column::Requirements>> column_regions;
#ifdef HYPERION_USE_CASACORE
  std::map<hyperion::string, Column::Requirements> mrc_reqs;
#endif
  {
    std::map<ColumnSpace, Column::Req> cs_reqs;
    for (PointInDomainIterator<1> pid(tdom); pid(); pid++) {
      auto css_pid = css.read(*pid);
      if (css_pid.is_empty())
        break;
      auto vfs_pid = vfs.read(*pid);
      auto nms_pid = nms.read(*pid);
      if (vfs_pid == no_column
          || column_requirements.count(nms_pid) == 0
          || column_requirements.at(nms_pid)) {
        assert((vfs_pid == no_column) == (nms_pid.size() == 0));
        Column::Requirements colreqs = default_column_requirements;
        if (vfs_pid != no_column && column_requirements.count(nms_pid) > 0)
          colreqs = column_requirements.at(nms_pid).value();
        column_regions[nms_pid] = {css_pid.metadata_lr, vss.read(*pid), colreqs};
        if (cs_reqs.count(css_pid) == 0) {
          cs_reqs[css_pid] = colreqs.column_space;
        } else {
          // FIXME: log a warning, and return empty result;
          // warning: inconsistent requirements on shared Column metadata regions
          assert(cs_reqs[css_pid] == colreqs.column_space);
        }
#ifdef HYPERION_USE_CASACORE
        auto rcs_pid = rcs.read(*pid);
        if (rcs_pid.size() > 0)
          mrc_reqs[rcs_pid] = colreqs;
#endif
        sel_flags[*pid] = 1;
        some_cols_selected = true;
      } else {
        sel_flags[*pid] = 0;
        all_cols_selected = false;
      }
    }
  }
  rt->unmap_region(ctx, col_select_pr);

#ifdef HYPERION_USE_CASACORE
  // apply mode of value column to its measure reference column
  for (auto& [nm, rq] : mrc_reqs) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
    auto& [mdlr, vlr, reqs] = column_regions[nm];
#pragma GCC diagnostic pop
    reqs = rq;
  }
#endif

  // create requirements, applying table_partition as needed
  std::map<ColumnSpace, LogicalPartition> partitions;
  // boolean elements in value of following maps is used to track whether the
  // requirement has already been added when iterating through columns
  std::map<LogicalRegion, std::tuple<bool, RegionRequirement>> md_reqs;
  std::map<
    std::tuple<LogicalRegion, PrivilegeMode, CoherenceProperty, MappingTagID>,
    std::tuple<bool, RegionRequirement>> val_reqs;
  for (PointInDomainIterator<1> pid(tdom);
       pid();
       pid++) {
    auto css_pid = css.read(*pid);
    if (css_pid.is_empty())
      break;
    auto nms_pid = nms.read(*pid);
    if (column_regions.count(nms_pid) > 0) {
      auto& rg = column_regions.at(nms_pid);
      auto& [mdlr, vlr, reqs] = rg;
      if (md_reqs.count(mdlr) == 0) {
        RegionRequirement req(
          mdlr,
          {ColumnSpace::AXIS_VECTOR_FID,
           ColumnSpace::AXIS_SET_UID_FID,
           ColumnSpace::INDEX_FLAG_FID},
          {ColumnSpace::AXIS_VECTOR_FID,
           ColumnSpace::AXIS_SET_UID_FID,
           ColumnSpace::INDEX_FLAG_FID},
          reqs.column_space.privilege,
          reqs.column_space.coherence,
          mdlr);
        md_reqs[mdlr] = {false, req};
      }
      auto vfs_pid = vfs.read(*pid);
      if (vfs_pid != no_column) {
        decltype(val_reqs)::key_type rg_rq =
          {vlr, reqs.values.privilege, reqs.values.coherence, reqs.tag};
        if (val_reqs.count(rg_rq) == 0) {
          LogicalRegion parent = vlr;
          if (column_parents.count(nms_pid) > 0)
            parent = column_parents.at(nms_pid);
          if (!table_partition.is_valid()) {
            val_reqs[rg_rq] =
              {false,
               RegionRequirement(
                 vlr,
                 reqs.values.privilege,
                 reqs.values.coherence,
                 parent,
                 reqs.tag)};
          } else {
            LogicalPartition lp;
            if (partitions.count(css_pid) == 0) {
              auto csp =
                table_partition.project_onto(ctx, rt, css_pid)
                .get_result<ColumnSpacePartition>();
              lp = rt->get_logical_partition(ctx, vlr, csp.column_ip);
              csp.destroy(ctx, rt);
              partitions[css_pid] = lp;
            } else {
              lp = partitions[css_pid];
            }
            val_reqs[rg_rq] =
              {false,
               RegionRequirement(
                 lp,
                 0,
                 reqs.values.privilege,
                 reqs.values.coherence,
                 parent,
                 reqs.tag)};
          }
        }
        std::get<1>(val_reqs[rg_rq]).add_field(vfs_pid, reqs.values.mapped);
      }
    }
  }
  std::vector<LogicalPartition> lps_result;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  for (auto& [csp, lp] : partitions)
#pragma GCC diagnostic pop
    lps_result.push_back(lp);

  // gather all requirements, in order set by this traversal of fields_pr
  std::vector<RegionRequirement> reqs_result;

  // start with the table fields_lr, partitioned according to col_select_lr when
  // necessary
  if (all_cols_selected) {
    reqs_result.push_back(
      table_fields_requirement(
        fields_pr.get_logical_region(),
        fields_parent,
        table_privilege));
  } else if (some_cols_selected) {
    auto cs = rt->create_index_space(ctx, Rect<1>(0, 1));
    auto ip =
      rt->create_partition_by_field(
        ctx,
        fields_pr.get_logical_region(),
        fields_parent,
        col_select_fid,
        cs);
    auto lp = rt->get_logical_partition(ctx, fields_parent, ip);
    auto lr = rt->get_logical_subregion_by_color(ctx, lp, 1);
    reqs_result.push_back(
      table_fields_requirement(
        fields_pr.get_logical_region(),
        lr,
        table_privilege));
    rt->destroy_index_partition(ctx, ip);
    rt->destroy_index_space(ctx, cs);
    lps_result.push_back(lp);
  }
  {
    auto fs = fields_pr.get_logical_region().get_field_space();
    auto fa = rt->create_field_allocator(ctx, fs);
    fa.free_field(col_select_fid);
  }

  // add requirements for all logical regions in all selected columns
  for (PointInDomainIterator<1> pid(tdom); pid(); pid++) {
    if (css.read(*pid).is_empty())
      break;
    auto nms_pid = nms.read(*pid);
    if (column_regions.count(nms_pid) > 0) {
      auto& rg = column_regions.at(nms_pid);
      auto& [mdlr, vlr, reqs] = rg;
      {
        auto& [added, rq] = md_reqs.at(mdlr);
        if (!added) {
          reqs_result.push_back(rq);
          added = true;
        }
      }
      decltype(val_reqs)::key_type rg_rq =
        {vlr, reqs.values.privilege, reqs.values.coherence, reqs.tag};
      if (vlr != LogicalRegion::NO_REGION) {
        auto& [added, rq] = val_reqs.at(rg_rq);
        if (!added) {
          reqs_result.push_back(rq);
          added = true;
        }
      }
      auto kws_pid = kws.read(*pid);
      auto nkw = kws_pid.size(rt);
      if (nkw > 0) {
        std::vector<FieldID> fids(nkw);
        std::iota(fids.begin(), fids.end(), 0);
        auto rqs =
          kws_pid.requirements(
            rt,
            fids,
            reqs.keywords.privilege,
            reqs.keywords.mapped)
          .value();
        reqs_result.push_back(rqs.type_tags);
        reqs_result.push_back(rqs.values);
      }

#ifdef HYPERION_USE_CASACORE
      auto mrs_pid = mrs.read(*pid);
      if (!mrs_pid.is_empty()) {
        auto [mrq, vrq, oirq] =
          mrs_pid.requirements(reqs.measref.privilege, reqs.measref.mapped);
        reqs_result.push_back(mrq);
        reqs_result.push_back(vrq);
        if (oirq)
          reqs_result.push_back(oirq.value());
      }
#endif
    }
  }
  return {reqs_result, lps_result};
}

TaskID Table::is_conformant_task_id;

const char* Table::is_conformant_task_name = "Table::is_conformant_task";

struct IsConformantArgs {
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
  std::optional<std::tuple<IndexSpace, PhysicalRegion>> index_cs;
  if (args->index_cs_is != IndexSpace::NO_SPACE)
    index_cs = {args->index_cs_is, regions[2]};
  return
    Table::is_conformant(
      rt,
      task->regions[0].region,
      regions[0],
      index_cs,
      args->cs_is,
      regions[1]);
}

Future /* bool */
Table::is_conformant(Context ctx, Runtime* rt, const ColumnSpace& cs) const {
  auto f_opt_index_cs = index_column_space(ctx, rt);
  IsConformantArgs args;
  args.cs_is = cs.column_is;
  TaskLauncher task(
    is_conformant_task_id,
    TaskArgument(&cs.column_is, sizeof(cs.column_is)));
  {
    RegionRequirement req(
      fields_lr,
      READ_ONLY,
      EXCLUSIVE,
      fields_parent.value_or(fields_lr));
    req.add_field(static_cast<FieldID>(TableFieldsFid::CS));
    task.add_region_requirement(req);
  }
  {
    RegionRequirement req(cs.metadata_lr, READ_ONLY, EXCLUSIVE, cs.metadata_lr);
    req.add_field(ColumnSpace::AXIS_SET_UID_FID);
    req.add_field(ColumnSpace::AXIS_VECTOR_FID);
    req.add_field(ColumnSpace::INDEX_FLAG_FID);
    task.add_region_requirement(req);
  }
  index_column_space_result_t opt_index_cs =
    f_opt_index_cs.get_result<index_column_space_result_t>();
  if (opt_index_cs) {
    auto& index_cs = opt_index_cs.value();
    args.index_cs_is = index_cs.column_is;
    RegionRequirement
      req(index_cs.metadata_lr, READ_ONLY, EXCLUSIVE, index_cs.metadata_lr);
    req.add_field(ColumnSpace::AXIS_SET_UID_FID);
    req.add_field(ColumnSpace::AXIS_VECTOR_FID);
    task.add_region_requirement(req);
  }
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
    Point<OBJECT_RANK> pt;
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
  const LogicalRegion& fields_parent,
  const PhysicalRegion& fields_pr,
  const std::optional<std::tuple<IndexSpace, PhysicalRegion>>&
  index_cs,
  const IndexSpace& cs_is,
  const PhysicalRegion& cs_md_pr) {

  if (!index_cs)
    return true;

  // if this ColumnSpace already exists in the Table, conformance must hold
  ColumnSpace cs(cs_is, cs_md_pr.get_logical_region());
  assert(!cs.is_empty());
  const ColumnSpaceAccessor<READ_ONLY>
    css(fields_pr, static_cast<FieldID>(TableFieldsFid::CS));
  for (PointInDomainIterator<1> pid(
         rt->get_index_space_domain(fields_parent.get_index_space()));
       pid();
       pid++) {
    auto css_pid = css.read(*pid);
    if (css_pid.is_empty())
      break;
    if (css_pid == cs)
      return true;
  }

  auto& [index_cs_is, index_cs_md_pr] = index_cs.value();
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
  // for conformance, cs cannot have its index flag set, and its axis uid must
  // be that of the index column space
  if (!cs_if[0] && index_cs_au[0] == cs_au[0]) {
    const auto index_ax = ColumnSpace::from_axis_vector(index_cs_av[0]);
    const auto cs_ax = ColumnSpace::from_axis_vector(cs_av[0]);
    // for conformance, the cs axis vector must have a prefix equal to the axis
    // vector of the index column space
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
#define CONFORM(IRANK, CRANK)                                     \
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
  }
  return result;
}

TaskID Table::add_columns_task_id;

const char* Table::add_columns_task_name = "Table::add_columns_task";

struct AddColumnsTaskArgs {
  std::array<
    std::tuple<ColumnSpace, bool, size_t, hyperion::string, TableField>,
    Table::MAX_COLUMNS> columns;
  std::array<LogicalRegion, Table::MAX_COLUMNS> vlrs;
  ssize_t index_cs_idx;
};

bool
Table::add_columns_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *rt) {

  const AddColumnsTaskArgs* args =
    static_cast<const AddColumnsTaskArgs*>(task->args);
  std::map<
    ColumnSpace,
    std::tuple<
      bool,
      size_t,
      std::vector<std::pair<hyperion::string, TableField>>>>
    columns;
  std::optional<std::tuple<IndexSpace, PhysicalRegion>> index_cs;
  if (args->index_cs_idx >= 0)
    index_cs =
      {args->vlrs[args->index_cs_idx].get_index_space(),
       regions[args->index_cs_idx]};
  for (auto& [csp, ixcs, idx, nm, tf]: args->columns) {
    if (!csp.is_valid())
      break;
    if (columns.count(csp) == 0)
      columns[csp] = {
        ixcs,
        idx,
        std::vector<std::pair<hyperion::string, TableField>>()};
    std::get<2>(columns[csp]).emplace_back(nm, tf);
  }
  std::vector<LogicalRegion> vlrs;
  for (auto& vlr: args->vlrs) {
    if (vlr == LogicalRegion::NO_REGION)
      break;
    vlrs.push_back(vlr);
  }
  const PhysicalRegion& fields_pr = regions.back();
  std::vector<PhysicalRegion> csp_md_prs(regions.begin(), regions.end() - 1);
  std::vector<
    std::tuple<
      ColumnSpace,
      bool,
      size_t,
      std::vector<std::pair<hyperion::string, TableField>>>>
    colv;
  colv.reserve(columns.size());
  for (auto& [csp, ixcs_idx_tfs] : columns) {
    auto& [ixcs, idx, tfs] = ixcs_idx_tfs;
    colv.emplace_back(csp, ixcs, idx, tfs);
  }
  return
    add_columns(
      ctx,
      rt,
      colv,
      vlrs,
      task->regions.back().region,
      fields_pr,
      index_cs,
      csp_md_prs);
}

Future /* bool */
Table::add_columns(
  Context ctx,
  Runtime* rt,
  const fields_t& new_columns) const {

  if (new_columns.size() == 0)
    return Future::from_value(rt, true);

  AddColumnsTaskArgs args;
  args.index_cs_idx = -1;
  TaskLauncher task(add_columns_task_id, TaskArgument(&args, sizeof(args)));
  std::vector<RegionRequirement> reqs;
  std::vector<ColumnSpace> new_csps;
  auto current_columns = columns(ctx, rt).get_result<columns_result_t>();
  std::map<ColumnSpace, size_t> current_csp_idxs;
  {
    size_t i = 0;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
    for (auto& [csp, ixcs, vlr, tfs] : current_columns.fields) {
#pragma GCC diagnostic pop
      current_csp_idxs[csp] = i;
      if (ixcs && args.index_cs_idx < 0)
        args.index_cs_idx = i;
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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
    for (auto& [csp, ixcs, nm_tfs]: new_columns) {
#pragma GCC diagnostic pop
      if (current_csp_idxs.count(csp) == 0) {
        current_csp_idxs[csp] = reqs.size();
        RegionRequirement
          req(csp.metadata_lr, READ_ONLY, EXCLUSIVE, csp.metadata_lr);
        req.add_field(ColumnSpace::AXIS_VECTOR_FID);
        req.add_field(ColumnSpace::AXIS_SET_UID_FID);
        req.add_field(ColumnSpace::INDEX_FLAG_FID);
        reqs.push_back(req);
      }
      auto idx = current_csp_idxs[csp];
      for (auto& [nm, tf]: nm_tfs)
        args.columns[i++] = {csp, ixcs, idx, string(nm), tf};
    }
    if (i < MAX_COLUMNS)
      args.columns[i] = {ColumnSpace(), false, 0, string(), TableField()};
  }
  reqs.push_back(
    table_fields_requirement(
      fields_lr,
      fields_parent.value_or(fields_lr),
      READ_WRITE));
  for (auto& req : reqs)
    task.add_region_requirement(req);
  return rt->execute_task(ctx, task);
}

bool
Table::add_columns(
  Context ctx,
  Runtime* rt,
  const std::vector<
    std::tuple<
      ColumnSpace,
      bool,
      size_t,
      std::vector<std::pair<hyperion::string, TableField>>>>& new_columns,
  const std::vector<LogicalRegion>& vlrs,
  const LogicalRegion& fields_parent,
  const PhysicalRegion& fields_pr,
  const std::optional<std::tuple<IndexSpace, PhysicalRegion>>& index_cs,
  const std::vector<PhysicalRegion>& csp_md_prs) {

  if (new_columns.size() == 0)
    return true;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  for (auto& [csp, ixcs, idx, nmtfs]: new_columns) {
#pragma GCC diagnostic pop
    const ColumnSpace::IndexFlagAccessor<READ_ONLY>
      ifl(csp_md_prs[idx], ColumnSpace::INDEX_FLAG_FID);
    if (ifl[0]) {
      // a ColumnSpace flagged as in index column cannot also be the Table's
      // index column space
      if (ixcs) {
        // FIXME: log warning: a ColumnSpace flagged as in index column cannot
        // also be the Table's index column space
        return false;
      }
      // new index columns can be added only if the current table has no index
      // column space
      if (index_cs) {
        // FIXME: log warning: new index columns can be added only if the
        // current table has no index column space
        return false;
      }
    }
  }

  // check conformance of all ColumnSpaces in new_columns
  //
  // TODO: make this optional?
  std::tuple<ssize_t, ColumnSpace, LogicalRegion>
    new_columns_ics = {-1, ColumnSpace(), LogicalRegion::NO_REGION};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  for (auto& [csp, ixcs, idx, nmtfs] : new_columns) {
#pragma GCC diagnostic pop
    if (ixcs) {
      // the third field is initialized only when the LogicalRegion can be
      // identified or created
      new_columns_ics = {idx, csp, LogicalRegion::NO_REGION};
    }
    if (!Table::is_conformant(
          rt,
          fields_parent,
          fields_pr,
          index_cs,
          csp.column_is,
          csp_md_prs[idx])) {
      // FIXME: log warning: cannot add non-conforming Columns to Table
      return false;
    }
  }

  // All columns must have a common axes uid. This condition has already been
  // checked by the calls to Table::is_conformant(), but only if index_cs
  // exists.
  std::tuple<std::vector<int>, std::optional<size_t>> new_index_axes;
  if (!index_cs) {
    assert(csp_md_prs.size() > 0);
    const ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
      au(csp_md_prs[0], ColumnSpace::AXIS_SET_UID_FID);
    ColumnSpace::AXIS_SET_UID_TYPE auid = au[0];
    for (size_t i = 1; i < csp_md_prs.size(); ++i) {
      const ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
        au(csp_md_prs[i], ColumnSpace::AXIS_SET_UID_FID);
      if (auid != au[0]) {
        // FIXME: log warning: cannot add Columns with different axes UID to a
        // single Table
        return false;
      }
    }
    new_index_axes = index_axes(csp_md_prs);

    // There must be a ColumnSpace given for the index column space. It ought to
    // be possible to synthesize such a ColumnSpace (and IndexSpace) if it were
    // not provided by the caller, but it would be expensive (TODO: optionally
    // sythesize a needed ColumnSpace?)
    if (!std::get<1>(new_index_axes)) {
      // FIXME: log warning: no ColumnSpace provided that matches the Table's
      // (row) index space
      return false;
    }
    if ((ssize_t)std::get<1>(new_index_axes).value()
        != std::get<0>(new_columns_ics)) {
      // FIXME: log warning; flagged index column ColumnSpace has incongruent
      // axes
      return false;
    }
  }

  // All ColumnSpaces must have unique axis vectors.
  {
    std::set<std::vector<int>> axes;
    for (auto& pr : csp_md_prs) {
      const ColumnSpace::AxisVectorAccessor<READ_ONLY>
        ax(pr, ColumnSpace::AXIS_VECTOR_FID);
      auto axv = ColumnSpace::from_axis_vector(ax[0]);
      if (axes.count(axv) > 0) {
        // FIXME: log warning: ColumnSpaces added to Table do not have unique
        // axis vectors
        return false;
      }
      axes.insert(axv);
    }
  }

  auto current_columns = columns(rt, fields_parent, fields_pr);
  auto current_column_map = column_map(current_columns);

  // column names must be unique
  {
    std::set<std::string> new_column_names;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
    for (auto& [csp, ixcs, idx, nmtfs]: new_columns) {
      for (auto& [hnm, tf]: nmtfs) {
#pragma GCC diagnostic pop
        std::string nm = hnm;
        if (current_column_map.count(nm) > 0
            || new_column_names.count(nm) > 0)
          return false;
        new_column_names.insert(nm);
      }
    }
  }
  // get ColumnSpace regions for current columns only
  std::vector<PhysicalRegion> current_csp_md_prs;
  for (auto& pr : csp_md_prs) {
    auto c =
      std::find_if(
        current_columns.fields.begin(),
        current_columns.fields.end(),
        [lr=pr.get_logical_region()](auto& csp_ixcs_vlr_tfs) {
          return lr == std::get<0>(csp_ixcs_vlr_tfs).metadata_lr;
        });
    if (c != current_columns.fields.end())
      current_csp_md_prs.push_back(pr);
  }

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
  const ColumnSpaceAccessor<READ_WRITE>
    css(fields_pr, static_cast<FieldID>(TableFieldsFid::CS));
  const ValueFidAccessor<READ_WRITE>
    vfs(fields_pr, static_cast<FieldID>(TableFieldsFid::VF));
  const ValuesAccessor<READ_WRITE>
    vss(fields_pr, static_cast<FieldID>(TableFieldsFid::VS));

  std::map<ColumnSpace, LogicalRegion> csp_vlrs;

  PointInDomainIterator<1> pid(
    rt->get_index_space_domain(fields_parent.get_index_space()));

  {
    size_t num_csp = 0;
    // gather up all ColumnSpaces
    while (pid()) {
      auto css_pid = css.read(*pid);
      if (css_pid == empty_cs)
        break;
      if (csp_vlrs.count(css_pid) == 0) {
        csp_vlrs[css_pid] = vss.read(*pid);
        ++num_csp;
      }
      pid++;
    }
  }
  if (!pid()) {
    // FIXME: log error: cannot add further columns to Table
    assert(pid());
  }

  for (auto& [csp, ixcs, idx, nm_tfs] : new_columns) {
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
      if (ixcs)
        std::get<2>(new_columns_ics) = csp_vlrs[csp];
    }
    LogicalRegion& values_lr = csp_vlrs[csp];
    std::set<FieldID> fids;
    FieldSpace fs = values_lr.get_field_space();
    rt->get_field_space_fields(fs, fids);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    for (auto& [nm, tf] : nm_tfs) {
      assert(pid());
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
      assert(pid());
      nms.write(*pid, nm);
      dts.write(*pid, tf.dt);
      kws.write(*pid, tf.kw);
#ifdef HYPERION_USE_CASACORE
      mrs.write(*pid, tf.mr);
      rcs.write(*pid, tf.rc.value_or(empty_rc));
#endif
      css.write(*pid, csp);
      vfs.write(*pid, tf.fid);
      vss.write(*pid, values_lr);
      fids.insert(tf.fid);
      pid++;
    }
  }

  // add empty column for index column space
  if (!index_cs) {
    assert(pid());
    nms.write(*pid, empty_nm);
    dts.write(*pid, empty_dt);
    kws.write(*pid, empty_kw);
#ifdef HYPERION_USE_CASACORE
    mrs.write(*pid, empty_mr);
    rcs.write(*pid, empty_rc);
#endif
    css.write(*pid, std::get<1>(new_columns_ics));
    vfs.write(*pid, no_column);
    vss.write(*pid, std::get<2>(new_columns_ics));
  }
  return true;
}

bool
Table::remove_columns(
  Context ctx,
  Runtime* rt,
  const std::unordered_set<std::string>& columns,
  bool destroy_orphan_column_spaces,
  bool destroy_field_data) {

  auto fields_pr =
    rt->map_region(
      ctx,
      table_fields_requirement(
        fields_lr,
        fields_parent.value_or(fields_lr),
        READ_WRITE));
  std::vector<ColumnSpace> css;
  std::vector<PhysicalRegion> cs_md_prs;
  auto tbl_columns = Table::columns(ctx, rt).get_result<columns_result_t>();
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  for (auto& [cs, ixcs, vlr, nm_tfs] : tbl_columns.fields) {
    for (auto& [nm, tf] : nm_tfs) {
#pragma GCC diagnostic pop
      if (columns.count(nm) > 0) {
        RegionRequirement
          req(cs.metadata_lr, READ_ONLY, EXCLUSIVE, cs.metadata_lr);
        req.add_field(ColumnSpace::INDEX_FLAG_FID);
        cs_md_prs.push_back(rt->map_region(ctx, req));
        css.push_back(cs);
        break;
      }
    }
  }
  std::set<hyperion::string> cols;
  for (auto& c : columns)
    cols.insert(c);
  auto result =
    Table::remove_columns(
      ctx,
      rt,
      cols,
      destroy_orphan_column_spaces,
      destroy_field_data,
      fields_parent.value_or(fields_lr),
      fields_pr,
      css,
      cs_md_prs);
  for (auto& pr : cs_md_prs)
    rt->unmap_region(ctx, pr);
  rt->unmap_region(ctx, fields_pr);
  return result;
}

bool
Table::remove_columns(
  Context ctx,
  Runtime* rt,
  const std::set<hyperion::string>& columns,
  bool destroy_orphan_column_spaces,
  bool destroy_field_data,
  const LogicalRegion& fields_parent,
  const PhysicalRegion& fields_pr,
  const std::vector<ColumnSpace>& cs,
  const std::vector<PhysicalRegion>& cs_md_prs) {

  if (columns.size() == 0)
    return true;

  std::map<ColumnSpace, std::tuple<LogicalRegion, FieldAllocator>> vlr_fa;
  ColumnSpace ics;
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
    const ColumnSpaceAccessor<READ_WRITE>
      css(fields_pr, static_cast<FieldID>(TableFieldsFid::CS));
    const ValueFidAccessor<READ_WRITE>
      vfs(fields_pr, static_cast<FieldID>(TableFieldsFid::VF));
    const ValuesAccessor<READ_WRITE>
      vss(fields_pr, static_cast<FieldID>(TableFieldsFid::VS));

    for (PointInDomainIterator<1> pid(
           rt->get_index_space_domain(fields_parent.get_index_space()));
         pid();
         pid++) {
      auto css_pid = css.read(*pid);
      if (css_pid.is_empty())
        break;
      if (vfs.read(*pid) != no_column && columns.count(nms.read(*pid)) > 0) {
        auto idx =
          std::distance(
            cs.begin(),
            std::find(cs.begin(), cs.end(), css_pid));
        assert(idx < (ssize_t)cs_md_prs.size());
        const ColumnSpace::IndexFlagAccessor<READ_ONLY>
          ixfl(cs_md_prs[idx], ColumnSpace::INDEX_FLAG_FID);
        if (ixfl[0]) {
          // FIXME: log warning: cannot remove a table index column
          return false;
        }
      }
    }

    PointInDomainIterator<1> src_pid(
      rt->get_index_space_domain(fields_parent.get_index_space()));
    PointInDomainIterator<1> dst_pid = src_pid;
    while (src_pid()) {
      auto css_src_pid = css.read(*src_pid);
      auto vfs_src_pid = vfs.read(*src_pid);
      if (vfs_src_pid == no_column)
        ics = css_src_pid;
      auto nms_src_pid = nms.read(*src_pid);
      bool remove =
        vfs_src_pid != no_column && columns.count(nms_src_pid) > 0;
      auto vss_src_pid = vss.read(*src_pid);
      if (remove) {
        if (vlr_fa.count(css_src_pid) == 0)
          vlr_fa[css_src_pid] =
            {vss_src_pid,
             rt->create_field_allocator(ctx, vss_src_pid.get_field_space())};
        std::get<1>(vlr_fa[css_src_pid]).free_field(vfs_src_pid);
        if (destroy_field_data) {
#ifdef HYPERION_USE_CASACORE
          mrs.read(*src_pid).destroy(ctx, rt);
#endif
          kws.read(*src_pid).destroy(ctx, rt);
        }
      } else if (src_pid[0] != dst_pid[0]) {
        nms.write(*dst_pid, nms_src_pid);
        dts.write(*dst_pid, dts.read(*src_pid));
        kws.write(*dst_pid, kws.read(*src_pid));
#ifdef HYPERION_USE_CASACORE
        mrs.write(*dst_pid, mrs.read(*src_pid));
        rcs.write(*dst_pid, rcs.read(*src_pid));
#endif
        css.write(*dst_pid, css_src_pid);
        vfs.write(*dst_pid, vfs_src_pid);
        vss.write(*dst_pid, vss_src_pid);
      }
      src_pid++;
      if (!remove)
        dst_pid++;
    }
    while (dst_pid()) {
      nms.write(*dst_pid, empty_nm);
      dts.write(*dst_pid, empty_dt);
      kws.write(*dst_pid, empty_kw);
#ifdef HYPERION_USE_CASACORE
      mrs.write(*dst_pid, empty_mr);
      rcs.write(*dst_pid, empty_rc);
#endif
      css.write(*dst_pid, empty_cs);
      vfs.write(*dst_pid, empty_vf);
      vss.write(*dst_pid, empty_vs);
      dst_pid++;
    }
  }
  std::vector<std::pair<ColumnSpace, std::tuple<LogicalRegion, FieldAllocator>>>
    csp_vlr_fa(vlr_fa.begin(), vlr_fa.end());
  for (auto& [csp, vlr_fa] : csp_vlr_fa) {
    if (csp != ics) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
      auto& [vlr, fa] = vlr_fa;
#pragma GCC diagnostic pop
      std::vector<FieldID> fids;
      rt->get_field_space_fields(vlr.get_field_space(), fids);
      if (fids.size() == 0) {
        if (destroy_orphan_column_spaces)
          csp.destroy(ctx, rt, true);
        rt->destroy_field_space(ctx, vlr.get_field_space());
        rt->destroy_logical_region(ctx, vlr);
      }
    }
  }
  return true;
}

void
Table::destroy(
  Context ctx,
  Runtime* rt,
  bool destroy_column_space_components,
  bool destroy_field_data) {

  if (fields_lr != LogicalRegion::NO_REGION) {
    auto fields_pr =
      rt->map_region(
        ctx,
        table_fields_requirement(
          fields_lr,
          fields_parent.value_or(fields_lr),
          READ_WRITE));
    const KeywordsAccessor<READ_WRITE>
      kws(fields_pr, static_cast<FieldID>(TableFieldsFid::KW));
#ifdef HYPERION_USE_CASACORE
    const MeasRefAccessor<READ_WRITE>
      mrs(fields_pr, static_cast<FieldID>(TableFieldsFid::MR));
#endif
    const ColumnSpaceAccessor<READ_WRITE>
      css(fields_pr, static_cast<FieldID>(TableFieldsFid::CS));
    const ValuesAccessor<READ_ONLY>
      vss(fields_pr, static_cast<FieldID>(TableFieldsFid::VS));

    std::set<ColumnSpace> destroyed_cs;
    std::set<LogicalRegion> destroyed_vlr;

    for (PointInDomainIterator<1> pid(
           rt->get_index_space_domain(fields_lr.get_index_space()));
         pid();
         pid++) {
      auto css_pid = css.read(*pid);
      if (css_pid.is_empty())
        break;
      if (destroy_field_data) {
        kws.read(*pid).destroy(ctx, rt);
#ifdef HYPERION_USE_CASACORE
        mrs.read(*pid).destroy(ctx, rt);
#endif
      }
      auto vss_pid = vss.read(*pid);
      if (vss_pid != LogicalRegion::NO_REGION
          && destroyed_vlr.count(vss_pid) == 0) {
        destroyed_vlr.insert(vss_pid);
        rt->destroy_field_space(ctx, vss_pid.get_field_space());
        rt->destroy_logical_region(ctx, vss_pid);
      }
      if (destroy_column_space_components
          && destroyed_cs.count(css_pid) == 0) {
        destroyed_cs.insert(css_pid);
        css_pid.destroy(ctx, rt, true);
      }
    }
    rt->unmap_region(ctx, fields_pr);

    rt->destroy_index_space(ctx, fields_lr.get_index_space());
    rt->destroy_field_space(ctx, fields_lr.get_field_space());
    rt->destroy_logical_region(ctx, fields_lr);
    fields_lr = LogicalRegion::NO_REGION;
  }
}

TaskID Table::columns_task_id;

const char* Table::columns_task_name = "Table::columns_task";

Table::columns_result_t
Table::columns_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context,
  Runtime *rt) {

  assert(regions.size() == 1);
  return columns(rt, task->regions[0].region, regions[0]);
}

Future /* columns_result_t */
Table::columns(Context ctx, Runtime *rt) const {
  TaskLauncher task(columns_task_id, TaskArgument(NULL, 0));
  task.add_region_requirement(
    table_fields_requirement(
      fields_lr,
      fields_parent.value_or(fields_lr),
      READ_ONLY));
  task.enable_inlining = true;
  return rt->execute_task(ctx, task);
}

Table::columns_result_t
Table::columns(
  Runtime *rt,
  const LogicalRegion& fields_parent,
  const PhysicalRegion& fields_pr) {

  std::map<
    ColumnSpace,
    std::tuple<
      bool,
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
  const ColumnSpaceAccessor<READ_ONLY>
    css(fields_pr, static_cast<FieldID>(TableFieldsFid::CS));
  const ValueFidAccessor<READ_ONLY>
    vfs(fields_pr, static_cast<FieldID>(TableFieldsFid::VF));
  const ValuesAccessor<READ_ONLY>
    vss(fields_pr, static_cast<FieldID>(TableFieldsFid::VS));

  ColumnSpace ics;
  for (PointInDomainIterator<1> pid(
         rt->get_index_space_domain(fields_parent.get_index_space()));
       pid();
       pid++) {
    auto css_pid = css.read(*pid);
    if (css_pid == empty_cs)
      break;
    auto vss_pid = vss.read(*pid);
    if (!css_pid.is_empty() && vss_pid != LogicalRegion::NO_REGION) {
      auto vfs_pid = vfs.read(*pid);
      if (vfs_pid == no_column)
        ics = css_pid;
      if (cols.count(css_pid) == 0)
        cols[css_pid] = {
          false,
          vss_pid,
          std::vector<columns_result_t::tbl_fld_t>()};
      if (vfs_pid != no_column) {
        auto rcs_pid = rcs.read(*pid);
        columns_result_t::tbl_fld_t tf = {
          nms.read(*pid),
          TableField(
            dts.read(*pid),
            vfs_pid,
            kws.read(*pid)
#ifdef HYPERION_USE_CASACORE
            , mrs.read(*pid)
            , ((rcs_pid.size() > 0)
             ? std::make_optional<hyperion::string>(rcs_pid)
             : std::nullopt)
#endif
            )};
        std::get<2>(cols[css_pid]).push_back(tf);
      }
    }
  }
  columns_result_t result;
  result.fields.reserve(cols.size());
  for (auto& [csp, ixcs_lr_tfs] : cols) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
    auto& [ixcs, lr, tfs] = ixcs_lr_tfs;
#pragma GCC diagnostic pop
    result.fields.emplace_back(csp, csp == ics, lr, tfs);
  }
  return result;
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

  auto index_cs =
    index_column_space(ctx, rt)
    .get_result<index_column_space_result_t>()
    .value();
  args.ics_is = index_cs.column_is;
  TaskLauncher task(partition_rows_task_id, TaskArgument(&args, sizeof(args)));
  RegionRequirement
    req(index_cs.metadata_lr, READ_ONLY, EXCLUSIVE, index_cs.metadata_lr);
  req.add_field(ColumnSpace::AXIS_VECTOR_FID);
  req.add_field(ColumnSpace::AXIS_SET_UID_FID);
  task.add_region_requirement(req);
  auto result = rt->execute_task(ctx, task);
  return result;
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
  std::array<std::pair<int, hyperion::string>, Table::MAX_COLUMNS> index_axes;
  bool allow_rows;
  char columns_buffer[]; // serialized Table::columns_result_t
};

LogicalRegion
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
  size_t rg = 2;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  for (auto& [csp, ixcs, vlr, tfs] : columns.fields) {
#pragma GCC diagnostic pop
    RegionRequirement values_req = task->regions[rg];
    PhysicalRegion values = regions[rg++];
    PhysicalRegion metadata = regions[rg++];
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
    for (auto& [nm, tf] : tfs) {
#pragma GCC diagnostic pop
      ColumnRegions cr;
      cr.values = {values_req.region, values};
      cr.metadata = metadata;
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
      // FIXME: this breaks for a column subset
      for (unsigned i = 0; !offset && i < MAX_COLUMNS; ++i) {
        if (vss.read(i) == vlr && vfs.read(i) == tf.fid)
          offset = i;
      }
      cregions.emplace_back(offset.value(), cr);
    }
  }
  return
    reindexed(
      ctx,
      rt,
      index_axes,
      args->allow_rows,
      task->regions[0].region,
      regions[0],
      regions[1],
      cregions);
}

Future /* LogicalRegion */
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
    reqs.push_back(
      table_fields_requirement(
        fields_lr,
        fields_parent.value_or(fields_lr),
        READ_ONLY));
    {
      auto index_cs =
        index_column_space(ctx, rt)
        .get_result<index_column_space_result_t>()
        .value();
      RegionRequirement
        req(index_cs.metadata_lr, READ_ONLY, EXCLUSIVE, index_cs.metadata_lr);
      req.add_field(ColumnSpace::AXIS_VECTOR_FID);
      reqs.push_back(req);
    }
    {
      std::vector<LogicalRegion> vlrs;
      auto pr = rt->map_region(ctx, reqs[0]);
      Table::columns_result_t columns =
        Table::columns(rt, fields_parent.value_or(fields_lr), pr);
      can_reindex =
        std::none_of(
          columns.fields.begin(),
          columns.fields.end(),
          [](auto& csp_ixcs_vlr_tfs) {
            return std::get<0>(csp_ixcs_vlr_tfs).is_empty();
          });
      if (can_reindex) {
        args_buffer_sz =
          sizeof(ReindexedTaskArgs) + columns.legion_buffer_size();
        args_buffer = std::move(std::make_unique<char[]>(args_buffer_sz));
        args = reinterpret_cast<ReindexedTaskArgs*>(args_buffer.get());
        columns.legion_serialize(args->columns_buffer);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
        for (auto& [csp, ixcs, vlr, tfs] : columns.fields) {
#pragma GCC diagnostic pop
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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
          for (auto& [nm, tf] : tfs) {
#pragma GCC diagnostic pop
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
        std::copy(
          index_axes.begin(),
          index_axes.end(),
          args->index_axes.begin());
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

template <unsigned TO, unsigned FROM>
static IndexSpaceT<TO>
truncate_index_space(Context ctx, Runtime* rt, const IndexSpaceT<FROM>& is) {

  static_assert(TO <= FROM);

  std::vector<Point<TO>> points;
  PointInDomainIterator<FROM> pid(rt->get_index_space_domain(is), false);
  if (pid()) {
    {
      Point<TO> pt;
      for (size_t i = 0; i < TO; ++i)
        pt[i] = pid[i];
      points.push_back(pt);
    }
    pid++;
    for (; pid(); pid++) {
      Point<TO> pt;
      for (size_t i = 0; i < TO; ++i)
        pt[i] = pid[i];
      if (pt != points.back())
        points.push_back(pt);
    }
  }
  return rt->create_index_space(ctx, points);
}

static ColumnSpace
truncate_column_space(
  Context ctx,
  Runtime* rt,
  const ColumnSpace& csp,
  unsigned rank) {

  auto ax = csp.axes(ctx, rt);
  assert(ax.size() == (unsigned)csp.column_is.get_dim());
  IndexSpace truncated_is;
  switch (rank * LEGION_MAX_DIM + ax.size()) {
#define TIS(RANK, CS_RANK)                                    \
    case (RANK * LEGION_MAX_DIM + CS_RANK): {                 \
      IndexSpaceT<CS_RANK> cs_is(csp.column_is);              \
      truncated_is =                                          \
        truncate_index_space<RANK, CS_RANK>(ctx, rt, cs_is);  \
      break;                                                  \
    }
    HYPERION_FOREACH_MN(TIS)
    default:
      assert(false);
      break;
  }
  ax.erase(ax.begin() + rank, ax.end());
  return
    ColumnSpace::create(
      ctx,
      rt,
      ax,
      csp.axes_uid(ctx, rt),
      truncated_is,
      false);
}

LogicalRegion
Table::reindexed(
  Context ctx,
  Runtime *rt,
  const std::vector<std::pair<int, std::string>>& index_axes,
  bool allow_rows,
  const LogicalRegion& fields_parent,
  const PhysicalRegion& fields_pr,
  const PhysicalRegion& index_cs_md_pr,
  const std::vector<std::tuple<coord_t, ColumnRegions>>& column_regions) {

  const ColumnSpace::AxisVectorAccessor<READ_ONLY>
    ax(index_cs_md_pr, ColumnSpace::AXIS_VECTOR_FID);
  std::vector<int> ixax = ColumnSpace::from_axis_vector(ax[0]);

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
      return LogicalRegion::NO_REGION;
    }
  }

  // construct map to associate column name with various Column fields provided
  // by the Table instance and the column_regions
  std::unordered_map<
    std::string,
    std::tuple<Column, ColumnRegions, std::optional<int>>> named_columns;
  {
    auto cols = Table::column_map(Table::columns(rt, fields_parent, fields_pr));
    const NameAccessor<READ_ONLY>
      nms(fields_pr, static_cast<FieldID>(TableFieldsFid::NM));
    for (auto& [i, cr] : column_regions) {
      auto nms_i = nms.read(i);
      auto& col = cols[nms_i];
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
      named_columns[nms_i] = {col1, cr, std::nullopt};
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
      return LogicalRegion::NO_REGION;
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
      index_cols[d] = {lr, pr};
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
  Table result_tbl;
  {
    bool have_index_column_space = false;
    std::vector<int> new_index_axes;
    new_index_axes.reserve(index_axes.size() + ((allow_rows ? 1 : 0)));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
    for (auto& [d, nm] : index_axes)
#pragma GCC diagnostic pop
      new_index_axes.push_back(d);
    if (allow_rows)
      new_index_axes.push_back(0);
    unsigned column_spaces_min_rank = 2 * LEGION_MAX_DIM;
    ColumnSpace min_rank_column_space;
    std::map<
      ColumnSpace,
      std::tuple<bool, std::vector<std::pair<std::string, TableField>>>>
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
          (okwrs ? Keywords::clone(ctx, rt, okwrs.value()) : Keywords())
#ifdef HYPERION_USE_CASACORE
          , (odrs ? MeasRef::clone(ctx, rt, odrs.value()) : MeasRef())
          , col.rc
#endif
          );
        if (ix || ifl[0]) {
          ColumnSpace icsp;
          if (ix)
            icsp =
              ColumnSpace::create(
                ctx,
                rt,
                {ix.value()},
                auid[0],
                // NB: take ownership of index space
                index_cols[ix.value()].first.get_index_space(),
                true);
          else
            icsp =
              ColumnSpace::create(
                ctx,
                rt,
                {av[0][0]},
                auid[0],
                rt->create_index_space(
                  ctx,
                  rt->get_index_space_domain(col.csp.column_is)),
                true);
          nmtfs[icsp] = {false, {{nm, tf}}};
        } else {
          tfs.emplace_back(nm, tf);
        }
      }
      if (reindexed.count(csp) > 0) {
        auto& rcsp = std::get<0>(reindexed[csp]);
        const auto ax = rcsp.axes(ctx, rt);
        bool ixcs = ax == new_index_axes;
        have_index_column_space = have_index_column_space || ixcs;
        if (ax.size() < column_spaces_min_rank) {
          column_spaces_min_rank = ax.size();
          min_rank_column_space = rcsp;
        }
        nmtfs[rcsp] = {ixcs, tfs};
      }
    }
    if (!have_index_column_space) {
      if (!min_rank_column_space.is_valid()) {
        // This case really can't occur, as it would imply that the index
        // columns completely index every column in the table, which would mean
        // that only index columns exist, and thus no "row" index. FIXME: We
        // should log a warning, but also just invent an index column space,
        // instead of generating an error.
        assert(false);
      } else {
        auto ics =
          truncate_column_space(
            ctx,
            rt,
            min_rank_column_space,
            new_index_axes.size());
        nmtfs[ics] = {true, {}};
      }
    }
    std::vector<
      std::tuple<
        ColumnSpace,
        bool,
        std::vector<std::pair<std::string, TableField>>>> cols;
    for (auto& [csp, ixcs_tfs] : nmtfs) {
      auto& [ixcs, tfs] = ixcs_tfs;
      cols.emplace_back(csp, ixcs, tfs);
    }
    result_tbl = Table::create(ctx, rt, cols);
  }

  // copy values from old table to new
  {
    const unsigned min_block_size = 1000000;
    CopyLauncher index_column_copier;
    auto dflds = result_tbl.columns(ctx, rt).get_result<columns_result_t>();
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
    for (auto& [dcsp, dixcs, dvlr, dtfs] : dflds.fields) {
#pragma GCC diagnostic pop
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
          slr = std::get<0>(crg.values);
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
          slr = col.vlr;
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
  return result_tbl.fields_lr;
}

TaskID Table::reindex_copy_values_task_id;

const char* Table::reindex_copy_values_task_name =
  "Table::reindex_copy_values_task";

// FIXME: use GenericAccessor rather than AffineAccessor, or at least leave it
// as a parameter
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
    // index_column_space_task
    index_column_space_task_id = Runtime::generate_static_task_id();
    TaskVariantRegistrar registrar(
      index_column_space_task_id,
      index_column_space_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    registrar.set_leaf();
    Runtime::preregister_task_variant<
      index_column_space_result_t,
      index_column_space_task>(
      registrar,
      index_column_space_task_name);
  }
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
    Runtime::preregister_task_variant<bool, add_columns_task>(
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
    Runtime::preregister_task_variant<LogicalRegion, reindexed_task>(
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
