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
#include <hyperion/PhysicalTable.h>
#include <hyperion/Keywords.h>

using namespace hyperion;
using namespace Legion;

PhysicalTable::PhysicalTable(
  LogicalRegion table_parent,
  PhysicalRegion table_pr,
  const std::unordered_map<std::string, std::shared_ptr<PhysicalColumn>>&
  columns)
  : m_table_parent(table_parent)
  , m_table_pr(table_pr)
  , m_columns(columns) {}

PhysicalTable::PhysicalTable(const PhysicalTable& other)
  : PhysicalTable(other.m_table_parent, other.m_table_pr, other.m_columns) {
  m_attached = other.m_attached;
}

PhysicalTable::PhysicalTable(PhysicalTable&& other)
  : PhysicalTable(
    std::move(other).m_table_parent,
    std::move(other).m_table_pr,
    std::move(other).m_columns) {
  m_attached = std::move(other).m_attached;
}

std::optional<
  std::tuple<
    PhysicalTable,
    std::vector<RegionRequirement>::const_iterator,
    std::vector<PhysicalRegion>::const_iterator>>
PhysicalTable::create(
  Runtime *rt,
  const std::vector<RegionRequirement>::const_iterator& reqs_begin,
  const std::vector<RegionRequirement>::const_iterator& reqs_end,
  const std::vector<PhysicalRegion>::const_iterator& prs_begin,
  const std::vector<PhysicalRegion>::const_iterator& prs_end) {

  std::optional<
    std::tuple<
      PhysicalTable,
      std::vector<RegionRequirement>::const_iterator,
      std::vector<PhysicalRegion>::const_iterator>> result;

  if (reqs_begin == reqs_end || prs_begin == prs_end)
    return result;

  std::vector<RegionRequirement>::const_iterator reqs = reqs_begin;
  std::vector<PhysicalRegion>::const_iterator prs = prs_begin;

  LogicalRegion table_parent = reqs->region;
  ++reqs;
  PhysicalRegion table_pr = *prs++;

  const Table::NameAccessor<READ_ONLY>
    nms(table_pr, static_cast<FieldID>(TableFieldsFid::NM));
  const Table::DatatypeAccessor<READ_ONLY>
    dts(table_pr, static_cast<FieldID>(TableFieldsFid::DT));
  const Table::KeywordsAccessor<READ_ONLY>
    kws(table_pr, static_cast<FieldID>(TableFieldsFid::KW));
#ifdef HYPERION_USE_CASACORE
  const Table::MeasRefAccessor<READ_ONLY>
    mrs(table_pr, static_cast<FieldID>(TableFieldsFid::MR));
  const Table::RefColumnAccessor<READ_ONLY>
    rcs(table_pr, static_cast<FieldID>(TableFieldsFid::RC));
#endif
  const Table::ColumnSpaceAccessor<READ_ONLY>
    css(table_pr, static_cast<FieldID>(TableFieldsFid::CS));
  const Table::ValueFidAccessor<READ_ONLY>
    vfs(table_pr, static_cast<FieldID>(TableFieldsFid::VF));
  const Table::ValuesAccessor<READ_ONLY>
    vss(table_pr, static_cast<FieldID>(TableFieldsFid::VS));

  std::unordered_map<std::string, std::shared_ptr<PhysicalColumn>> columns;
  std::unordered_map<std::string, std::string> refcols;

  std::map<ColumnSpace, PhysicalRegion> md_regions;
  std::map<
    std::tuple<FieldID, ColumnSpace>,
    std::tuple<LogicalRegion, std::variant<PhysicalRegion, LogicalRegion>>>
    value_regions;

  unsigned idx_rank = index_rank(rt, table_parent, table_pr);
  for (PointInDomainIterator<1> pid(
         rt->get_index_space_domain(table_parent.get_index_space()));
       pid();
       pid++) {

    auto css_pid = css.read(*pid);
    if (css_pid.is_empty())
      break;
    auto nms_pid = nms.read(*pid);
    if (md_regions.count(css_pid) == 0) {
      if (reqs == reqs_end || prs == prs_end)
        return result;
      md_regions[css_pid] = *prs;
      ++reqs;
      ++prs;
    }
    auto vss_pid = vss.read(*pid);
    auto& metadata = md_regions.at(css_pid);
    LogicalRegion parent;
    std::variant<PhysicalRegion, LogicalRegion> values = vss_pid;
    std::optional<Keywords::pair<PhysicalRegion>> kw_prs;
    std::optional<MeasRef::DataRegions> mr_drs;
    auto vfs_pid = vfs.read(*pid);
    if (vfs_pid != Table::no_column) {
      std::tuple<FieldID, ColumnSpace> fid_cs = {vfs_pid, css_pid};
      if (value_regions.count(fid_cs) == 0) {
        if (reqs == reqs_end || prs == prs_end)
          return result;
        auto& cs = std::get<1>(fid_cs);
        for (auto& fid : reqs->privilege_fields)
          value_regions[{fid, cs}] = {reqs->region, *prs};
        ++reqs;
        ++prs;
      }
      std::tie(parent, values) = value_regions.at(fid_cs);
      auto kws_pid = kws.read(*pid);
      if (!kws_pid.is_empty()) {
        Keywords::pair<PhysicalRegion> kwpair;
        if (reqs == reqs_end || prs == prs_end)
          return result;
        ++reqs;
        kwpair.type_tags = *prs++;
        if (reqs == reqs_end || prs == prs_end)
          return result;
        ++reqs;
        kwpair.values = *prs++;
        kw_prs = kwpair;
      }
#ifdef HYPERION_USE_CASACORE
      auto mrs_pid = mrs.read(*pid);
      if (!mrs_pid.is_empty()) {
        MeasRef::DataRegions drs;
        if (reqs == reqs_end || prs == prs_end)
          return result;
        ++reqs;
        drs.metadata = *prs++;
        if (reqs == reqs_end || prs == prs_end)
          return result;
        ++reqs;
        drs.values = *prs++;
        if (mrs_pid.index_lr != LogicalRegion::NO_REGION) {
          if (reqs == reqs_end || prs == prs_end)
            return result;
          ++reqs;
          drs.index = *prs++;
        }
        mr_drs = drs;
        auto rcs_pid = rcs.read(*pid);
        if (rcs_pid.size() > 0)
          refcols[nms_pid] = rcs_pid;
      }
#endif
    } else {
      parent = vss_pid;
    }
    auto dts_pid = dts.read(*pid);
    columns.emplace(
      nms_pid,
      std::make_shared<PhysicalColumn>(
        rt,
        dts_pid,
        vfs_pid,
        idx_rank,
        metadata,
        parent,
        values,
        kw_prs
#ifdef HYPERION_USE_CASACORE
        , mr_drs
        , std::nullopt
#endif
        ));
  }
#ifdef HYPERION_USE_CASACORE
  for (auto& [nm, ppc] : columns) {
    if (refcols.count(nm) > 0) {
      auto& rc = refcols[nm];
      ppc->set_refcol(rc, columns.at(rc));
    }
  }
#endif
  return
    std::make_tuple(PhysicalTable(table_parent, table_pr, columns), reqs, prs);
}

Table
PhysicalTable::table() const {
  std::unordered_map<std::string, LogicalRegion> column_parents;
  for (auto& [nm, ppc] : m_columns)
    column_parents[nm] = ppc->parent();
  return
    Table(m_table_pr.get_logical_region(), m_table_parent, column_parents);
}

std::optional<ColumnSpace::AXIS_SET_UID_TYPE>
PhysicalTable::axes_uid() const {
  std::optional<ColumnSpace::AXIS_SET_UID_TYPE> result;
  if (m_columns.size() > 0) {
    const ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
      au(
        std::get<1>(*m_columns.begin())->m_metadata,
        ColumnSpace::AXIS_SET_UID_FID);
    result = au[0];
  }
  return result;
}

std::optional<std::shared_ptr<PhysicalColumn>>
PhysicalTable::column(const std::string& name) const {
  std::optional<std::shared_ptr<PhysicalColumn>> result;
  if (m_columns.count(name) > 0)
    result = m_columns.at(name);
  return result;
}

std::optional<Point<1>>
PhysicalTable::index_column_space(Runtime* rt) const {
  return index_column_space(rt, m_table_parent, m_table_pr);
}

std::optional<Point<1>>
PhysicalTable::index_column_space(
  Runtime* rt,
  const LogicalRegion& parent,
  const PhysicalRegion& pr) {

  std::optional<Point<1>> result;
  const Table::ColumnSpaceAccessor<READ_ONLY>
    css(pr, static_cast<FieldID>(TableFieldsFid::CS));
  const Table::ValueFidAccessor<READ_ONLY>
    vfs(pr, static_cast<FieldID>(TableFieldsFid::VF));
  for (PointInDomainIterator<1> pid(
         rt->get_index_space_domain(parent.get_index_space()));
       pid() && !result;
       pid++) {
    if (css.read(*pid).is_empty())
      break;
    if (vfs.read(*pid) == Table::no_column)
      result = *pid;
  }
  return result;
}

std::optional<std::shared_ptr<PhysicalColumn>>
PhysicalTable::index_column(Runtime* rt) const {
  return
    map(
      index_column_space(rt),
      [this](const auto& ics) {
        const Table::NameAccessor<READ_ONLY>
          nms(m_table_pr, static_cast<FieldID>(TableFieldsFid::NM));
        return m_columns.at(nms.read(ics));
      });
}

unsigned
PhysicalTable::index_rank(Runtime* rt) const {
  return index_rank(rt, m_table_parent, m_table_pr);
}

unsigned
PhysicalTable::index_rank(
  Runtime* rt,
  const LogicalRegion& parent,
  const PhysicalRegion& pr) {
  unsigned result = 0;
  auto ics = index_column_space(rt, parent, pr);
  if (ics) {
    const Table::ColumnSpaceAccessor<READ_ONLY>
      css(pr, static_cast<FieldID>(TableFieldsFid::CS));
    result = (unsigned)css.read(ics.value()).column_is.get_dim();
  }
  return result;
}

bool
PhysicalTable::is_conformant(
  Runtime* rt,
  const IndexSpace& cs_is,
  const PhysicalRegion& cs_md_pr) const {

  auto icsp = index_column_space(rt);
  std::optional<std::tuple<IndexSpace, PhysicalRegion>> ics;
  const Table::ColumnSpaceAccessor<READ_ONLY>
    css(m_table_pr, static_cast<FieldID>(TableFieldsFid::CS));
  const Table::NameAccessor<READ_ONLY>
    nms(m_table_pr, static_cast<FieldID>(TableFieldsFid::NM));
  if (icsp)
    ics =
      std::make_tuple(
        css.read(icsp.value()).column_is,
        m_columns.at(nms.read(icsp.value()))->m_metadata);
  return
    Table::is_conformant(
      rt,
      m_table_parent,
      m_table_pr,
      ics,
      cs_is,
      cs_md_pr);
}

std::tuple<std::vector<RegionRequirement>, std::vector<LogicalPartition>>
PhysicalTable::requirements(
  Context ctx,
  Runtime* rt,
  const ColumnSpacePartition& table_partition,
  PrivilegeMode table_privilege,
  const std::map<std::string, std::optional<Column::Requirements>>&
    column_requirements,
  const std::optional<Column::Requirements>&
    default_column_requirements) const {

  std::unordered_map<std::string, LogicalRegion> column_parents;
  for (auto& [nm, ppc] : m_columns)
    column_parents[nm] = ppc->m_parent;

  return
    Table::requirements(
      ctx,
      rt,
      m_table_parent,
      m_table_pr,
      column_parents,
      table_partition,
      table_privilege,
      column_requirements,
      default_column_requirements);
}

decltype(Table::columns_result_t::fields)
PhysicalTable::column_fields(Runtime *rt) const {
  return Table::columns(rt, m_table_parent, m_table_pr).fields;
}

bool
PhysicalTable::add_columns(
  Context ctx,
  Runtime* rt,
  const std::vector<
    std::tuple<
      ColumnSpace,
      bool,
      std::vector<std::pair<std::string, TableField>>>>& cols) {

  std::optional<std::tuple<IndexSpace, PhysicalRegion>> index_cs =
    map(
      index_column(rt),
      [](const auto& ppc) {
        return
          std::make_tuple(ppc->m_parent.get_index_space(), ppc->m_metadata);
      });

  std::vector<
    std::tuple<
      ColumnSpace,
      bool,
      size_t,
      std::vector<std::pair<hyperion::string, TableField>>>> indexed_cols;
  std::map<LogicalRegion, size_t> cs_idxs; // metadata_lr
  std::vector<LogicalRegion> val_lrs;
  std::vector<PhysicalRegion> cs_md_prs;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  for (auto& [nm, ppc] : m_columns) {
#pragma GCC diagnostic pop
    auto md_lr = ppc->m_metadata.get_logical_region();
    if (cs_idxs.count(md_lr) == 0) {
      auto idx = cs_md_prs.size();
      cs_idxs[md_lr] = idx;
      cs_md_prs.push_back(ppc->m_metadata);
      val_lrs.push_back(
        std::visit(overloaded {
            [](const PhysicalRegion& pr) { return pr.get_logical_region(); },
            [](const LogicalRegion& lr) { return lr; }
          },
          ppc->values()));
    }
  }
  for (auto& [cs, ixcs, nm_tfs] : cols) {
    if (cs_idxs.count(cs.metadata_lr) == 0) {
      cs_idxs[cs.metadata_lr] = cs_md_prs.size();
      RegionRequirement
        req(cs.metadata_lr, READ_ONLY, EXCLUSIVE, cs.metadata_lr);
      req.add_field(ColumnSpace::AXIS_VECTOR_FID);
      req.add_field(ColumnSpace::AXIS_SET_UID_FID);
      req.add_field(ColumnSpace::INDEX_FLAG_FID);
      cs_md_prs.push_back(rt->map_region(ctx, req));
    }
    std::vector<std::pair<hyperion::string, TableField>> hnm_tfs;
    for (auto& [nm, tf] : nm_tfs)
      hnm_tfs.emplace_back(nm, tf);
    indexed_cols.emplace_back(cs, ixcs, cs_idxs[cs.metadata_lr], hnm_tfs);
  }
  bool result =
    Table::add_columns(
      ctx,
      rt,
      indexed_cols,
      val_lrs,
      m_table_parent,
      m_table_pr,
      index_cs,
      cs_md_prs);

  if (result) {
    // create (unmapped) PhysicalColumns for added columns
    unsigned idx_rank = index_rank(rt);
    std::unordered_map<std::string, std::string> refcols;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
    for (auto& [cs, ixcs, vlr, nm_tfs] : column_fields(rt)) {
#pragma GCC diagnostic pop
      std::vector<FieldID> new_fields;
      for (auto& [nm, tf] : nm_tfs)
        if (m_columns.count(nm) == 0)
          new_fields.push_back(tf.fid);
      if (new_fields.size() > 0) {
        std::variant<PhysicalRegion, LogicalRegion> vpr = vlr;
        auto nocol =
          std::find(new_fields.begin(), new_fields.end(), Table::no_column);
        if (nocol != new_fields.end())
          new_fields.erase(nocol);
        if (new_fields.size() > 0) {
          RegionRequirement req(vlr, READ_WRITE, EXCLUSIVE, vlr);
          req.add_fields(new_fields, false);
          vpr = rt->map_region(ctx, req);
        }
        auto& md_pr = cs_md_prs[cs_idxs[cs.metadata_lr]];
        for (auto& [nm, tf] : nm_tfs) {
          // create kws for Keywords
          std::optional<Keywords::pair<PhysicalRegion>> kws;
          if (!tf.kw.is_empty()) {
            std::vector<FieldID> fids;
            fids.resize(tf.kw.size(rt));
            std::iota(fids.begin(), fids.end(), 0);
            auto reqs = tf.kw.requirements(rt, fids, READ_WRITE, false).value();
            auto prs =
              reqs.map([&ctx, rt](const auto& rq) { return rt->map_region(ctx, rq); });
            kws = prs;
          }
          // create mr_drs for MeasRef
          std::optional<MeasRef::DataRegions> mr_drs;
          if (!tf.mr.is_empty()) {
            auto [mrq, vrq, oirq] = tf.mr.requirements(READ_WRITE, false);
            MeasRef::DataRegions prs;
            prs.metadata = rt->map_region(ctx, mrq);
            prs.values = rt->map_region(ctx, vrq);
            if (oirq)
              prs.index = rt->map_region(ctx, oirq.value());
            mr_drs = prs;
          }
          if (tf.rc)
            refcols[nm] = tf.rc.value();
          if (m_columns.count(nm) == 0) {
            m_columns.emplace(
              nm,
              std::make_shared<PhysicalColumn>(
                rt,
                tf.dt,
                tf.fid,
                idx_rank,
                md_pr,
                vlr,
                vpr,
                kws
#ifdef HYPERION_USE_CASACORE
                , mr_drs
                , std::nullopt
#endif
                ));
          }
        }
      }
    }
    for (auto& [c, rc] : refcols)
      m_columns[c]->set_refcol(rc, m_columns[rc]);
  }
  return result;
}

bool
PhysicalTable::remove_columns(
  Context ctx,
  Runtime* rt,
  const std::unordered_set<std::string>& cols,
  bool destroy_orphan_column_spaces,
  bool destroy_field_data) {

  std::vector<ColumnSpace> css;
  std::vector<PhysicalRegion> cs_md_prs;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  for (auto& [cs, ixcs, vlr, nm_tfs] : column_fields(rt)) {
    for (auto& [nm, tf] : nm_tfs) {
#pragma GCC diagnostic pop
      if (cols.count(nm) > 0) {
        css.push_back(cs);
        cs_md_prs.push_back(m_columns.at(nm)->m_metadata);
        break;
      }
    }
  }
  std::set<hyperion::string> hcols;
  for (auto& c : cols)
    hcols.insert(c);

  return
    Table::remove_columns(
      ctx,
      rt,
      hcols,
      destroy_orphan_column_spaces,
      destroy_field_data,
      m_table_parent,
      m_table_pr,
      css,
      cs_md_prs);
}

template <typename F>
static void
for_all_column_regions(
  const std::shared_ptr<PhysicalColumn>& column,
  std::set<PhysicalRegion>& done,
  F fn) {

  auto md = column->metadata();
  if (done.count(md) == 0) {
    fn(md);
    done.insert(md);
  }
  if (std::holds_alternative<PhysicalRegion>(column->values())) {
    auto& v = std::get<PhysicalRegion>(column->values());
    if (done.count(v) == 0) {
      fn(v);
      done.insert(v);
    }
  }
  if (column->kws()) {
    auto& kws = column->kws().value();
    fn(kws.type_tags);
    done.insert(kws.type_tags);
    fn(kws.values);
    done.insert(kws.values);
  }
#ifdef HYPERION_USE_CASACORE
  if (column->mr_drs()) {
    auto& dr = column->mr_drs().value();
    fn(dr.metadata);
    done.insert(dr.metadata);
    fn(dr.values);
    done.insert(dr.values);
    if (dr.index) {
      fn(dr.index.value());
      done.insert(dr.index.value());
    }
  }
  if (column->refcol()) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
    auto& [nm, prc] = column->refcol().value();
#pragma GCC diagnostic pop
    for_all_column_regions(prc, done, fn);
  }
#endif
}

void
PhysicalTable::unmap_regions(Context ctx, Runtime* rt) const {

  std::set<PhysicalRegion> unmapped;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  for (auto& [nm, pc] : columns())
#pragma GCC diagnostic pop
    for_all_column_regions(
      pc,
      unmapped,
      [&ctx, rt](auto& pr) { rt->unmap_region(ctx, pr); });
  rt->unmap_region(ctx, m_table_pr);
}

void
PhysicalTable::remap_regions(Context ctx, Runtime* rt) const {

  std::set<PhysicalRegion> remapped;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  for (auto& [nm, pc] : columns())
#pragma GCC diagnostic pop
    for_all_column_regions(
      pc,
      remapped,
      [&ctx, rt](auto& pr) { rt->remap_region(ctx, pr); });
  rt->remap_region(ctx, m_table_pr);
}

ColumnSpacePartition
PhysicalTable::partition_rows(
  Context ctx,
  Runtime* rt,
  const std::vector<std::optional<size_t>>& block_sizes) const {

  auto ic = index_column(rt).value();
  return
    Table::partition_rows(
      ctx,
      rt,
      block_sizes,
      ic->parent().get_index_space(),
      ic->metadata());
}

LogicalRegion
PhysicalTable::reindexed(
  Context ctx,
  Runtime* rt,
  const std::vector<std::pair<int, std::string>>& index_axes,
  bool allow_rows) const {

  auto oic = index_column(rt);
  if (!oic)
    return LogicalRegion::NO_REGION;
  auto ic = oic.value();

  const Table::ColumnSpaceAccessor<READ_ONLY>
    css(m_table_pr, static_cast<FieldID>(TableFieldsFid::CS));
  const Table::NameAccessor<READ_ONLY>
    nms(m_table_pr, static_cast<FieldID>(TableFieldsFid::NM));

  std::vector<std::tuple<coord_t, Table::ColumnRegions>> cregions;
  for (PointInDomainIterator<1> pid(
         rt->get_index_space_domain(m_table_parent.get_index_space()));
       pid();
       pid++) {
    if (css.read(*pid).is_empty())
      break;
    auto& ppc = m_columns.at(nms.read(*pid));
    if (std::holds_alternative<PhysicalRegion>(ppc->values())) {
      Table::ColumnRegions cr;
      cr.values = {ppc->parent(), std::get<PhysicalRegion>(ppc->values())};
      cr.metadata = ppc->metadata();
      if (ppc->kws()) {
        auto& kws = ppc->kws().value();
        cr.kw_type_tags = kws.type_tags;
        cr.kw_values = kws.values;
      }
      if (ppc->mr_drs()) {
        auto& mr_drs = ppc->mr_drs().value();
        cr.mr_metadata = mr_drs.metadata;
        cr.mr_values = mr_drs.values;
        cr.mr_index = mr_drs.index;
      }
      cregions.emplace_back(*pid, cr);
    } else {
      // the column values have not been mapped, which is an error unless this
      // column has no values (e.g, it's the index column)
      assert(ppc->fid() == Table::no_column);
    }
  }
  return
    Table::reindexed(
      ctx,
      rt,
      index_axes,
      allow_rows,
      m_table_parent,
      m_table_pr,
      ic->m_metadata,
      cregions);
}

bool
PhysicalTable::attach_columns(
  Context ctx,
  Runtime* rt,
  const CXX_FILESYSTEM_NAMESPACE::path& file_path,
  const std::unordered_map<std::string, std::string>& column_paths,
  const std::unordered_map<std::string, std::tuple<bool, bool, bool>>&
  column_modes) {

  std::map<
    std::tuple<LogicalRegion, std::tuple<bool, bool, bool>>,
    std::vector<std::tuple<FieldID, std::string>>>
    regions;
  for (auto& [nm, pc] : m_columns) {
    if (column_paths.count(nm) > 0 && pc->fid() != Table::no_column) {
      if (column_modes.count(nm) == 0) {
        // FIXME: log warning message: missing column path and/or mode
        return false;
      }
      if (m_attached.count(nm) > 0) {
        // FIXME: log warning message: column is already attached; this could
        // perhaps be relaxed if the attachment parameters were maintained
        return false;
      }
      std::tuple<LogicalRegion, std::tuple<bool, bool, bool>> key =
        {pc->parent(), column_modes.at(nm)};
      if (regions.count(key) == 0)
        regions[key] = std::vector<std::tuple<FieldID, std::string>>();
      regions[key].emplace_back(pc->fid(), nm);
    }
  }
  for (auto& [parent_modes, fid_nms] : regions) {
    std::map<FieldID, const char*> field_map;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
    for (auto& [fid, nm] : fid_nms)
#pragma GCC diagnostic pop
      field_map[fid] = column_paths.at(nm).c_str();
    auto& [parent, modes] = parent_modes;
    auto& [read_only, restricted, mapped] = modes;
    LogicalRegion lr =
      std::visit(overloaded {
          [&ctx, rt](const PhysicalRegion& pr) {
            auto result = pr.get_logical_region();
            rt->unmap_region(ctx, pr);
            return result;
          },
          [](const LogicalRegion& lr) {
            return lr;
          }
        },
        m_columns.at(std::get<1>(fid_nms[0]))->values());
    AttachLauncher attach(EXTERNAL_HDF5_FILE, lr, parent, restricted, mapped);
    attach.attach_hdf5(
      file_path.c_str(),
      field_map,
      read_only ? LEGION_FILE_READ_ONLY : LEGION_FILE_READ_WRITE);
    auto pr1 = rt->attach_external_resource(ctx, attach);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
    for (auto& [fid, nm] : fid_nms)
#pragma GCC diagnostic pop
      m_attached[nm] = pr1;
  }
  return true;
}

void
PhysicalTable::detach_columns(
  Context ctx,
  Runtime* rt,
  const std::unordered_set<std::string>& columns) {

  // TODO: we must detach all columns sharing the PhysicalRegion, should there
  // be an error when not all such columns are named in "columns"?
  std::set<PhysicalRegion> detached;
  for (auto& col : columns) {
    if (m_attached.count(col) > 0) {
      auto pr = m_attached.at(col);
      if (detached.count(pr) == 0) {
        rt->detach_external_resource(ctx, pr);
        detached.insert(pr);
      }
    }
  }
  for (auto& [nm, pr] : m_attached)
    if (detached.count(pr) > 0)
      m_attached.erase(nm);
}

void
PhysicalTable::acquire_columns(Context ctx, Runtime* rt) {

}

void
PhysicalTable::release_columns(Context ctx, Runtime* rt) {

}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
