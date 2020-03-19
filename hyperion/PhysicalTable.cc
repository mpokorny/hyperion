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

std::optional<
  std::tuple<
    PhysicalTable,
    std::vector<RegionRequirement>::const_iterator,
    std::vector<PhysicalRegion>::const_iterator>>
PhysicalTable::create(
  Legion::Runtime *rt,
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
    std::tuple<LogicalRegion, PhysicalRegion>> value_regions;

  unsigned idx_rank = index_rank(rt, table_parent, table_pr);
  for (PointInDomainIterator<1> pid(
         rt->get_index_space_domain(table_parent.get_index_space()));
       pid();
       pid++) {

    auto css_pid = css.read(*pid);
    if (css_pid.is_empty())
      break;
    if (md_regions.count(css_pid) == 0) {
      if (reqs == reqs_end || prs == prs_end)
        return result;
      md_regions[css_pid] = *prs;
      ++reqs;
      ++prs;
    }
    auto& metadata = md_regions.at(css_pid);
    LogicalRegion parent;
    std::optional<PhysicalRegion> values;
    std::optional<Keywords::pair<PhysicalRegion>> kw_prs;
    std::optional<MeasRef::DataRegions> mr_drs;
    auto nms_pid = nms.read(*pid);
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
      ppc->update_refcol(rt, std::make_tuple(rc, columns.at(rc)));
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
    column_parents[nm] = ppc->m_parent;
  return
    Table(m_table_pr.get_logical_region(), m_table_parent, column_parents);
}

std::optional<std::shared_ptr<PhysicalColumn>>
PhysicalTable::column(const std::string& name) const {
  std::optional<std::shared_ptr<PhysicalColumn>> result;
  if (m_columns.count(name) > 0)
    result = m_columns.at(name);
  return result;
}

std::optional<Point<1>>
PhysicalTable::index_column_space(Legion::Runtime* rt) const {
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
  const std::map<
    std::string,
    std::optional<
      std::tuple<bool, PrivilegeMode, CoherenceProperty>>>& column_modes,
  bool columns_mapped,
  PrivilegeMode columns_privilege,
  CoherenceProperty columns_coherence) const {

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
      column_modes,
      columns_mapped,
      columns_privilege,
      columns_coherence);
}

Table::columns_result_t
PhysicalTable::columns(Runtime *rt) const {
  return Table::columns(rt, m_table_parent, m_table_pr);
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
  std::vector<std::optional<PhysicalRegion>> val_prs;
  std::vector<PhysicalRegion> cs_md_prs;
  for (auto& [nm, ppc] : m_columns) {
    auto md_lr = ppc->m_metadata.get_logical_region();
    if (cs_idxs.count(md_lr) == 0) {
      auto idx = cs_md_prs.size();
      cs_idxs[md_lr] = idx;
      cs_md_prs.push_back(ppc->m_metadata);
      val_lrs.push_back(
        map(
          ppc->m_values,
          [](const auto& pr) { return pr.get_logical_region(); })
        .value_or(LogicalRegion::NO_REGION));
      val_prs.push_back(ppc->m_values);
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
    for (auto& [cs, ixcs, vlr, nm_tfs] : columns(rt).fields) {
      std::vector<FieldID> new_fields;
      for (auto& [nm, tf] : nm_tfs)
        if (m_columns.count(nm) == 0)
          new_fields.push_back(tf.fid);
      if (new_fields.size() > 0) {
        std::optional<PhysicalRegion> vpr;
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
                std::nullopt
#ifdef HYPERION_USE_CASACORE
                , std::nullopt
                , std::nullopt
#endif
                ));
          }
        }
      }
    }
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
  for (auto& [cs, ixcs, vlr, nm_tfs] : PhysicalTable::columns(rt).fields) {
    for (auto& [nm, tf] : nm_tfs) {
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

  std::vector<std::tuple<Legion::coord_t, Table::ColumnRegions>> cregions;
  for (PointInDomainIterator<1> pid(
         rt->get_index_space_domain(m_table_parent.get_index_space()));
       pid();
       pid++) {
    if (css.read(*pid).is_empty())
      break;
    auto& ppc = m_columns.at(nms.read(*pid));
    if (ppc->m_values) {
      Table::ColumnRegions cr;
      cr.values = {ppc->m_parent, ppc->m_values.value()};
      cr.metadata = ppc->m_metadata;
      if (ppc->m_kws) {
        cr.kw_type_tags = ppc->m_kws.value().type_tags;
        cr.kw_values = ppc->m_kws.value().values;
      }
      if (ppc->m_mr_drs) {
        cr.mr_metadata = ppc->m_mr_drs.value().metadata;
        cr.mr_values = ppc->m_mr_drs.value().values;
        cr.mr_index = ppc->m_mr_drs.value().index;
      }
      cregions.emplace_back(*pid, cr);
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

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
