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
  std::unordered_map<std::string, PhysicalColumn> columns)
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
  // Maintain PhysicalColumns in a vector for now, since any sort of map would
  // require, below, a trivial constructor for PhysicalColumn. We'll convert it
  // to an unordered_map a bit later.
  std::vector<std::pair<std::string, PhysicalColumn>> colv;

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

  std::unordered_map<
    std::string,
    std::tuple<
      std::vector<std::shared_ptr<casacore::MRBase>>,
      std::unordered_map<unsigned, unsigned>,
      std::optional<std::string>>> partial_mrbs;

  std::map<ColumnSpace, PhysicalRegion> md_regions;
  std::map<
    std::tuple<FieldID, ColumnSpace>,
    std::tuple<LogicalRegion, PhysicalRegion>> value_regions;

  unsigned idx_rank = index_rank(rt, table_parent, table_pr);
  for (PointInDomainIterator<1> pid(
         rt->get_index_space_domain(table_parent.get_index_space()));
       pid() && !css[*pid].is_empty();
       pid++) {

    if (md_regions.count(css[*pid]) == 0) {
      if (reqs == reqs_end || prs == prs_end)
        return result;
      md_regions[css[*pid]] = *prs;
      ++reqs;
      ++prs;
    }
    auto& metadata = md_regions.at(css[*pid]);
    LogicalRegion parent;
    std::optional<PhysicalRegion> values;
    std::unordered_map<std::string, std::any> kwmap;
    assert((vss[*pid] == LogicalRegion::NO_REGION)
           == (vfs[*pid] == Table::no_column));
    if (vss[*pid] != LogicalRegion::NO_REGION) {
      std::tuple<FieldID, ColumnSpace> fid_cs = {vfs[*pid], css[*pid]};
      if (value_regions.count(fid_cs) == 0) {
        if (reqs == reqs_end || prs == prs_end)
          return result;
        auto& cs = std::get<1>(fid_cs);
        for (auto& fid : reqs->privilege_fields) {
          value_regions[{fid, cs}] = {reqs->region, *prs};
        }
        ++reqs;
        ++prs;
      }
      std::tie(parent, values.value()) = value_regions.at(fid_cs);
      if (!kws[*pid].is_empty()) {
        Keywords::pair<PhysicalRegion> kwprs;
        if (reqs == reqs_end || prs == prs_end)
          return result;
        ++reqs;
        kwprs.type_tags = *prs++;
        if (reqs == reqs_end || prs == prs_end)
          return result;
        ++reqs;
        kwprs.values = *prs++;
        kwmap = Keywords::to_map(rt, kwprs);
      }
#ifdef HYPERION_USE_CASACORE
      if (!mrs[*pid].is_empty()) {
        MeasRef::DataRegions drs;
        if (reqs == reqs_end || prs == prs_end)
          return result;
        ++reqs;
        drs.metadata = *prs++;
        if (reqs == reqs_end || prs == prs_end)
          return result;
        ++reqs;
        drs.values = *prs++;
        if (mrs[*pid].index_lr != LogicalRegion::NO_REGION) {
          if (reqs == reqs_end || prs == prs_end)
            return result;
          ++reqs;
          drs.index = *prs++;
        }
        auto [mrb, rmap] = MeasRef::make(rt, drs);
        std::vector<std::shared_ptr<casacore::MRBase>> smrb;
        std::move(mrb.begin(), mrb.end(), std::back_inserter(smrb));
        if (rcs[*pid].size() > 0)
          partial_mrbs[nms[*pid]] = {std::move(smrb), rmap, rcs[*pid]};
        else
          partial_mrbs[nms[*pid]] = {std::move(smrb), rmap, std::nullopt};
      }
#endif
    }
    colv.emplace_back(
      nms[*pid],
      PhysicalColumn(
        dts[*pid],
        vfs[*pid],
        idx_rank,
        metadata,
        parent,
        values,
        kwmap
#ifdef HYPERION_USE_CASACORE
        , std::nullopt
#endif
        ));
  }
#ifdef HYPERION_USE_CASACORE
  for (auto& [nm, pc] : colv) {
    if (partial_mrbs.count(nm) > 0) {
      auto& [mrb, rmap, rcol] = partial_mrbs[nm];
      if (rcol) {
        auto rpc =
          std::find_if(
            colv.begin(),
            colv.end(),
            [rc=rcol.value()](auto& n1_pc1) {
              return rc == std::get<0>(n1_pc1);
            });
        pc.m_mrb =
          std::make_tuple(
            std::move(mrb),
            rmap,
            rpc->second.m_values.value(),
            rpc->second.m_fid);
        colv.erase(rpc);
      } else {
        pc.m_mrb = mrb[0];
      }
    }
  }
#endif
  std::unordered_map<std::string, PhysicalColumn>
    columns(colv.begin(), colv.end());
  return
    std::make_tuple(PhysicalTable(table_parent, table_pr, columns), reqs, prs);
}

Table
PhysicalTable::table() const {
  std::unordered_map<std::string, LogicalRegion> column_parents;
  for (auto& [nm, pc] : m_columns)
    column_parents[nm] = pc.m_parent;
  return
    Table(m_table_pr.get_logical_region(), m_table_parent, column_parents);
}

std::optional<PhysicalColumn>
PhysicalTable::column(const std::string& name) const {
  std::optional<PhysicalColumn> result;
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
       pid() && !result && !css[*pid].is_empty();
       pid++)
    if (vfs[*pid] == Table::no_column)
      result = *pid;
  return result;
}

std::optional<PhysicalColumn>
PhysicalTable::index_column(Runtime* rt) const {
  return
    map(
      index_column_space(rt),
      [this](const auto& ics) {
        const Table::NameAccessor<READ_ONLY>
          nms(m_table_pr, static_cast<FieldID>(TableFieldsFid::NM));
        return m_columns.at(nms[ics]);
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
    result = (unsigned)css[ics.value()].column_is.get_dim();
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
        css[icsp.value()].column_is,
        m_columns.at(nms[icsp.value()]).m_metadata);
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
  for (auto& [nm, pc] : m_columns)
    column_parents[nm] = pc.m_parent;

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
      [](const auto& pc) {
        return std::make_tuple(pc.m_parent.get_index_space(), pc.m_metadata);
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
  for (auto& [nm, pc] : m_columns) {
    auto md_lr = pc.m_metadata.get_logical_region();
    if (cs_idxs.count(md_lr) == 0) {
      auto idx = cs_md_prs.size();
      cs_idxs[md_lr] = idx;
      cs_md_prs.push_back(pc.m_metadata);
      val_lrs.push_back(
        map(
          pc.m_values,
          [](const auto& pr) { return pr.get_logical_region(); })
        .value_or(LogicalRegion::NO_REGION));
      val_prs.push_back(pc.m_values);
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
              PhysicalColumn(
                tf.dt,
                tf.fid,
                idx_rank,
                md_pr,
                vlr,
                vpr,
                {}
#ifdef HYPERION_USE_CASACORE
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

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
