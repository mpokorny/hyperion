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
  const std::string& axes_uid,
  const std::vector<int>& index_axes,
  const PhysicalRegion& index_col_md,
  const std::tuple<LogicalRegion, PhysicalRegion>& index_col,
  const LogicalRegion& index_col_parent,
  const std::unordered_map<std::string, std::shared_ptr<PhysicalColumn>>&
    columns)
  : m_axes_uid(axes_uid)
  , m_index_axes(index_axes)
  , m_index_col_md(index_col_md)
  , m_index_col(index_col)
  , m_index_col_parent(index_col_parent)
  , m_columns(columns) {
}

PhysicalTable::PhysicalTable(const PhysicalTable& other)
  : PhysicalTable(
    other.m_axes_uid,
    other.m_index_axes,
    other.m_index_col_md,
    other.m_index_col,
    other.m_index_col_parent,
    other.m_columns) {
  m_attached = other.m_attached;
}

PhysicalTable::PhysicalTable(PhysicalTable&& other)
  : PhysicalTable(
    std::move(other).m_axes_uid,
    std::move(other).m_index_axes,
    std::move(other).m_index_col_md,
    std::move(other).m_index_col,
    std::move(other).m_index_col_parent,
    std::move(other).m_columns) {
  m_attached = std::move(other).m_attached;
}

CXX_OPTIONAL_NAMESPACE::optional<
  std::tuple<
    PhysicalTable,
    std::vector<RegionRequirement>::const_iterator,
    std::vector<PhysicalRegion>::const_iterator>>
PhysicalTable::create(
  Runtime *rt,
  const Table::Desc& desc,
  const std::vector<RegionRequirement>::const_iterator& reqs_begin,
  const std::vector<RegionRequirement>::const_iterator& reqs_end,
  const std::vector<PhysicalRegion>::const_iterator& prs_begin,
  const std::vector<PhysicalRegion>::const_iterator& prs_end) {

  CXX_OPTIONAL_NAMESPACE::optional<
    std::tuple<
      PhysicalTable,
      std::vector<RegionRequirement>::const_iterator,
      std::vector<PhysicalRegion>::const_iterator>> result;

  if (reqs_begin == reqs_end || prs_begin == prs_end)
    return result;

  std::vector<RegionRequirement>::const_iterator reqs = reqs_begin;
  std::vector<PhysicalRegion>::const_iterator prs = prs_begin;

  PhysicalRegion index_col_md = *prs++;
  ++reqs; // don't need this for index_col_md
  if (reqs == reqs_end || prs == prs_end)
    return result;
  LogicalRegion index_col_parent = reqs->parent;
  std::tuple<LogicalRegion, PhysicalRegion> index_col =
    {reqs++->region, *prs++};

  std::unordered_map<std::string, std::shared_ptr<PhysicalColumn>> columns;
  std::unordered_map<std::string, std::string> refcols;
  std::map<LogicalRegion, PhysicalRegion> md_regions;
  std::map<
    std::tuple<LogicalRegion, FieldID>,
    std::tuple<LogicalRegion, LogicalRegion, PhysicalRegion>>
    value_regions;

  unsigned index_rank = ColumnSpace::size(desc.index_axes);

  for (size_t i = 0; i < desc.num_columns; ++i) {
    auto& cdesc = desc.columns[i];
    if (md_regions.count(cdesc.region) == 0) {
      if (reqs == reqs_end || prs == prs_end)
        return result;
      md_regions[cdesc.region] = *prs;
      ++reqs;
      ++prs;
    }
    std::tuple<LogicalRegion, FieldID> vkey = {cdesc.region, cdesc.fid};
    if (value_regions.count(vkey) == 0) {
      if (reqs == reqs_end || prs == prs_end)
        return result;
      assert(cdesc.region == reqs->parent);
      for (auto& fid : reqs->privilege_fields)
        value_regions[{cdesc.region, fid}]
          = {reqs->region, reqs->parent, *prs};
      ++reqs;
      ++prs;
    }

    CXX_OPTIONAL_NAMESPACE::optional<Keywords::pair<PhysicalRegion>> kw_prs;
    if (cdesc.n_kw > 0) {
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
    CXX_OPTIONAL_NAMESPACE::optional<MeasRef::DataRegions> mr_drs;
    if (cdesc.n_mr > 0) {
      MeasRef::DataRegions drs;
      if (reqs == reqs_end || prs == prs_end)
        return result;
      ++reqs;
      drs.metadata = *prs++;
      if (reqs == reqs_end || prs == prs_end)
        return result;
      ++reqs;
      drs.values = *prs++;
      if (cdesc.n_mr > 2) {
        if (reqs == reqs_end || prs == prs_end)
          return result;
        ++reqs;
        drs.index = *prs++;
      }
      mr_drs = drs;
      if (cdesc.refcol.size() > 0)
        refcols[cdesc.name] = cdesc.refcol;
    }
#endif
    auto& region_parent_values = value_regions.at(vkey);
#if HAVE_CXX17
    auto& [region, parent, values] = region_parent_values;
#else // !HAVE_CXX17
    auto& region = std::get<0>(region_parent_values);
    auto& parent = std::get<1>(region_parent_values);
    auto& values = std::get<2>(region_parent_values);
#endif // HAVE_CXX17
    unsigned col_rank = static_cast<unsigned>(cdesc.region.get_dim());
    assert(col_rank == 1 || col_rank >= index_rank);
    columns.emplace(
      cdesc.name,
      std::make_shared<PhysicalColumn>(
        rt,
        cdesc.dt,
        cdesc.fid,
        std::min(index_rank, col_rank),
        md_regions.at(parent),
        region,
        parent,
        values,
        kw_prs
#ifdef HYPERION_USE_CASACORE
        , mr_drs
        , CXX_OPTIONAL_NAMESPACE::nullopt
#endif
        ));
  }
#ifdef HYPERION_USE_CASACORE

#if HAVE_CXX17
  for (auto& [nm, ppc] : columns) {
    if (refcols.count(nm) > 0) {
      auto& rc = refcols[nm];
      ppc->set_refcol(rc, columns.at(rc));
    }
  }
#else // !HAVE_CXX17
  for (auto& nm_ppc : columns) {
    auto& nm = std::get<0>(nm_ppc);
    auto& ppc = std::get<1>(nm_ppc);
    if (refcols.count(nm) > 0) {
      auto& rc = refcols[nm];
      ppc->set_refcol(rc, columns.at(rc));
    }
  }
#endif // HAVE_CXX17

#endif
  return
    std::make_tuple(
      PhysicalTable(
        desc.axes_uid,
        ColumnSpace::from_axis_vector(desc.index_axes),
        index_col_md,
        index_col,
        index_col_parent,
        columns),
      reqs,
      prs);
}

CXX_OPTIONAL_NAMESPACE::optional<
  std::tuple<
    std::vector<PhysicalTable>,
    std::vector<RegionRequirement>::const_iterator,
    std::vector<PhysicalRegion>::const_iterator>>
PhysicalTable::create_many(
  Runtime *rt,
  const std::vector<Table::Desc>& desc,
  const std::vector<RegionRequirement>::const_iterator& reqs_begin,
  const std::vector<RegionRequirement>::const_iterator& reqs_end,
  const std::vector<PhysicalRegion>::const_iterator& prs_begin,
  const std::vector<PhysicalRegion>::const_iterator& prs_end) {

  std::remove_cv_t<std::remove_reference_t<decltype(reqs_begin)>> rit =
    reqs_begin;
  std::remove_cv_t<std::remove_reference_t<decltype(prs_begin)>> pit =
    prs_begin;
  std::vector<PhysicalTable> tables;
  auto descp = desc.begin();
  while (descp != desc.end() && rit != reqs_end && pit != prs_end) {
    auto opt = create(rt, *descp++, rit, reqs_end, pit, prs_end);
    if (!opt)
      return CXX_OPTIONAL_NAMESPACE::nullopt;
    tables.push_back(std::move(std::get<0>(opt.value())));
    rit = std::get<1>(opt.value());
    pit = std::get<2>(opt.value());
  }
  return std::make_tuple(tables, rit, pit);
}

Table
PhysicalTable::table(Context ctx, Runtime* rt) const {
  std::unordered_map<std::string, Column> columns = get_columns();
  return
    Table(
      index_column_space(ctx, rt),
      std::get<0>(m_index_col),
      m_index_col_parent,
      columns);
}

CXX_OPTIONAL_NAMESPACE::optional<std::string>
PhysicalTable::axes_uid() const {
  CXX_OPTIONAL_NAMESPACE::optional<std::string> result;
  if (m_axes_uid.size() > 0)
    result = m_axes_uid;
  return result;
}

std::vector<int>
PhysicalTable::index_axes() const {
  return m_index_axes;
}

unsigned
PhysicalTable::index_rank() const {
  return index_axes().size();
}

ColumnSpace
PhysicalTable::index_column_space(Context ctx, Runtime* rt) const {
  return
    ColumnSpace::clone(
      ctx,
      rt,
      std::get<0>(m_index_col).get_index_space(),
      m_index_col_md);
}

IndexSpace
PhysicalTable::index_column_space_index_space() const {
  return std::get<0>(m_index_col).get_index_space();
}

const PhysicalRegion&
PhysicalTable::index_column_space_metadata() const {
  return m_index_col_md;
}

CXX_OPTIONAL_NAMESPACE::optional<std::shared_ptr<PhysicalColumn>>
PhysicalTable::column(const std::string& name) const {
  CXX_OPTIONAL_NAMESPACE::optional<std::shared_ptr<PhysicalColumn>> result;
  if (m_columns.count(name) > 0)
    result = m_columns.at(name);
  return result;
}

const std::unordered_map<std::string, std::shared_ptr<PhysicalColumn>>&
PhysicalTable::columns() const {
  return m_columns;
}

bool
PhysicalTable::is_conformant(
  Runtime* rt,
  const IndexSpace& cs_is,
  const PhysicalRegion& cs_md_pr) const {

  std::unordered_map<std::string, Column> cols = get_columns();
  return
    Table::is_conformant(
      rt,
      cols,
      {std::get<0>(m_index_col).get_index_space(), m_index_col_md},
      cs_is,
      cs_md_pr);
}

std::tuple<
  std::vector<RegionRequirement>,
  std::vector<ColumnSpacePartition>,
  Table::Desc>
PhysicalTable::requirements(
  Context ctx,
  Runtime* rt,
  const ColumnSpacePartition& table_partition,
  const std::map<
    std::string,
    CXX_OPTIONAL_NAMESPACE::optional<Column::Requirements>>&
    column_requirements,
  const CXX_OPTIONAL_NAMESPACE::optional<Column::Requirements>&
    default_column_requirements) const {

  std::unordered_map<std::string, Column> cols = get_columns();

  ColumnSpace index_col_cs(
    std::get<0>(m_index_col).get_index_space(),
    m_index_col_md.get_logical_region());

  // collect requirement parameters for each column
  std::map<std::string, Column::Requirements> column_reqs;
  {
#ifdef HYPERION_USE_CASACORE
    std::map<std::string, Column::Requirements> mrc_reqs;
#endif
    std::map<LogicalRegion, Column::Req> lr_mdreqs;
    for (auto& nm_ppc : m_columns) {
      // just use c++14 construct here, as the loop body is too large to
      // duplicate
      auto& nm = std::get<0>(nm_ppc);
      auto& ppc = std::get<1>(nm_ppc);
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
        if (lr_mdreqs.count(ppc->region()) == 0) {
          lr_mdreqs[ppc->region()] = colreqs.column_space;
        } else {
          // FIXME: log a warning, and return empty result;
          // warning: inconsistent requirements on shared Column metadata
          // regions
          assert(lr_mdreqs[ppc->region()] == colreqs.column_space);
        }
#ifdef HYPERION_USE_CASACORE
        if (ppc->refcol())
          mrc_reqs[std::get<0>(ppc->refcol().value())] = colreqs;
#endif
      }
    }

#ifdef HYPERION_USE_CASACORE
    // apply mode of value column to its measure reference column
#if HAVE_CXX17
    for (auto& [nm, rq] : mrc_reqs)
      column_reqs.at(nm) = rq;
#else // !HAVE_CXX17
    for (auto& nm_rq : mrc_reqs) {
      auto& nm = std::get<0>(nm_rq);
      auto& rq = std::get<1>(nm_rq);
      column_reqs.at(nm) = rq;
    }
#endif // HAVE_CXX17
#endif
  }
  // create requirements, applying table_partition as needed
  std::map<LogicalRegion, std::tuple<ColumnSpacePartition, LogicalPartition>>
    partitions;
  if (table_partition.is_valid()) {
    auto& lr = std::get<0>(m_index_col);
    if (table_partition.column_space.column_is != lr.get_index_space()) {
      auto csp =
        table_partition.project_onto(
          ctx,
          rt,
          lr.get_index_space(),
          m_index_col_md);
      auto lp = rt->get_logical_partition(ctx, lr, csp.column_ip);
      csp.destroy(ctx, rt);
      partitions[lr] = {csp, lp};
    } else {
      auto lp = rt->get_logical_partition(ctx, lr, table_partition.column_ip);
      partitions[lr] = {ColumnSpacePartition(), lp};
    }
  }

  // boolean elements in value of following maps is used to track whether the
  // requirement has already been added when iterating through columns
  std::map<LogicalRegion, std::tuple<bool, RegionRequirement>> md_reqs;
  std::map<
    std::tuple<LogicalRegion, PrivilegeMode, CoherenceProperty, MappingTagID>,
    std::tuple<bool, RegionRequirement>> val_reqs;
  for (auto& nm_ppc : m_columns) {
    // again, just use c++14 here because of size of loop body
    auto& nm = std::get<0>(nm_ppc);
    auto& ppc = std::get<1>(nm_ppc);
    if (column_reqs.count(nm) > 0) {
      auto& reqs = column_reqs.at(nm);
      auto cs = ppc->column_space();
      if (md_reqs.count(ppc->region()) == 0)
        md_reqs[ppc->region()] =
          {false,
           cs.requirements(
             reqs.column_space.privilege,
             reqs.column_space.coherence)};
      decltype(val_reqs)::key_type rg_rq =
        {ppc->region(), reqs.values.privilege, reqs.values.coherence, reqs.tag};
      if (val_reqs.count(rg_rq) == 0) {
        if (!table_partition.is_valid()) {
          val_reqs[rg_rq] =
            {false,
             RegionRequirement(
               ppc->region(),
               reqs.values.privilege,
               reqs.values.coherence,
               ppc->region(),
               reqs.tag)};
        } else {
          LogicalPartition lp;
          if (partitions.count(ppc->region()) == 0) {
            auto csp =
              table_partition.project_onto(
                ctx,
                rt,
                ppc->region().get_index_space(),
                ppc->metadata());
            assert(csp.column_space == cs);
            lp = rt->get_logical_partition(ctx, ppc->region(), csp.column_ip);
            csp.destroy(ctx, rt);
            partitions[ppc->region()] = {csp, lp};
          } else {
            lp = std::get<1>(partitions[ppc->region()]);
          }
          val_reqs[rg_rq] =
            {false,
             RegionRequirement(
               lp,
               0,
               reqs.values.privilege,
               reqs.values.coherence,
               ppc->region(),
               reqs.tag)};
        }
      }
      std::get<1>(val_reqs[rg_rq]).add_field(ppc->fid(), reqs.values.mapped);
    }
  }
  std::vector<ColumnSpacePartition> csps_result;
  for (auto& lr_csplp : partitions) {
#if HAVE_CXX17
    auto& [csp, lp] = std::get<1>(lr_csplp);
#else // !HAVE_CXX17
    auto& csplp = std::get<1>(lr_csplp);
    auto& csp = std::get<0>(csplp);
#endif
    if (csp.is_valid()) // avoid returning table_partition
      csps_result.push_back(csp);
  }

  // gather all requirements, in order set by this traversal of fields
  std::vector<RegionRequirement> reqs_result;

  // start with index_col ColumnSpace metadata
  reqs_result.push_back(index_col_cs.requirements(READ_ONLY, EXCLUSIVE));
  // next, index_col index space partition
  {
    auto& lr = std::get<0>(m_index_col);
    if (table_partition.is_valid()) {
      RegionRequirement req(
        std::get<1>(partitions[lr]),
        0,
        {Table::m_index_col_fid},
        {}, // always remains unmapped!
        WRITE_ONLY,
        SIMULTANEOUS,
        lr);
      reqs_result.push_back(req);
    } else {
      RegionRequirement req(
        lr,
        {Table::m_index_col_fid},
        {}, // always remains unmapped!
        WRITE_ONLY,
        SIMULTANEOUS,
        lr);
      reqs_result.push_back(req);
    }
  }
  Table::Desc desc_result;
  desc_result.axes_uid = m_axes_uid;
  desc_result.index_axes = ColumnSpace::to_axis_vector(m_index_axes);
  desc_result.num_columns = column_reqs.size();
  assert(desc_result.num_columns <= desc_result.columns.size());

  // add requirements for all logical regions in all selected columns
  size_t desc_idx = 0;
  for (auto& nm_ppc : m_columns) {
    // again, c++14 only, due to size of loop body
    auto& nm = std::get<0>(nm_ppc);
    auto& ppc = std::get<1>(nm_ppc);
    if (column_reqs.count(nm) > 0) {
      auto cdesc = ppc->column().desc(nm);
      auto& reqs = column_reqs.at(nm);
      {
        auto& added_rq = md_reqs.at(ppc->region());
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
        {ppc->region(), reqs.values.privilege, reqs.values.coherence, reqs.tag};
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
        auto rqs =
          Keywords::requirements(
            rt,
            ppc->kws().value(),
            reqs.keywords.privilege,
            reqs.keywords.mapped);
        reqs_result.push_back(rqs.type_tags);
        reqs_result.push_back(rqs.values);
      }

#ifdef HYPERION_USE_CASACORE
      if (cdesc.n_mr > 0) {
        auto rqs =
          MeasRef::requirements(
            ppc->mr_drs().value(),
            reqs.measref.privilege,
            reqs.measref.mapped);
        auto& mrq = std::get<0>(rqs);
        auto& vrq = std::get<1>(rqs);
        auto& oirq = std::get<2>(rqs);
        assert(cdesc.n_mr == 2 || cdesc.n_mr == 3);
        reqs_result.push_back(mrq);
        reqs_result.push_back(vrq);
        if (oirq) {
          assert(cdesc.n_mr == 3);
          reqs_result.push_back(oirq.value());
        }
      }
#endif
      desc_result.columns[desc_idx++] = cdesc;
    }
  }
  return {reqs_result, csps_result, desc_result};
}

bool
PhysicalTable::add_columns(
  Context ctx,
  Runtime* rt,
  std::vector<
    std::tuple<
      ColumnSpace,
      std::vector<std::pair<std::string, TableField>>>>&& cols) {

  std::vector<
    std::tuple<
      ColumnSpace,
      size_t,
      std::vector<std::pair<hyperion::string, TableField>>>> new_columns;
  std::map<LogicalRegion, size_t> cs_idxs; // metadata_lr
  std::vector<PhysicalRegion> cs_md_prs;
  std::unordered_map<std::string, Column> current_cols;
  for (auto& nm_ppc : m_columns) {
    auto& nm = std::get<0>(nm_ppc);
    auto& ppc = std::get<1>(nm_ppc);
    auto md_lr = ppc->m_metadata.get_logical_region();
    if (cs_idxs.count(md_lr) == 0) {
      auto idx = cs_md_prs.size();
      cs_idxs[md_lr] = idx;
      cs_md_prs.push_back(ppc->m_metadata);
    }
    current_cols[nm] = ppc->column();
  }
  for (auto& cs_nm_tfs : cols) {
    auto& cs = std::get<0>(cs_nm_tfs);
    auto& nm_tfs = std::get<1>(cs_nm_tfs);
    if (cs_idxs.count(cs.metadata_lr) == 0) {
      cs_idxs[cs.metadata_lr] = cs_md_prs.size();
      cs_md_prs.push_back(
        rt->map_region(
          ctx,
          cs.requirements(READ_ONLY, EXCLUSIVE)));
    }
    std::vector<std::pair<hyperion::string, TableField>> hnm_tfs;
#if HAVE_CXX17
    for (auto& [nm, tf] : nm_tfs)
      hnm_tfs.emplace_back(nm, tf);
#else // !HAVE_CXX17
    for (auto& nm_tf : nm_tfs) {
      auto& nm = std::get<0>(nm_tf);
      auto& tf = std::get<1>(nm_tf);
      hnm_tfs.emplace_back(nm, tf);
    }
#endif // HAVE_CXX17
    new_columns.emplace_back(cs, cs_idxs[cs.metadata_lr], hnm_tfs);
  }
  std::tuple<IndexSpace, PhysicalRegion> index_cs =
    {std::get<0>(m_index_col).get_index_space(), std::get<1>(m_index_col)};
  auto added =
    Table::add_columns(
      ctx,
      rt,
      std::move(new_columns),
      current_cols,
      cs_md_prs,
      index_cs);

  // create (unmapped) PhysicalColumns for added column values
  std::map<
    LogicalRegion,
    std::tuple<
      std::vector<FieldID>,
      CXX_OPTIONAL_NAMESPACE::optional<PhysicalRegion>>> new_fields;
  for (auto& nm_col : added) {
    auto& col = std::get<1>(nm_col);
    if (new_fields.count(col.region) == 0)
      new_fields[col.region] = {{col.fid}, CXX_OPTIONAL_NAMESPACE::nullopt};
    else
      std::get<0>(new_fields[col.region]).push_back(col.fid);
  }
  for (auto& lr_fids_pr : new_fields) {
    auto& lr = std::get<0>(lr_fids_pr);
    auto& fids_pr = std::get<1>(lr_fids_pr);
    auto& fids = std::get<0>(fids_pr);
    auto& pr = std::get<1>(fids_pr);
    RegionRequirement req(lr, WRITE_DISCARD, EXCLUSIVE, lr);
    req.add_fields(fids, false);
    pr = rt->map_region(ctx, req);
  }

  unsigned idx_rank = index_rank();
  std::unordered_map<std::string, std::string> refcols;
  for (auto& nm_col : added) {
    auto& nm = std::get<0>(nm_col);
    auto& col = std::get<1>(nm_col);
    // create kws for Keywords
    CXX_OPTIONAL_NAMESPACE::optional<Keywords::pair<PhysicalRegion>> kws;
    if (!col.kw.is_empty()) {
      std::vector<FieldID> fids;
      fids.resize(col.kw.size(rt));
      std::iota(fids.begin(), fids.end(), 0);
      auto reqs = col.kw.requirements(rt, fids, READ_WRITE, false).value();
      auto prs =
        reqs.map(
          [&ctx, rt](const auto& rq) {
            return rt->map_region(ctx, rq);
          });
      kws = prs;
    }
#ifdef HYPERION_USE_CASACORE
    // create mr_drs for MeasRef
    CXX_OPTIONAL_NAMESPACE::optional<MeasRef::DataRegions> mr_drs;
    if (!col.mr.is_empty()) {
      auto rqs = col.mr.requirements(READ_WRITE, false);
      auto& mrq = std::get<0>(rqs);
      auto& vrq = std::get<1>(rqs);
      auto& oirq = std::get<2>(rqs);
      MeasRef::DataRegions prs;
      prs.metadata = rt->map_region(ctx, mrq);
      prs.values = rt->map_region(ctx, vrq);
      if (oirq)
        prs.index = rt->map_region(ctx, oirq.value());
      mr_drs = prs;
    }
    if (col.rc)
      refcols[nm] = col.rc.value();
#endif
    assert(m_columns.count(nm) == 0);
    unsigned col_rank =
      ColumnSpace::size(
        ColumnSpace::axes(cs_md_prs[cs_idxs[col.cs.metadata_lr]]));
    assert(col_rank == 1 || col_rank >= idx_rank);
    m_columns.emplace(
      nm,
      std::make_shared<PhysicalColumn>(
        rt,
        col.dt,
        col.fid,
        std::min(idx_rank, col_rank),
        cs_md_prs[cs_idxs[col.cs.metadata_lr]],
        col.region,
        col.parent,
        std::get<1>(new_fields.at(col.region)),
        kws
#ifdef HYPERION_USE_CASACORE
        , mr_drs
        , CXX_OPTIONAL_NAMESPACE::nullopt
#endif
        ));
  }
#ifdef HYPERION_USE_CASACORE
  for (auto& nm_col : added) {
    auto& nm = std::get<0>(nm_col);
    if (refcols.count(nm) > 0) {
      auto& rc = refcols.at(nm);
      m_columns.at(nm)->set_refcol(rc, m_columns.at(rc));
    }
  }
#endif
  return added.size() > 0;
}

bool
PhysicalTable::remove_columns(
  Context ctx,
  Runtime* rt,
  const std::unordered_set<std::string>& cols) {

  std::vector<ColumnSpace> css;
  std::vector<PhysicalRegion> cs_md_prs;
  for (auto& nm_pcol : m_columns) {
    auto& nm = std::get<0>(nm_pcol);
    auto& pcol = std::get<1>(nm_pcol);
    if (cols.count(nm) > 0) {
      if (std::find(css.begin(), css.end(), pcol->column_space()) == css.end()) {
        css.push_back(pcol->column_space());
        cs_md_prs.push_back(pcol->metadata());
      }
    }
  }

  std::unordered_map<std::string, Column> columns = get_columns();

  bool result =
    Table::remove_columns(
      ctx,
      rt,
      cols,
      columns,
      css,
      cs_md_prs);

  if (result) {
    std::map<LogicalRegion, ColumnSpace> lrcss;
    for (auto& nm_col : columns) {
    auto& nm = std::get<0>(nm_col);
    auto& col = std::get<1>(nm_col);
      if (lrcss.count(col.region) == 0)
        lrcss[col.region] = col.cs;
      col.kw.destroy(ctx, rt);
#ifdef HYPERION_USE_CASACORE
      col.mr.destroy(ctx, rt);
#endif
      m_columns.erase(nm);
      m_attached.erase(nm);
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
  if (column->values()) {
    auto& v = column->values().value();
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
  if (column->refcol())
    for_all_column_regions(std::get<1>(column->refcol().value()), done, fn);
#endif
}

void
PhysicalTable::unmap_regions(Context ctx, Runtime* rt) const {

  std::set<PhysicalRegion> unmapped;
  for (auto& nm_pc : m_columns) {
    for_all_column_regions(
      std::get<1>(nm_pc),
      unmapped,
      [&ctx, rt](auto& pr) { rt->unmap_region(ctx, pr); });
  }
  rt->unmap_region(ctx, m_index_col_md);
  rt->unmap_region(ctx, std::get<1>(m_index_col));
}

void
PhysicalTable::remap_regions(Context ctx, Runtime* rt) const {

  std::set<PhysicalRegion> remapped;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  for (auto& nm_pc : m_columns) {
#pragma GCC diagnostic pop
    for_all_column_regions(
      std::get<1>(nm_pc),
      remapped,
      [&ctx, rt](auto& pr) { rt->remap_region(ctx, pr); });
  }
  rt->remap_region(ctx, m_index_col_md);
  rt->remap_region(ctx, std::get<1>(m_index_col));
}

ColumnSpacePartition
PhysicalTable::partition_rows(
  Context ctx,
  Runtime* rt,
  const std::vector<CXX_OPTIONAL_NAMESPACE::optional<size_t>>& block_sizes)
  const {

  return
    Table::partition_rows(
      ctx,
      rt,
      block_sizes,
      std::get<0>(m_index_col).get_index_space(),
      m_index_col_md);
}

TaskID PhysicalTable::reindex_copy_values_task_id;

const char* PhysicalTable::reindex_copy_values_task_name =
  "PhysicalTable::reindex_copy_values_task";

struct ReindexCopyValuesTaskArgs {
  hyperion::TypeTag dt;
  FieldID fid;
};

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
PhysicalTable::reindex_copy_values_task(
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
  const ColumnSpace& cs,
  unsigned rank) {

  auto ax = cs.axes(ctx, rt);
  assert(ax.size() == (unsigned)cs.column_is.get_dim());
  IndexSpace truncated_is;
  switch (rank * LEGION_MAX_DIM + ax.size()) {
#define TIS(RANK, CS_RANK)                                \
    case (RANK * LEGION_MAX_DIM + CS_RANK): {             \
      IndexSpaceT<CS_RANK> is(cs.column_is);              \
      truncated_is =                                      \
        truncate_index_space<RANK, CS_RANK>(ctx, rt, is); \
      break;                                              \
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
      cs.axes_uid(ctx, rt),
      truncated_is,
      false);
}

Table
PhysicalTable::reindexed(
  Context ctx,
  Runtime* rt,
  const std::vector<std::pair<int, std::string>>& index_axes,
  bool allow_rows) const {

  std::vector<int> ixax = this->index_axes();

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

  // can only reindex on an axis if table has a column with the associated name
  {
    std::set<std::string> missing;
    std::for_each(
      index_axes_extension,
      index_axes.end(),
      [this, &missing](auto& d_nm) {
        auto& nm = std::get<1>(d_nm);
        if (m_columns.count(nm) == 0)
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
  std::map<ColumnSpace, std::vector<std::string>> cs_cols;
  for (auto& nm_pcol : m_columns) {
#if HAVE_CXX17
    auto& [nm, pcol] = nm_pcol;
#else // !HAVE_CXX17
    auto& nm = std::get<0>(nm_pcol);
    auto& pcol = std::get<1>(nm_pcol);
#endif // HAVE_CXX17
    auto cs = pcol->column_space();
    if (cs_cols.count(cs) == 0)
      cs_cols[cs] = {nm};
    else
      cs_cols[cs].push_back(nm);
  }

  // compute new index column indexes, and map the index regions
  std::unordered_map<std::string, int> column_index;
  std::unordered_map<int, std::pair<LogicalRegion, PhysicalRegion>> index_cols;
  std::for_each(
    index_axes_extension,
    index_axes.end(),
    [this, &column_index, &index_cols, &ctx, rt](auto& d_nm) {
#if HAVE_CXX17
      auto& [d, nm] = d_nm;
#else // !HAVE_CXX17
      auto& d = std::get<0>(d_nm);
      auto& nm = std::get<1>(d_nm);
#endif // HAVE_CXX17
      auto lr = m_columns.at(nm)->create_index(ctx, rt);
      RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
      req.add_field(Column::COLUMN_INDEX_ROWS_FID);
      auto pr = rt->map_region(ctx, req);
      index_cols[d] = {lr, pr};
      column_index[nm] = d;
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

    for (auto& cs_nms : cs_cols) {
      auto& cs = std::get<0>(cs_nms);
      auto& nms = std::get<1>(cs_nms);
      for (auto& nm : nms) {
        auto& col = m_columns.at(nm);
        const ColumnSpace::IndexFlagAccessor<READ_ONLY>
          ifl(col->metadata(), ColumnSpace::INDEX_FLAG_FID);
        if (!ifl[0] && column_index.count(nm) == 0) {
          assert((unsigned)cs.column_is.get_dim() >= ixax.size());
          unsigned element_rank =
            (unsigned)cs.column_is.get_dim() - ixax.size();
          assert(reindexed.count(cs) == 0);
          reindexed[cs] =
            ColumnSpace::reindexed(
              ctx,
              rt,
              element_rank,
              ixcols,
              allow_rows,
              cs.column_is,
              col->metadata());
          break;
        }
      }
    }
  }

  // create the reindexed table
  Table result_tbl;
  {
    std::vector<int> new_index_axes;
    new_index_axes.reserve(index_axes.size() + ((allow_rows ? 1 : 0)));
    for (auto& d_nm : index_axes)
      new_index_axes.push_back(std::get<0>(d_nm));
    if (allow_rows)
      new_index_axes.push_back(0);
    unsigned column_spaces_min_rank = 2 * LEGION_MAX_DIM;
    ColumnSpace min_rank_column_space;
    std::map<ColumnSpace, std::vector<std::pair<std::string, TableField>>>
      nmtfs;
    std::string axuid;
    for (auto& cs_nms : cs_cols) {
      auto& cs = std::get<0>(cs_nms);
      auto& nms = std::get<1>(cs_nms);
      std::vector<std::pair<std::string, TableField>> tfs;
      for (auto& nm : nms) {
        auto& col = m_columns.at(nm);
        const ColumnSpace::IndexFlagAccessor<READ_ONLY>
          ifl(col->metadata(), ColumnSpace::INDEX_FLAG_FID);
        const ColumnSpace::AxisVectorAccessor<READ_ONLY>
          av(col->metadata(), ColumnSpace::AXIS_VECTOR_FID);
        const ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
          auid(col->metadata(), ColumnSpace::AXIS_SET_UID_FID);
        axuid = auid[0];
#ifdef HYPERION_USE_CASACORE
        CXX_OPTIONAL_NAMESPACE::optional<MeasRef::DataRegions> odrs
          = col->mr_drs();
#endif
        CXX_OPTIONAL_NAMESPACE::optional<Keywords::pair<PhysicalRegion>> okwrs
          = col->kws();
        TableField tf(
          col->dt(),
          col->fid(),
          (okwrs ? Keywords::clone(ctx, rt, okwrs.value()) : Keywords())
#ifdef HYPERION_USE_CASACORE
          , (odrs ? MeasRef::clone(ctx, rt, odrs.value()) : MeasRef())
          , map(
            col->refcol(),
            [](const auto& nm_c){ return std::get<0>(nm_c); })
#endif
          );
        if (column_index.count(nm) > 0 || ifl[0]) {
          ColumnSpace ics;
          if (column_index.count(nm) > 0)
            ics =
              ColumnSpace::create(
                ctx,
                rt,
                {column_index.at(nm)},
                auid[0],
                // NB: take ownership of index space
                index_cols[column_index.at(nm)].first.get_index_space(),
                true);
          else
            ics =
              ColumnSpace::create(
                ctx,
                rt,
                {av[0][0]},
                auid[0],
                rt->create_index_space(
                  ctx,
                  rt->get_index_space_domain(col->column_space().column_is)),
                true);
          nmtfs[ics] = {{nm, tf}};
        } else {
          tfs.emplace_back(nm, tf);
        }
      }
      if (reindexed.count(cs) > 0) {
        auto& rcs = std::get<0>(reindexed[cs]);
        const auto ax = rcs.axes(ctx, rt);
        if (ax.size() < column_spaces_min_rank) {
          column_spaces_min_rank = ax.size();
          min_rank_column_space = rcs;
        }
        nmtfs[rcs] = tfs;
      }
    }
    if (!min_rank_column_space.is_valid()) {
      // This case really can't occur, as it would imply that the index
      // columns completely index every column in the table, which would mean
      // that only index columns exist, and thus no "row" index. FIXME: We
      // should log a warning, but also just invent an index column space,
      // instead of generating an error.
      assert(false);
    }
    ColumnSpace reindexed_cs =
      truncate_column_space(
        ctx,
        rt,
        min_rank_column_space,
        new_index_axes.size());
    Table::fields_t cols(nmtfs.begin(), nmtfs.end());
    result_tbl =
      Table::create(ctx, rt, std::move(reindexed_cs), std::move(cols));
  }

  // copy values from old table to new
  {
    const unsigned min_block_size = 1000000;
    CopyLauncher index_column_copier;
    auto dcols = result_tbl.columns();
    // collect columns by value LogicalRegion
    std::map<LogicalRegion, std::map<std::string, Column>> grouped_dcols;
    for (auto& nm_dcol : dcols) {
      auto& nm = std::get<0>(nm_dcol);
      auto& dcol = std::get<1>(nm_dcol);
      if (grouped_dcols.count(dcol.region) == 0)
        grouped_dcols[dcol.region] = {{nm, dcol}};
      else
        grouped_dcols[dcol.region][nm] = dcol;
    }
    // now copy values by group
    for (auto& dcg_dcols : grouped_dcols) {
      auto& dcols = std::get<1>(dcg_dcols);
      // check first element of dcols to determine whether we've got an index
      // column in the result Table
      auto& nm = std::get<0>(*dcols.begin());
      auto& dcol = std::get<1>(*dcols.begin());
      auto dcs_md_pr =
        rt->map_region(ctx, dcol.cs.requirements(READ_ONLY, EXCLUSIVE));
      const ColumnSpace::AxisVectorAccessor<READ_ONLY>
        dav(dcs_md_pr, ColumnSpace::AXIS_VECTOR_FID);
      const ColumnSpace::IndexFlagAccessor<READ_ONLY>
        difl(dcs_md_pr, ColumnSpace::INDEX_FLAG_FID);

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
          auto col = m_columns.at(nm);
          slr = col->region();
          sfid = col->fid();
        }
        RegionRequirement src(slr, {sfid}, {sfid}, READ_ONLY, EXCLUSIVE, slr);
        RegionRequirement dst(
          dcol.region,
          {dcol.fid},
          {dcol.fid},
          WRITE_ONLY,
          EXCLUSIVE,
          dcol.region);
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
          auto col = m_columns.at(nm);
          rctlr = std::get<1>(reindexed[col->column_space()]);
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
              dcol.region.get_index_space(),
              rctlp,
              rctlr,
              ColumnSpace::REINDEXED_ROW_RECTS_FID,
              cs,
              DISJOINT_COMPLETE_KIND);
          dlp = rt->get_logical_partition(ctx, dcol.region, dip);
          slr = col->region();
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
        for (auto& nm_dcol : dcols) {
          auto& nm = std::get<0>(nm_dcol);
          auto& dcol = std::get<1>(nm_dcol);
          auto col = m_columns.at(nm);
          args.dt = col->dt();
          args.fid = col->fid();
          assert(dcol.fid == col->fid());
          task.region_requirements.resize(1);
          task.add_region_requirement(
            RegionRequirement(slr, READ_ONLY, EXCLUSIVE, slr));
          task.add_field(1, col->fid());
          task.add_region_requirement(
            RegionRequirement(dlp, 0, WRITE_ONLY, EXCLUSIVE, dcol.region));
          task.add_field(2, dcol.fid);
          rt->execute_index_space(ctx, task);
        }
        rt->destroy_index_partition(ctx, dlp.get_index_partition());
        rt->destroy_index_partition(ctx, rctlp.get_index_partition());
      }
      rt->unmap_region(ctx, dcs_md_pr);
    }
    rt->issue_copy_operation(ctx, index_column_copier);
  }

  for (auto& d_lr_pr : index_cols) {
    auto& lr_pr = std::get<1>(d_lr_pr);
    auto& lr = std::get<0>(lr_pr);
    auto& pr = std::get<1>(lr_pr);
    rt->unmap_region(ctx, pr);
    auto fs = lr.get_field_space();
    // DON'T do this: rt->destroy_index_space(ctx, lr.get_index_space());
    rt->destroy_logical_region(ctx, lr);
    rt->destroy_field_space(ctx, fs);
  }
  for (auto& cs_rcs_rlr_lcid : reindexed) {
    auto& rcs_rlr_lcid = std::get<1>(cs_rcs_rlr_lcid);
    auto& rlr = std::get<1>(rcs_rlr_lcid);
    auto& lcid = std::get<2>(rcs_rlr_lcid);
    rt->release_layout(lcid);
    auto fs = rlr.get_field_space();
    // DON'T destroy index space
    rt->destroy_logical_region(ctx, rlr);
    rt->destroy_field_space(ctx, fs);
  }
  return result_tbl;
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
  for (auto& nm_pc : m_columns) {
    auto& nm = std::get<0>(nm_pc);
    auto& pc = std::get<1>(nm_pc);
    if (column_paths.count(nm) > 0) {
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
        {pc->region(), column_modes.at(nm)};
      if (regions.count(key) == 0)
        regions[key] = std::vector<std::tuple<FieldID, std::string>>();
      regions[key].emplace_back(pc->fid(), nm);
    }
  }
  for (auto& parent_modes_fid_nms : regions) {
    auto& parent_modes = std::get<0>(parent_modes_fid_nms);
    auto& fid_nms= std::get<1>(parent_modes_fid_nms);
    std::map<FieldID, const char*> field_map;
    for (auto& fid_nm : fid_nms) {
      auto& fid = std::get<0>(fid_nm);
      auto& nm = std::get<1>(fid_nm);
      field_map[fid] = column_paths.at(nm).c_str();
    }
    auto& parent = std::get<0>(parent_modes);
    auto& modes = std::get<1>(parent_modes);
    auto& read_only = std::get<0>(modes);
    auto& restricted = std::get<1>(modes);
    auto& mapped = std::get<2>(modes);
    LogicalRegion lr = m_columns.at(std::get<1>(fid_nms[0]))->values_lr();
    AttachLauncher attach(EXTERNAL_HDF5_FILE, lr, parent, restricted, mapped);
    attach.attach_hdf5(
      file_path.c_str(),
      field_map,
      read_only ? LEGION_FILE_READ_ONLY : LEGION_FILE_READ_WRITE);
    auto pr1 = rt->attach_external_resource(ctx, attach);
    for (auto& fid_nm : fid_nms) {
      auto& nm = std::get<1>(fid_nm);
      m_columns.at(nm)->m_values = pr1;
      m_attached[nm] = pr1;
    }
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
  for (auto& nm : columns) {
    if (m_attached.count(nm) > 0) {
      PhysicalRegion pr = m_attached.at(nm);
      m_columns.at(nm)->m_values = CXX_OPTIONAL_NAMESPACE::nullopt;
      if (detached.count(pr) == 0) {
        rt->detach_external_resource(ctx, pr);
        detached.insert(pr);
      }
    }
  }
  for (auto it = m_attached.begin(); it != m_attached.end();) {
    if (detached.count(it->second) > 0)
      it = m_attached.erase(it);
    else
      ++it;
  }
}

void
PhysicalTable::acquire_columns(Context ctx, Runtime* rt) {

}

void
PhysicalTable::release_columns(Context ctx, Runtime* rt) {

}

std::unordered_map<std::string, Column>
PhysicalTable::get_columns() const {
  std::unordered_map<std::string, Column> result;
#if HAVE_CXX17
  for (auto& [nm, ppc] : m_columns)
    result[nm] = ppc->column();
#else // !HAVE_CXX17
  for (auto& nm_ppc : m_columns) {
    auto& nm = std::get<0>(nm_ppc);
    auto& ppc = std::get<1>(nm_ppc);
    result[nm] = ppc->column();
  }
#endif // HAVE_CXX17
  return result;
}

void
PhysicalTable::preregister_tasks() {
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
