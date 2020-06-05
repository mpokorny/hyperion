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
#include <hyperion/PhysicalColumn.h>
#include <hyperion/Column.h>

using namespace hyperion;
using namespace Legion;

Column
PhysicalColumn::column() const {

#ifdef HYPERION_USE_CASACORE
  MeasRef mr;
  if (m_mr_drs) {
#if HAVE_CXX17
    auto& [mrg, vrg, oirg] = m_mr_drs.value();
#else // !HAVE_CXX17
    MeasRef::DataRegions drs = m_mr_drs.value();
    PhysicalRegion mrg = drs.metadata;
    PhysicalRegion vrg = drs.values;
    CXX_OPTIONAL_NAMESPACE::optional<PhysicalRegion> oirg = drs.index;
#endif // HAVE_CXX17
    mr =
      MeasRef(
        mrg.get_logical_region(),
        vrg.get_logical_region(),
        (oirg ? oirg.value().get_logical_region() : LogicalRegion::NO_REGION));
  }
#endif

  Keywords kws;
  if (m_kws){
    auto& kwv = m_kws.value();
    kws =
      Keywords(
        Keywords::pair<LogicalRegion>{
          kwv.type_tags.get_logical_region(),
          kwv.values.get_logical_region()});
  }

  return
    Column(
      m_dt,
      m_fid,
      column_space(),
      m_region,
      m_parent,
      kws
#ifdef HYPERION_USE_CASACORE
      , mr
      , map(m_refcol, [](const auto& nm_pc){ return std::get<0>(nm_pc); })
#endif
      );
}

ColumnSpace
PhysicalColumn::column_space() const {
  return
    ColumnSpace(m_region.get_index_space(), m_metadata.get_logical_region());
}

ColumnSpace::AXIS_VECTOR_TYPE
PhysicalColumn::axes() const {
  return ColumnSpace::axes(m_metadata);
}

#ifdef HYPERION_USE_CASACORE
std::vector<std::shared_ptr<casacore::MRBase>>
PhysicalColumn::mrbases(Runtime* rt) const {
  std::vector<std::shared_ptr<casacore::MRBase>> result;
  auto omrb = mrb(rt);
  if (omrb) {
#if HAVE_CXX17
    std::visit(overloaded {
        [&result](simple_mrb_t& smrb) {
          result.push_back(smrb);
        },
        [&result](ref_mrb_t& rmrb) {
          result = std::get<0>(rmrb);
        }
      },
      omrb.value());
#else // !HAVE_CXX17
    result = std::get<0>(omrb.value().ref);
#endif // HAVE_CXX17
  }
  return result;
}

CXX_OPTIONAL_NAMESPACE::optional<PhysicalColumn::mrb_t>
PhysicalColumn::mrb(Runtime* rt) const {

  CXX_OPTIONAL_NAMESPACE::optional<mrb_t> result;
  if (m_mr_drs) {
#if HAVE_CXX17
    auto [mrb, rmap] = MeasRef::make(rt, m_mr_drs.value());
#else // !HAVE_CXX17
    std::vector<std::unique_ptr<casacore::MRBase>> mrb;
    std::unordered_map<unsigned, unsigned> rmap;
    std::tie(mrb, rmap) = MeasRef::make(rt, m_mr_drs.value());
#endif // HAVE_CXX17
    std::vector<std::shared_ptr<casacore::MRBase>> smrb;
    std::move(mrb.begin(), mrb.end(), std::back_inserter(smrb));
    result = mrb_t();
    if (m_refcol) {
      const std::shared_ptr<PhysicalColumn>& ppc =
        std::get<1>(m_refcol.value());
#if HAVE_CXX17
      result =
        std::make_tuple(
          std::move(smrb),
          rmap,
          ppc->m_values.value(),
          ppc->m_fid);
#else
      result.value().is_simple = false;
      result.value().ref =
        std::make_tuple(
          std::move(smrb),
          rmap,
          ppc->m_values.value(),
          ppc->m_fid);
#endif
    } else {
#if HAVE_CXX17
      result = smrb[0];
#else // !HAVE_CXX17
      result.value().is_simple = true;
      std::get<0>(result.value().ref) = std::move(smrb);
#endif // HAVE_CXX17
    }
  }
  return result;
}

#endif

LogicalRegion
PhysicalColumn::create_index(Context ctx, Runtime* rt) const {

  auto col = column();
  return col.create_index(ctx, rt);
}


// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
