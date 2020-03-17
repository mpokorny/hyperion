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
    auto& [mrg, vrg, oirg] = m_mr_drs.value();
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

  return Column(
    m_dt,
    m_fid,
#ifdef HYPERION_USE_CASACORE
    mr,
    m_refcol_name,
#endif
    kws,
    ColumnSpace(m_parent.get_index_space(), m_metadata.get_logical_region()),
    m_parent);
}

std::optional<std::any>
PhysicalColumn::kw(const std::string& key) const {
  std::optional<std::any> result;
  if (m_kw_map.count(key) > 0)
    result = m_kw_map.at(key);
  return result;
}

#ifdef HYPERION_USE_CASACORE
void
PhysicalColumn::update_refcol(
  Runtime* rt,
  const std::optional<std::tuple<std::string, PhysicalColumn>>& refcol) {

  if (m_mr_drs) {
    auto [mrb, rmap] = MeasRef::make(rt, m_mr_drs.value());
    std::vector<std::shared_ptr<casacore::MRBase>> smrb;
    std::move(mrb.begin(), mrb.end(), std::back_inserter(smrb));
    if (refcol) {
      auto& [nm, pc] = refcol.value();
      m_mrb =
        std::make_tuple(
          std::move(smrb),
          rmap,
          pc.m_values.value(),
          pc.m_fid);
      m_refcol_name = nm;
    } else {
      m_mrb = smrb[0];
      m_refcol_name.reset();
    }
  }
}

std::vector<std::shared_ptr<casacore::MRBase>>
PhysicalColumn::mrbs() const {
  std::vector<std::shared_ptr<casacore::MRBase>> result;
  if (m_mrb) {
    mrb_t mrb = m_mrb.value();
    std::visit(overloaded {
        [&result](simple_mrb_t& smrb) {
          result.push_back(smrb);
        },
        [&result](ref_mrb_t& rmrb) {
          result = std::get<0>(rmrb);
        }
      },
      mrb);
  }
  return result;
}
#endif

LogicalRegion
PhysicalColumn::create_index(Context ctx, Runtime* rt) const {

  auto col = Column();
  return col.create_index(ctx, rt);
}


// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
