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
#include <hyperion/MSFieldColumns.h>
#include <hyperion/MeasRefContainer.h>
#include <hyperion/MeasRefDict.h>

#include <casacore/casa/BasicMath/Math.h>

using namespace hyperion;
using namespace Legion;

namespace cc = casacore;

MSFieldColumns::MSFieldColumns(
  Context ctx,
  Runtime* rt,
  const RegionRequirement& rows_requirement,
  const std::unordered_map<std::string, std::vector<Legion::PhysicalRegion>>&
  regions)
  : m_rows_requirement(rows_requirement) {

  for (auto& [nm, prs] : regions) {
    if (prs.size() > 0) {
      auto col = C::lookup_col(nm);
      if (col) {
        m_regions[col.value()] = prs[0];
      } else {
#ifdef HYPERION_USE_CASACORE
        auto col = C::lookup_measure_col(nm);
        if (col) {
          switch (col.value()) {
          case C::col_t::MS_FIELD_COL_TIME:
            m_time_mr =
              MeasRefDict::get<M_EPOCH>(
                MeasRefContainer::make_dict(ctx, rt, prs.begin(), prs.end())
                .get("Epoch").value());
            break;
          case C::col_t::MS_FIELD_COL_DELAY_DIR:
            m_delay_dir_mr =
              MeasRefDict::get<M_DIRECTION>(
                MeasRefContainer::make_dict(ctx, rt, prs.begin(), prs.end())
                .get("Direction").value());
            break;
          case C::col_t::MS_FIELD_COL_PHASE_DIR:
            MeasRefDict::get<M_DIRECTION>(
              MeasRefContainer::make_dict(ctx, rt, prs.begin(), prs.end())
              .get("Direction").value());
            break;
          case C::col_t::MS_FIELD_COL_REFERENCE_DIR:
            m_reference_dir_mr =
              MeasRefDict::get<M_DIRECTION>(
                MeasRefContainer::make_dict(ctx, rt, prs.begin(), prs.end())
                .get("Direction").value());
            break;
          default:
            break;
          }
        }
#endif //HYPERION_USE_CASACORE
      }
    }
  }
}

cc::MDirection
MSFieldColumns::interpolateDirMeas(
  const std::vector<cc::MDirection>& dir_poly,
  double interTime,
  double timeOrigin) {

  if ((dir_poly.size() == 1)
      || cc::nearAbs(interTime, timeOrigin)) {
    return dir_poly[0];
  } else {
    cc::Vector<double> dir(dir_poly[0].getAngle().getValue());
    double dt = interTime - timeOrigin;
    double fac = 1.0;
    for (size_t i = 1; i < dir_poly.size(); ++i) {
      fac *= dt;
      auto tmp = dir_poly[i].getAngle().getValue();
      auto a = tmp.begin();
      for (double& d : dir)
        d += fac * *a++;
    }
    return cc::MDirection(cc::MVDirection(dir), dir_poly[0].getRef());
  }
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
