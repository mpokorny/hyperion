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
#include <hyperion/MSTableColumns.h>
#include <hyperion/MeasRefContainer.h>
#include <hyperion/MeasRefDict.h>

#include <casacore/casa/BasicMath/Math.h>

using namespace hyperion;
using namespace Legion;

namespace cc = casacore;

#define COL(C) HYPERION_COLUMN_NAME(FIELD, C)

MSFieldColumns::MSFieldColumns(
  Context ctx,
  Runtime* rt,
  const RegionRequirement& rows_requirement,
  const std::unordered_map<std::string, std::vector<Legion::PhysicalRegion>>&
  regions)
  : m_rows_requirement(rows_requirement) {

  if (regions.count(COL(NAME)) > 0)
    m_name_region = regions.at(COL(NAME))[0];
  if (regions.count(COL(CODE)) > 0)
    m_code_region = regions.at(COL(CODE))[0];
  if (regions.count(COL(TIME)) > 0)
    m_time_region = regions.at(COL(TIME))[0];
  if (regions.count(COL(NUM_POLY)) > 0)
    m_num_poly_region = regions.at(COL(NUM_POLY))[0];
  if (regions.count(COL(DELAY_DIR)) > 0)
    m_delay_dir_region = regions.at(COL(DELAY_DIR))[0];
  if (regions.count(COL(PHASE_DIR)) > 0)
    m_phase_dir_region = regions.at(COL(PHASE_DIR))[0];
  if (regions.count(COL(REFERENCE_DIR)) > 0)
    m_reference_dir_region = regions.at(COL(REFERENCE_DIR))[0];
  if (regions.count(COL(SOURCE_ID)) > 0)
    m_source_id_region = regions.at(COL(SOURCE_ID))[0];
  if (regions.count(COL(EPHEMERIS_ID)) > 0)
    m_ephemeris_id_region = regions.at(COL(EPHEMERIS_ID))[0];
  if (regions.count(COL(FLAG_ROW)) > 0)
    m_flag_row_region = regions.at(COL(FLAG_ROW))[0];

#ifdef HYPERION_USE_CASACORE
  if (regions.count("TIME_MEAS_REF") > 0) {
    auto prs = regions.at("TIME_MEAS_REF");
    if (prs.size() > 0)
      m_time_mr =
        MeasRefDict::get<M_EPOCH>(
          MeasRefContainer::make_dict(ctx, rt, prs.begin(), prs.end())
          .get("Epoch").value());
  }

  if (regions.count("DELAY_DIR_MEAS_REF") > 0) {
    auto prs = regions.at("DELAY_DIR_MEAS_REF");
    if (prs.size() > 0)
      m_delay_dir_mr =
        MeasRefDict::get<M_DIRECTION>(
          MeasRefContainer::make_dict(ctx, rt, prs.begin(), prs.end())
          .get("Direction").value());
  }

  if (regions.count("PHASE_DIR_MEAS_REF") > 0) {
    auto prs = regions.at("PHASE_DIR_MEAS_REF");
    if (prs.size() > 0)
      m_phase_dir_mr =
        MeasRefDict::get<M_DIRECTION>(
          MeasRefContainer::make_dict(ctx, rt, prs.begin(), prs.end())
          .get("Direction").value());
  }

  if (regions.count("REFERENCE_DIR_MEAS_REF") > 0) {
    auto prs = regions.at("REFERENCE_DIR_MEAS_REF");
    if (prs.size() > 0)
      m_reference_dir_mr =
        MeasRefDict::get<M_DIRECTION>(
          MeasRefContainer::make_dict(ctx, rt, prs.begin(), prs.end())
          .get("Direction").value());
  }
#endif // HYPERION_USE_CASACORE
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
