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
#include <hyperion/MSAntennaColumns.h>
#include <hyperion/MSTableColumns.h>
#include <hyperion/MeasRefContainer.h>
#include <hyperion/MeasRefDict.h>

using namespace hyperion;
using namespace Legion;

typedef MSTableColumns<MS_ANTENNA> TC;

#define COL(C) HYPERION_COLUMN_NAME(ANTENNA, C)

MSAntennaColumns::MSAntennaColumns(
  Context ctx,
  Runtime* rt,
  const RegionRequirement& rows_requirement,
  const std::unordered_map<std::string, std::vector<Legion::PhysicalRegion>>&
  regions)
  : m_rows_requirement(rows_requirement) {

  if (regions.count(COL(NAME)) > 0)
    m_name_region = regions.at(COL(NAME))[0];
  if (regions.count(COL(STATION)) > 0)
    m_station_region = regions.at(COL(STATION))[0];
  if (regions.count(COL(TYPE)) > 0)
    m_type_region = regions.at(COL(TYPE))[0];
  if (regions.count(COL(MOUNT)) > 0)
    m_mount_region = regions.at(COL(MOUNT))[0];
  if (regions.count(COL(POSITION)) > 0)
    m_position_region = regions.at(COL(POSITION))[0];
  if (regions.count(COL(OFFSET)) > 0)
    m_offset_region = regions.at(COL(OFFSET))[0];
  if (regions.count(COL(DISH_DIAMETER)) > 0)
    m_dish_diameter_region = regions.at(COL(DISH_DIAMETER))[0];
  if (regions.count(COL(ORBIT_ID)) > 0)
    m_orbit_id_region = regions.at(COL(ORBIT_ID))[0];
  if (regions.count(COL(MEAN_ORBIT)) > 0)
    m_mean_orbit_region = regions.at(COL(MEAN_ORBIT))[0];
  if (regions.count(COL(PHASED_ARRAY_ID)) > 0)
    m_phased_array_id_region = regions.at(COL(PHASED_ARRAY_ID))[0];
  if (regions.count(COL(FLAG_ROW)) > 0)
    m_flag_row_region = regions.at(COL(FLAG_ROW))[0];

#ifdef HYPERION_USE_CASACORE
  if (regions.count("POSITION_MEAS_REF") > 0) {
    auto prs = regions.at("POSITION_MEAS_REF");
    if (prs.size() > 0)
      m_position_mr =
        MeasRefDict::get<M_POSITION>(
          MeasRefContainer::make_dict(ctx, rt, prs.begin(), prs.end())
          .get("Position").value());
  }

  if (regions.count("OFFSET_MEAS_REF") > 0) {
    auto prs = regions.at("OFFSET_MEAS_REF");
    if (prs.size() > 0)
      m_offset_mr =
        MeasRefDict::get<M_POSITION>(
          MeasRefContainer::make_dict(ctx, rt, prs.begin(), prs.end())
          .get("Position").value());
  }
#endif // HYPERION_USE_CASACORE
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
