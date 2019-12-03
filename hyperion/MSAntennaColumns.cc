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

using namespace hyperion;
using namespace Legion;

MSAntennaColumns::MSAntennaColumns(
  Context ctx,
  Runtime* rt,
  const RegionRequirement& rows_requirement,
  const std::optional<PhysicalRegion>& name_region,
  const std::optional<PhysicalRegion>& station_region,
  const std::optional<PhysicalRegion>& type_region,
  const std::optional<PhysicalRegion>& mount_region,
  const std::optional<PhysicalRegion>& position_region,
#ifdef HYPERION_USE_CASACORE
  const std::vector<PhysicalRegion>&
  position_position_regions,
#endif
  const std::optional<PhysicalRegion>& offset_region,
#ifdef HYPERION_USE_CASACORE
  const std::vector<PhysicalRegion>&
  offset_position_regions,
#endif
  const std::optional<PhysicalRegion>& dish_diameter_region,
  const std::optional<PhysicalRegion>& orbit_id_region,
  const std::optional<PhysicalRegion>& mean_orbit_region,
  const std::optional<PhysicalRegion>& phased_array_id_region,
  const std::optional<PhysicalRegion>& flag_row_region)
  : m_name_region(name_region)
  , m_station_region(station_region)
  , m_type_region(type_region)
  , m_mount_region(mount_region)
  , m_position_region(position_region)
  , m_offset_region(offset_region)
  , m_dish_diameter_region(dish_diameter_region)
  , m_orbit_id_region(orbit_id_region)
  , m_mean_orbit_region(mean_orbit_region)
  , m_phased_array_id_region(phased_array_id_region)
  , m_flag_row_region(flag_row_region) {

#ifdef HYPERION_USE_CASACORE
  if (position_position_regions.size() > 0)
    m_position_position =
      MeasRefDict::get<M_POSITION>(
        MeasRefContainer::make_dict(
          ctx,
          rt,
          position_position_regions.begin(),
          position_position_regions.end())
        .get("Position").value());

  if (offset_position_regions.size() > 0)
    m_offset_position =
      MeasRefDict::get<M_POSITION>(
        MeasRefContainer::make_dict(
          ctx,
          rt,
          offset_position_regions.begin(),
          offset_position_regions.end())
        .get("Position").value());
#endif // HYPERION_USE_CASACORE
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
