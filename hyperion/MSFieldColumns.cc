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
  const std::optional<PhysicalRegion>& name_region,
  const std::optional<PhysicalRegion>& code_region,
  const std::optional<PhysicalRegion>& time_region,
#ifdef HYPERION_USE_CASACORE
  const std::vector<PhysicalRegion>& time_epoch_regions,
#endif
  const std::optional<PhysicalRegion>& num_poly_region,
  const std::optional<PhysicalRegion>& delay_dir_region,
#ifdef HYPERION_USE_CASACORE
  const std::vector<PhysicalRegion>& delay_dir_direction_regions,
#endif
  const std::optional<PhysicalRegion>& phase_dir_region,
#ifdef HYPERION_USE_CASACORE
  const std::vector<PhysicalRegion>& phase_dir_direction_regions,
#endif
  const std::optional<PhysicalRegion>& reference_dir_region,
#ifdef HYPERION_USE_CASACORE
  const std::vector<PhysicalRegion>& reference_dir_direction_regions,
#endif
  const std::optional<PhysicalRegion>& source_id_region,
  const std::optional<PhysicalRegion>& ephemeris_id_region,
  const std::optional<PhysicalRegion>& flag_row_region)
  : m_rows_requirement(rows_requirement)
  , m_name_region(name_region)
  , m_code_region(code_region)
  , m_time_region(time_region)
  , m_num_poly_region(num_poly_region)
  , m_delay_dir_region(delay_dir_region)
  , m_phase_dir_region(phase_dir_region)
  , m_reference_dir_region(reference_dir_region)
  , m_source_id_region(source_id_region)
  , m_ephemeris_id_region(ephemeris_id_region)
  , m_flag_row_region(flag_row_region) {

#ifdef HYPERION_USE_CASACORE
  if (time_epoch_regions.size() > 0)
    m_time_epoch =
      MeasRefDict::get<M_EPOCH>(
        MeasRefContainer::make_dict(
          ctx,
          rt,
          time_epoch_regions.begin(),
          time_epoch_regions.end())
        .get("Epoch").value());

  if (delay_dir_direction_regions.size() > 0)
    m_delay_dir_direction =
      MeasRefDict::get<M_DIRECTION>(
        MeasRefContainer::make_dict(
          ctx,
          rt,
          delay_dir_direction_regions.begin(),
          delay_dir_direction_regions.end())
        .get("Direction").value());

  if (phase_dir_direction_regions.size() > 0)
    m_phase_dir_direction =
      MeasRefDict::get<M_DIRECTION>(
        MeasRefContainer::make_dict(
          ctx,
          rt,
          phase_dir_direction_regions.begin(),
          phase_dir_direction_regions.end())
        .get("Direction").value());

  if (reference_dir_direction_regions.size() > 0)
    m_reference_dir_direction =
      MeasRefDict::get<M_DIRECTION>(
        MeasRefContainer::make_dict(
          ctx,
          rt,
          reference_dir_direction_regions.begin(),
          reference_dir_direction_regions.end())
        .get("Direction").value());
#endif // HYPERION_USE_CASACORE
}

static cc::MDirection
interpolateDirMeas(
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

template <typename MF>
static cc::MDirection
dirMeas(
  const DataType<HYPERION_TYPE_DOUBLE>::ValueType* dd,
  const DataType<HYPERION_TYPE_INT>::ValueType& np,
  const DataType<HYPERION_TYPE_DOUBLE>::ValueType& t,
  double interTime,
  MF mf) {

  // TODO: support ephemerides as in cc::MSFieldColumns
  std::vector<cc::MDirection> dir_poly;
  dir_poly.reserve(np + 1);
  for (int i = 0; i < np + 1; ++i) {
    dir_poly.push_back(mf(dd));
    dd += 2;
  }
  return interpolateDirMeas(dir_poly, interTime, t);
}

cc::MDirection
MSFieldColumns::delayDirMeas(
  const DataType<HYPERION_TYPE_DOUBLE>::ValueType* dd,
  const DataType<HYPERION_TYPE_INT>::ValueType& np,
  const DataType<HYPERION_TYPE_DOUBLE>::ValueType& t,
  double interTime) const {

  return
    dirMeas(
      dd,
      np,
      t,
      interTime,
      [this](const DataType<HYPERION_TYPE_DOUBLE>::ValueType* d) {
        return delayDirMeas(d);
      });
}

cc::MDirection
MSFieldColumns::phaseDirMeas(
  const DataType<HYPERION_TYPE_DOUBLE>::ValueType* dd,
  const DataType<HYPERION_TYPE_INT>::ValueType& np,
  const DataType<HYPERION_TYPE_DOUBLE>::ValueType& t,
  double interTime) const {

  return
    dirMeas(
      dd,
      np,
      t,
      interTime,
      [this](const DataType<HYPERION_TYPE_DOUBLE>::ValueType* d) {
        return phaseDirMeas(d);
      });
}

cc::MDirection
MSFieldColumns::referenceDirMeas(
  const DataType<HYPERION_TYPE_DOUBLE>::ValueType* dd,
  const DataType<HYPERION_TYPE_INT>::ValueType& np,
  const DataType<HYPERION_TYPE_DOUBLE>::ValueType& t,
  double interTime) const {
  
  return
    dirMeas(
      dd,
      np,
      t,
      interTime,
      [this](const DataType<HYPERION_TYPE_DOUBLE>::ValueType* d) {
        return referenceDirMeas(d);
      });
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
