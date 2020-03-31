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
#include <hyperion/MSFieldTable.h>

#include <casacore/casa/BasicMath/Math.h>

using namespace hyperion;
using namespace Legion;

namespace cc = casacore;

cc::MDirection
MSFieldTable::interpolate_dir_meas(
  const std::vector<cc::MDirection>& dir_poly,
  double inter_time,
  double time_origin) {

  if ((dir_poly.size() == 1)
      || cc::nearAbs(inter_time, time_origin)) {
    return dir_poly[0];
  } else {
    cc::Vector<double> dir(dir_poly[0].getAngle().getValue());
    double dt = inter_time - time_origin;
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
