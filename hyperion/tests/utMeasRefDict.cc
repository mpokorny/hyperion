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
#include <hyperion/testing/TestSuiteDriver.h>
#include <hyperion/testing/TestRecorder.h>
#include <hyperion/MeasRefDict.h>

#include <casacore/measures/Measures/MeasData.h>

using namespace hyperion;
using namespace Legion;

enum {
  MEAS_REF_DICT_TEST_SUITE,
};

#define TE(f) testing::TestEval([&](){ return f; }, #f)

bool
check_dict_value_type(
  Context ctx,
  Runtime* rt,
  const MeasRefDict::Ref& value,
  const MeasRef& ref) {

  bool result = false;
  switch (ref.mclass(ctx, rt)) {
#define CHECK(M)                                                        \
    case M:                                                             \
      result = MeasRefDict::holds<M>(value);                            \
    break;
    HYPERION_FOREACH_MCLASS(CHECK)
  default:
    assert(false);
    break;
  }
  return result;
}

void
meas_ref_dict_test_suite(
  const Task *task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  register_tasks(ctx, rt);

  testing::TestRecorder<READ_WRITE> recorder(
    testing::TestLog<READ_WRITE>(
      task->regions[0].region,
      regions[0],
      task->regions[1].region,
      regions[1],
      ctx,
      rt));

  {
    casacore::MEpoch::Ref ref_tai(casacore::MEpoch::TAI);
    MeasRef mr_tai = MeasRef::create(ctx, rt, "EPOCH", ref_tai);
    casacore::MDirection::Ref ref_j2000(casacore::MDirection::J2000);
    MeasRef mr_j2000 = MeasRef::create(ctx, rt, "DIRECTION", ref_j2000);
    casacore::MPosition::Ref ref_wgs84(casacore::MPosition::WGS84);
    MeasRef mr_wgs84 = MeasRef::create(ctx, rt, "POSITION", ref_wgs84);
    casacore::MFrequency::Ref ref_geo(casacore::MFrequency::GEO);
    MeasRef mr_geo = MeasRef::create(ctx, rt, "FREQUENCY", ref_geo);
    casacore::MRadialVelocity::Ref ref_topo(casacore::MRadialVelocity::TOPO);
    MeasRef mr_topo = MeasRef::create(ctx, rt, "RADIAL_VELOCITY", ref_topo);
    casacore::MDoppler::Ref ref_z(casacore::MDoppler::Z);
    MeasRef mr_z = MeasRef::create(ctx, rt, "DOPPLER", ref_z);
    std::vector<const MeasRef*>
      mrs{&mr_tai, &mr_j2000, &mr_wgs84, &mr_geo, &mr_topo, &mr_z};

    {
      MeasRefDict dict(ctx, rt, mrs);
      recorder.expect_false(
        "Empty optional value returned for non-existent MeasRef name",
        TE(dict.get("FOOBAR").has_value()));
      for (auto& mr : mrs) {
        auto name = mr->name(ctx, rt);
        auto ref = dict.get(name);
        recorder.assert_true(
          std::string("Non-empty optional value returned for MeasRef ") + name,
          TE(ref.has_value()));
        recorder.expect_true(
          std::string("Contained value for MeasRef ")
          + name + " has expected type",
          TE(check_dict_value_type(ctx, rt, ref.value(), *mr)));
      }
    }

    for (auto& mr : mrs)
      const_cast<MeasRef*>(mr)->destroy(ctx, rt);
  }
}

int
main(int argc, char* argv[]) {

  testing::TestSuiteDriver driver =
    testing::TestSuiteDriver::make<meas_ref_dict_test_suite>(
      MEAS_REF_DICT_TEST_SUITE,
      "meas_ref_dict_test_suite");

  return driver.start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
