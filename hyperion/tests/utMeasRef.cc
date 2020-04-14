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
#include <hyperion/testing/TestSuiteDriver.h>
#include <hyperion/testing/TestRecorder.h>
#include <hyperion/MeasRef.h>

#include <casacore/measures/Measures/MeasData.h>
#include <casacore/measures/Measures/MCEpoch.h>

using namespace hyperion;
using namespace Legion;

enum {
  MEAS_REF_TEST_SUITE,
};

#define TE(f) testing::TestEval([&](){ return f; }, #f)

void
meas_ref_test_suite(
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

  casacore::MEpoch::Ref reftai(casacore::MEpoch::TAI);
  casacore::MEpoch::Ref refutc(casacore::MEpoch::UTC);
  casacore::Quantity mjd2000(casacore::MeasData::MJD2000, "d");
  casacore::Quantity mjdb1950(casacore::MeasData::MJDB1950, "d");
  {
    MeasRef mr_tai = MeasRef::create(ctx, rt, reftai);
    recorder.expect_true(
      "MEpoch::Ref region has expected measure class",
      TE(mr_tai.mclass(ctx, rt) == M_EPOCH));
    recorder.expect_false(
      "Readback of MEpoch::Ref region cannot be instantiated as another class",
      testing::TestEval(
        [&mr_tai, &ctx, rt](){
          auto ref = std::get<0>(mr_tai.make<casacore::MDirection>(ctx, rt));
          return ref.size() == 1;
        }));
    recorder.expect_true(
      "MeasRef equivalence relation is reflexive",
      TE(mr_tai.equiv(ctx, rt, mr_tai)));
    auto ref = std::get<0>(mr_tai.make<casacore::MEpoch>(ctx, rt));
    recorder.assert_true(
      "Instance of MEpoch::Ref region has expected class",
      TE(ref.size() == 1));
    recorder.expect_true(
      "Instance of MEpoch::Ref region has expected MEpoch::Ref type",
      TE(ref[0]->getType() == reftai.getType()));
    recorder.expect_true(
      "Instance of MEpoch::Ref region has expected offset",
      TE(ref[0]->offset() == nullptr));
    recorder.expect_true(
      "Instance of MEpoch::Ref region has expected frame",
      testing::TestEval(
        [&ref](){
          auto frame = ref[0]->getFrame();
          return
            frame.epoch() == nullptr && frame.position() == nullptr
            && frame.direction() == nullptr && frame.radialVelocity() == nullptr
            && frame.comet() == nullptr;
        }));
    // TODO: add more comparisons -- casacore::MeasRef equality operator is
    // insufficient, as it relies on pointer comparisons, so will have to
    // dissect the MeasRefs and compare parts (which?) individually by value

    {
      casacore::MEpoch val_2000(mjd2000, reftai);
      casacore::MEpoch val_ref(mjd2000, *ref[0]);
      recorder.expect_true(
        "MEpoch value using MeasRef reference equals MEpoch value using original reference",
        val_2000.get("s") == val_ref.get("s"));
    }
    mr_tai.destroy(ctx, rt);
  }
  {
    casacore::MEpoch val_1950(mjdb1950, reftai);
    casacore::MEpoch::Ref ref1950(casacore::MEpoch::TAI, val_1950);
    casacore::MVEpoch v20_50(
      casacore::Quantity(
        casacore::MeasData::MJD2000 - casacore::MeasData::MJDB1950,
        "d"));
    casacore::MEpoch e20_50(v20_50, ref1950);
    MeasRef mr_1950 = MeasRef::create(ctx, rt, ref1950);
    auto ref = std::get<0>(mr_1950.make<casacore::MEpoch>(ctx, rt));
    recorder.expect_true(
      "Instance of MEpoch::Ref with offset region has same MEpoch value as original",
      testing::TestEval(
        [&v20_50, &ref]() {
          auto ep = casacore::MEpoch(v20_50, *ref[0]);
          return ep.getValue() == v20_50;
        }));

    casacore::MEpoch val_ref(v20_50, *ref[0]);
    recorder.expect_true(
      "MEpoch value using MeasRef reference with offset equals MEpoch value using original reference",
      e20_50.get("s") == val_ref.get("s"));

    casacore::MEpoch::Convert tai_to_utc20_50(e20_50, refutc);
    casacore::MEpoch::Convert tai_to_utc(val_ref, refutc);
    recorder.expect_true(
      "Conversion using MeasRef reference equals conversion using original reference",
      tai_to_utc20_50().get("d") == tai_to_utc().get("d"));

    mr_1950.destroy(ctx, rt);
  }
}

int
main(int argc, char* argv[]) {

  testing::TestSuiteDriver driver =
    testing::TestSuiteDriver::make<meas_ref_test_suite>(
      MEAS_REF_TEST_SUITE,
      "meas_ref_test_suite");

  return driver.start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
