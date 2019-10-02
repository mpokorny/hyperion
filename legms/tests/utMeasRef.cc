#include "testing/TestSuiteDriver.h"
#include "testing/TestRecorder.h"

#include "MeasRef.h"

#ifdef LEGMS_USE_CASACORE
#include <casacore/measures/Measures/MeasData.h>
#include <casacore/casa/System/AppState.h>

using namespace legms;
using namespace Legion;

enum {
  MEAS_REF_TEST_SUITE,
};

class CasacoreState
  : public casacore::AppState {
public:

  CasacoreState() {}

  std::list<std::string>
  dataPath() const override {
    static std::list<std::string>
      result{"/users/mpokorny/projects/casa.git/data"};
    return result;
  }

  bool
  initialized() const override {
    return true;
  }
};

#define TE(f) testing::TestEval([&](){ return f; }, #f)

void
meas_ref_test_suite(
  const Task *task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  casacore::AppStateSource::initialize(new CasacoreState);
  
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
  casacore::Quantity mjdb1950(casacore::MeasData::MJDB1950, "d");
  {
    MeasRef mr_tai = MeasRef::create(ctx, rt, "EPOCH", reftai);
    recorder.expect_true(
      "MEpoch::Ref region has expected name",
      TE(mr_tai.name(ctx, rt) == "EPOCH"));
    recorder.expect_true(
      "MEpoch::Ref region has expected measure class",
      TE(mr_tai.mclass(ctx, rt) == M_EPOCH));
    recorder.expect_false(
      "Readback of MEpoch::Ref region cannot be instantiated as another class",
      testing::TestEval(
        [&mr_tai, &ctx, rt](){
          auto r = mr_tai.make<casacore::MDirection>(ctx, rt);
          return r.has_value();
        }));
    auto ref = mr_tai.make<casacore::MEpoch>(ctx, rt);
    recorder.assert_true(
      "Instance of MEpoch::Ref region has expected class",
      TE(ref.has_value()));
    recorder.expect_true(
      "Instance of MEpoch::Ref region has expected MEpoch::Ref type",
      TE(ref.value().getType() == reftai.getType()));
    recorder.expect_true(
      "Instance of MEpoch::Ref region has expected offset",
      TE(ref.value().offset() == nullptr));
    recorder.expect_true(
      "Instance of MEpoch::Ref region has expected frame",
      testing::TestEval(
        [&ref](){
          auto frame = ref.value().getFrame();
          return
            frame.epoch() == nullptr && frame.position() == nullptr
            && frame.direction() == nullptr && frame.radialVelocity() == nullptr
            && frame.comet() == nullptr;
        }));
    // TODO: add more comparisons -- casacore::MeasRef equality operator is
    // insufficient, as it relies on pointer comparisons, so will have to
    // dissect the MeasRefs and compare parts (which?) individually by value
    mr_tai.destroy(ctx, rt);
  }
  {
    casacore::MEpoch val1950(mjdb1950, reftai);
    casacore::MEpoch::Ref ref1950(casacore::MEpoch::TAI, val1950);
    casacore::MVEpoch v20_50(
      casacore::Quantity(
        casacore::MeasData::MJD2000 - casacore::MeasData::MJDB1950,
        "d"));
    casacore::MEpoch e20_50(v20_50, ref1950);
    MeasRef mr_1950 = MeasRef::create(ctx, rt, "EPOCH", ref1950);
    auto ref = mr_1950.make<casacore::MEpoch>(ctx, rt);
    recorder.expect_true(
      "Instance of MEpoch::Ref region with offset has same MEpoch value as original",
      testing::TestEval(
        [&v20_50, &e20_50, &ref]() {
          auto ep = casacore::MEpoch(v20_50, ref.value());
          return ep.getValue() == v20_50;
        }));
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

#endif // LEGMS_USE_CASACORE

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
