#include "testing/TestSuiteDriver.h"
#include "testing/TestRecorder.h"

#include "Measures.h"
#include <casacore/measures/Measures/MEpoch.h>
#include <casacore/measures/Measures/MeasData.h>
#include <casacore/measures/Measures/MCEpoch.h>
#include <casacore/casa/System/AppState.h>

#ifdef LEGMS_USE_CASACORE

using namespace legms;
using namespace Legion;

enum {
  MEASURES_TEST_SUITE,
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

void
measures_test_suite(
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
  casacore::MEpoch::Ref refutc(casacore::MEpoch::UTC);
  casacore::Quantity mjd2000(casacore::MeasData::MJD2000, "d");
  casacore::Quantity mjdb1950(casacore::MeasData::MJDB1950, "d");
  {
    casacore::MEpoch tai_val(mjd2000, reftai);
    MeasureRegion tai_region = MeasureRegion::create_from(ctx, rt, tai_val);
    auto val = tai_region.make<casacore::MEpoch>(ctx, rt);
    recorder.expect_true(
      "Readback of region initialized from MEpoch has expected value",
      tai_val.get("s") == val->get("s"));
    tai_region.destroy(ctx, rt);
  }
  {
    casacore::MEpoch val1950(mjdb1950, reftai);
    casacore::MEpoch::Ref ref1950(casacore::MEpoch::TAI, val1950);
    casacore::MEpoch val20_50(
      casacore::Quantity(
        casacore::MeasData::MJD2000 - casacore::MeasData::MJDB1950,
        "d"),
      ref1950);
    MeasureRegion reg20_50 = MeasureRegion::create_from(ctx, rt, val20_50);
    auto val = reg20_50.make<casacore::MEpoch>(ctx, rt);
    recorder.expect_true(
      "Readback of region initialized from MEpoch with reference has expected value",
      val20_50.get("s") == val->get("s"));
    casacore::MEpoch::Convert tai_to_utc20_50(val20_50, refutc);
    casacore::MEpoch::Convert tai_to_utc(*val, refutc);
    recorder.expect_true(
      "Conversion using instantiated MEpoch equals conversion using original",
      tai_to_utc20_50().get("d") == tai_to_utc().get("d"));
    reg20_50.destroy(ctx, rt);
  }
}

int
main(int argc, char* argv[]) {

  testing::TestSuiteDriver driver =
    testing::TestSuiteDriver::make<measures_test_suite>(
      MEASURES_TEST_SUITE,
      "measures_test_suite");

  return driver.start(argc, argv);
}

#endif // LEGMS_USE_CASACORE

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
