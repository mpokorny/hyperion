#include "testing/TestSuiteDriver.h"
#include "testing/TestRecorder.h"

#include "Measures.h"
#include <casacore/measures/Measures/MEpoch.h>
#include <casacore/measures/Measures/MeasData.h>

#ifdef LEGMS_USE_CASACORE

using namespace legms;
using namespace Legion;

enum {
  MEASURES_TEST_SUITE,
};

void
measures_test_suite(
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
  casacore::MEpoch utc_val(mjd2000, reftai);
  {
    MeasureRegion utc_region = MeasureRegion::create_from(ctx, rt, utc_val);
    auto val = utc_region.make<casacore::MEpoch>(ctx, rt);
    recorder.expect_true(
      "Readback of region initialized from MEpoch has expected value",
      utc_val.get("s") == val->get("s"));
    utc_region.destroy(ctx, rt);
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
