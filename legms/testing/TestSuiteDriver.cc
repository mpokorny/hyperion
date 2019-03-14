#include "TestSuiteDriver.h"
#include "TestLog.h"
#include "utility.h"

using namespace legms;
using namespace legms::testing;
using namespace Legion;

int
TestSuiteDriver::start(int argc, char* argv[]) {
  return Runtime::start(argc, argv);
}

void
TestSuiteDriver::impl(
  const Task* task,
  const std::vector<PhysicalRegion>&,
  Context context,
  Runtime* runtime,
  const TestSuiteDriver::TaskArgs& args) {

  // initialize the test log
  TestLogReference logref(args.log_length, context, runtime);
  TestLog<WRITE_DISCARD>(logref, context, runtime).initialize();

  // run the test suite
  TaskLauncher test(args.test_suite_task, TaskArgument());
  auto reqs = logref.requirements<WRITE_DISCARD>();
  test.add_region_requirement(reqs[0]);
  test.add_region_requirement(reqs[1]);
  runtime->execute_task(context, test);

  // print out the test log
  std::ostringstream oss;
  TestLog<READ_ONLY>(logref, context, runtime).for_each(
    [&oss](auto& it) {
      auto test_result = *it;
      switch (test_result.state) {
      case TestState::SUCCESS:
        oss << "PASS\t"
            << test_result.name
            << std::endl;
        break;
      case TestState::FAILURE:
        oss << "FAIL\t"
            << test_result.name;
        if (test_result.fail_info.size() > 0)
          oss << "\t"
              << test_result.fail_info;
        oss << std::endl;
        break;
      case TestState::SKIPPED:
        oss << "SKIPPED\t"
            << test_result.name
            << std::endl;
        break;
      case TestState::UNKNOWN:
        break;
      }
    });
  std::cout << oss.str();

}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
