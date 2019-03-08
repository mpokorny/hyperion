#include <iostream>

#include "TestLog.h"

using namespace legms;
using namespace Legion;

enum {
  TEST_RUNNER_TASK_ID,
};

void
test_runner_task(
  const Task*,
  const std::vector<PhysicalRegion>&,
  Context ctx,
  Runtime* runtime) {

  // initialize the test log
  testing::TestLogReference logref(100, ctx, runtime);
  testing::TestLog<WRITE_DISCARD>(logref).for_each(
    [](auto& it) {
      it <<= testing::TestResult<READ_ONLY>{
        testing::TestState::UNKNOWN,
          false,
          "",
          ""};
    });

  // print out the test log
  std::ostringstream oss;
  testing::TestLog<READ_ONLY>(logref).for_each(
    [&oss](auto& it) {
      auto test_result = *it;
      switch (test_result.state) {
      case testing::TestState::SUCCESS:
        oss << "PASS: "
            << test_result.location
            << std::endl;
        break;
      case testing::TestState::FAILURE:
        oss << "FAIL: "
            << test_result.location
            << ": "
            << test_result.description
            << std::endl;
        break;
      case testing::TestState::SKIPPED:
        oss << "SKIPPED: "
            << test_result.location
            << std::endl;
        break;
      case testing::TestState::UNKNOWN:
        break;
      }
    });
}

int
main(int argc, char* argv[]) {

  Runtime::set_top_level_task_id(TEST_RUNNER_TASK_ID);
  SerdezManager::register_ops();

  {
    TaskVariantRegistrar registrar(TEST_RUNNER_TASK_ID, "test_runner");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<test_runner_task>(registrar, "top_level");
  }

  return Runtime::start(argc, argv);

}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
