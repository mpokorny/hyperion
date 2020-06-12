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
#include <hyperion/testing/TestLog.h>
#include <hyperion/utility.h>

using namespace hyperion;
using namespace hyperion::testing;
using namespace Legion;

void
TestSuiteDriver::reporter_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context context,
  Runtime* runtime) {

  TestLog<READ_WRITE>
    log(
      task->regions[0].region,
      regions[0],
      task->regions[1].region,
      regions[1],
      context,
      runtime);

  // print out the test log
  std::ostringstream oss;
  log.for_each(
    [&oss](const TestLogIterator<READ_WRITE>& it) {
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

Legion::TaskID TestSuiteDriver::DRIVER_TASK_ID;
Legion::TaskID TestSuiteDriver::REPORTER_TASK_ID;

int
TestSuiteDriver::start(int argc, char* argv[]) {
  return Runtime::start(argc, argv);
}

void
TestSuiteDriver::driver_task(
  const Task* task,
  const std::vector<PhysicalRegion>&,
  Context context,
  Runtime* runtime,
  const TestSuiteDriver::TaskArgs& args) {

  // initialize the test log
  TestLogReference logref(args.log_length, context, runtime);

  // run the test suite
  {
    TaskLauncher test(args.test_suite_task, TaskArgument());
    auto reqs = logref.requirements<READ_WRITE>();
    test.add_region_requirement(reqs[0]);
    test.add_region_requirement(reqs[1]);
    runtime->unmap_all_regions(context);
    runtime->execute_task(context, test);
  }
  {
    TaskLauncher report(REPORTER_TASK_ID, TaskArgument(NULL, 0));
    // NB: although READ_ONLY privileges ought to be sufficient for the reporter
    // task, TestLogReference<READ_ONLY> appears broken (for the abort state
    // region, it seems), so we use READ_WRITE privileges instead
    auto reqs = logref.requirements<READ_WRITE>();
    report.add_region_requirement(reqs[0]);
    report.add_region_requirement(reqs[1]);
    runtime->execute_task(context, report);
  }
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
