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
#include <hyperion/testing/TestExpression.h>

#include <iostream>
#include <stdexcept>

using namespace hyperion;
using namespace Legion;

enum {
  TEST_RECORDER_TEST_SUITE_ID,
};

std::string
verify_result(
  const testing::TestResult<READ_WRITE>& tr,
  const testing::TestResult<READ_ONLY>& expected) {

  const char* sep = "";
  std::ostringstream oss;
  if (tr.name != expected.name) {
    oss << "'name' expected '" << expected.name
        << "', got '" << tr.name << "'";
    sep = "; ";
  }
  if (tr.state != expected.state) {
    oss << sep
        << "'state' expected " << expected.state
        << ", got " << tr.state;
    sep = "; ";
  }
  if (tr.abort != expected.abort) {
    oss << sep
        << "'abort' expected " << expected.abort
        << ", got " << tr.abort;
    sep = "; ";
  }
  if (tr.fail_info != expected.fail_info) {
    oss << sep
        << "'fail_info' expected '" << expected.fail_info
        << "', got '" << tr.fail_info << "'";
    sep = "; ";
  }
  return oss.str();
}

void
test_recorder_test_suite(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *runtime) {

  testing::TestLog<READ_WRITE>
    log(
      task->regions[0].region,
      regions[0],
      task->regions[1].region,
      regions[1],
      ctx,
      runtime);
  testing::TestRecorder<READ_WRITE> recorder(log);

  testing::TestResult<READ_ONLY> dummy_success{
    testing::TestState::SUCCESS,
    false,
    "Dummy success",
    ""};
  recorder.append(dummy_success.name, dummy_success.state);

  testing::TestResult<READ_ONLY> dummy_success_testresult{
    testing::TestState::SUCCESS,
    false,
    "Dummy success TestResult",
    ""};
  recorder.append(dummy_success_testresult);

  testing::TestResult<READ_ONLY> dummy_failure{
    testing::TestState::FAILURE,
    false,
    "Dummy failure",
    "Expected FAILURE"};
  recorder.append(dummy_failure);

  testing::TestResult<READ_ONLY> dummy_append_success{
    testing::TestState::SUCCESS,
    false,
    "Dummy append_success",
    ""};  
  recorder.append_success(dummy_append_success.name);

  testing::TestResult<READ_ONLY> dummy_append_failure{
    testing::TestState::FAILURE,
    false,
    "Dummy append_failure",
    "Expected failure"};  
  recorder.append_failure(
    dummy_append_failure.name,
    dummy_append_failure.fail_info);

  testing::TestResult<READ_ONLY> dummy_append_skipped{
    testing::TestState::SKIPPED,
    false,
    "Dummy append_skipped",
    ""};  
  recorder.append_skipped(dummy_append_skipped.name);

  auto log_readback = log.iterator();
  {
    const char* name = "Read back dummy success";
    testing::TestResult<READ_WRITE> test_result(*log_readback);
    std::string errors = verify_result(test_result, dummy_success);
    recorder.expect_true(name, errors.size() == 0, errors);
  }
  ++log_readback;

  {
    const char *name = "Read back dummy success TestResult";
    testing::TestResult<READ_WRITE> test_result(*log_readback);
    std::string errors = verify_result(test_result, dummy_success_testresult);
    recorder.expect_true(name, errors.size() == 0, errors);
  }
  ++log_readback;

  {
    const char *name = "Read back dummy failure";
    testing::TestResult<READ_WRITE> test_result(*log_readback);
    std::string errors = verify_result(test_result, dummy_failure);
    recorder.expect_true(name, errors.size() == 0, errors);
  }
  ++log_readback;

  {
    const char *name = "Read back dummy append success";
    testing::TestResult<READ_WRITE> test_result(*log_readback);
    std::string errors = verify_result(test_result, dummy_append_success);
    recorder.expect_true(name, errors.size() == 0, errors);
  }
  ++log_readback;

  {
    const char *name = "Read back dummy append failure";
    testing::TestResult<READ_WRITE> test_result(*log_readback);
    std::string errors = verify_result(test_result, dummy_append_failure);
    recorder.expect_true(name, errors.size() == 0, errors);
  }
  ++log_readback;

  {
    const char *name = "Read back dummy append skipped";
    testing::TestResult<READ_WRITE> test_result(*log_readback);
    std::string errors = verify_result(test_result, dummy_append_skipped);
    recorder.expect_true(name, errors.size() == 0, errors);
  }
  ++log_readback;

  {
    unsigned val = 42;
    Future fval = Future::from_value(runtime, val);
    recorder.expect_true(
      "Evaluate Future in TestExpression",
      testing::TestFuture<decltype(val)>(fval) == testing::TestVal(val));
  }
  ++log_readback;

  testing::TestResult<READ_ONLY> dummy_append_abort{
    testing::TestState::FAILURE,
      true,
      "Dummy append_abort",
      "Expected abort"};
  recorder.append_failure(
    dummy_append_abort.name,
    dummy_append_abort.fail_info,
    dummy_append_abort.abort);

  {
    const char *name = "Read back dummy append abort";
    testing::TestResult<READ_WRITE> test_result(*log_readback);
    std::string errors = verify_result(test_result, dummy_append_abort);
    // note that the following logs a "SKIPPED" result, since the logged ABORT
    // causes all following tests to be skipped; nevertheless, if the test
    // runner compares outputs, the error message "errors" will signify an error
    // in logging
    recorder.expect_true(name, errors.size() == 0, errors);
  }
  ++log_readback;
}

int
main(int argc, char* argv[]) {

  testing::TestSuiteDriver driver =
    testing::TestSuiteDriver::make<test_recorder_test_suite>(
      TEST_RECORDER_TEST_SUITE_ID,
      "test_recorder_test_suite");

  return driver.start(argc, argv);

}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
