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
#include <iostream>

#include <legms/testing/TestLog.h>
#include <legms/testing/TestSuiteDriver.h>

using namespace legms;
using namespace Legion;

#define LOG_LENGTH 50

enum {
  TEST_LOG_TEST_SUITE_ID,
  TEST_LOG_SUBTASK_ID,
};

void
test_log_subtask(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *runtime) {

  assert(regions.size() == 2);
  std::string name = "subtask" + std::to_string(task->index_point[0]);
  testing::TestLog<READ_WRITE>
    log(
      task->regions[0].region,
      regions[0],
      task->regions[1].region,
      regions[1],
      ctx,
      runtime);
  auto log_output = log.iterator();
  log_output <<= testing::TestResult<READ_ONLY>{
    testing::TestState::SUCCESS,
      false,
      name,
      ""};
}

void
test_log_test_suite(
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
  size_t length = 0;
  bool all_unknown = true;
  log.for_each(
    [&length, &all_unknown](auto& it) {
      ++length;
      all_unknown = all_unknown && (*it).state == testing::TestState::UNKNOWN;
    });
  testing::TestResult<READ_WRITE> test_result;
  auto log_output = log.iterator();

  test_result.state = (
    (length == LOG_LENGTH)
    ? testing::TestState::SUCCESS
    : testing::TestState::FAILURE);
  test_result.abort = false;
  test_result.name = "Initial log size is correct";
  if (test_result.state != testing::TestState::SUCCESS) {
    std::ostringstream oss;
    oss << length << " lines, rather than the expected " << LOG_LENGTH;
    test_result.fail_info = oss.str();
  }
  log_output <<= test_result;
  ++log_output;

  test_result.state = (
    all_unknown
    ? testing::TestState::SUCCESS
    : testing::TestState::FAILURE);
  test_result.abort = false;
  test_result.name = "Log initialized all results to UNKNOWN";
  test_result.fail_info = "";
  log_output <<= test_result;
  ++log_output;

  test_result.state = (
    log.contains_abort()
    ? testing::TestState::FAILURE
    : testing::TestState::SUCCESS);
  test_result.abort = false;
  test_result.name = "Initial log not in ABORT state";
  log_output <<= test_result;
  ++log_output;

  test_result.state = testing::TestState::SKIPPED;
  test_result.abort = false;
  test_result.name = "Expected SKIPPED";
  log_output <<= test_result;
  ++log_output;

  test_result.state = testing::TestState::FAILURE;
  test_result.abort = false;
  test_result.name = "Expected FAILURE";
  test_result.fail_info = "OK";
  log_output <<= test_result;
  ++log_output;

  test_result.state = testing::TestState::FAILURE;
  test_result.abort = true;
  test_result.name = "Expected ABORT";
  test_result.fail_info = "OK";
  log_output <<= test_result;
  ++log_output;

  test_result.state = (
    log.contains_abort()
    ? testing::TestState::SUCCESS
    : testing::TestState::FAILURE);
  test_result.abort = false;
  test_result.name = "Log in ABORT state after ABORT result";
  test_result.fail_info = "";
  log_output <<= test_result;
  ++log_output;

  if (test_result.state == testing::TestState::SUCCESS)
    test_result.state = (
      log.contains_abort()
      ? testing::TestState::SUCCESS
      : testing::TestState::FAILURE);
  else
    test_result.state = testing::TestState::SKIPPED;
  test_result.abort = false;
  test_result.name = "ABORT state is irreversible";
  log_output <<= test_result;
  ++log_output;

  IndexSpace subtask_is(runtime->create_index_space(ctx, Rect<1>(0, 1)));
  auto remaining_log =
    log.get_log_references_by_state({testing::TestState::UNKNOWN})[0];
  IndexPartition subtask_log_ip =
    runtime->create_equal_partition(
      ctx,
      remaining_log.log_region().get_index_space(),
      subtask_is);
  LogicalPartitionT<1> subtask_logs(
    runtime->get_logical_partition(
      ctx,
      remaining_log.log_region(),
      subtask_log_ip));
  IndexTaskLauncher subtasks(
    TEST_LOG_SUBTASK_ID,
    subtask_is,
    TaskArgument(NULL, 0),
    ArgumentMap());
  auto reqs =
    remaining_log.requirements<READ_WRITE>(
      subtask_logs,
      log.log_reference().log_region(),
      0);
  std::for_each(
    reqs.begin(),
    reqs.end(),
    [&subtasks](auto& req) { subtasks.add_region_requirement(req); });
  runtime->execute_index_space(ctx, subtasks);
}

int
main(int argc, char* argv[]) {

  testing::TestSuiteDriver driver =
    testing::TestSuiteDriver::make<test_log_test_suite>(
      TEST_LOG_TEST_SUITE_ID,
      "test_log_test_suite",
      LOG_LENGTH);

  {
    TaskVariantRegistrar registrar(TEST_LOG_SUBTASK_ID, "subtask_suite");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<test_log_subtask>(
      registrar,
      "subtask_suite");
  }

  return driver.start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
