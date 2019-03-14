#include <iostream>

#include "TestLog.h"
#include "TestSuiteDriver.h"

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

  std::string name = "subtask" + std::to_string(task->index_point[0]);
  testing::TestLog<READ_WRITE> log(regions[0], regions[1], ctx, runtime);
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

  testing::TestLog<READ_WRITE> log(regions[0], regions[1], ctx, runtime);
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

  auto subtask_is = runtime->create_index_space(ctx, Rect<1>(0, 1));
  auto remaining_log =
    log.get_log_references_by_state({testing::TestState::UNKNOWN})[0];
  IndexPartitionT<1> subtask_log_ip(
    runtime->create_equal_partition(
      ctx,
      remaining_log.log_region().get_index_space(),
      subtask_is));
  LogicalPartitionT<1> subtask_logs(
    runtime->get_logical_partition(
      ctx,
      regions[0].get_logical_region(),
      subtask_log_ip));
  IndexTaskLauncher subtasks(
    TEST_LOG_SUBTASK_ID,
    subtask_is,
    TaskArgument(),
    ArgumentMap());
  auto reqs = log.log_reference().requirements<READ_WRITE>(subtask_logs, 0);
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
