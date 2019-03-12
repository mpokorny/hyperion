#include <iostream>

#include "TestLog.h"

using namespace legms;
using namespace Legion;

enum {
  TEST_SUITE_DRIVER_TASK_ID,
  TEST_LOG_TEST_SUITE_ID,
  TEST_LOG_SUBTASK_ID,
};

#define LOG_LENGTH 100

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

void
test_suite_driver_task(
  const Task*,
  const std::vector<PhysicalRegion>&,
  Context context,
  Runtime* runtime) {

  // initialize the test log
  testing::TestLogReference logref(LOG_LENGTH, context, runtime);
  testing::TestLog<WRITE_DISCARD>(logref, context, runtime).initialize();

  TaskLauncher test(TEST_LOG_TEST_SUITE_ID, TaskArgument());
  auto reqs = logref.requirements<READ_WRITE>();
  test.add_region_requirement(reqs[0]);
  test.add_region_requirement(reqs[1]);
  runtime->execute_task(context, test);

  // print out the test log
  std::ostringstream oss;
  testing::TestLog<READ_ONLY>(logref, context, runtime).for_each(
    [&oss](auto& it) {
      auto test_result = *it;
      switch (test_result.state) {
      case testing::TestState::SUCCESS:
        oss << "PASS: "
            << test_result.name
            << std::endl;
        break;
      case testing::TestState::FAILURE:
        oss << "FAIL: "
            << test_result.name;
        if (test_result.fail_info.size() > 0)
          oss << ": "
              << test_result.fail_info;
        oss << std::endl;
        break;
      case testing::TestState::SKIPPED:
        oss << "SKIPPED: "
            << test_result.name
            << std::endl;
        break;
      case testing::TestState::UNKNOWN:
        break;
      }
    });
  std::cout << oss.str();
}

int
main(int argc, char* argv[]) {

  Runtime::set_top_level_task_id(TEST_SUITE_DRIVER_TASK_ID);
  SerdezManager::register_ops();

  {
    TaskVariantRegistrar registrar(TEST_SUITE_DRIVER_TASK_ID, "test_driver");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<test_suite_driver_task>(
      registrar,
      "test_driver");
  }
  {
    TaskVariantRegistrar registrar(TEST_LOG_TEST_SUITE_ID, "test_suite");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<test_log_test_suite>(
      registrar,
      "test_suite");
  }
  {
    TaskVariantRegistrar registrar(TEST_LOG_SUBTASK_ID, "subtask_suite");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<test_log_subtask>(
      registrar,
      "subtask_suite");
  }

  return Runtime::start(argc, argv);

}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
