#ifndef LEGMS_TESTING_TEST_SUITE_DRIVER_H_
#define LEGMS_TESTING_TEST_SUITE_DRIVER_H_

#pragma GCC visibility push(default)
#include <array>
#pragma GCC visibility pop

#include "legms.h"
#include "utility.h"

#ifndef TEST_SUITE_DRIVER_LOG_LENGTH
# define TEST_SUITE_DRIVER_LOG_LENGTH 100
#endif

namespace legms {
namespace testing {

class LEGMS_API TestSuiteDriver {
public:

  static Legion::TaskID TASK_ID;
  static constexpr const char* TASK_NAME = "test_suite_driver_task";

  typedef void (*task_ptr_t)(
    const Legion::Task*,
    const std::vector<Legion::PhysicalRegion>&,
    Legion::Context,
    Legion::Runtime*);

  template <typename UDT>
  using task_udt_ptr_t =
    void (*)(
      const Legion::Task*,
      const std::vector<Legion::PhysicalRegion>&,
      Legion::Context,
      Legion::Runtime*,
      const UDT&);

  template <task_ptr_t TASK_PTR>
  static TestSuiteDriver
  make(
    Legion::TaskID test_suite_task,
    const char* test_suite_name,
    size_t log_length=TEST_SUITE_DRIVER_LOG_LENGTH) {

    TestSuiteDriver driver(test_suite_task, log_length);
    Legion::TaskVariantRegistrar registrar(test_suite_task, test_suite_name);
    registrar.add_constraint(
      Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    Legion::Runtime::preregister_task_variant<TASK_PTR>(
      registrar,
      test_suite_name);
    return driver;
  }

  template <typename UDT, task_udt_ptr_t<UDT> TASK_PTR>
  static TestSuiteDriver
  make(
    Legion::TaskID test_suite_task,
    const char* test_suite_name,
    const UDT& user_data,
    size_t log_length=TEST_SUITE_DRIVER_LOG_LENGTH) {

    TestSuiteDriver driver(test_suite_task, log_length);
    Legion::TaskVariantRegistrar registrar(test_suite_task, test_suite_name);
    registrar.add_constraint(
      Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    Legion::Runtime::preregister_task_variant<TASK_PTR>(
      registrar,
      user_data,
      test_suite_name);
    return driver;
  }

  int
  start(int argc, char* argv[]);

  struct TaskArgs {
    Legion::TaskID test_suite_task;
    size_t log_length;
  };

  static void
  impl(
    const Legion::Task*,
    const std::vector<Legion::PhysicalRegion>&,
    Legion::Context context,
    Legion::Runtime* runtime,
    const TaskArgs& args);

protected:

  TestSuiteDriver(Legion::TaskID test_suite_task, size_t log_length) {

    m_args.test_suite_task = test_suite_task;
    m_args.log_length = log_length;

    TASK_ID = Legion::Runtime::generate_static_task_id();
    Legion::TaskVariantRegistrar registrar(TASK_ID, TASK_NAME);
    registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    Legion::Runtime::preregister_task_variant<TaskArgs, impl>(
      registrar,
      m_args,
      TASK_NAME);
    Legion::Runtime::set_top_level_task_id(TASK_ID);
    preregister_all();
  }

private:

  TaskArgs m_args;
};

} // end namespace testing
} // end namespace legms

#endif // LEGMS_TESTING_TEST_DRIVER_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
