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
#ifndef HYPERION_TESTING_TEST_SUITE_DRIVER_H_
#define HYPERION_TESTING_TEST_SUITE_DRIVER_H_

#include <hyperion/hyperion.h>
#include <hyperion/utility.h>

#include <array>

#ifndef TEST_SUITE_DRIVER_LOG_LENGTH
# define TEST_SUITE_DRIVER_LOG_LENGTH 100
#endif

namespace hyperion {
namespace testing {

class HYPERION_API TestSuiteDriver {
public:

  static Legion::TaskID DRIVER_TASK_ID;
  static constexpr const char* DRIVER_TASK_NAME =
    "test_suite_driver_task";

  static Legion::TaskID REPORTER_TASK_ID;
  static constexpr const char* REPORTER_TASK_NAME =
    "test_suite_driver_reporter_task";

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
  driver_task(
    const Legion::Task*,
    const std::vector<Legion::PhysicalRegion>&,
    Legion::Context context,
    Legion::Runtime* runtime,
    const TaskArgs& args);

  static void
  reporter_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context context,
    Legion::Runtime* runtime);

protected:

  TestSuiteDriver(Legion::TaskID test_suite_task, size_t log_length) {

    m_args.test_suite_task = test_suite_task;
    m_args.log_length = log_length;

    {
      DRIVER_TASK_ID = Legion::Runtime::generate_static_task_id();
      Legion::TaskVariantRegistrar
        registrar(DRIVER_TASK_ID, DRIVER_TASK_NAME);
      registrar.add_constraint(
        Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
      Legion::Runtime::preregister_task_variant<TaskArgs, driver_task>(
        registrar,
        m_args,
        DRIVER_TASK_NAME);
      Legion::Runtime::set_top_level_task_id(DRIVER_TASK_ID);
    }
    {
      REPORTER_TASK_ID = Legion::Runtime::generate_static_task_id();
      Legion::TaskVariantRegistrar
        registrar(REPORTER_TASK_ID, REPORTER_TASK_NAME);
      registrar.add_constraint(
        Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
      Legion::Runtime::preregister_task_variant<reporter_task>(
        registrar,
        REPORTER_TASK_NAME);
    }
    preregister_all();
  }

private:

  TaskArgs m_args;
};

} // end namespace testing
} // end namespace hyperion

#endif // HYPERION_TESTING_TEST_DRIVER_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
