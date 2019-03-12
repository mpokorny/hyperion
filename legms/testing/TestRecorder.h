#ifndef LEGMS_TESTING_TEST_RECORDER_H_
#define LEGMS_TESTING_TEST_RECORDER_H_

#include "TestLog.h"

namespace legms {
namespace testing {

class TestRecorder {
public:

  TestRecorder(const TestLog<READ_WRITE>& log)
    : m_log_iter(log.iterator()) {
  }

  virtual ~TestRecorder() {
  }

  template <legion_privilege_mode_t MODE>
  void
  append(const TestResult<MODE>& tr) {
    m_log_iter <<= tr;
    ++m_log_iter;
  }

  inline void
  append(
    const std::string& name,
    int state,
    const std::string& fail_info = "",
    bool abort = false) {

    append(TestResult<READ_ONLY>{ state, abort, name, fail_info});
  }

  inline void
  append_success(const std::string& name) {
    append(name, testing::TestState::SUCCESS);
  }

  inline void
  append_failure(
    const std::string& name,
    const std::string& fail_info = "",
    bool abort = false) {

    append(name, testing::TestState::FAILURE, fail_info, abort);
  }

  inline void
  append_skipped(const std::string& name) {
    append(name, testing::TestState::SKIPPED);
  }

private:

  TestLogIterator<READ_WRITE> m_log_iter;
};

} // end namespace testing
} // end namespace legms

#endif // LEGMS_TESTING_TEST_RECORDER_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
