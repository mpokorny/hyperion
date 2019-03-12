#ifndef LEGMS_TESTING_TEST_RECORDER_H_
#define LEGMS_TESTING_TEST_RECORDER_H_

#include "TestLog.h"

namespace legms {
namespace testing {

class TestRecorder {
public:

  TestRecorder(const TestLog<READ_WRITE>& log)
    : m_log(
      log.get_log_references_by_state({TestState::UNKNOWN})[0],
      log.m_context,
      log.m_runtime)
    , m_log_iter(m_log.iterator()) {
  }

  TestRecorder(const TestLog<WRITE_DISCARD>& log)
    : m_log(
      log.get_log_references_by_state({TestState::UNKNOWN})[0],
      log.m_context,
      log.m_runtime)
    , m_log_iter(m_log.iterator()) {
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
    bool abort = false,
    const std::string& fail_info = "") {

    append(TestResult<READ_ONLY>{ state, abort, name, fail_info});
  }

private:

  TestLog<READ_WRITE> m_log;

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
