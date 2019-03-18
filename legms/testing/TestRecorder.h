#ifndef LEGMS_TESTING_TEST_RECORDER_H_
#define LEGMS_TESTING_TEST_RECORDER_H_

#include <exception>

#include "TestLog.h"
#include "TestExpression.h"

namespace legms {
namespace testing {

template <legion_privilege_mode_t MODE>
class TestRecorder {

public:

  TestRecorder(const TestLog<MODE>& log)
    : m_log(log)
    , m_log_iter(m_log.iterator())
    , m_aborted(m_log.contains_abort()) {
  }

  TestRecorder(TestLog<MODE>&& log)
    : m_log(std::forward<TestLog<MODE>>(log))
    , m_log_iter(m_log.iterator())
    , m_aborted(m_log.contains_abort()) {
  }

  virtual ~TestRecorder() {
  }

  template <legion_privilege_mode_t M>
  void
  append(const TestResult<M>& tr) {
    if (!m_aborted || tr.state == testing::TestState::SKIPPED) {
      m_log_iter <<= tr;
      m_aborted = m_aborted || tr.abort;
    } else {
      TestResult<READ_ONLY> tr1{
        testing::TestState::SKIPPED, false, tr.name, ""};
      m_log_iter <<= tr1;
    }
    ++m_log_iter;
  }

  inline void
  append(
    const std::string& name,
    int state,
    const std::string& fail_info = "",
    bool abort = false) {

    append(TestResult<READ_ONLY>{ state, abort, name, fail_info });
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

  template <template <typename> typename E>
  void
  expect_bool(
    const std::string& name,
    const TestExpression<bool, E>& expr,
    const std::string fail_info,
    bool state,
    bool assert) {

    if (m_aborted) {
      append_skipped(name);
    } else {
      try {
        if (expr() == state)
          append_success(name);
        else
          append_failure(
            name,
            ((fail_info.size() > 0) ? fail_info : expr.reason(!state)),
            assert);
      } catch (const std::exception& e) {
        std::string fail_info = "unexpected exception: ";
        append_failure(name, fail_info + e.what(), assert);
      }
    }
  }

  template <template <typename> typename E>
  void
  expect_true(
    const std::string& name,
    const TestExpression<bool, E>& expr,
    const std::string& fail_info="") {
    expect_bool(name, expr, fail_info, true, false);
  }

  void
  expect_true(
    const std::string& name,
    bool val,
    const std::string& fail_info="") {
    expect_bool(name, TestVal(val), fail_info, true, false);
  }

  template <template <typename> typename E>
  void
  assert_true(
    const std::string& name,
    const TestExpression<bool, E>& expr,
    const std::string& fail_info="") {
    expect_bool(name, expr, fail_info, true, true);
  }

  void
  assert_true(
    const std::string& name,
    bool val,
    const std::string& fail_info="") {
    expect_bool(name, TestVal(val), fail_info, true, true);
  }

  template <template <typename> typename E>
  void
  expect_false(
    const std::string& name,
    const TestExpression<bool, E>& expr,
    const std::string& fail_info="") {

    expect_bool(name, expr, fail_info, false, false);
  }

  void
  expect_false(
    const std::string& name,
    bool val,
    const std::string& fail_info="") {

    expect_bool(name, TestVal(val), fail_info, false, false);
  }

  template <template <typename> typename E>
  void
  assert_false(
    const std::string& name,
    const TestExpression<bool, E>& expr,
    const std::string& fail_info="") {

    expect_bool(name, expr, fail_info, false, true);
  }

  void
  assert_false(
    const std::string& name,
    bool val,
    const std::string& fail_info="") {
    expect_bool(name, TestVal(val), fail_info, false, true);
  }

  template <typename Exc, typename T, template <typename> typename E>
  void
  expect_throw(
    const std::string& name,
    const TestExpression<T, E>& expr,
    bool assert=false) {

    try {
      (void)expr()();
      append_failure(name, "no exception thrown", assert);
    } catch (const Exc& e) {
      append_success(name);
    } catch (const std::exception& e) {
      std::string fail_info = "unexpected exception: ";
      append_failure(name, fail_info + e.what(), assert);
    }
  }

  template <typename Exc, typename F>
  void
  expect_throw(const std::string& name, const F& f, bool assert=false) {
    expect_throw<Exc>(name, TestEval(f), assert);
  }

  template <typename Exc, typename T, template <typename> typename E>
  void
  assert_throw(const std::string& name, const TestExpression<T, E>& expr) {
    expect_throw<Exc>(name, expr, true);
  }

  template <typename Exc, typename F>
  void
  assert_throw(const std::string& name, const F& f) {
    expect_throw<Exc>(name, TestEval(f), true);
  }

private:

  TestLog<MODE> m_log;

  TestLogIterator<MODE> m_log_iter;

  bool m_aborted;
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
