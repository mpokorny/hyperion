#ifndef LEGMS_TESTING_TEST_EXPRESSION_H_
#define LEGMS_TESTING_TEST_EXPRESSION_H_

#include <optional>
#include <sstream>

#include "legion.h"
#include "TestRecorder.h"

namespace legms{
namespace testing {

template <template <typename> typename E,
          template <typename> typename F>
class BoolOr;

template <template <typename> typename E,
          template <typename> typename F>
class BoolAnd;

template <typename T,
          template <typename> typename E,
          typename U,
          template <typename> typename F>
class BoolEq;

template <typename T, template <typename> typename E>
struct TestExpression {

  T
  operator()() const {
    return static_cast<const E<T>&>(*this)();
  }

  template <template <typename> typename F>
  BoolOr<E, F>
  operator||(const TestExpression<bool, F>& right) const {
    return BoolOr(*this, right);
  }

  template <template <typename> typename F>
  BoolAnd<E, F>
  operator&&(const TestExpression<bool, F>& right) const {
    return BoolAnd(*this, right);
  }

  template <typename U,
            template <typename> typename F>
  BoolEq<T, E, U, F>
  operator==(const TestExpression<U, F>& right) const {
    return BoolEq(*this, right);
  }

  template <typename U>
  BoolEq<T, E, U, TestVal>
  operator==(const TestVal<U>& right) const {
    return BoolEq(*this, right);
  }

  template <typename U>
  BoolEq<T, E, U, TestVal>
  operator==(const U& right) const {
    return BoolEq(*this, TestVal(right));
  }

  std::string
  reason(bool state) const {
    return static_cast<const E<T>&>(*this).reason(state);
  }
};

template <typename T>
std::string
to_string(T v) {
  std::ostringstream oss;
  oss << v;
  return oss.str();
}

template <typename T>
struct TestVal
  : public TestExpression<T, TestVal> {

  TestVal(T val, const std::optional<std::string>& repr=std::nullopt)
    : m_val(val)
    , m_repr(repr.value_or(to_string(val))) {}

  template <template <typename> typename E>
  TestVal(TestExpression<T, E> expr)
    : m_val(expr()) {}

  T
  operator()() const {
    return m_val;
  }

  std::string
  reason(bool state) const {
    std::ostringstream oss;
    if (state)
      oss << "Value is TRUE";
    else
      oss << "Value is FALSE";
    return oss.str();
  }

  const std::string&
  repr() const {
    return m_repr;
  }

private:

  T m_val;

  std::string m_repr;
};

template <typename F, typename T>
struct TestEval;

template <typename F, typename T>
struct TestEvalF {
  template <typename S> using Expr = TestEval<F, S>;
};

template <typename F,
          typename T = typename std::invoke_result_t<F>>
struct TestEval
  : public TestExpression<T, TestEvalF<F, T>::template Expr> {

  TestEval(const F& f, const std::string& repr="F")
    : m_f(f)
    , m_repr(repr) {}

  T
  operator()() const {
    return m_f();
  }

  std::string
  reason(bool state) const {
    std::ostringstream oss;
    if (state)
      oss << "Evaluation is TRUE";
    else
      oss << "Evaluation is FALSE";
    return oss.str();
  }

  const std::string&
  repr() const {
    return m_repr;
  }

private:

  const F& m_f;

  std::string m_repr;
};

template <typename E, typename T>
struct BoolExprT {
  template <typename S> using Expr = E;
};

template <typename E,
          typename T = bool>
class BoolExpr
  : public TestExpression<bool, BoolExprT<E, T>::template Expr> {
};

template <template <typename> typename E,
          template <typename> typename F>
class BoolOr
  : public BoolExpr<BoolOr<E, F>> {
public:

  BoolOr(
    const TestExpression<bool, E>& left,
    const TestExpression<bool, F>& right,
    const std::optional<std::string>& repr=std::nullopt)
    : m_left(static_cast<const E<bool>&>(left))
    , m_right(static_cast<const F<bool>&>(right)) {

    if (!repr) {
      std::ostringstream oss;
      oss << "(" << m_left.repr() << " OR " << m_right.repr() << ")";
      m_repr = oss.str();
    } else {
      m_repr = repr.value();
    }
  }

  bool
  operator()() const {
    return m_left() || m_right();
  }

  std::string
  reason(bool state) const {
    std::ostringstream oss;
    if (state) {
      oss << "Either " << m_left.repr()
          << " or " << m_right.repr()
          << " is TRUE";
    } else {
      oss << "Neither " << m_left.repr()
          << " nor " << m_right.repr()
          << " is TRUE";
    }
    return oss.str();
  }

  const std::string&
  repr() const {
    return m_repr;
  }

private:

  const E<bool>& m_left;
  const F<bool>& m_right;
  std::string m_repr;
};

template <template <typename> typename E,
          template <typename> typename F>
class BoolAnd
  : public BoolExpr<BoolAnd<E, F>> {
public:

  BoolAnd(
    const TestExpression<bool, E>& left,
    const TestExpression<bool, F>& right,
    const std::optional<std::string>& repr=std::nullopt)
    : m_left(static_cast<const E<bool>&>(left))
    , m_right(static_cast<const F<bool>&>(right)) {

    if (!repr) {
      std::ostringstream oss;
      oss << "(" << m_left.repr() << " AND " << m_right.repr() << ")";
      m_repr = oss.str();
    } else {
      m_repr = repr.value();
    }
  }

  bool
  operator()() const {
    return m_left() && m_right();
  }

  std::string
  reason(bool state) const {
    std::ostringstream oss;
    if (state) {
      oss << "Both " << m_left.repr()
          << " and " << m_right.repr()
          << " are TRUE";
    } else {
      oss << "Either " << m_left.repr()
          << " or " << m_right.repr()
          << " is FALSE";
    }
    return oss.str();
  }

  const std::string&
  repr() const {
    return m_repr;
  }

private:

  const E<bool>& m_left;
  const F<bool>& m_right;
  std::string m_repr;
};

template <typename T,
          template <typename> typename E,
          typename U,
          template <typename> typename F>
class BoolEq
  : public BoolExpr<BoolEq<T, E, U, F>> {
public:

  BoolEq(
    const TestExpression<T, E>& left,
    const TestExpression<U, F>& right,
    const std::optional<std::string>& repr=std::nullopt)
    : m_left(static_cast<const E<T>&>(left))
    , m_right(static_cast<const F<U>&>(right)) {

    if (!repr) {
      std::ostringstream oss;
      oss << "(" << m_left.repr() << " EQ " << m_right.repr() << ")";
      m_repr = oss.str();
    } else {
      m_repr = repr.value();
    }
  }

  bool
  operator()() const {
    return m_left() == m_right();
  }

  std::string
  reason(bool state) const {
    std::ostringstream oss;
    if (state)
      oss << m_left.repr() << " == " << m_right.repr();
    else
      oss << m_left.repr() << " != " << m_right.repr();
    return oss.str();
  }

  const std::string&
  repr() const {
    return m_repr;
  }

private:

  const E<T>& m_left;
  const F<U>& m_right;
  std::string m_repr;
};

template <typename T>
struct TestFuture;

template <typename T>
struct TestFuture
  : public TestExpression<T, TestFuture> {

  TestFuture(const Legion::Future& f, const std::string& repr="Future")
    : m_f(f)
    , m_repr(repr) {}

  T
  operator()() const {
    return m_f.get_result<T>();
  }

  std::string
  reason(bool state) const {
    std::ostringstream oss;
    if (state)
      oss << "Value is TRUE";
    else
      oss << "Value is FALSE";
    return oss.str();
  }

  const std::string&
  repr() const {
    return m_repr;
  }

private:

  const Legion::Future& m_f;

  std::string m_repr;
};

} // end namespace testing
} // end namespace legms

#endif //LEGMS_TESTING_TEST_EXPRESSION_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
