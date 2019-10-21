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
#ifndef HYPERION_TESTING_TEST_EXPRESSION_H_
#define HYPERION_TESTING_TEST_EXPRESSION_H_

#pragma GCC visibility push(default)
#include <optional>
#include <ostream>
#include <sstream>
#include <typeinfo>
#include <type_traits>
#pragma GCC visibility pop

#include <hyperion/hyperion.h>
#include <hyperion/testing/TestRecorder.h>

namespace hyperion{
namespace testing {

template <template <typename> typename E,
          template <typename> typename F>
class BoolOr;

template <template <typename> typename E,
          template <typename> typename F>
class BoolAnd;

template <typename T,
          template <typename> typename E,
          template <typename> typename F>
class BoolEq;

template <typename T>
struct TestVal;

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

  template <template <typename> typename F>
  BoolEq<T, E, F>
  operator==(const TestExpression<T, F>& right) const {
    return BoolEq(*this, right);
  }

  BoolEq<T, E, TestVal>
  operator==(const T& right) const {
    return BoolEq(*this, TestVal(right));
  }

  std::string
  reason(bool state) const {
    return static_cast<const E<T>&>(*this).reason(state);
  }
};

template <typename T>
struct insertion_op {
  std::ostream& operator()(std::ostream& s, const T& t) {
    return s << t;
  }
};

template <typename T,
          typename =
          typename std::invoke_result_t<insertion_op<T>, std::ostringstream, T>>
std::string
to_string(T v) {
  std::ostringstream oss;
  oss << v;
  return oss.str();
}

template <typename T>
std::string
to_string(T) {
  return typeid(T).name();
}

template <typename T>
struct TestVal
  : public TestExpression<T, TestVal> {

  TestVal(const T& val, const std::optional<std::string>& repr=std::nullopt)
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

  F m_f;

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
      oss << "(" << m_left.repr() << " || " << m_right.repr() << ")";
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

  E<bool> m_left;
  F<bool> m_right;
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
      oss << "(" << m_left.repr() << " && " << m_right.repr() << ")";
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

  E<bool> m_left;
  F<bool> m_right;
  std::string m_repr;
};

template <typename T,
          template <typename> typename E,
          template <typename> typename F>
class BoolEq
  : public BoolExpr<BoolEq<T, E, F>> {
public:

  BoolEq(
    const TestExpression<T, E>& left,
    const TestExpression<T, F>& right,
    const std::optional<std::string>& repr=std::nullopt)
    : m_left(static_cast<const E<T>&>(left))
    , m_right(static_cast<const F<T>&>(right)) {

    if (!repr) {
      std::ostringstream oss;
      oss << "(" << m_left.repr() << " == " << m_right.repr() << ")";
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

  E<T> m_left;
  F<T> m_right;
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

  Legion::Future m_f;

  std::string m_repr;
};

} // end namespace testing
} // end namespace hyperion

#endif //HYPERION_TESTING_TEST_EXPRESSION_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
