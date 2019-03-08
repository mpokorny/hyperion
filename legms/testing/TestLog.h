#ifndef LEGMS_TESTING_TEST_LOG_H_
#define LEGMS_TESTING_TEST_LOG_H_

#include <memory>

#include "legion.h"

#include "utility.h"

namespace legms {
namespace testing {

enum TestState {
  SUCCESS,
  FAILURE,
  UNKNOWN
};

struct TestLogReference {
public:

  Legion::LogicalRegion log_handle;
  Legion::LogicalRegion log_parent;
  Legion::LogicalRegion abort_state_handle;

  enum {
    STATE_FID,
    ABORT_FID,
    LOCATION_FID,
    DESCRIPTION_FID
  };

  static TestLogReference
  create(size_t length, Legion::Context context, Legion::Runtime* runtime);

  std::vector<Legion::RegionRequirement>
  rw_requirements() const;

  std::vector<Legion::RegionRequirement>
  ro_requirements() const;

  std::vector<Legion::RegionRequirement>
  wd_requirements() const;

  Legion::LogicalPartition
  partition_log_by_state(
    Legion::Context context,
    Legion::Runtime* runtime) const;

  template <legion_privilege_mode_t MODE>
  using state_accessor =
    Legion::FieldAccessor<
    MODE,
    int,
    1,
    Legion::coord_t,
    Legion::AffineAccessor<int, 1, Legion::coord_t>,
    false>;

  template <legion_privilege_mode_t MODE>
  using abort_accessor =
    Legion::FieldAccessor<
    MODE,
    bool,
    1,
    Legion::coord_t,
    Legion::AffineAccessor<bool, 1, Legion::coord_t>,
    false>;

  template <legion_privilege_mode_t MODE>
  using location_accessor =
    Legion::FieldAccessor<
    MODE,
    std::string,
    1,
    Legion::coord_t,
    Legion::AffineAccessor<std::string, 1, Legion::coord_t>,
    false>;;

  template <legion_privilege_mode_t MODE>
  using description_accessor =
    Legion::FieldAccessor<
    MODE,
    std::string,
    1,
    Legion::coord_t,
    Legion::AffineAccessor<std::string, 1, Legion::coord_t>,
    false>;

  typedef Legion::ReductionAccessor<
    bool_or_redop,
    false,
    1,
    Legion::coord_t,
    Legion::AffineAccessor<bool, 1, Legion::coord_t>,
    false> abort_state_reduce_accessor;

  typedef Legion::FieldAccessor<
    READ_ONLY,
    bool,
    1,
    Legion::coord_t,
    Legion::AffineAccessor<bool, 1, Legion::coord_t>,
    false> abort_state_read_accessor;
};

template <legion_privilege_mode_t MODE>
class TestLogIterator {

  virtual
  ~TestLogIterator() {}

  TestLogIterator(
    Legion::PhysicalRegion* region,
    Legion::Runtime* runtime)
    : m_region(region)
    , m_pir(
      Legion::PointInRectIterator<1>(
        runtime->get_index_space_domain(
          region->get_logical_region().get_index_space())))
    , m_state(
      TestLogReference::state_accessor<MODE>(
        *region,
        TestLogReference::STATE_FID))
    , m_abort(
      TestLogReference::abort_accessor<MODE>(
        *region,
        TestLogReference::ABORT_FID))
    , m_location(
      TestLogReference::location_accessor<MODE>(
        *region,
        TestLogReference::LOCATION_FID))
    , m_description(
      TestLogReference::description_accessor<MODE>(
        *region,
        TestLogReference::DESCRIPTION_FID)) {
  }

  TestLogIterator(const TestLogIterator& other)
    : m_region(other.m_region)
    , m_pir(other.m_pir)
    , m_state(other.m_state)
    , m_abort(other.m_abort)
    , m_location(other.m_location)
    , m_description(other.m_description) {
  }

  TestLogIterator(TestLogIterator&& other)
    : m_region(other.m_region)
    , m_pir(std::move(other).m_pir)
    , m_state(std::move(other).m_state)
    , m_abort(std::move(other).m_abort)
    , m_location(std::move(other).m_location)
    , m_description(std::move(other).m_description) {
  }

  TestLogIterator&
  operator=(const TestLogIterator& other) {
    TestLogIterator tmp(other);
    swap(tmp);
    return *this;
  }

  TestLogIterator&
  operator=(TestLogIterator&& other) {
    m_region = other.m_region;
    m_pir = other.m_pir;
    return *this;
  }

  bool
  operator==(const TestLogIterator& rhs) {
    return m_region == rhs.m_region && *m_pir == *rhs.m_pir;
  }

  bool
  operator!=(const TestLogIterator& rhs) {
    return !operator==(rhs);
  }

  TestLogIterator&
  operator++() {
    m_pir++;
    return *this;
  }

  TestLogIterator
  operator++(int) {
    TestLogIterator result(*this);
    operator++();
    return result;
  }

  bool
  at_end() const {
    return m_pir();
  }

  friend void
  swap(TestLogIterator& lhs, TestLogIterator& rhs) {
    lhs.swap(rhs);
  }

private:

  void
  swap(TestLogIterator& other) {
    std::swap(m_region, other.m_region);
    std::swap(m_pir, other.m_pir);
  }

  Legion::PhysicalRegion* m_region;

  Legion::PointInRectIterator<1> m_pir;

  TestLogReference::state_accessor<MODE> m_state;
  TestLogReference::abort_accessor<MODE> m_abort;
  TestLogReference::location_accessor<MODE> m_location;
  TestLogReference::description_accessor<MODE> m_description;

};

template <legion_privilege_mode_t MODE>
struct TestResult;

template <>
struct TestResult<READ_ONLY> {
  const int& state;
  const bool& abort;
  const std::string& location;
  const std::string& description;
};

template <>
struct TestResult<READ_WRITE> {
  int state;
  bool abort;
  std::string location;
  std::string description;
};

template <>
struct TestResult<WRITE_DISCARD> {
  int state;
  bool abort;
  std::string location;
  std::string description;
};

template <legion_privilege_mode_t MODE>
class TestLog {

public:

  TestLog(
    Legion::PhysicalRegion* region,
    Legion::Runtime* runtime)
    : m_region(region)
    , m_runtime(runtime) {
  }

public:

  TestLogIterator<MODE>
  iterator() const {
    return TestLogIterator<MODE>(m_region, m_runtime);
  }

private:

  Legion::PhysicalRegion* m_region;

  Legion::Runtime* m_runtime;
};

template <>
class TestLogIterator<READ_ONLY> {

  virtual
  ~TestLogIterator() {}

  TestLogIterator(
    Legion::PhysicalRegion* region,
    Legion::Runtime* runtime)
    : m_region(region)
    , m_pir(
      Legion::PointInRectIterator<1>(
        runtime->get_index_space_domain(
          region->get_logical_region().get_index_space())))
    , m_state(
      TestLogReference::state_accessor<READ_ONLY>(
        *region,
        TestLogReference::STATE_FID))
    , m_abort(
      TestLogReference::abort_accessor<READ_ONLY>(
        *region,
        TestLogReference::ABORT_FID))
    , m_location(
      TestLogReference::location_accessor<READ_ONLY>(
        *region,
        TestLogReference::LOCATION_FID))
    , m_description(
      TestLogReference::description_accessor<READ_ONLY>(
        *region,
        TestLogReference::DESCRIPTION_FID)) {
  }

  TestLogIterator(const TestLogIterator& other)
    : m_region(other.m_region)
    , m_pir(other.m_pir)
    , m_state(other.m_state)
    , m_abort(other.m_abort)
    , m_location(other.m_location)
    , m_description(other.m_description)
    , m_abort_state(other.m_abort_state) {
  }

  TestLogIterator(TestLogIterator&& other)
    : m_region(other.m_region)
    , m_pir(std::move(other).m_pir)
    , m_state(std::move(other).m_state)
    , m_abort(std::move(other).m_abort)
    , m_location(std::move(other).m_location)
    , m_description(std::move(other).m_description)
    , m_abort_state(std::move(other).m_abort_state) {
  }

  TestLogIterator&
  operator=(const TestLogIterator& other) {
    TestLogIterator tmp(other);
    swap(tmp);
    return *this;
  }

  TestLogIterator&
  operator=(TestLogIterator&& other) {
    m_region = other.m_region;
    m_pir = other.m_pir;
    m_state = other.m_state;
    m_abort = other.m_abort;
    m_location = other.m_location;
    m_description = other.m_description;
    m_abort_state = other.m_abort_state;
    return *this;
  }

  bool
  operator==(const TestLogIterator& rhs) {
    return m_region == rhs.m_region && *m_pir == *rhs.m_pir;
  }

  bool
  operator!=(const TestLogIterator& rhs) {
    return !operator==(rhs);
  }

  TestLogIterator&
  operator++() {
    m_pir++;
    return *this;
  }

  TestLogIterator
  operator++(int) {
    TestLogIterator result(*this);
    operator++();
    return result;
  }

  bool
  at_end() const {
    return m_pir();
  }

  TestResult<READ_ONLY>
  operator*() const {
    return TestResult<READ_ONLY>{
      m_state[*m_pir],
        m_abort[*m_pir],
        m_location[*m_pir],
        m_description[*m_pir] };
  }

  friend void
  swap(TestLogIterator& lhs, TestLogIterator& rhs) {
    lhs.swap(rhs);
  }

private:

  void
  swap(TestLogIterator& other) {
    std::swap(m_region, other.m_region);
    std::swap(m_pir, other.m_pir);
    std::swap(m_state, other.m_state);
    std::swap(m_abort, other.m_abort);
    std::swap(m_location, other.m_location);
    std::swap(m_description, other.m_description);
    std::swap(m_abort_state, other.m_abort_state);
  }

  Legion::PhysicalRegion* m_region;

  Legion::PointInRectIterator<1> m_pir;

  TestLogReference::state_accessor<READ_ONLY> m_state;
  TestLogReference::abort_accessor<READ_ONLY> m_abort;
  TestLogReference::location_accessor<READ_ONLY> m_location;
  TestLogReference::description_accessor<READ_ONLY> m_description;
  TestLogReference::abort_state_read_accessor m_abort_state;
};

template <>
class TestLogIterator<READ_WRITE> {

  virtual
  ~TestLogIterator() {}

  TestLogIterator(
    Legion::PhysicalRegion* region,
    Legion::Runtime* runtime)
    : m_region(region)
    , m_pir(
      Legion::PointInRectIterator<1>(
        runtime->get_index_space_domain(
          region->get_logical_region().get_index_space())))
    , m_state(
      TestLogReference::state_accessor<READ_WRITE>(
        *region,
        TestLogReference::STATE_FID))
    , m_abort(
      TestLogReference::abort_accessor<READ_WRITE>(
        *region,
        TestLogReference::ABORT_FID))
    , m_location(
      TestLogReference::location_accessor<READ_WRITE>(
        *region,
        TestLogReference::LOCATION_FID))
    , m_description(
      TestLogReference::description_accessor<READ_WRITE>(
        *region,
        TestLogReference::DESCRIPTION_FID)) {
  }

  TestLogIterator(const TestLogIterator& other)
    : m_region(other.m_region)
    , m_pir(other.m_pir)
    , m_state(other.m_state)
    , m_abort(other.m_abort)
    , m_location(other.m_location)
    , m_description(other.m_description)
    , m_abort_state(other.m_abort_state) {
  }

  TestLogIterator(TestLogIterator&& other)
    : m_region(other.m_region)
    , m_pir(std::move(other).m_pir)
    , m_state(std::move(other).m_state)
    , m_abort(std::move(other).m_abort)
    , m_location(std::move(other).m_location)
    , m_description(std::move(other).m_description)
    , m_abort_state(std::move(other).m_abort_state) {
  }

  TestLogIterator&
  operator=(const TestLogIterator& other) {
    TestLogIterator tmp(other);
    swap(tmp);
    return *this;
  }

  TestLogIterator&
  operator=(TestLogIterator&& other) {
    m_region = other.m_region;
    m_pir = other.m_pir;
    m_state = other.m_state;
    m_abort = other.m_abort;
    m_location = other.m_location;
    m_description = other.m_description;
    m_abort_state = other.m_abort_state;
    return *this;
  }

  bool
  operator==(const TestLogIterator& rhs) {
    return m_region == rhs.m_region && *m_pir == *rhs.m_pir;
  }

  bool
  operator!=(const TestLogIterator& rhs) {
    return !operator==(rhs);
  }

  TestLogIterator&
  operator++() {
    m_pir++;
    return *this;
  }

  TestLogIterator
  operator++(int) {
    TestLogIterator result(*this);
    operator++();
    return result;
  }

  bool
  at_end() const {
    return m_pir();
  }

  TestResult<READ_WRITE>
  operator*() const {
    return TestResult<READ_WRITE>{
      m_state[*m_pir],
        m_abort[*m_pir],
        m_location[*m_pir],
        m_description[*m_pir]};
  }

  void
  operator<<=(const TestResult<READ_WRITE>& tr) const {
    m_state[*m_pir] = tr.state;
    m_abort[*m_pir] = tr.abort;
    m_location[*m_pir] = tr.location;
    m_description[*m_pir] = tr.description;
    m_abort_state[*m_pir] <<= tr.abort;
  }

  friend void
  swap(TestLogIterator& lhs, TestLogIterator& rhs) {
    lhs.swap(rhs);
  }

private:

  void
  swap(TestLogIterator& other) {
    std::swap(m_region, other.m_region);
    std::swap(m_pir, other.m_pir);
    std::swap(m_state, other.m_state);
    std::swap(m_abort, other.m_abort);
    std::swap(m_location, other.m_location);
    std::swap(m_description, other.m_description);
    std::swap(m_abort_state, other.m_abort_state);
  }

  Legion::PhysicalRegion* m_region;

  Legion::PointInRectIterator<1> m_pir;

  TestLogReference::state_accessor<READ_WRITE> m_state;
  TestLogReference::abort_accessor<READ_WRITE> m_abort;
  TestLogReference::location_accessor<READ_WRITE> m_location;
  TestLogReference::description_accessor<READ_WRITE> m_description;
  TestLogReference::abort_state_reduce_accessor m_abort_state;
};

template <>
class TestLogIterator<WRITE_DISCARD> {

  virtual
  ~TestLogIterator() {}

  TestLogIterator(
    Legion::PhysicalRegion* region,
    Legion::Runtime* runtime)
    : m_region(region)
    , m_pir(
      Legion::PointInRectIterator<1>(
        runtime->get_index_space_domain(
          region->get_logical_region().get_index_space())))
    , m_state(
      TestLogReference::state_accessor<WRITE_DISCARD>(
        *region,
        TestLogReference::STATE_FID))
    , m_abort(
      TestLogReference::abort_accessor<WRITE_DISCARD>(
        *region,
        TestLogReference::ABORT_FID))
    , m_location(
      TestLogReference::location_accessor<WRITE_DISCARD>(
        *region,
        TestLogReference::LOCATION_FID))
    , m_description(
      TestLogReference::description_accessor<WRITE_DISCARD>(
        *region,
        TestLogReference::DESCRIPTION_FID)) {
  }

  TestLogIterator(const TestLogIterator& other)
    : m_region(other.m_region)
    , m_pir(other.m_pir)
    , m_state(other.m_state)
    , m_abort(other.m_abort)
    , m_location(other.m_location)
    , m_description(other.m_description)
    , m_abort_state(other.m_abort_state) {
  }

  TestLogIterator(TestLogIterator&& other)
    : m_region(other.m_region)
    , m_pir(std::move(other).m_pir)
    , m_state(std::move(other).m_state)
    , m_abort(std::move(other).m_abort)
    , m_location(std::move(other).m_location)
    , m_description(std::move(other).m_description)
    , m_abort_state(std::move(other).m_abort_state) {
  }

  TestLogIterator&
  operator=(const TestLogIterator& other) {
    TestLogIterator tmp(other);
    swap(tmp);
    return *this;
  }

  TestLogIterator&
  operator=(TestLogIterator&& other) {
    m_region = other.m_region;
    m_pir = other.m_pir;
    m_state = other.m_state;
    m_abort = other.m_abort;
    m_location = other.m_location;
    m_description = other.m_description;
    m_abort_state = other.m_abort_state;
    return *this;
  }

  bool
  operator==(const TestLogIterator& rhs) {
    return m_region == rhs.m_region && *m_pir == *rhs.m_pir;
  }

  bool
  operator!=(const TestLogIterator& rhs) {
    return !operator==(rhs);
  }

  TestLogIterator&
  operator++() {
    m_pir++;
    return *this;
  }

  TestLogIterator
  operator++(int) {
    TestLogIterator result(*this);
    operator++();
    return result;
  }

  bool
  at_end() const {
    return m_pir();
  }

  TestResult<WRITE_DISCARD>
  operator*() const {
    return TestResult<WRITE_DISCARD>{
      m_state[*m_pir],
        m_abort[*m_pir],
        m_location[*m_pir],
        m_description[*m_pir]};
  }

  void
  operator<<=(const TestResult<WRITE_DISCARD>& tr) const {
    m_state[*m_pir] = tr.state;
    m_abort[*m_pir] = tr.abort;
    m_location[*m_pir] = tr.location;
    m_description[*m_pir] = tr.description;
    m_abort_state[*m_pir] <<= tr.abort;
  }

  friend void
  swap(TestLogIterator& lhs, TestLogIterator& rhs) {
    lhs.swap(rhs);
  }

private:

  void
  swap(TestLogIterator& other) {
    std::swap(m_region, other.m_region);
    std::swap(m_pir, other.m_pir);
    std::swap(m_state, other.m_state);
    std::swap(m_abort, other.m_abort);
    std::swap(m_location, other.m_location);
    std::swap(m_description, other.m_description);
    std::swap(m_abort_state, other.m_abort_state);
  }

  Legion::PhysicalRegion* m_region;

  Legion::PointInRectIterator<1> m_pir;

  TestLogReference::state_accessor<WRITE_DISCARD> m_state;
  TestLogReference::abort_accessor<WRITE_DISCARD> m_abort;
  TestLogReference::location_accessor<WRITE_DISCARD> m_location;
  TestLogReference::description_accessor<WRITE_DISCARD> m_description;
  TestLogReference::abort_state_reduce_accessor m_abort_state;
};

} // end namespace testing
} // end namespace legms

#endif // LEGMS_TESTING_TEST_LOG_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
