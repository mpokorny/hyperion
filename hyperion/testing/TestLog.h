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
#ifndef HYPERION_TESTING_TEST_LOG_H_
#define HYPERION_TESTING_TEST_LOG_H_

#include <hyperion/hyperion.h>
#include <hyperion/utility.h>

#pragma GCC visibility push(default)
# include <array>
# include <memory>
# include <optional>
# include <tuple>
#pragma GCC visibility pop

namespace hyperion {
namespace testing {

enum TestState {
  SUCCESS,
  FAILURE,
  SKIPPED,
  UNKNOWN
};

template <legion_privilege_mode_t MODE>
class TestLog;

struct HYPERION_API TestLogReference {
public:

  TestLogReference(
    size_t length,
    Legion::Context context,
    Legion::Runtime* runtime);

  TestLogReference(
    Legion::LogicalRegionT<1> log_handle,
    Legion::LogicalRegionT<1> log_parent,
    Legion::LogicalRegionT<1> abort_state_handle);

  enum {
    STATE_FID,
    ABORT_FID,
    NAME_FID,
    FAIL_INFO_FID
  };

  static constexpr int log_requirement_index = 0;
  static constexpr int abort_state_requirement_index = 1;

  virtual ~TestLogReference();

  Legion::LogicalRegionT<1>
  log_region() const {
    return m_log_handle;
  }

  Legion::LogicalRegionT<1>
  abort_state_region() const {
    return m_abort_state_handle;
  }

  std::array<Legion::RegionRequirement, 2>
  rw_requirements(
    Legion::LogicalRegionT<1> log_child,
    Legion::LogicalRegionT<1> log_parent) const;

  std::array<Legion::RegionRequirement, 2>
  ro_requirements(
    Legion::LogicalRegionT<1> log_child,
    Legion::LogicalRegionT<1> log_parent) const;

  std::array<Legion::RegionRequirement, 2>
  wd_requirements(
    Legion::LogicalRegionT<1> log_child,
    Legion::LogicalRegionT<1> log_parent) const;

  std::array<Legion::RegionRequirement, 2>
  rw_requirements(
    Legion::LogicalPartitionT<1> log_partition,
    Legion::LogicalRegionT<1> log_parent,
    int projection_id) const;

  std::array<Legion::RegionRequirement, 2>
  ro_requirements(
    Legion::LogicalPartitionT<1> log_partition,
    Legion::LogicalRegionT<1> log_parent,
    int projection_id) const;

  std::array<Legion::RegionRequirement, 2>
  wd_requirements(
    Legion::LogicalPartitionT<1> log_partition,
    Legion::LogicalRegionT<1> log_parent,
    int projection_id) const;

  std::array<Legion::RegionRequirement, 2>
  rw_requirements() const {
    return rw_requirements(m_log_handle, m_log_parent);
  }

  std::array<Legion::RegionRequirement, 2>
  ro_requirements() const {
    return ro_requirements(m_log_handle, m_log_parent);
  };

  std::array<Legion::RegionRequirement, 2>
  wd_requirements() const {
    return wd_requirements(m_log_handle, m_log_parent);
  }

  template <legion_privilege_mode_t MODE>
  std::array<Legion::RegionRequirement, 2>
  requirements(
    Legion::LogicalRegionT<1> log_child,
    Legion::LogicalRegionT<1> log_parent) const;

  template <legion_privilege_mode_t MODE>
  std::array<Legion::RegionRequirement, 2>
  requirements(
    Legion::LogicalPartitionT<1> log_partition,
    Legion::LogicalRegionT<1> log_parent,
    int projection_id=0) const;

  template <legion_privilege_mode_t MODE>
  std::array<Legion::RegionRequirement, 2>
  requirements() const;

  Legion::LogicalPartitionT<1>
  create_partition_by_log_state(
    Legion::Context context,
    Legion::Runtime* runtime) const;

  template <legion_privilege_mode_t MODE>
  using state_accessor =
    Legion::FieldAccessor<
    MODE,
    Legion::Point<1>,
    1,
    Legion::coord_t,
    Legion::AffineAccessor<
      Legion::Point<1>,
      1,
      Legion::coord_t>,
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
  using name_accessor =
    Legion::FieldAccessor<
    MODE,
    std::string,
    1,
    Legion::coord_t,
    Legion::AffineAccessor<std::string, 1, Legion::coord_t>,
    false>;

  template <legion_privilege_mode_t MODE>
  using fail_info_accessor =
    Legion::FieldAccessor<
    MODE,
    std::string,
    1,
    Legion::coord_t,
    Legion::AffineAccessor<std::string, 1, Legion::coord_t>,
    false>;

  template <legion_privilege_mode_t MODE>
  struct abort_state_accessor {};

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

private:

  Legion::LogicalRegionT<1> m_log_handle;
  Legion::LogicalRegionT<1> m_log_parent;
  Legion::LogicalRegionT<1> m_abort_state_handle;

private:

  friend class TestLog<READ_ONLY>;
  friend class TestLog<READ_WRITE>;
  friend class TestLog<WRITE_DISCARD>;

  bool m_own_regions;
  Legion::Context m_context;
  Legion::Runtime* m_runtime;

};

template <>
struct TestLogReference::abort_state_accessor<READ_ONLY> {
  typedef TestLogReference::abort_state_read_accessor t;
};

template <>
struct TestLogReference::abort_state_accessor<READ_WRITE> {
  typedef TestLogReference::abort_state_reduce_accessor t;
};

template <>
struct TestLogReference::abort_state_accessor<WRITE_DISCARD> {
  typedef TestLogReference::abort_state_reduce_accessor t;
};

template <> inline
std::array<Legion::RegionRequirement, 2>
TestLogReference::requirements<READ_ONLY>() const {
  return ro_requirements();
};

template <> inline
std::array<Legion::RegionRequirement, 2>
TestLogReference::requirements<READ_WRITE>() const {
  return rw_requirements();
};

template <> inline
std::array<Legion::RegionRequirement, 2>
TestLogReference::requirements<WRITE_DISCARD>() const {
  return wd_requirements();
};

template <> inline
std::array<Legion::RegionRequirement, 2>
TestLogReference::requirements<READ_ONLY>(
  Legion::LogicalRegionT<1> log_child,
  Legion::LogicalRegionT<1> log_parent) const {
  return ro_requirements(log_child, log_parent);
};

template <> inline
std::array<Legion::RegionRequirement, 2>
TestLogReference::requirements<READ_WRITE>(
  Legion::LogicalRegionT<1> log_child,
  Legion::LogicalRegionT<1> log_parent) const {
  return rw_requirements(log_child, log_parent);
};

template <> inline
std::array<Legion::RegionRequirement, 2>
TestLogReference::requirements<WRITE_DISCARD>(
  Legion::LogicalRegionT<1> log_child,
  Legion::LogicalRegionT<1> log_parent) const {
  return wd_requirements(log_child, log_parent);
};

template <> inline
std::array<Legion::RegionRequirement, 2>
TestLogReference::requirements<READ_ONLY>(
  Legion::LogicalPartitionT<1> log_partition,
  Legion::LogicalRegionT<1> log_parent,
  int projection_id) const {
  return ro_requirements(log_partition, log_parent, projection_id);
};

template <> inline
std::array<Legion::RegionRequirement, 2>
TestLogReference::requirements<READ_WRITE>(
  Legion::LogicalPartitionT<1> log_partition,
  Legion::LogicalRegionT<1> log_parent,
  int projection_id) const {
  return rw_requirements(log_partition, log_parent, projection_id);
};

template <> inline
std::array<Legion::RegionRequirement, 2>
TestLogReference::requirements<WRITE_DISCARD>(
  Legion::LogicalPartitionT<1> log_partition,
  Legion::LogicalRegionT<1> log_parent,
  int projection_id) const {
  return wd_requirements(log_partition, log_parent, projection_id);
};

template <legion_privilege_mode_t MODE>
class TestLogIterator {

public:

  bool
  operator==(const TestLogIterator& rhs) {
    return m_region == rhs.m_region && *m_pid == *rhs.m_pid;
  }

  bool
  operator!=(const TestLogIterator& rhs) {
    return !operator==(rhs);
  }

  TestLogIterator&
  operator++() {
    m_pid++;
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
    return !m_pid();
  }

  friend void
  swap(TestLogIterator& lhs, TestLogIterator& rhs) {
    lhs.swap(rhs);
  }

private:

  void
  swap(TestLogIterator& other) {
    std::swap(m_region, other.m_region);
    std::swap(m_pid, other.m_pid);
  }

  Legion::PhysicalRegion* m_region;

  Legion::PointInDomainIterator<1> m_pid;

  TestLogReference::state_accessor<MODE> m_state;
  TestLogReference::abort_accessor<MODE> m_abort;
  TestLogReference::name_accessor<MODE> m_name;
  TestLogReference::fail_info_accessor<MODE> m_fail_info;
  typename TestLogReference::abort_state_accessor<MODE>::t m_abort_state;
};

template <legion_privilege_mode_t MODE>
struct TestResult;

template <>
struct TestResult<READ_ONLY> {
  const coord_t& state;
  const bool& abort;
  const std::string& name;
  const std::string& fail_info;
};

template <>
struct TestResult<READ_WRITE> {
  coord_t state;
  bool abort;
  std::string name;
  std::string fail_info;
};

template <>
struct TestResult<WRITE_DISCARD> {
  coord_t state;
  bool abort;
  std::string name;
  std::string fail_info;
};

template <legion_privilege_mode_t MODE>
class TestLog {
public:

  TestLogIterator<MODE>
  iterator() const;

  bool
  contains_abort() const;

  template <typename CB>
  void
  for_each(CB cb);

};

template <>
class TestLogIterator<READ_ONLY> {
public:

  virtual
  ~TestLogIterator() {}

  TestLogIterator(
    const Legion::LogicalRegion& log_lr,
    const Legion::PhysicalRegion* log_pr,
    const Legion::PhysicalRegion* abort_state_region,
    Legion::Runtime* runtime)
    : m_log_pr(log_pr)
    , m_pid(
      Legion::PointInDomainIterator<1>(
        runtime->get_index_space_domain(log_lr.get_index_space())))
    , m_state(
      TestLogReference::state_accessor<READ_ONLY>(
        *log_pr,
        TestLogReference::STATE_FID))
    , m_abort(
      TestLogReference::abort_accessor<READ_ONLY>(
        *log_pr,
        TestLogReference::ABORT_FID))
    , m_name(
      TestLogReference::name_accessor<READ_ONLY>(
        *log_pr,
        TestLogReference::NAME_FID))
    , m_fail_info(
      TestLogReference::fail_info_accessor<READ_ONLY>(
        *log_pr,
        TestLogReference::FAIL_INFO_FID))
    , m_abort_state(
      TestLogReference::abort_state_accessor<READ_ONLY>::t(
        *abort_state_region,
        0)) {
  }

  TestLogIterator(const TestLogIterator& other)
    : m_log_pr(other.m_log_pr)
    , m_pid(other.m_pid)
    , m_state(other.m_state)
    , m_abort(other.m_abort)
    , m_name(other.m_name)
    , m_fail_info(other.m_fail_info)
    , m_abort_state(other.m_abort_state) {
  }

  TestLogIterator(TestLogIterator&& other)
    : m_log_pr(other.m_log_pr)
    , m_pid(std::move(other).m_pid)
    , m_state(std::move(other).m_state)
    , m_abort(std::move(other).m_abort)
    , m_name(std::move(other).m_name)
    , m_fail_info(std::move(other).m_fail_info)
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
    m_log_pr = other.m_log_pr;
    m_pid = other.m_pid;
    m_state = other.m_state;
    m_abort = other.m_abort;
    m_name = other.m_name;
    m_fail_info = other.m_fail_info;
    m_abort_state = other.m_abort_state;
    return *this;
  }

  bool
  operator==(const TestLogIterator& rhs) {
    return m_log_pr == rhs.m_log_pr && *m_pid == *rhs.m_pid;
  }

  bool
  operator!=(const TestLogIterator& rhs) {
    return !operator==(rhs);
  }

  TestLogIterator&
  operator++() {
    m_pid++;
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
    return !m_pid();
  }

  TestResult<READ_ONLY>
  operator*() const {
    return TestResult<READ_ONLY>{
      m_state[*m_pid],
        m_abort[*m_pid],
        m_name[*m_pid],
        m_fail_info[*m_pid] };
  }

  friend void
  swap(TestLogIterator& lhs, TestLogIterator& rhs) {
    lhs.swap(rhs);
  }

private:

  void
  swap(TestLogIterator& other) {
    std::swap(m_log_pr, other.m_log_pr);
    std::swap(m_pid, other.m_pid);
    std::swap(m_state, other.m_state);
    std::swap(m_abort, other.m_abort);
    std::swap(m_name, other.m_name);
    std::swap(m_fail_info, other.m_fail_info);
    std::swap(m_abort_state, other.m_abort_state);
  }

  const Legion::PhysicalRegion* m_log_pr;

  Legion::PointInDomainIterator<1> m_pid;

  TestLogReference::state_accessor<READ_ONLY> m_state;
  TestLogReference::abort_accessor<READ_ONLY> m_abort;
  TestLogReference::name_accessor<READ_ONLY> m_name;
  TestLogReference::fail_info_accessor<READ_ONLY> m_fail_info;
  TestLogReference::abort_state_accessor<READ_ONLY>::t m_abort_state;
};

template <>
class TestLogIterator<READ_WRITE> {
public:

  virtual
  ~TestLogIterator() {}

  TestLogIterator(
    const Legion::LogicalRegion& log_lr,
    const Legion::PhysicalRegion* log_pr,
    const Legion::PhysicalRegion* abort_state_region,
    Legion::Runtime* runtime)
    : m_log_pr(log_pr)
    , m_pid(
      Legion::PointInDomainIterator<1>(
        runtime->get_index_space_domain(log_lr.get_index_space())))
    , m_state(
      TestLogReference::state_accessor<READ_WRITE>(
        *log_pr,
        TestLogReference::STATE_FID))
    , m_abort(
      TestLogReference::abort_accessor<READ_WRITE>(
        *log_pr,
        TestLogReference::ABORT_FID))
    , m_name(
      TestLogReference::name_accessor<READ_WRITE>(
        *log_pr,
        TestLogReference::NAME_FID))
    , m_fail_info(
      TestLogReference::fail_info_accessor<READ_WRITE>(
        *log_pr,
        TestLogReference::FAIL_INFO_FID))
    , m_abort_state(
      TestLogReference::abort_state_accessor<READ_WRITE>::t(
        *abort_state_region,
        0,
        OpsManager::BOOL_OR_REDOP)) {
  }

  TestLogIterator(const TestLogIterator& other)
    : m_log_pr(other.m_log_pr)
    , m_pid(other.m_pid)
    , m_state(other.m_state)
    , m_abort(other.m_abort)
    , m_name(other.m_name)
    , m_fail_info(other.m_fail_info)
    , m_abort_state(other.m_abort_state) {
  }

  TestLogIterator(TestLogIterator&& other)
    : m_log_pr(other.m_log_pr)
    , m_pid(std::move(other).m_pid)
    , m_state(std::move(other).m_state)
    , m_abort(std::move(other).m_abort)
    , m_name(std::move(other).m_name)
    , m_fail_info(std::move(other).m_fail_info)
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
    m_log_pr = other.m_log_pr;
    m_pid = other.m_pid;
    m_state = other.m_state;
    m_abort = other.m_abort;
    m_name = other.m_name;
    m_fail_info = other.m_fail_info;
    m_abort_state = other.m_abort_state;
    return *this;
  }

  bool
  operator==(const TestLogIterator& rhs) {
    return m_log_pr == rhs.m_log_pr && *m_pid == *rhs.m_pid;
  }

  bool
  operator!=(const TestLogIterator& rhs) {
    return !operator==(rhs);
  }

  TestLogIterator&
  operator++() {
    m_pid++;
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
    return !m_pid();
  }

  TestResult<READ_WRITE>
  operator*() const {
    return TestResult<READ_WRITE>{
      m_state[*m_pid],
        m_abort[*m_pid],
        m_name[*m_pid],
        m_fail_info[*m_pid]};
  }

  template <legion_privilege_mode_t MODE>
  void
  operator<<=(const TestResult<MODE>& tr) const {
    m_state[*m_pid] = tr.state;
    m_abort[*m_pid] = tr.abort;
    m_name[*m_pid] = tr.name;
    m_fail_info[*m_pid] = tr.fail_info;
    m_abort_state[0] <<= tr.abort;
  }

  friend void
  swap(TestLogIterator& lhs, TestLogIterator& rhs) {
    lhs.swap(rhs);
  }

private:

  void
  swap(TestLogIterator& other) {
    std::swap(m_log_pr, other.m_log_pr);
    std::swap(m_pid, other.m_pid);
    std::swap(m_state, other.m_state);
    std::swap(m_abort, other.m_abort);
    std::swap(m_name, other.m_name);
    std::swap(m_fail_info, other.m_fail_info);
    std::swap(m_abort_state, other.m_abort_state);
  }

  const Legion::PhysicalRegion* m_log_pr;

  Legion::PointInDomainIterator<1> m_pid;

  TestLogReference::state_accessor<READ_WRITE> m_state;
  TestLogReference::abort_accessor<READ_WRITE> m_abort;
  TestLogReference::name_accessor<READ_WRITE> m_name;
  TestLogReference::fail_info_accessor<READ_WRITE> m_fail_info;
  TestLogReference::abort_state_accessor<READ_WRITE>::t m_abort_state;
};

template <>
class TestLogIterator<WRITE_DISCARD> {
public:

  virtual
  ~TestLogIterator() {}

  TestLogIterator(
    const Legion::LogicalRegion& log_lr,
    const Legion::PhysicalRegion* log_pr,
    const Legion::PhysicalRegion* abort_state_region,
    Legion::Runtime* runtime)
    : m_log_pr(log_pr)
    , m_pid(
      Legion::PointInDomainIterator<1>(
        runtime->get_index_space_domain(log_lr.get_index_space())))
    , m_state(
      TestLogReference::state_accessor<WRITE_DISCARD>(
        *log_pr,
        TestLogReference::STATE_FID))
    , m_abort(
      TestLogReference::abort_accessor<WRITE_DISCARD>(
        *log_pr,
        TestLogReference::ABORT_FID))
    , m_name(
      TestLogReference::name_accessor<WRITE_DISCARD>(
        *log_pr,
        TestLogReference::NAME_FID))
    , m_fail_info(
      TestLogReference::fail_info_accessor<WRITE_DISCARD>(
        *log_pr,
        TestLogReference::FAIL_INFO_FID))
    , m_abort_state(
      TestLogReference::abort_state_accessor<WRITE_DISCARD>::t(
        *abort_state_region,
        0,
        OpsManager::BOOL_OR_REDOP)) {
  }

  TestLogIterator(const TestLogIterator& other)
    : m_log_pr(other.m_log_pr)
    , m_pid(other.m_pid)
    , m_state(other.m_state)
    , m_abort(other.m_abort)
    , m_name(other.m_name)
    , m_fail_info(other.m_fail_info)
    , m_abort_state(other.m_abort_state) {
  }

  TestLogIterator(TestLogIterator&& other)
    : m_log_pr(other.m_log_pr)
    , m_pid(std::move(other).m_pid)
    , m_state(std::move(other).m_state)
    , m_abort(std::move(other).m_abort)
    , m_name(std::move(other).m_name)
    , m_fail_info(std::move(other).m_fail_info)
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
    m_log_pr = other.m_log_pr;
    m_pid = other.m_pid;
    m_state = other.m_state;
    m_abort = other.m_abort;
    m_name = other.m_name;
    m_fail_info = other.m_fail_info;
    m_abort_state = other.m_abort_state;
    return *this;
  }

  bool
  operator==(const TestLogIterator& rhs) {
    return m_log_pr == rhs.m_log_pr && *m_pid == *rhs.m_pid;
  }

  bool
  operator!=(const TestLogIterator& rhs) {
    return !operator==(rhs);
  }

  TestLogIterator&
  operator++() {
    m_pid++;
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
    return !m_pid();
  }

  TestResult<WRITE_DISCARD>
  operator*() const {
    return TestResult<WRITE_DISCARD>{
      m_state[*m_pid],
        m_abort[*m_pid],
        m_name[*m_pid],
        m_fail_info[*m_pid]};
  }

  template <legion_privilege_mode_t MODE>
  void
  operator<<=(const TestResult<MODE>& tr) const {
    m_state[*m_pid] = tr.state;
    m_abort[*m_pid] = tr.abort;
    m_name[*m_pid] = tr.name;
    m_fail_info[*m_pid] = tr.fail_info;
    m_abort_state[0] <<= tr.abort;
  }

  friend void
  swap(TestLogIterator& lhs, TestLogIterator& rhs) {
    lhs.swap(rhs);
  }

private:

  void
  swap(TestLogIterator& other) {
    std::swap(m_log_pr, other.m_log_pr);
    std::swap(m_pid, other.m_pid);
    std::swap(m_state, other.m_state);
    std::swap(m_abort, other.m_abort);
    std::swap(m_name, other.m_name);
    std::swap(m_fail_info, other.m_fail_info);
    std::swap(m_abort_state, other.m_abort_state);
  }

  const Legion::PhysicalRegion* m_log_pr;

  Legion::PointInDomainIterator<1> m_pid;

  TestLogReference::state_accessor<WRITE_DISCARD> m_state;
  TestLogReference::abort_accessor<WRITE_DISCARD> m_abort;
  TestLogReference::name_accessor<WRITE_DISCARD> m_name;
  TestLogReference::fail_info_accessor<WRITE_DISCARD> m_fail_info;
  TestLogReference::abort_state_accessor<WRITE_DISCARD>::t
  m_abort_state;
};

template <>
class TestLog<READ_ONLY> {

public:

  // abort_state_lr isn't really needed currently, but is present for future use
  TestLog(
    const Legion::LogicalRegion& log_lr,
    const Legion::PhysicalRegion& log_pr,
    const Legion::LogicalRegion& abort_state_lr,
    const Legion::PhysicalRegion& abort_state_pr,
    Legion::Context context,
    Legion::Runtime* runtime)
    : m_log_lr(Legion::LogicalRegionT<1>(log_lr))
    , m_log_pr(&log_pr)
    , m_abort_state_lr(Legion::LogicalRegionT<1>(abort_state_lr))
    , m_abort_state_pr(&abort_state_pr)
    , m_context(context)
    , m_runtime(runtime)
    , m_abort_state(abort_state_pr, 0) {
  }

  TestLog(
    TestLogReference& logref,
    Legion::Context context,
    Legion::Runtime* runtime)
    : m_context(context)
    , m_runtime(runtime) {

    assert(runtime != nullptr);

    auto reqs = logref.requirements<READ_ONLY>();
    m_own_log_pr =
      runtime->map_region(
        context,
        reqs[TestLogReference::log_requirement_index]);
    m_log_pr = &m_own_log_pr.value();
    m_log_lr = reqs[TestLogReference::log_requirement_index].region;
    m_own_abort_state_pr =
      runtime->map_region(
        context,
        reqs[TestLogReference::abort_state_requirement_index]);
    m_abort_state_pr = &m_own_abort_state_pr.value();
    m_abort_state_lr =
      reqs[TestLogReference::abort_state_requirement_index].region;
    m_abort_state =
      TestLogReference::abort_state_accessor<READ_ONLY>::t(
        *m_abort_state_pr,
        0);
  }

  TestLog(const TestLog& other)
    : m_log_lr(other.m_log_lr)
    , m_log_pr(other.m_log_pr)
    , m_abort_state_lr(other.m_abort_state_lr)
    , m_abort_state_pr(other.m_abort_state_pr)
    , m_context(other.m_context)
    , m_runtime(other.m_runtime)
    , m_abort_state(other.m_abort_state) {
  }

  TestLog(TestLog&& other)
    : m_log_lr(other.m_log_lr)
    , m_log_pr(other.m_log_pr)
    , m_abort_state_lr(other.m_abort_state_lr)
    , m_abort_state_pr(other.m_abort_state_pr)
    , m_context(other.m_context)
    , m_runtime(other.m_runtime)
    , m_abort_state(other.m_abort_state)
    , m_own_log_pr(std::move(other).m_own_log_pr)
    , m_own_abort_state_pr(std::move(other).m_own_abort_state_pr) {
  }

  TestLog&
  operator=(const TestLog& other) {
    TestLog tmp(other);
    swap(tmp);
    return *this;
  }

  TestLog&
  operator=(TestLog&& other) {
    m_log_lr = other.m_log_lr;
    m_log_pr = other.m_log_pr;
    m_abort_state_lr = other.m_abort_state_lr;
    m_abort_state_pr = other.m_abort_state_pr;
    m_context = other.m_context;
    m_runtime = other.m_runtime;
    m_abort_state = other.m_abort_state;
    m_own_log_pr = std::move(other).m_own_log_pr;
    m_own_abort_state_pr = std::move(other).m_own_abort_state_pr;
    return *this;
  }

  virtual ~TestLog() {
    if (m_own_log_pr)
      m_runtime->unmap_region(m_context, m_own_log_pr.value());
    if (m_own_abort_state_pr)
      m_runtime->unmap_region(m_context, m_own_abort_state_pr.value());
  }

public:

  TestLogIterator<READ_ONLY>
  iterator() const {
    return
      TestLogIterator<READ_ONLY>(
        m_log_lr,
        m_log_pr,
        m_abort_state_pr,
        m_runtime);
  }

  bool
  contains_abort() const {
    return m_abort_state[0];
  }

  template <typename CB>
  void
  for_each(CB cb) {
    auto it = iterator();
    while (!it.at_end()) {
      cb(it);
      ++it;
    }
  }

  TestLogReference
  log_reference() const {
    return
      TestLogReference(
        m_log_lr,
        Legion::LogicalRegionT<1>(m_log_pr->get_logical_region()),
        Legion::LogicalRegionT<1>(m_abort_state_pr->get_logical_region()));
  }

  std::vector<TestLogReference>
  get_log_references_by_state(const std::vector<TestState>& states) const {
    auto logref = log_reference();
    auto lp = logref.create_partition_by_log_state(m_context, m_runtime);

    std::vector<TestLogReference> result;
    std::transform(
      states.begin(),
      states.end(),
      std::back_inserter(result),
      [this, &logref, &lp](auto& st) {
        Legion::LogicalRegionT<1> log(
          m_runtime->get_logical_subregion_by_color(lp, st));
        return
          TestLogReference(
            log,
            Legion::LogicalRegionT<1>(m_log_pr->get_logical_region()),
            logref.abort_state_region());
      });
    return result;
  }

private:

  void
  swap(TestLog& other) {
    using std::swap;
    swap(m_log_lr, other.m_log_lr);
    swap(m_log_pr, other.m_log_pr);
    swap(m_abort_state_lr, other.m_abort_state_lr);
    swap(m_abort_state_pr, other.m_abort_state_pr);
    swap(m_context, other.m_context);
    swap(m_runtime, other.m_runtime);
    swap(m_abort_state, other.m_abort_state);
    swap(m_own_log_pr, other.m_own_log_pr);
    swap(m_own_abort_state_pr, other.m_own_abort_state_pr);
  }

  Legion::LogicalRegionT<1> m_log_lr;

  const Legion::PhysicalRegion* m_log_pr;

  Legion::LogicalRegionT<1> m_abort_state_lr;

  const Legion::PhysicalRegion* m_abort_state_pr;

  Legion::Context m_context;

  Legion::Runtime* m_runtime;

  TestLogReference::abort_state_accessor<READ_ONLY>::t m_abort_state;

  std::optional<Legion::PhysicalRegion> m_own_log_pr;

  std::optional<Legion::PhysicalRegion> m_own_abort_state_pr;
};

template <>
class TestLog<READ_WRITE> {

public:

  TestLog(
    const Legion::LogicalRegion& log_lr,
    const Legion::PhysicalRegion& log_pr,
    const Legion::LogicalRegion& abort_state_lr,
    const Legion::PhysicalRegion& abort_state_pr,
    Legion::Context context,
    Legion::Runtime* runtime)
    : m_log_lr(Legion::LogicalRegionT<1>(log_lr))
    , m_log_pr(&log_pr)
    , m_abort_state_lr(Legion::LogicalRegionT<1>(abort_state_lr))
    , m_abort_state_pr(&abort_state_pr)
    , m_context(context)
    , m_runtime(runtime)
    , m_abort_state(abort_state_pr, 0, OpsManager::BOOL_OR_REDOP) {
  }

  TestLog(
    TestLogReference& logref,
    Legion::Context context,
    Legion::Runtime* runtime)
    : m_context(context)
    , m_runtime(runtime) {

    assert(runtime != nullptr);

    auto reqs = logref.requirements<READ_WRITE>();
    m_own_log_pr =
      runtime->map_region(
        context,
        reqs[TestLogReference::log_requirement_index]);
    m_log_pr = &m_own_log_pr.value();
    m_log_lr = reqs[TestLogReference::log_requirement_index].region;
    m_own_abort_state_pr =
      runtime->map_region(
        context,
        reqs[TestLogReference::abort_state_requirement_index]);
    m_abort_state_pr = &m_own_abort_state_pr.value();
    m_abort_state_lr =
      reqs[TestLogReference::abort_state_requirement_index].region;
    m_abort_state =
      TestLogReference::abort_state_accessor<READ_WRITE>::t(
        *m_abort_state_pr,
        0,
        OpsManager::BOOL_OR_REDOP);
  }

  TestLog(const TestLog& other)
    : m_log_lr(other.m_log_lr)
    , m_log_pr(other.m_log_pr)
    , m_abort_state_lr(other.m_abort_state_lr)
    , m_abort_state_pr(other.m_abort_state_pr)
    , m_context(other.m_context)
    , m_runtime(other.m_runtime)
    , m_abort_state(other.m_abort_state) {
  }

  TestLog(TestLog&& other)
    : m_log_lr(other.m_log_lr)
    , m_log_pr(other.m_log_pr)
    , m_abort_state_lr(other.m_abort_state_lr)
    , m_abort_state_pr(other.m_abort_state_pr)
    , m_context(other.m_context)
    , m_runtime(other.m_runtime)
    , m_abort_state(other.m_abort_state)
    , m_own_log_pr(std::move(other).m_own_log_pr)
    , m_own_abort_state_pr(std::move(other).m_own_abort_state_pr) {
  }

  TestLog&
  operator=(const TestLog& other) {
    TestLog tmp(other);
    swap(tmp);
    return *this;
  }

  TestLog&
  operator=(TestLog&& other) {
    m_log_lr = other.m_log_lr;
    m_log_pr = other.m_log_pr;
    m_abort_state_lr = other.m_abort_state_lr;
    m_abort_state_pr = other.m_abort_state_pr;
    m_context = other.m_context;
    m_runtime = other.m_runtime;
    m_abort_state = other.m_abort_state;
    m_own_log_pr = std::move(other).m_own_log_pr;
    m_own_abort_state_pr = std::move(other).m_own_abort_state_pr;
    return *this;
  }

  virtual ~TestLog() {
    if (m_own_log_pr)
      m_runtime->unmap_region(m_context, m_own_log_pr.value());
    if (m_own_abort_state_pr)
      m_runtime->unmap_region(m_context, m_own_abort_state_pr.value());
  }

public:

  TestLogIterator<READ_WRITE>
  iterator() const {
    return
      TestLogIterator<READ_WRITE>(
        m_log_lr,
        m_log_pr,
        m_abort_state_pr,
        m_runtime);
  }

  bool
  contains_abort() const {
    return *m_abort_state.ptr(0);
  }

  template <typename CB>
  void
  for_each(CB cb) {
    auto it = iterator();
    while (!it.at_end()) {
      cb(it);
      ++it;
    }
  }

  TestLogReference
  log_reference() const {
    return
      TestLogReference(
        m_log_lr,
        Legion::LogicalRegionT<1>(m_log_pr->get_logical_region()),
        Legion::LogicalRegionT<1>(m_abort_state_pr->get_logical_region()));
  }

  std::vector<TestLogReference>
  get_log_references_by_state(const std::vector<TestState>& states) const {
    auto logref = log_reference();
    auto lp = logref.create_partition_by_log_state(m_context, m_runtime);

    std::vector<TestLogReference> result;
    std::transform(
      states.begin(),
      states.end(),
      std::back_inserter(result),
      [this, &logref, &lp](auto& st) {
        Legion::LogicalRegionT<1> log(
          m_runtime->get_logical_subregion_by_color(lp, st));
        return
          TestLogReference(
            log,
            Legion::LogicalRegionT<1>(m_log_pr->get_logical_region()),
            logref.abort_state_region());
      });
    return result;
  }

private:

  void
  swap(TestLog& other) {
    using std::swap;
    swap(m_log_lr, other.m_log_lr);
    swap(m_log_pr, other.m_log_pr);
    swap(m_abort_state_lr, other.m_abort_state_lr);
    swap(m_abort_state_pr, other.m_abort_state_pr);
    swap(m_context, other.m_context);
    swap(m_runtime, other.m_runtime);
    swap(m_abort_state, other.m_abort_state);
    swap(m_own_log_pr, other.m_own_log_pr);
    swap(m_own_abort_state_pr, other.m_own_abort_state_pr);
  }

  Legion::LogicalRegionT<1> m_log_lr;

  const Legion::PhysicalRegion* m_log_pr;

  Legion::LogicalRegionT<1> m_abort_state_lr;

  const Legion::PhysicalRegion* m_abort_state_pr;

  Legion::Context m_context;

  Legion::Runtime* m_runtime;

  TestLogReference::abort_state_accessor<READ_WRITE>::t m_abort_state;

  std::optional<Legion::PhysicalRegion> m_own_log_pr;

  std::optional<Legion::PhysicalRegion> m_own_abort_state_pr;
};

template <>
class TestLog<WRITE_DISCARD> {

public:

  TestLog(
    const Legion::LogicalRegion& log_lr,
    const Legion::PhysicalRegion& log_pr,
    const Legion::LogicalRegion& abort_state_lr,
    const Legion::PhysicalRegion& abort_state_pr,
    Legion::Context context,
    Legion::Runtime* runtime)
    : m_log_lr(Legion::LogicalRegionT<1>(log_lr))
    , m_log_pr(&log_pr)
    , m_abort_state_lr(Legion::LogicalRegionT<1>(abort_state_lr))
    , m_abort_state_pr(&abort_state_pr)
    , m_context(context)
    , m_runtime(runtime)
    , m_abort_state(abort_state_pr, 0, OpsManager::BOOL_OR_REDOP) {
  }

  TestLog(
    TestLogReference& logref,
    Legion::Context context,
    Legion::Runtime* runtime)
    : m_context(context)
    , m_runtime(runtime) {

    assert(runtime != nullptr);

    auto reqs = logref.requirements<WRITE_DISCARD>();
    m_own_log_pr =
      runtime->map_region(
        context,
        reqs[TestLogReference::log_requirement_index]);
    m_log_pr = &m_own_log_pr.value();
    m_log_lr = reqs[TestLogReference::log_requirement_index].region;
    m_own_abort_state_pr =
      runtime->map_region(
        context,
        reqs[TestLogReference::abort_state_requirement_index]);
    m_abort_state_pr = &m_own_abort_state_pr.value();
    m_abort_state_lr =
      reqs[TestLogReference::abort_state_requirement_index].region;
    m_abort_state =
      TestLogReference::abort_state_accessor<WRITE_DISCARD>::t(
        *m_abort_state_pr,
        0,
        OpsManager::BOOL_OR_REDOP);
  }

  TestLog(const TestLog& other)
    : m_log_lr(other.m_log_lr)
    , m_log_pr(other.m_log_pr)
    , m_abort_state_lr(other.m_abort_state_lr)
    , m_abort_state_pr(other.m_abort_state_pr)
    , m_context(other.m_context)
    , m_runtime(other.m_runtime)
    , m_abort_state(other.m_abort_state) {
  }

  TestLog(TestLog&& other)
    : m_log_lr(other.m_log_lr)
    , m_log_pr(other.m_log_pr)
    , m_abort_state_lr(other.m_abort_state_lr)
    , m_abort_state_pr(other.m_abort_state_pr)
    , m_context(other.m_context)
    , m_runtime(other.m_runtime)
    , m_abort_state(other.m_abort_state)
    , m_own_log_pr(std::move(other).m_own_log_pr)
    , m_own_abort_state_pr(std::move(other).m_own_abort_state_pr) {
  }

  TestLog&
  operator=(const TestLog& other) {
    TestLog tmp(other);
    swap(tmp);
    return *this;
  }

  TestLog&
  operator=(TestLog&& other) {
    m_log_lr = other.m_log_lr;
    m_log_pr = other.m_log_pr;
    m_abort_state_lr = other.m_abort_state_lr;
    m_abort_state_pr = other.m_abort_state_pr;
    m_context = other.m_context;
    m_runtime = other.m_runtime;
    m_abort_state = other.m_abort_state;
    m_own_log_pr = std::move(other).m_own_log_pr;
    m_own_abort_state_pr = std::move(other).m_own_abort_state_pr;
    return *this;
  }

  virtual ~TestLog() {
    if (m_own_log_pr)
      m_runtime->unmap_region(m_context, m_own_log_pr.value());
    if (m_own_abort_state_pr)
      m_runtime->unmap_region(m_context, m_own_abort_state_pr.value());
  }

public:

  TestLogIterator<WRITE_DISCARD>
  iterator() const {
    return
      TestLogIterator<WRITE_DISCARD>(
        m_log_lr,
        m_log_pr,
        m_abort_state_pr,
        m_runtime);
  }

  bool
  contains_abort() const {
    return *m_abort_state.ptr(0);
  }

  template <typename CB>
  void
  for_each(CB cb) {
    auto it = iterator();
    while (!it.at_end()) {
      cb(it);
      ++it;
    }
  }

  TestLogReference
  log_reference() const {
    return
      TestLogReference(
        m_log_lr,
        Legion::LogicalRegionT<1>(m_log_pr->get_logical_region()),
        Legion::LogicalRegionT<1>(m_abort_state_pr->get_logical_region()));
  }

  std::vector<TestLogReference>
  get_log_references_by_state(const std::vector<TestState>& states) const {
    auto logref = log_reference();
    auto lp = logref.create_partition_by_log_state(m_context, m_runtime);

    std::vector<TestLogReference> result;
    std::transform(
      states.begin(),
      states.end(),
      std::back_inserter(result),
      [this, &logref, &lp](auto& st) {
        Legion::LogicalRegionT<1> log(
          m_runtime->get_logical_subregion_by_color(lp, st));
        return
          TestLogReference(
            log,
            Legion::LogicalRegionT<1>(m_log_pr->get_logical_region()),
            logref.abort_state_region());
      });
    return result;
  }

private:

  void
  swap(TestLog& other) {
    using std::swap;
    swap(m_log_lr, other.m_log_lr);
    swap(m_log_pr, other.m_log_pr);
    swap(m_abort_state_lr, other.m_abort_state_lr);
    swap(m_abort_state_pr, other.m_abort_state_pr);
    swap(m_context, other.m_context);
    swap(m_runtime, other.m_runtime);
    swap(m_abort_state, other.m_abort_state);
    swap(m_own_log_pr, other.m_own_log_pr);
    swap(m_own_abort_state_pr, other.m_own_abort_state_pr);
  }

  Legion::LogicalRegionT<1> m_log_lr;

  const Legion::PhysicalRegion* m_log_pr;

  Legion::LogicalRegionT<1> m_abort_state_lr;

  const Legion::PhysicalRegion* m_abort_state_pr;

  Legion::Context m_context;

  Legion::Runtime* m_runtime;

  TestLogReference::abort_state_accessor<WRITE_DISCARD>::t m_abort_state;

  std::optional<Legion::PhysicalRegion> m_own_log_pr;

  std::optional<Legion::PhysicalRegion> m_own_abort_state_pr;
};

} // end namespace testing
} // end namespace hyperion

#endif // HYPERION_TESTING_TEST_LOG_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
