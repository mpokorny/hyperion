#include "TestLog.h"

#include <algorithm>

#include "utility.h"

using namespace legms::testing;
using namespace Legion;

TestLogReference::TestLogReference(
  size_t length,
  Context context,
  Runtime* runtime)
  : m_own_regions(true)
  , m_context(context)
  , m_runtime(runtime) {

  {
    FieldSpace fs = runtime->create_field_space(context);

    FieldAllocator fa = runtime->create_field_allocator(context, fs);
    fa.allocate_field(sizeof(TestState), STATE_FID);
    fa.allocate_field(sizeof(bool), ABORT_FID);
    fa.allocate_field(
      sizeof(std::string),
      NAME_FID,
      SerdezManager::CASACORE_STRING_SID);
    fa.allocate_field(
      sizeof(std::string),
      FAIL_INFO_FID,
      SerdezManager::CASACORE_STRING_SID);

    IndexSpaceT<1> is =
      runtime->create_index_space(context, Rect<1>(0, length - 1));

    m_log_handle = runtime->create_logical_region(context, is, fs);
    m_log_parent = m_log_handle;

    runtime->destroy_field_space(context, fs);
    runtime->destroy_index_space(context, is);
  }

  {
    FieldSpace fs = runtime->create_field_space(context);

    FieldAllocator fa = runtime->create_field_allocator(context, fs);
    fa.allocate_field(sizeof(bool), 0);

    IndexSpaceT<1> is =
      runtime->create_index_space(context, Rect<1>(0, 0));

    m_abort_state_handle = runtime->create_logical_region(context, is, fs);

    runtime->destroy_field_space(context, fs);
    runtime->destroy_index_space(context, is);
  }
}

TestLogReference::TestLogReference(
  LogicalRegionT<1> log_handle,
  LogicalRegionT<1> log_parent,
  LogicalRegionT<1> abort_state_handle)
  : m_log_handle(log_handle)
  , m_log_parent(log_parent)
  , m_abort_state_handle(abort_state_handle)
  , m_own_regions(false)
  , m_runtime(nullptr) {}

TestLogReference::~TestLogReference() {
  if (m_own_regions) {
    m_runtime->destroy_logical_region(m_context, m_log_handle);
    m_runtime->destroy_logical_region(m_context, m_abort_state_handle);
  }
}

std::array<RegionRequirement, 2>
TestLogReference::rw_requirements(LogicalRegionT<1> child) const {

  RegionRequirement
    log_req(
      child,
      {STATE_FID, ABORT_FID, NAME_FID, FAIL_INFO_FID},
      {STATE_FID, ABORT_FID, NAME_FID, FAIL_INFO_FID},
      READ_WRITE,
      EXCLUSIVE,
      m_log_handle);

  RegionRequirement
    abort_state_req(
      m_abort_state_handle,
      {0},
      {0},
      SerdezManager::BOOL_OR_REDOP,
      ATOMIC,
      m_abort_state_handle);

  std::array<RegionRequirement, 2> result;
  result[log_requirement_index] = log_req;
  result[abort_state_requirement_index] = abort_state_req;
  return result;
}

std::array<RegionRequirement, 2>
TestLogReference::ro_requirements(LogicalRegionT<1> child) const {

  RegionRequirement
    log_req(
      child,
      {STATE_FID, ABORT_FID, NAME_FID, FAIL_INFO_FID},
      {STATE_FID, ABORT_FID, NAME_FID, FAIL_INFO_FID},
      READ_ONLY,
      EXCLUSIVE,
      m_log_handle);

  RegionRequirement
    abort_state_req(
      m_abort_state_handle,
      {0},
      {0},
      {READ_ONLY},
      ATOMIC,
      m_abort_state_handle);

  std::array<RegionRequirement, 2> result;
  result[log_requirement_index] = log_req;
  result[abort_state_requirement_index] = abort_state_req;
  return result;
}

std::array<RegionRequirement, 2>
TestLogReference::wd_requirements(LogicalRegionT<1> child) const {

  RegionRequirement
    log_req(
      child,
      {STATE_FID, ABORT_FID, NAME_FID, FAIL_INFO_FID},
      {STATE_FID, ABORT_FID, NAME_FID, FAIL_INFO_FID},
      WRITE_DISCARD,
      EXCLUSIVE,
      m_log_handle);

  RegionRequirement
    abort_state_req(
      m_abort_state_handle,
      {0},
      {0},
      SerdezManager::BOOL_OR_REDOP,
      ATOMIC,
      m_abort_state_handle);

  std::array<RegionRequirement, 2> result;
  result[log_requirement_index] = log_req;
  result[abort_state_requirement_index] = abort_state_req;
  return result;
}

std::array<RegionRequirement, 2>
TestLogReference::rw_requirements(
  LogicalPartitionT<1> log_partition,
  int projection_id) const {

  RegionRequirement
    log_req(
      log_partition,
      projection_id,
      {STATE_FID, ABORT_FID, NAME_FID, FAIL_INFO_FID},
      {STATE_FID, ABORT_FID, NAME_FID, FAIL_INFO_FID},
      READ_WRITE,
      EXCLUSIVE,
      m_log_handle);

  RegionRequirement
    abort_state_req(
      m_abort_state_handle,
      {0},
      {0},
      SerdezManager::BOOL_OR_REDOP,
      ATOMIC,
      m_abort_state_handle);

  std::array<RegionRequirement, 2> result;
  result[log_requirement_index] = log_req;
  result[abort_state_requirement_index] = abort_state_req;
  return result;
}

std::array<RegionRequirement, 2>
TestLogReference::ro_requirements(
  LogicalPartitionT<1> log_partition,
  int projection_id) const {

  RegionRequirement
    log_req(
      log_partition,
      projection_id,
      {STATE_FID, ABORT_FID, NAME_FID, FAIL_INFO_FID},
      {STATE_FID, ABORT_FID, NAME_FID, FAIL_INFO_FID},
      READ_ONLY,
      EXCLUSIVE,
      m_log_handle);

  RegionRequirement
    abort_state_req(
      m_abort_state_handle,
      {0},
      {0},
      {READ_ONLY},
      ATOMIC,
      m_abort_state_handle);

  std::array<RegionRequirement, 2> result;
  result[log_requirement_index] = log_req;
  result[abort_state_requirement_index] = abort_state_req;
  return result;
}

std::array<RegionRequirement, 2>
TestLogReference::wd_requirements(
  LogicalPartitionT<1> log_partition,
  int projection_id) const {

  RegionRequirement
    log_req(
      log_partition,
      projection_id,
      {STATE_FID, ABORT_FID, NAME_FID, FAIL_INFO_FID},
      {STATE_FID, ABORT_FID, NAME_FID, FAIL_INFO_FID},
      WRITE_DISCARD,
      EXCLUSIVE,
      m_log_handle);

  RegionRequirement
    abort_state_req(
      m_abort_state_handle,
      {0},
      {0},
      SerdezManager::BOOL_OR_REDOP,
      ATOMIC,
      m_abort_state_handle);

  std::array<RegionRequirement, 2> result;
  result[log_requirement_index] = log_req;
  result[abort_state_requirement_index] = abort_state_req;
  return result;
}

Legion::LogicalPartitionT<1>
TestLogReference::create_partition_by_log_state(
  Context context,
  Runtime* runtime) const {

  IndexSpaceT<1,TEST_STATE_TYPE> states(
    runtime->create_index_space(
      context,
      Rect<1,TEST_STATE_TYPE>(TestState::SUCCESS, TestState::UNKNOWN)));
  IndexPartitionT<1> ip(
    runtime->create_partition_by_field(
      context,
      m_log_handle,
      m_log_parent,
      STATE_FID,
      states));
  LogicalPartitionT<1> result =
    runtime->get_logical_partition(m_log_handle, ip);

  runtime->destroy_index_space(context, states);
  return result;
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
