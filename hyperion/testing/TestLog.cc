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
#include <hyperion/testing/TestLog.h>

#include <algorithm>

using namespace hyperion::testing;
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
    fa.allocate_field(sizeof(Point<1>), STATE_FID);
    fa.allocate_field(sizeof(bool), ABORT_FID);
    fa.allocate_field(
      sizeof(std::string),
      NAME_FID,
      OpsManager::serdez_id(OpsManager::STD_STRING_SID));
    fa.allocate_field(
      sizeof(std::string),
      FAIL_INFO_FID,
      OpsManager::serdez_id(OpsManager::STD_STRING_SID));

    IndexSpaceT<1> is =
      runtime->create_index_space(context, Rect<1>(0, length - 1));

    m_log_handle = runtime->create_logical_region(context, is, fs);
    m_log_parent = m_log_handle;

    runtime->fill_field<Point<1>>(
      context,
      m_log_handle,
      m_log_handle,
      STATE_FID,
      Point<1>(TestState::UNKNOWN));
    runtime->fill_field<bool>(
      context,
      m_log_handle,
      m_log_handle,
      ABORT_FID,
      false);
    RegionRequirement req(m_log_handle, WRITE_DISCARD, EXCLUSIVE, m_log_handle);
    req.add_field(NAME_FID);
    req.add_field(FAIL_INFO_FID);
    PhysicalRegion pr = runtime->map_region(context, req);
    const FieldAccessor<
      WRITE_ONLY,
      std::string,
      1,
      coord_t,
      AffineAccessor<std::string, 1, coord_t>,
      false> names(pr, NAME_FID);
    const FieldAccessor<
      WRITE_ONLY,
      std::string,
      1,
      coord_t,
      AffineAccessor<std::string, 1, coord_t>,
      false> fails(pr, FAIL_INFO_FID);
    for (PointInDomainIterator<1> pid(runtime->get_index_space_domain(is));
         pid();
         pid++) {
      ::new(names.ptr(*pid)) std::string;
      ::new(fails.ptr(*pid)) std::string;
    }
    runtime->unmap_region(context, pr);
  }

  {
    FieldSpace fs = runtime->create_field_space(context);

    FieldAllocator fa = runtime->create_field_allocator(context, fs);
    fa.allocate_field(sizeof(bool), 0);

    IndexSpaceT<1> is =
      runtime->create_index_space(context, Rect<1>(0, 0));

    m_abort_state_handle = runtime->create_logical_region(context, is, fs);

    runtime->fill_field<bool>(
      context,
      m_abort_state_handle,
      m_abort_state_handle,
      0,
      false);
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
    {
      auto fs = m_log_handle.get_field_space();
      auto is = m_log_handle.get_index_space();
      m_runtime->destroy_logical_region(m_context, m_log_handle);
      m_runtime->destroy_index_space(m_context, is);
      m_runtime->destroy_field_space(m_context, fs);
    }
    {
      auto fs = m_abort_state_handle.get_field_space();
      auto is = m_abort_state_handle.get_index_space();
      m_runtime->destroy_logical_region(m_context, m_abort_state_handle);
      m_runtime->destroy_index_space(m_context, is);
      m_runtime->destroy_field_space(m_context, fs);
    }
  }
}

std::array<RegionRequirement, 2>
TestLogReference::rw_requirements(
  LogicalRegionT<1> child,
  LogicalRegionT<1> log_parent) const {

  RegionRequirement
    log_req(
      child,
      {STATE_FID, ABORT_FID, NAME_FID, FAIL_INFO_FID},
      {STATE_FID, ABORT_FID, NAME_FID, FAIL_INFO_FID},
      READ_WRITE,
      EXCLUSIVE,
      log_parent);

  RegionRequirement
    abort_state_req(
      m_abort_state_handle,
      {0},
      {0},
      OpsManager::reduction_id(OpsManager::BOOL_OR_REDOP),
      ATOMIC,
      m_abort_state_handle);

  std::array<RegionRequirement, 2> result;
  result[log_requirement_index] = log_req;
  result[abort_state_requirement_index] = abort_state_req;
  return result;
}

std::array<RegionRequirement, 2>
TestLogReference::ro_requirements(
  LogicalRegionT<1> child,
  Legion::LogicalRegionT<1> log_parent) const {

  RegionRequirement
    log_req(
      child,
      {STATE_FID, ABORT_FID, NAME_FID, FAIL_INFO_FID},
      {STATE_FID, ABORT_FID, NAME_FID, FAIL_INFO_FID},
      READ_ONLY,
      EXCLUSIVE,
      log_parent);

  RegionRequirement
    abort_state_req(
      m_abort_state_handle,
      {0},
      {0},
      READ_ONLY,
      ATOMIC,
      m_abort_state_handle);

  std::array<RegionRequirement, 2> result;
  result[log_requirement_index] = log_req;
  result[abort_state_requirement_index] = abort_state_req;
  return result;
}

std::array<RegionRequirement, 2>
TestLogReference::wd_requirements(
  LogicalRegionT<1> child,
  Legion::LogicalRegionT<1> log_parent) const {

  RegionRequirement
    log_req(
      child,
      {STATE_FID, ABORT_FID, NAME_FID, FAIL_INFO_FID},
      {STATE_FID, ABORT_FID, NAME_FID, FAIL_INFO_FID},
      WRITE_DISCARD,
      EXCLUSIVE,
      log_parent);

  RegionRequirement
    abort_state_req(
      m_abort_state_handle,
      {0},
      {0},
      OpsManager::reduction_id(OpsManager::BOOL_OR_REDOP),
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
  LogicalRegionT<1> log_parent,
  int projection_id) const {

  RegionRequirement
    log_req(
      log_partition,
      projection_id,
      {STATE_FID, ABORT_FID, NAME_FID, FAIL_INFO_FID},
      {STATE_FID, ABORT_FID, NAME_FID, FAIL_INFO_FID},
      READ_WRITE,
      EXCLUSIVE,
      log_parent);

  RegionRequirement
    abort_state_req(
      m_abort_state_handle,
      {0},
      {0},
      OpsManager::reduction_id(OpsManager::BOOL_OR_REDOP),
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
  LogicalRegionT<1> log_parent,
  int projection_id) const {

  RegionRequirement
    log_req(
      log_partition,
      projection_id,
      {STATE_FID, ABORT_FID, NAME_FID, FAIL_INFO_FID},
      {STATE_FID, ABORT_FID, NAME_FID, FAIL_INFO_FID},
      READ_ONLY,
      EXCLUSIVE,
      log_parent);

  RegionRequirement
    abort_state_req(
      m_abort_state_handle,
      {0},
      {0},
      READ_ONLY,
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
  LogicalRegionT<1> log_parent,
  int projection_id) const {

  RegionRequirement
    log_req(
      log_partition,
      projection_id,
      {STATE_FID, ABORT_FID, NAME_FID, FAIL_INFO_FID},
      {STATE_FID, ABORT_FID, NAME_FID, FAIL_INFO_FID},
      WRITE_DISCARD,
      EXCLUSIVE,
      log_parent);

  RegionRequirement
    abort_state_req(
      m_abort_state_handle,
      {0},
      {0},
      OpsManager::reduction_id(OpsManager::BOOL_OR_REDOP),
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

  IndexSpaceT<1> states(
    runtime->create_index_space(
      context,
      Rect<1>(TestState::SUCCESS, TestState::UNKNOWN)));
  IndexPartitionT<1> ip(
    runtime->create_partition_by_field(
      context,
      m_log_handle,
      m_log_parent,
      STATE_FID,
      states));

  LogicalPartitionT<1> result =
    runtime->get_logical_partition(m_log_handle, ip);

  //runtime->destroy_index_space(context, states);
  return result;
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
