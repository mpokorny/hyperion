#include "TestLog.h"

#include "utility.h"

using namespace legms::testing;
using namespace Legion;

TestLogReference
TestLogReference::create(
  size_t length,
  Legion::Context context,
  Legion::Runtime* runtime) {

  FieldSpace fs = runtime->create_field_space(context);

  FieldAllocator fa = runtime->create_field_allocator(context, fs);
  fa.allocate_field(sizeof(TestState), STATE_FID);
  fa.allocate_field(sizeof(bool), ABORT_FID);
  fa.allocate_field(
    sizeof(std::string),
    LOCATION_FID,
    SerdezManager::CASACORE_STRING_SID);
  fa.allocate_field(
    sizeof(std::string),
    DESCRIPTION_FID,
    SerdezManager::CASACORE_STRING_SID);

  IndexSpaceT<1> is =
    runtime->create_index_space(context, Rect<1>(0, length - 1));

  LogicalRegion log_handle =
    runtime->create_logical_region(context, is, fs);

  runtime->destroy_field_space(context, fs);
  runtime->destroy_index_space(context, is);

  return TestLogReference{ log_handle, log_handle, LogicalRegion::NO_REGION };
}

std::vector<RegionRequirement>
TestLogReference::rw_requirements() const {
  RegionRequirement result(
    log_handle,
    {STATE_FID, ABORT_FID, LOCATION_FID, DESCRIPTION_FID},
    {STATE_FID, ABORT_FID, LOCATION_FID, DESCRIPTION_FID},
    READ_WRITE,
    EXCLUSIVE,
    log_parent);
  return {result};
}

std::vector<RegionRequirement>
TestLogReference::ro_requirements() const {
  RegionRequirement result(
    log_handle,
    {STATE_FID, ABORT_FID, LOCATION_FID, DESCRIPTION_FID},
    {STATE_FID, ABORT_FID, LOCATION_FID, DESCRIPTION_FID},
    READ_ONLY,
    EXCLUSIVE,
    log_parent);
  return {result};
}

std::vector<RegionRequirement>
TestLogReference::wd_requirements() const {
  RegionRequirement result(
    log_handle,
    {STATE_FID, ABORT_FID, LOCATION_FID, DESCRIPTION_FID},
    {STATE_FID, ABORT_FID, LOCATION_FID, DESCRIPTION_FID},
    WRITE_DISCARD,
    EXCLUSIVE,
    log_parent);

  return {result};
}

LogicalPartition
TestLogReference::partition_log_by_state(
  Context context,
  Runtime* runtime) const {

  IndexSpaceT<1,int> states(
    runtime->create_index_space(
      context,
      Rect<1,int>(TestState::SUCCESS, TestState::UNKNOWN)));
  IndexPartitionT<1> states_partition(
    runtime->create_partition_by_field(
      context,
      log_handle,
      log_parent,
      STATE_FID,
      states));
  LogicalPartition result(
    runtime->get_logical_partition(context, log_handle, states_partition));
  runtime->destroy_index_partition(context, states_partition);
  runtime->destroy_index_space(context, states);
  return result;
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
