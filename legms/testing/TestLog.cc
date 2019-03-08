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

  LogicalRegion handle =
    runtime->create_logical_region(context, is, fs);

  runtime->destroy_field_space(context, fs);
  runtime->destroy_index_space(context, is);

  return TestLogReference{ handle, handle };
}

RegionRequirement
TestLogReference::rw_requirement() const {
  RegionRequirement result(
    handle,
    {STATE_FID, ABORT_FID, LOCATION_FID, DESCRIPTION_FID},
    {STATE_FID, ABORT_FID, LOCATION_FID, DESCRIPTION_FID},
    READ_WRITE,
    EXCLUSIVE,
    parent);
  return result;
}

RegionRequirement
TestLogReference::ro_requirement() const {
  RegionRequirement result(
    handle,
    {STATE_FID, ABORT_FID, LOCATION_FID, DESCRIPTION_FID},
    {STATE_FID, ABORT_FID, LOCATION_FID, DESCRIPTION_FID},
    READ_ONLY,
    EXCLUSIVE,
    parent);
  return result;
}

RegionRequirement
TestLogReference::wd_requirement() const {
  RegionRequirement result(
    handle,
    {STATE_FID, ABORT_FID, LOCATION_FID, DESCRIPTION_FID},
    {STATE_FID, ABORT_FID, LOCATION_FID, DESCRIPTION_FID},
    WRITE_DISCARD,
    EXCLUSIVE,
    parent);

  return result;
}

LogicalPartition
TestLogReference::partition_by_state(Context context, Runtime* runtime) const {

  IndexSpaceT<1,int> states(
    runtime->create_index_space(
      context,
      Rect<1,int>(TestState::SUCCESS, TestState::UNKNOWN)));
  IndexPartitionT<1> states_partition(
    runtime->create_partition_by_field(
      context,
      handle,
      parent,
      STATE_FID,
      states));
  LogicalPartition result(
    runtime->get_logical_partition(context, handle, states_partition));
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
