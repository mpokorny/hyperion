#include <vector>

#include "c_util.h"
#include "Column_c.h"
#include "Column.h"

#include "legion/legion_c_util.h"

using namespace legms;
using namespace legms::CObjectWrapper;

const char *
column_name(column_t column) {
  return unwrap(column)->name().c_str();
}

unsigned
column_num_keywords(column_t column) {
  return unwrap(column)->keywords().size();
}

const char*
column_axes_uid(column_t column) {
  return unwrap(column)->axes_uid().c_str();
}

void
column_axes(column_t column, int* axes) {
  auto ax = unwrap(column)->axes();
  std::memcpy(axes, ax.data(), ax.size() * sizeof(int));
}

unsigned
column_rank(column_t column) {
  return unwrap(column)->rank();
}

::type_tag_t
column_datatype(column_t column) {
  return unwrap(column)->datatype();
}

legion_index_space_t
column_index_space(column_t column) {
  return Legion::CObjectWrapper::wrap(unwrap(column)->index_space());
}

legion_logical_region_t
column_logical_region(column_t column) {
  return Legion::CObjectWrapper::wrap(unwrap(column)->logical_region());
}

column_partition_t
column_partition_on_axes(
  column_t column,
  unsigned num_axes,
  const int* axes) {

  std::vector<int> ax(num_axes);
  std::memcpy(ax.data(), axes, num_axes * sizeof(int));
  return wrap(unwrap(column)->partition_on_axes(ax));
}

column_partition_t
column_projected_column_partition(
  column_t column,
  column_partition_t column_partition) {
  return
    wrap(
      unwrap(column)->projected_column_partition(unwrap(column_partition)));
}

legion_field_id_t
column_value_fid() {
  return Column::value_fid;
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
