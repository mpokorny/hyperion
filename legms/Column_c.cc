#include <vector>

#include "c_util.h"
#include "Column_c.h"
#include "Column.h"

#include "legion/legion_c_util.h"

using namespace legms;
using namespace legms::CObjectWrapper;

const char *
legms_column_name(legms_column_t column) {
  return unwrap(column)->name().c_str();
}

const char*
legms_column_axes_uid(legms_column_t column) {
  return unwrap(column)->axes_uid();
}

void
legms_column_axes(legms_column_t column, int* axes) {
  auto ax = unwrap(column)->axes();
  std::memcpy(axes, ax.data(), ax.size() * sizeof(int));
}

unsigned
legms_column_rank(legms_column_t column) {
  return unwrap(column)->rank();
}

::legms_type_tag_t
legms_column_datatype(legms_column_t column) {
  return unwrap(column)->datatype();
}

legion_index_space_t
legms_column_index_space(legms_column_t column) {
  return Legion::CObjectWrapper::wrap(unwrap(column)->index_space());
}

legion_logical_region_t
legms_column_logical_region(legms_column_t column) {
  return Legion::CObjectWrapper::wrap(unwrap(column)->logical_region());
}

legms_column_partition_t
legms_column_partition_on_axes(
  legms_column_t column,
  unsigned num_axes,
  const int* axes) {

  std::vector<int> ax(num_axes);
  std::memcpy(ax.data(), axes, num_axes * sizeof(int));
  return wrap(unwrap(column)->partition_on_axes(ax));
}

legms_column_partition_t
legms_column_projected_column_partition(
  legms_column_t column,
  legms_column_partition_t column_partition) {
  return
    wrap(
      unwrap(column)->projected_column_partition(unwrap(column_partition)));
}

legion_field_id_t
legms_column_value_fid() {
  return Column::value_fid;
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
