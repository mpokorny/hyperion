#include "Column_c.h"
#include "c_util.h"

#include "legion/legion_c_util.h"

using namespace legms;
using namespace legms::CObjectWrapper;

const char *
column_name(column_t column) {
  return unwrap(column)->name().c_str();
}

unsigned
column_row_rank(column_t column) {
  return unwrap(column)->row_rank();
}

unsigned
column_rank(column_t column) {
  return unwrap(column)->rank();
}

size_t
column_num_rows(column_t column) {
  return unwrap(column)->num_rows();
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
column_projected_column_partition(
  column_t column,
  column_partition_t column_partition) {
  return wrap(
    unwrap(column)->projected_column_partition(unwrap(column_partition)));
}

legion_field_id_t
column_value_fid() {
  return Column::value_fid;
}

legion_field_id_t
column_row_number_fid() {
  return Column::row_number_fid;
}

void
column_register_tasks(legion_runtime_t runtime) {
  Column::register_tasks(Legion::CObjectWrapper::unwrap(runtime));
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
