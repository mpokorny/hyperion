#include <vector>

#include "c_util.h"
#include "Column_c.h"
#include "Column.h"

#include "legion/legion_c_util.h"

using namespace legms;
// need the following for column_partition_t...should remove that requirement
using namespace legms::CObjectWrapper;

static std::unique_ptr<Column>
to_column(
  legion_context_t context,
  legion_runtime_t runtime,
  const column_t& col) {

  return
    ColumnGenArgs(col)(
      Legion::CObjectWrapper::unwrap(context)->context(),
      Legion::CObjectWrapper::unwrap(runtime));
}

column_partition_t
column_partition_on_axes(
  legion_context_t context,
  legion_runtime_t runtime,
  const column_t* column,
  unsigned num_axes,
  const int* axes) {

  std::vector<int> ax(num_axes);
  std::memcpy(ax.data(), axes, num_axes * sizeof(int));

  return wrap(to_column(context, runtime, *column)->partition_on_axes(ax));
}

column_partition_t
column_projected_column_partition(
  legion_context_t context,
  legion_runtime_t runtime,
  const column_t* column,
  column_partition_t column_partition) {

  return
    wrap(
      to_column(context, runtime, *column)
      ->projected_column_partition(unwrap(column_partition)));
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
