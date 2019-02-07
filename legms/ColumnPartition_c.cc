#include "ColumnPartition_c.h"
#include "c_util.h"

#include "legion/legion_c_util.h"

using namespace legms;
using namespace legms::CObjectWrapper;

legion_index_partition_t
column_partition_index_partition(column_partition_t column_partition) {
  return
    Legion::CObjectWrapper::wrap(unwrap(column_partition)->index_partition());
}

const int *
column_partition_axes(column_partition_t column_partition) {
  return unwrap(column_partition)->axes().data();
}

size_t
column_partition_num_axes(column_partition_t column_partition) {
  return unwrap(column_partition)->axes().size();
}

void
column_partition_destroy(column_partition_t column_partition) {
  destroy(column_partition);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
