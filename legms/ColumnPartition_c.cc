#include "c_util.h"
#include "ColumnPartition_c.h"
#include "ColumnPartition.h"

#include "legion/legion_c_util.h"

using namespace legms;
using namespace legms::CObjectWrapper;

legion_index_partition_t
legms_column_partition_index_partition(
  legms_column_partition_t column_partition) {
  return
    Legion::CObjectWrapper::wrap(unwrap(column_partition)->index_partition());
}

const int *
legms_column_partition_axes(legms_column_partition_t column_partition) {
  return unwrap(column_partition)->axes().data();
}

size_t
legms_column_partition_num_axes(legms_column_partition_t column_partition) {
  return unwrap(column_partition)->axes().size();
}

void
legms_column_partition_destroy(legms_column_partition_t column_partition) {
  destroy(column_partition);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
