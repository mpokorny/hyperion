#ifndef LEGMS_COLUMN_PARTITION_C_H_
#define LEGMS_COLUMN_PARTITION_C_H_

#include "utility_c.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct column_partition_t {
  void* impl;
} column_partition_t;

legion_index_partition_t
column_partition_index_partition(
  column_partition_t column_partition);

const int *
column_partition_axes(column_partition_t column_partition);

size_t
column_partition_num_axes(column_partition_t column_partition);

void
column_partition_destroy(column_partition_t column_partition);

#ifdef __cplusplus
}
#endif

#endif // LEGMS_COLUMN_PARTITION_C_H_

// Local Variables:
// mode: c
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
