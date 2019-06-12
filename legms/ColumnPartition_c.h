#ifndef LEGMS_COLUMN_PARTITION_C_H_
#define LEGMS_COLUMN_PARTITION_C_H_

#include "utility_c.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct legms_column_partition_t {
  void* impl;
} legms_column_partition_t;

legion_index_partition_t
legms_column_partition_index_partition(
  legms_column_partition_t column_partition);

const int *
legms_column_partition_axes(legms_column_partition_t column_partition);

size_t
legms_column_partition_num_axes(legms_column_partition_t column_partition);

void
legms_column_partition_destroy(legms_column_partition_t column_partition);

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
