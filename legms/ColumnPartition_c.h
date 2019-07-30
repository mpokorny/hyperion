#ifndef LEGMS_COLUMN_PARTITION_C_H_
#define LEGMS_COLUMN_PARTITION_C_H_

#include "utility_c.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LEGMS_API column_partition_t {
  void* impl;
} column_partition_t;

LEGMS_API legion_index_partition_t
column_partition_index_partition(
  column_partition_t column_partition);

LEGMS_API const int *
column_partition_axes(column_partition_t column_partition);

LEGMS_API size_t
column_partition_num_axes(column_partition_t column_partition);

LEGMS_API void
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
