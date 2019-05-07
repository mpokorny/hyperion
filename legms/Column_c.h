#ifndef LEGMS_COLUMN_C_H_
#define LEGMS_COLUMN_C_H_

#include "legms_c.h"
#include "utility_c.h"
#include "ColumnPartition_c.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct column_t { void* impl; } column_t;

const char *
column_name(column_t column);

unsigned
column_rank(column_t column);

legion_index_space_t
column_index_space(column_t column);

legion_logical_region_t
column_logical_region(column_t column);

column_partition_t
column_partition_on_axes(
  column_t column,
  /* first element of axes is the vector's length -1 */
  const int* axes);

column_partition_t
column_projected_column_partition(
  column_t column,
  column_partition_t column_partition);

legion_field_id_t
column_value_fid();

#ifdef __cplusplus
}
#endif

#endif // LEGMS_COLUMN_C_H_

// Local Variables:
// mode: c
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
