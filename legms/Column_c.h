#ifndef LEGMS_COLUMN_C_H_
#define LEGMS_COLUMN_C_H_

#include "legms_c.h"
#include "utility_c.h"
#include "ColumnPartition_c.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct legms_column_t { void* impl; } legms_column_t;

const char*
legms_column_name(legms_column_t column);

const char*
legms_column_axes_uid(legms_column_t column);

// use legms_column_rank() to find required minimum length of vector "axes"
void
legms_column_axes(legms_column_t column, int* axes);

unsigned
legms_column_rank(legms_column_t column);

legms_type_tag_t
legms_column_datatype(legms_column_t column);

legion_index_space_t
legms_column_index_space(legms_column_t column);

legion_logical_region_t
legms_column_logical_region(legms_column_t column);

legms_column_partition_t
legms_column_partition_on_axes(
  legms_column_t column,
  unsigned num_axes,
  const int* axes);

legms_column_partition_t
legms_column_projected_column_partition(
  legms_column_t column,
  legms_column_partition_t column_partition);

legion_field_id_t
legms_column_value_fid();

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
