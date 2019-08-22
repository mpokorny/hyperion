#ifndef LEGMS_COLUMN_C_H_
#define LEGMS_COLUMN_C_H_

#include "legms_c.h"
#include "utility_c.h"
#include "ColumnPartition_c.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LEGMS_API column_t {
  legion_logical_region_t metadata;
  legion_logical_region_t axes;
  legion_logical_region_t values;
  legion_logical_region_t keyword_type_tags;
  legion_logical_region_t keyword_values;
} column_t;

/* metadata field types: [char*, char*, type_tag_t] */
LEGMS_API const legion_field_id_t*
column_metadata_fs();

/* axes field types: [int] */
LEGMS_API const legion_field_id_t*
column_axes_fs();

/* values field types: [metadata[0][2]] */
LEGMS_API const legion_field_id_t*
column_values_fs();

LEGMS_API unsigned
column_rank(legion_runtime_t rt, column_t col);

LEGMS_API int
column_is_empty(column_t col);

LEGMS_API column_partition_t
column_partition_on_axes(
  legion_context_t ctx,
  legion_runtime_t rt,
  column_t col,
  unsigned num_axes,
  const int* axes);

LEGMS_API column_partition_t
column_projected_column_partition(
  legion_context_t ctx,
  legion_runtime_t rt,
  column_t col,
  column_partition_t cp);

LEGMS_API void
column_destroy(legion_context_t ctx, legion_runtime_t rt, column_t col);

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
