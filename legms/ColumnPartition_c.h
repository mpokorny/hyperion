#ifndef LEGMS_COLUMN_PARTITION_C_H_
#define LEGMS_COLUMN_PARTITION_C_H_

#include <legms/utility_c.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LEGMS_API column_partition_t {
  legion_logical_region_t axes_uid;
  legion_logical_region_t axes;
  legion_index_partition_t index_partition;
} column_partition_t;

// axes_uid field types: [char*]
LEGMS_API const legion_field_id_t*
column_partition_axes_uid_fs();

// axes field types: [int]
LEGMS_API const legion_field_id_t*
column_partition_axes_fs();

LEGMS_API legion_index_space_t
column_partition_color_space(legion_runtime_t rt, column_partition_t cp);

LEGMS_API void
column_partition_destroy(
  legion_context_t ctx,
  legion_runtime_t rt,
  column_partition_t cp,
  int destroy_color_space);

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
