#ifndef LEGMS_COLUMN_C_H_
#define LEGMS_COLUMN_C_H_

#include "legms_c.h"
#include "utility_c.h"
#include "ColumnPartition_c.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct column_t {
  char name[LEGMS_MAX_STRING_SIZE];
  char axes_uid[LEGMS_MAX_STRING_SIZE];
  type_tag_t datatype;
  unsigned num_axes;
  int axes[LEGION_MAX_DIM];
  legion_logical_region_t values;
  unsigned num_keywords;
  type_tag_t keyword_datatypes[LEGMS_MAX_NUM_KEYWORDS];
  legion_logical_region_t keywords;
} column_t;

column_partition_t
column_partition_on_axes(
  legion_context_t context,
  legion_runtime_t runtime,
  const column_t* column,
  unsigned num_axes,
  const int* axes);

column_partition_t
column_projected_column_partition(
  legion_context_t context,
  legion_runtime_t runtime,
  const column_t* column,
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
