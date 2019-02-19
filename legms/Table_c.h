#ifndef LEGMS_TABLE_C_H_
#define LEGMS_TABLE_C_H_

#include "legion/legion_c.h"
#include "Column_c.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct table_t { void* impl; } table_t;

const char*
table_name(table_t table);

unsigned
table_rank(table_t table);

int
table_is_empty(table_t table);

char** // NULL-terminated array of string pointers
table_column_names(table_t table);

column_t
table_column(table_t table, const char* name);

const char *
table_min_rank_column_name(table_t table);

const char *
table_max_rank_column_name(table_t table);

#if 0
column_row_number_t
table_num_rows(table_t table);

column_partition_t
table_row_block_partition(table_t table, size_t block_length);

column_partition_t
table_all_rows_partition(table_t table);

column_partition_t
table_row_partition(
  table_t table,
  // NULL-terminated array of vectors; first element
  // of each vector is its length -1
  column_row_number_t** rowp,
  int include_unselected,
  int sorted_selections);
#endif

void
table_destroy(table_t table);

table_t
table_from_ms(
  legion_context_t context,
  legion_runtime_t runtime,
  const char* path,
  // NULL-terminated array of string pointers
  const char** column_selections);

#ifdef __cplusplus
}
#endif

#endif /* LEGMS_TABLE_C_H_ */

// Local Variables:
// mode: c
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
