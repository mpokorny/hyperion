#ifndef LEGMS_TABLE_C_H_
#define LEGMS_TABLE_C_H_

#include "utility_c.h"
#include "Column_c.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct table_t { void* impl; } table_t;

const char*
table_name(table_t table);

int
table_is_empty(table_t table);

unsigned
table_num_columns(table_t table);

// use table_num_columns() to find required minimum length of vector
// "names"; returned strings must be freed by caller
void
table_column_names(table_t table, char** names);

int
table_has_column(table_t table, const char* name);

column_t
table_column(table_t table, const char* name);

const char*
table_min_rank_column_name(table_t table);

const char*
table_max_rank_column_name(table_t table);

const char*
table_axes_uid(table_t table);

unsigned
table_num_index_axes(table_t table);

const int*
table_index_axes(table_t table);

table_t
table_reindexed(
  table_t table,
  const int* axes,
  unsigned num_axes,
  int allow_rows);

void
table_destroy(table_t table);

#ifdef USE_CASACORE
table_t
table_from_ms(
  legion_context_t context,
  legion_runtime_t runtime,
  const char* path,
  // NULL-terminated array of string pointers
  const char** column_selections);
#endif

#ifdef USE_HDF5
table_t
init_from_h5(
  legion_context_t context,
  legion_runtime_t runtime,
  const char* path);
#endif

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
