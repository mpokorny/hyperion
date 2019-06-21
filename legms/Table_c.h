#ifndef LEGMS_TABLE_C_H_
#define LEGMS_TABLE_C_H_

#include "utility_c.h"
#include "Column_c.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct legms_table_t { void* impl; } legms_table_t;

const char*
legms_table_name(legms_table_t table);

int
legms_table_is_empty(legms_table_t table);

unsigned
legms_table_num_columns(legms_table_t table);

// use legms_table_num_columns() to find required minimum length of vector
// "names"; returned strings must be freed by caller
void
legms_table_column_names(legms_table_t table, char** names);

int
legms_table_has_column(legms_table_t table, const char* name);

legms_column_t
legms_table_column(legms_table_t table, const char* name);

const char*
legms_table_min_rank_column_name(legms_table_t table);

const char*
legms_table_max_rank_column_name(legms_table_t table);

const char*
legms_table_axes_uid(legms_table_t table);

unsigned
legms_table_num_index_axes(legms_table_t table);

const int*
legms_table_index_axes(legms_table_t table);

legms_table_t
legms_table_reindexed(
  legms_table_t table,
  const int* axes,
  unsigned num_axes,
  int allow_rows);

void
legms_table_destroy(legms_table_t table);

#ifdef USE_CASACORE
legms_table_t
legms_table_from_ms(
  legion_context_t context,
  legion_runtime_t runtime,
  const char* path,
  // NULL-terminated array of string pointers
  const char** column_selections);
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
