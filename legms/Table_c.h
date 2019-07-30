#ifndef LEGMS_TABLE_C_H_
#define LEGMS_TABLE_C_H_

#include "utility_c.h"
#include "Column_c.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LEGMS_API table_t {
  char name[LEGMS_MAX_STRING_SIZE];
  char axes_uid[LEGMS_MAX_STRING_SIZE];
  unsigned num_index_axes;
  int index_axes[LEGION_MAX_DIM];
  unsigned num_columns;
  column_t columns[LEGMS_MAX_NUM_TABLE_COLUMNS];
  unsigned num_keywords;
  type_tag_t keyword_datatypes[LEGMS_MAX_NUM_KEYWORDS];
  legion_logical_region_t keywords;
} table_t;

LEGMS_API int
table_has_column(const table_t* table, const char* name);

LEGMS_API const column_t*
table_column(const table_t* table, const char* name);

LEGMS_API table_t
table_reindexed(
  legion_context_t context,
  legion_runtime_t runtime,
  const table_t* table,
  unsigned num_axes,
  const int* axes,
  int allow_rows);

LEGMS_API void
table_partition_by_value(
  legion_context_t context,
  legion_runtime_t runtime,
  const table_t* table,
  unsigned num_axes,
  const int* axes,
  /* length of col_names and col_partitions arrays must equal value of
   * table_num_columns() */
  char** col_names,
  legion_logical_partition_t* col_partitions);

#ifdef LEGMS_USE_CASACORE
LEGMS_API table_t
table_from_ms(
  legion_context_t context,
  legion_runtime_t runtime,
  const char* path,
  // NULL-terminated array of string pointers
  const char** column_selections);
#endif

#ifdef LEGMS_USE_HDF5
// returns NULL-terminated array of strings -- caller must free everything
LEGMS_API char **
tables_in_h5(const char* path);

// returns NULL-terminated array of strings -- caller must free everything
LEGMS_API char **
columns_in_h5(const char* path, const char* table_path);

LEGMS_API table_t
table_from_h5(
  legion_context_t context,
  legion_runtime_t runtime,
  const char* path,
  const char* table_path,
  unsigned num_column_selections,
  const char** column_selections);

// use table_num_keywords() to find required minimum length of vectors
// "keywords" and "paths"; returned strings must be freed by caller
LEGMS_API void
table_keyword_paths(
  legion_context_t context,
  legion_runtime_t runtime,
  const table_t* table,
  char** keywords,
  char** paths);

// returned string "*path" must be freed by caller
LEGMS_API void
table_column_value_path(
  legion_context_t context,
  legion_runtime_t runtime,
  const table_t* table,
  const char* colname,
  char** path);

// use column_num_keywords() to find required minimum length of vector
// "keywords" and "paths"; returned strings must be freed by caller
LEGMS_API void
table_column_keyword_paths(
  legion_context_t context,
  legion_runtime_t runtime,
  const table_t* table,
  const char* colname,
  char** keywords,
  char** paths);

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
