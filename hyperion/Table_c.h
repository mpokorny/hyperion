/*
 * Copyright 2019 Associated Universities, Inc. Washington DC, USA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef HYPERION_TABLE_C_H_
#define HYPERION_TABLE_C_H_

#include <hyperion/utility_c.h>
#include <hyperion/Column_c.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HYPERION_API table_t {
  legion_logical_region_t metadata;
  legion_logical_region_t axes;
  legion_logical_region_t columns;
  legion_logical_region_t keyword_type_tags;
  legion_logical_region_t keyword_values;
} table_t;

/* metadata field types: [char*, char*] */
HYPERION_API const legion_field_id_t*
table_metadata_fs();

/* axes field types: [int] */
HYPERION_API const legion_field_id_t*
table_axes_fs();

HYPERION_API int
table_is_empty(legion_context_t ctx, legion_runtime_t rt, table_t t);

HYPERION_API char**
table_column_names(legion_context_t ctx, legion_runtime_t rt, table_t tab);

HYPERION_API column_t
table_column(
  legion_context_t ctx,
  legion_runtime_t rt,
  table_t tab,
  const char* name);

HYPERION_API void
table_destroy(
  legion_context_t ctx,
  legion_runtime_t rt,
  table_t tab,
  int destroy_columns);

/* HYPERION_API table_t */
/* table_reindexed( */
/*   legion_context_t context, */
/*   legion_runtime_t runtime, */
/*   const table_t* table, */
/*   unsigned num_axes, */
/*   const int* axes, */
/*   int allow_rows); */

/* HYPERION_API void */
/* table_partition_by_value( */
/*   legion_context_t context, */
/*   legion_runtime_t runtime, */
/*   const table_t* table, */
/*   unsigned num_axes, */
/*   const int* axes, */
/*   /\* length of col_names and col_partitions arrays must equal value of */
/*    * table_num_columns() *\/ */
/*   char** col_names, */
/*   legion_logical_partition_t* col_partitions); */

/* #ifdef HYPERION_USE_CASACORE */
/* HYPERION_API table_t */
/* table_from_ms( */
/*   legion_context_t context, */
/*   legion_runtime_t runtime, */
/*   const char* path, */
/*   // NULL-terminated array of string pointers */
/*   const char** column_selections); */
/* #endif */

/* #ifdef HYPERION_USE_HDF5 */
/* // returns NULL-terminated array of strings -- caller must free everything */
/* HYPERION_API char ** */
/* tables_in_h5(const char* path); */

/* // returns NULL-terminated array of strings -- caller must free everything */
/* HYPERION_API char ** */
/* columns_in_h5(const char* path, const char* table_path); */

/* HYPERION_API table_t */
/* table_from_h5( */
/*   legion_context_t context, */
/*   legion_runtime_t runtime, */
/*   const char* path, */
/*   const char* table_path, */
/*   unsigned num_column_selections, */
/*   const char** column_selections); */

/* // use table_num_keywords() to find required minimum length of vectors */
/* // "keywords" and "paths"; returned strings must be freed by caller */
/* HYPERION_API void */
/* table_keyword_paths( */
/*   legion_context_t context, */
/*   legion_runtime_t runtime, */
/*   const table_t* table, */
/*   char** keywords, */
/*   char** paths); */

/* // returned string "*path" must be freed by caller */
/* HYPERION_API void */
/* table_column_value_path( */
/*   legion_context_t context, */
/*   legion_runtime_t runtime, */
/*   const table_t* table, */
/*   const char* colname, */
/*   char** path); */

/* // use column_num_keywords() to find required minimum length of vector */
/* // "keywords" and "paths"; returned strings must be freed by caller */
/* HYPERION_API void */
/* table_column_keyword_paths( */
/*   legion_context_t context, */
/*   legion_runtime_t runtime, */
/*   const table_t* table, */
/*   const char* colname, */
/*   char** keywords, */
/*   char** paths); */

/* #endif */

#ifdef __cplusplus
}
#endif

#endif /* HYPERION_TABLE_C_H_ */

// Local Variables:
// mode: c
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
