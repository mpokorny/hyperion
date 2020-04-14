/*
 * Copyright 2020 Associated Universities, Inc. Washington DC, USA.
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
#ifndef HYPERION_COLUMN_C_H_
#define HYPERION_COLUMN_C_H_

#include <hyperion/hyperion_c.h>
#include <hyperion/utility_c.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HYPERION_API column_t {
  legion_logical_region_t metadata;
  legion_logical_region_t axes;
  legion_logical_region_t values;
  legion_logical_region_t keyword_type_tags;
  legion_logical_region_t keyword_values;
} column_t;

/* metadata field types: [char*, char*, type_tag_t] */
HYPERION_API const legion_field_id_t*
column_metadata_fs();

/* axes field types: [int] */
HYPERION_API const legion_field_id_t*
column_axes_fs();

/* values field types: [metadata[0][2]] */
HYPERION_API const legion_field_id_t*
column_values_fs();

HYPERION_API unsigned
column_rank(legion_runtime_t rt, column_t col);

HYPERION_API int
column_is_empty(column_t col);

/* HYPERION_API column_partition_t */
/* column_partition_on_axes( */
/*   legion_context_t ctx, */
/*   legion_runtime_t rt, */
/*   column_t col, */
/*   unsigned num_axes, */
/*   const int* axes); */

/* HYPERION_API column_partition_t */
/* column_projected_column_partition( */
/*   legion_context_t ctx, */
/*   legion_runtime_t rt, */
/*   column_t col, */
/*   column_partition_t cp); */

HYPERION_API void
column_destroy(legion_context_t ctx, legion_runtime_t rt, column_t col);

#ifdef __cplusplus
}
#endif

#endif // HYPERION_COLUMN_C_H_

// Local Variables:
// mode: c
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
