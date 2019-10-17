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
