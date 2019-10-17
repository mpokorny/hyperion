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
#ifndef HYPERION_GRIDS_C_H_
#define HYPERION_GRIDS_C_H_

#include <hyperion/utility_c.h>

#ifdef __cplusplus
extern "C" {
#endif

HYPERION_API void
block_and_halo_partitions_1d(
  legion_context_t ctx,
  legion_runtime_t runtime,
  legion_index_space_t grid,
  legion_point_1d_t block_size,
  legion_point_1d_t border,
  legion_index_partition_t* block_ip,
  legion_index_partition_t* halo_ip);

HYPERION_API void
block_and_halo_partitions_2d(
  legion_context_t ctx,
  legion_runtime_t runtime,
  legion_index_space_t grid,
  legion_point_2d_t block_size,
  legion_point_2d_t border,
  legion_index_partition_t* block_ip,
  legion_index_partition_t* halo_ip);

HYPERION_API void
block_and_halo_partitions_3d(
  legion_context_t ctx,
  legion_runtime_t runtime,
  legion_index_space_t grid,
  legion_point_3d_t block_size,
  legion_point_3d_t border,
  legion_index_partition_t* block_ip,
  legion_index_partition_t* halo_ip);

#ifdef __cplusplus
}
#endif

#endif // HYPERION_GRIDS_C_H_

// Local Variables:
// mode: c
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
