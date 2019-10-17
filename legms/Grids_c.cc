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
#include "Grids_c.h"
#include "Grids.h"

#include <hyperion/hyperion_c.h>

#pragma GCC visibility push(default)
#include <legion/legion_c_util.h>
#pragma GCC visibility pop

void
block_and_halo_partitions_1d(
  legion_context_t ctx,
  legion_runtime_t runtime,
  legion_index_space_t grid,
  legion_point_1d_t block_size,
  legion_point_1d_t border,
  legion_index_partition_t* block_ip,
  legion_index_partition_t* halo_ip) {

  Legion::IndexSpaceT<1> grid1(Legion::CObjectWrapper::unwrap(grid));
  auto [bip, hip] =
    hyperion::block_and_halo_partitions(
      Legion::CObjectWrapper::unwrap(ctx)->context(),
      Legion::CObjectWrapper::unwrap(runtime),
      grid1,
      Legion::CObjectWrapper::unwrap(block_size),
      Legion::CObjectWrapper::unwrap(border));
  *block_ip = Legion::CObjectWrapper::wrap(bip);
  *halo_ip = Legion::CObjectWrapper::wrap(hip);
}

void
block_and_halo_partitions_2d(
  legion_context_t ctx,
  legion_runtime_t runtime,
  legion_index_space_t grid,
  legion_point_2d_t block_size,
  legion_point_2d_t border,
  legion_index_partition_t* block_ip,
  legion_index_partition_t* halo_ip) {

  Legion::IndexSpaceT<2> grid2(Legion::CObjectWrapper::unwrap(grid));
  auto [bip, hip] =
    hyperion::block_and_halo_partitions(
      Legion::CObjectWrapper::unwrap(ctx)->context(),
      Legion::CObjectWrapper::unwrap(runtime),
      grid2,
      Legion::CObjectWrapper::unwrap(block_size),
      Legion::CObjectWrapper::unwrap(border));
  *block_ip = Legion::CObjectWrapper::wrap(bip);
  *halo_ip = Legion::CObjectWrapper::wrap(hip);
}

void
block_and_halo_partitions_3d(
  legion_context_t ctx,
  legion_runtime_t runtime,
  legion_index_space_t grid,
  legion_point_3d_t block_size,
  legion_point_3d_t border,
  legion_index_partition_t* block_ip,
  legion_index_partition_t* halo_ip) {

  Legion::IndexSpaceT<3> grid3(Legion::CObjectWrapper::unwrap(grid));
  auto [bip, hip] =
    hyperion::block_and_halo_partitions(
      Legion::CObjectWrapper::unwrap(ctx)->context(),
      Legion::CObjectWrapper::unwrap(runtime),
      grid3,
      Legion::CObjectWrapper::unwrap(block_size),
      Legion::CObjectWrapper::unwrap(border));
  *block_ip = Legion::CObjectWrapper::wrap(bip);
  *halo_ip = Legion::CObjectWrapper::wrap(hip);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
