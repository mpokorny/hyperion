#include "Grids_c.h"
#include "Grids.h"

#include "legion.h"
#include "legion/legion_c_util.h"

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
    legms::block_and_halo_partitions(
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
    legms::block_and_halo_partitions(
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
    legms::block_and_halo_partitions(
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
