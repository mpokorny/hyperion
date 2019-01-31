#ifndef LEGMS_GRIDS_C_H_
#define LEGMS_GRIDS_C_H_

#include "legion/legion_c_util.h"

#ifdef __cplusplus
extern "C" {
#endif

void
block_and_halo_partitions_1d(
  legion_context_t ctx,
  legion_runtime_t runtime,
  legion_index_space_t grid,
  legion_point_1d_t block_size,
  legion_point_1d_t border,
  legion_index_partition_t* block_ip,
  legion_index_partition_t* halo_ip);

void
block_and_halo_partitions_2d(
  legion_context_t ctx,
  legion_runtime_t runtime,
  legion_index_space_t grid,
  legion_point_2d_t block_size,
  legion_point_2d_t border,
  legion_index_partition_t* block_ip,
  legion_index_partition_t* halo_ip);

void
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

#endif // LEGMS_GRIDS_C_H_

// Local Variables:
// mode: c
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
