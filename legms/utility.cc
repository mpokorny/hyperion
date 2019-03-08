#include<algorithm>

#include "utility.h"
#include "Column.h"
#include "TableReadTask.h"
#include "tree_index_space.h"

using namespace legms;
using namespace Legion;

std::once_flag SerdezManager::initialized;

FieldID
legms::add_field(
  casacore::DataType datatype,
  FieldAllocator fa,
  FieldID field_id) {

  FieldID result;

#define ALLOC_FLD(tp)                           \
  tp:                                           \
    result = fa.allocate_field(                 \
      sizeof(DataType<tp>::ValueType),          \
      field_id,                                 \
      DataType<tp>::serdez_id);

  switch (datatype) {

  case ALLOC_FLD(casacore::DataType::TpBool)
    break;

  case ALLOC_FLD(casacore::DataType::TpChar)
    break;

  case ALLOC_FLD(casacore::DataType::TpUChar)
    break;

  case ALLOC_FLD(casacore::DataType::TpShort)
    break;

  case ALLOC_FLD(casacore::DataType::TpUShort)
    break;

  case ALLOC_FLD(casacore::DataType::TpInt)
    break;

  case ALLOC_FLD(casacore::DataType::TpUInt)
    break;

  // case ALLOC_FLD(casacore::DataType::TpInt64)
  //   break;

  case ALLOC_FLD(casacore::DataType::TpFloat)
    break;

  case ALLOC_FLD(casacore::DataType::TpDouble)
    break;

  case ALLOC_FLD(casacore::DataType::TpComplex)
    break;

  case ALLOC_FLD(casacore::DataType::TpDComplex)
    break;

  case ALLOC_FLD(casacore::DataType::TpString)
    break;

  case casacore::DataType::TpQuantity:
    assert(false); // TODO: implement quantity-valued columns
    break;

  case casacore::DataType::TpRecord:
    assert(false); // TODO: implement record-valued columns
    break;

  case casacore::DataType::TpTable:
    assert(false); // TODO: implement table-valued columns
    break;

  default:
    assert(false);
    break;
  }

#undef ALLOC_FLD

  return result;
}

ProjectedIndexPartitionTask::ProjectedIndexPartitionTask(
  IndexSpace launch_space,
  LogicalPartition lp,
  LogicalRegion lr,
  args* global_arg)
  : IndexTaskLauncher(
    TASK_ID,
    launch_space,
    TaskArgument(
      global_arg,
      sizeof(ProjectedIndexPartitionTask::args)
      + global_arg->prjdim * sizeof(global_arg->dmap[0])),
    ArgumentMap()){

  add_region_requirement(
    RegionRequirement(lp, 0, WRITE_DISCARD, EXCLUSIVE, lr));
  add_field(0, IMAGE_RANGES_FID);
}

void
ProjectedIndexPartitionTask::dispatch(Context ctx, Runtime* runtime) {
  runtime->execute_index_space(ctx, *this);
}

template <int IPDIM, int PRJDIM>
void
pipt_impl(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context,
  Runtime *runtime) {

  const ProjectedIndexPartitionTask::args* targs =
    static_cast<const ProjectedIndexPartitionTask::args*>(task->args);
  Rect<PRJDIM> bounds = targs->bounds;

  const FieldAccessor<
    WRITE_DISCARD,
    Rect<PRJDIM>,
    IPDIM,
    coord_t,
    Realm::AffineAccessor<Rect<PRJDIM>, IPDIM, coord_t>,
    false>
    image_ranges(regions[0], ProjectedIndexPartitionTask::IMAGE_RANGES_FID);

  DomainT<IPDIM> domain =
    runtime->get_index_space_domain(
      task->regions[0].region.get_index_space());
  for (PointInDomainIterator<IPDIM> pid(domain);
       pid();
       pid++) {

    Rect<PRJDIM> r;
    for (size_t i = 0; i < PRJDIM; ++i)
      if (0 <= targs->dmap[i]) {
        r.lo[i] = r.hi[i] = pid[targs->dmap[i]];
      } else {
        r.lo[i] = bounds.lo[i];
        r.hi[i] = bounds.hi[i];
      }
    image_ranges[*pid] = r;
  }
}

void
ProjectedIndexPartitionTask::base_impl(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *runtime) {

  const args* targs = static_cast<const args*>(task->args);
  IndexSpace is = task->regions[0].region.get_index_space();
  switch (is.get_dim()) {
#if MAX_DIM >= 1
  case 1:
    switch (targs->prjdim) {
#if MAX_DIM >= 1
    case 1:
      pipt_impl<1, 1>(task, regions, ctx, runtime);
      break;
#endif
#if MAX_DIM >= 2
    case 2:
      pipt_impl<1, 2>(task, regions, ctx, runtime);
      break;
#endif
#if MAX_DIM >= 3
    case 3:
      pipt_impl<1, 3>(task, regions, ctx, runtime);
      break;
#endif
#if MAX_DIM >= 4
    case 4:
      pipt_impl<1, 4>(task, regions, ctx, runtime);
      break;
#endif
    default:
      assert(false);
      break;
    }
    break;
#endif
#if MAX_DIM >= 2
  case 2:
    switch (targs->prjdim) {
#if MAX_DIM >= 1
    case 1:
      pipt_impl<2, 1>(task, regions, ctx, runtime);
      break;
#endif
#if MAX_DIM >= 2
    case 2:
      pipt_impl<2, 2>(task, regions, ctx, runtime);
      break;
#endif
#if MAX_DIM >= 3
    case 3:
      pipt_impl<2, 3>(task, regions, ctx, runtime);
      break;
#endif
#if MAX_DIM >= 4
    case 4:
      pipt_impl<2, 4>(task, regions, ctx, runtime);
      break;
#endif
    default:
      assert(false);
      break;
    }
    break;
#endif
#if MAX_DIM >= 3
  case 3:
    switch (targs->prjdim) {
#if MAX_DIM >= 1
    case 1:
      pipt_impl<3, 1>(task, regions, ctx, runtime);
      break;
#endif
#if MAX_DIM >= 2
    case 2:
      pipt_impl<3, 2>(task, regions, ctx, runtime);
      break;
#endif
#if MAX_DIM >= 3
    case 3:
      pipt_impl<3, 3>(task, regions, ctx, runtime);
      break;
#endif
#if MAX_DIM >= 4
    case 4:
      pipt_impl<3, 4>(task, regions, ctx, runtime);
      break;
#endif
    default:
      assert(false);
      break;
    }
    break;
#endif
#if MAX_DIM >= 4
  case 4:
    switch (targs->prjdim) {
#if MAX_DIM >= 1
    case 1:
      pipt_impl<4, 1>(task, regions, ctx, runtime);
      break;
#endif
#if MAX_DIM >= 2
    case 2:
      pipt_impl<4, 2>(task, regions, ctx, runtime);
      break;
#endif
#if MAX_DIM >= 3
    case 3:
      pipt_impl<4, 3>(task, regions, ctx, runtime);
      break;
#endif
#if MAX_DIM >= 4
    case 4:
      pipt_impl<4, 4>(task, regions, ctx, runtime);
      break;
#endif
    default:
      assert(false);
      break;
    }
    break;
#endif
  default:
    assert(false);
    break;
  }
}

void
ProjectedIndexPartitionTask::register_task(Runtime* runtime) {
  TASK_ID =
    runtime->generate_library_task_ids("legms::ProjectedIndexPartitionTask", 1);
  TaskVariantRegistrar registrar(TASK_ID, TASK_NAME);
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  registrar.set_leaf();
  registrar.set_idempotent();
  runtime->register_task_variant<base_impl>(registrar);
}

TaskID ProjectedIndexPartitionTask::TASK_ID;

IndexPartition
projected_index_partition(
  Context ctx,
  Runtime* runtime,
  IndexPartition ip,
  IndexSpace prj_is,
  const std::vector<int>& dmap) {

  if (prj_is == IndexSpace::NO_SPACE)
    return IndexPartition::NO_PART;

  switch (ip.get_dim()) {
#if MAX_DIM >= 1
  case 1:
    switch (prj_is.get_dim()) {
#if MAX_DIM >= 1
    case 1:
      return
        projected_index_partition(
          ctx,
          runtime,
          IndexPartitionT<1>(ip),
          IndexSpaceT<1>(prj_is),
          {dmap[0]});
      break;
#endif
#if MAX_DIM >= 2
    case 2:
      return
        projected_index_partition(
          ctx,
          runtime,
          IndexPartitionT<1>(ip),
          IndexSpaceT<2>(prj_is),
          {dmap[0], dmap[1]});
      break;
#endif
#if MAX_DIM >= 3
    case 3:
      return
        projected_index_partition(
          ctx,
          runtime,
          IndexPartitionT<1>(ip),
          IndexSpaceT<3>(prj_is),
          {dmap[0], dmap[1], dmap[2]});
      break;
#endif
#if MAX_DIM >= 4
    case 4:
      return
        projected_index_partition(
          ctx,
          runtime,
          IndexPartitionT<1>(ip),
          IndexSpaceT<4>(prj_is),
          {dmap[0], dmap[1], dmap[2], dmap[3]});
      break;
#endif
    default:
      assert(false);
      break;
    }
    break;
#endif
#if MAX_DIM >= 2
  case 2:
    switch (prj_is.get_dim()) {
#if MAX_DIM >= 1
    case 1:
      return
        projected_index_partition(
          ctx,
          runtime,
          IndexPartitionT<2>(ip),
          IndexSpaceT<1>(prj_is),
          {dmap[0]});
      break;
#endif
#if MAX_DIM >= 2
    case 2:
      return
        projected_index_partition(
          ctx,
          runtime,
          IndexPartitionT<2>(ip),
          IndexSpaceT<2>(prj_is),
          {dmap[0], dmap[1]});
      break;
#endif
#if MAX_DIM >= 3
    case 3:
      return
        projected_index_partition(
          ctx,
          runtime,
          IndexPartitionT<2>(ip),
          IndexSpaceT<3>(prj_is),
          {dmap[0], dmap[1], dmap[2]});
      break;
#endif
#if MAX_DIM >= 4
    case 4:
      return
        projected_index_partition(
          ctx,
          runtime,
          IndexPartitionT<2>(ip),
          IndexSpaceT<4>(prj_is),
          {dmap[0], dmap[1], dmap[2], dmap[3]});
      break;
#endif
    default:
      assert(false);
      break;
    }
    break;
#endif
#if MAX_DIM >= 3
  case 3:
    switch (prj_is.get_dim()) {
#if MAX_DIM >= 1
    case 1:
      return
        projected_index_partition(
          ctx,
          runtime,
          IndexPartitionT<3>(ip),
          IndexSpaceT<1>(prj_is),
          {dmap[0]});
      break;
#endif
#if MAX_DIM >= 2
    case 2:
      return
        projected_index_partition(
          ctx,
          runtime,
          IndexPartitionT<3>(ip),
          IndexSpaceT<2>(prj_is),
          {dmap[0], dmap[1]});
      break;
#endif
#if MAX_DIM >= 3
    case 3:
      return
        projected_index_partition(
          ctx,
          runtime,
          IndexPartitionT<3>(ip),
          IndexSpaceT<3>(prj_is),
          {dmap[0], dmap[1], dmap[2]});
      break;
#endif
#if MAX_DIM >= 4
    case 4:
      return
        projected_index_partition(
          ctx,
          runtime,
          IndexPartitionT<3>(ip),
          IndexSpaceT<4>(prj_is),
          {dmap[0], dmap[1], dmap[2], dmap[3]});
      break;
#endif
    default:
      assert(false);
      break;
    }
    break;
#endif
#if MAX_DIM >= 4
  case 4:
    switch (prj_is.get_dim()) {
#if MAX_DIM >= 1
    case 1:
      return
        projected_index_partition(
          ctx,
          runtime,
          IndexPartitionT<4>(ip),
          IndexSpaceT<1>(prj_is),
          {dmap[0]});
      break;
#endif
#if MAX_DIM >= 2
    case 2:
      return
        projected_index_partition(
          ctx,
          runtime,
          IndexPartitionT<4>(ip),
          IndexSpaceT<2>(prj_is),
          {dmap[0], dmap[1]});
      break;
#endif
#if MAX_DIM >= 3
    case 3:
      return
        projected_index_partition(
          ctx,
          runtime,
          IndexPartitionT<4>(ip),
          IndexSpaceT<3>(prj_is),
          {dmap[0], dmap[1], dmap[2]});
      break;
#endif
#if MAX_DIM >= 4
    case 4:
      return
        projected_index_partition(
          ctx,
          runtime,
          IndexPartitionT<4>(ip),
          IndexSpaceT<4>(prj_is),
          {dmap[0], dmap[1], dmap[2], dmap[3]});
      break;
#endif
    default:
      assert(false);
      break;
    }
    break;
#endif
  default:
    assert(false);
    break;
  }
}

template <int IS_DIM, int PART_DIM>
static IndexPartitionT<IS_DIM>
create_partition_on_axes(
  Context ctx,
  Runtime* runtime,
  IndexSpaceT<IS_DIM> is,
  const std::vector<int>& dims) {

  Rect<IS_DIM> is_rect = runtime->get_index_space_domain(is);

  // partition color space
  Rect<PART_DIM> cs_rect;
  for (auto n = 0; n < PART_DIM; ++n) {
    cs_rect.lo[n] = is_rect.lo[dims[n]];
    cs_rect.hi[n] = is_rect.hi[dims[n]];
  }

  // transform matrix from partition color space to index space
  Transform<IS_DIM, PART_DIM> transform;
  for (auto m = 0; m < IS_DIM; ++m)
    for (auto n = 0; n < PART_DIM; ++n)
      transform[m][n] = 0;
  for (auto n = 0; n < PART_DIM; ++n)
    transform[dims[n]][n] = 1;

  // partition extent
  Rect<IS_DIM> extent = is_rect;
  for (auto n = 0; n < PART_DIM; ++n) {
    extent.lo[dims[n]] = 0;
    extent.hi[dims[n]] = 0;
  }

  IndexSpaceT<PART_DIM> cs = runtime->create_index_space(ctx, cs_rect);
  IndexPartitionT<IS_DIM> result =
    runtime->create_partition_by_restriction(ctx, is, cs, transform, extent);
  runtime->destroy_index_space(ctx, cs);
  return result;
}

IndexPartition
create_partition_on_axes(
  Context ctx,
  Runtime* runtime,
  IndexSpace is,
  const std::vector<int>& dims) {

  assert(has_unique_values(dims));
  assert(
    std::all_of(
      dims.begin(),
      dims.end(),
      [nd=static_cast<int>(is.get_dim())](auto d) {
        return 0 <= d && d < nd;
      }));

  switch (is.get_dim()) {
#if MAX_DIM >= 1
  case 1:
    switch (dims.size()) {
#if MAX_DIM >= 1
    case 1:
      return
        create_partition_on_axes<1,1>(ctx, runtime, IndexSpaceT<1>(is), dims);
      break;
#endif
    default:
      assert(false);
      break;
    }
    break;
#endif
#if MAX_DIM >= 2
  case 2:
    switch (dims.size()) {
#if MAX_DIM >= 1
    case 1:
      return
        create_partition_on_axes<2,1>(ctx, runtime, IndexSpaceT<2>(is), dims);
      break;
#endif
#if MAX_DIM >= 2
    case 2:
      return
        create_partition_on_axes<2,2>(ctx, runtime, IndexSpaceT<2>(is), dims);
      break;
#endif
    default:
      assert(false);
      break;
    }
    break;
#endif
#if MAX_DIM >= 3
  case 3:
    switch (dims.size()) {
#if MAX_DIM >= 1
    case 1:
      return
        create_partition_on_axes<3,1>(ctx, runtime, IndexSpaceT<3>(is), dims);
      break;
#endif
#if MAX_DIM >= 2
    case 2:
      return
        create_partition_on_axes<3,2>(ctx, runtime, IndexSpaceT<3>(is), dims);
      break;
#endif
#if MAX_DIM >= 3
    case 3:
      return
        create_partition_on_axes<3,3>(ctx, runtime, IndexSpaceT<3>(is), dims);
      break;
#endif
    default:
      assert(false);
      break;
    }
    break;
#endif
#if MAX_DIM >= 4
  case 4:
    switch (dims.size()) {
#if MAX_DIM >= 1
    case 1:
      return
        create_partition_on_axes<4,1>(ctx, runtime, IndexSpaceT<4>(is), dims);
      break;
#endif
#if MAX_DIM >= 2
    case 2:
      return
        create_partition_on_axes<4,2>(ctx, runtime, IndexSpaceT<4>(is), dims);
      break;
#endif
#if MAX_DIM >= 3
    case 3:
      return
        create_partition_on_axes<4,3>(ctx, runtime, IndexSpaceT<4>(is), dims);
      break;
#endif
#if MAX_DIM >= 4
    case 4:
      return
        create_partition_on_axes<4,4>(ctx, runtime, IndexSpaceT<4>(is), dims);
      break;
#endif
    default:
      assert(false);
      break;
    }
    break;
#endif
  default:
    assert(false);
    break;
  }
  return IndexPartition::NO_PART;
}


void
register_tasks(Runtime* runtime) {
  TableReadTask::register_task(runtime);
  TreeIndexSpace::register_tasks(runtime);
  Column::register_tasks(runtime);
  ProjectedIndexPartitionTask::register_task(runtime);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
