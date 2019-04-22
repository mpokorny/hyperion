#include<algorithm>

#include "utility.h"
#include "Column.h"
#include "TableReadTask.h"
#include "tree_index_space.h"

using namespace legms;
using namespace Legion;

FieldID
legms::add_field(
  casacore::DataType datatype,
  FieldAllocator fa,
  FieldID field_id) {

  FieldID result;

#define ALLOC_FLD(tp)                           \
  case tp:                                      \
    result = fa.allocate_field(                 \
      sizeof(DataType<tp>::ValueType),          \
      field_id,                                 \
      DataType<tp>::serdez_id);                 \
    break;

  switch (datatype) {
    FOREACH_DATATYPE(ALLOC_FLD);
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

  const ProjectedIndexPartitionTask::args* targs =
    static_cast<const ProjectedIndexPartitionTask::args*>(task->args);
  IndexSpace is = task->regions[0].region.get_index_space();
  switch (is.get_dim()) {
#if LEGMS_MAX_DIM >= 1
  case 1:
    switch (targs->prjdim) {
#if LEGMS_MAX_DIM >= 1
    case 1:
      ::pipt_impl<1, 1>(task, regions, ctx, runtime);
      break;
#endif
#if LEGMS_MAX_DIM >= 2
    case 2:
      ::pipt_impl<1, 2>(task, regions, ctx, runtime);
      break;
#endif
#if LEGMS_MAX_DIM >= 3
    case 3:
      ::pipt_impl<1, 3>(task, regions, ctx, runtime);
      break;
#endif
#if LEGMS_MAX_DIM >= 4
    case 4:
      ::pipt_impl<1, 4>(task, regions, ctx, runtime);
      break;
#endif
    default:
      assert(false);
      break;
    }
    break;
#endif
#if LEGMS_MAX_DIM >= 2
  case 2:
    switch (targs->prjdim) {
#if LEGMS_MAX_DIM >= 1
    case 1:
      ::pipt_impl<2, 1>(task, regions, ctx, runtime);
      break;
#endif
#if LEGMS_MAX_DIM >= 2
    case 2:
      ::pipt_impl<2, 2>(task, regions, ctx, runtime);
      break;
#endif
#if LEGMS_MAX_DIM >= 3
    case 3:
      ::pipt_impl<2, 3>(task, regions, ctx, runtime);
      break;
#endif
#if LEGMS_MAX_DIM >= 4
    case 4:
      ::pipt_impl<2, 4>(task, regions, ctx, runtime);
      break;
#endif
    default:
      assert(false);
      break;
    }
    break;
#endif
#if LEGMS_MAX_DIM >= 3
  case 3:
    switch (targs->prjdim) {
#if LEGMS_MAX_DIM >= 1
    case 1:
      ::pipt_impl<3, 1>(task, regions, ctx, runtime);
      break;
#endif
#if LEGMS_MAX_DIM >= 2
    case 2:
      ::pipt_impl<3, 2>(task, regions, ctx, runtime);
      break;
#endif
#if LEGMS_MAX_DIM >= 3
    case 3:
      ::pipt_impl<3, 3>(task, regions, ctx, runtime);
      break;
#endif
#if LEGMS_MAX_DIM >= 4
    case 4:
      ::pipt_impl<3, 4>(task, regions, ctx, runtime);
      break;
#endif
    default:
      assert(false);
      break;
    }
    break;
#endif
#if LEGMS_MAX_DIM >= 4
  case 4:
    switch (targs->prjdim) {
#if LEGMS_MAX_DIM >= 1
    case 1:
      ::pipt_impl<4, 1>(task, regions, ctx, runtime);
      break;
#endif
#if LEGMS_MAX_DIM >= 2
    case 2:
      ::pipt_impl<4, 2>(task, regions, ctx, runtime);
      break;
#endif
#if LEGMS_MAX_DIM >= 3
    case 3:
      ::pipt_impl<4, 3>(task, regions, ctx, runtime);
      break;
#endif
#if LEGMS_MAX_DIM >= 4
    case 4:
      ::pipt_impl<4, 4>(task, regions, ctx, runtime);
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
  TaskVariantRegistrar registrar(TASK_ID, TASK_NAME, false);
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  registrar.set_leaf();
  registrar.set_idempotent();
  registrar.set_replicable();
  runtime->register_task_variant<base_impl>(registrar);
}

TaskID ProjectedIndexPartitionTask::TASK_ID;

IndexPartition
legms::projected_index_partition(
  Context ctx,
  Runtime* runtime,
  IndexPartition ip,
  IndexSpace prj_is,
  const std::vector<int>& dmap) {

  if (prj_is == IndexSpace::NO_SPACE)
    return IndexPartition::NO_PART;

  switch (ip.get_dim()) {
#if LEGMS_MAX_DIM >= 1
  case 1:
    switch (prj_is.get_dim()) {
#if LEGMS_MAX_DIM >= 1
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
#if LEGMS_MAX_DIM >= 2
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
#if LEGMS_MAX_DIM >= 3
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
#if LEGMS_MAX_DIM >= 4
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
#if LEGMS_MAX_DIM >= 2
  case 2:
    switch (prj_is.get_dim()) {
#if LEGMS_MAX_DIM >= 1
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
#if LEGMS_MAX_DIM >= 2
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
#if LEGMS_MAX_DIM >= 3
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
#if LEGMS_MAX_DIM >= 4
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
#if LEGMS_MAX_DIM >= 3
  case 3:
    switch (prj_is.get_dim()) {
#if LEGMS_MAX_DIM >= 1
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
#if LEGMS_MAX_DIM >= 2
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
#if LEGMS_MAX_DIM >= 3
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
#if LEGMS_MAX_DIM >= 4
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
#if LEGMS_MAX_DIM >= 4
  case 4:
    switch (prj_is.get_dim()) {
#if LEGMS_MAX_DIM >= 1
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
#if LEGMS_MAX_DIM >= 2
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
#if LEGMS_MAX_DIM >= 3
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
#if LEGMS_MAX_DIM >= 4
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
  const std::vector<AxisPartition<int>>& parts) {

  Rect<IS_DIM> is_rect = runtime->get_index_space_domain(is).bounds;

  // partition color space
  Rect<PART_DIM> cs_rect;
  for (auto n = 0; n < PART_DIM; ++n) {
    const auto& part = parts[n];
    coord_t m =
      ((is_rect.hi[part.dim] - is_rect.lo[part.dim] /*+ 1*/ - part.offset)
       + part.stride /*- 1*/)
      / part.stride;
    cs_rect.lo[n] = 0;
    cs_rect.hi[n] = m - 1;
  }

  // transform matrix from partition color space to index space delta
  Transform<IS_DIM, PART_DIM> transform;
  for (auto m = 0; m < IS_DIM; ++m)
    for (auto n = 0; n < PART_DIM; ++n)
      transform[m][n] = 0;
  for (auto n = 0; n < PART_DIM; ++n) {
    const auto& part = parts[n];
    transform[part.dim][n] = part.stride;
  }

  // partition extent
  Rect<IS_DIM> extent = is_rect;
  for (auto n = 0; n < PART_DIM; ++n) {
    const auto& part = parts[n];
    extent.lo[part.dim] = part.offset + part.lo;
    extent.hi[part.dim] = part.offset + part.hi;
  }

  IndexSpaceT<PART_DIM> cs = runtime->create_index_space(ctx, cs_rect);
  IndexPartitionT<IS_DIM> result =
    runtime->create_partition_by_restriction(ctx, is, cs, transform, extent);
  runtime->destroy_index_space(ctx, cs);
  return result;
}

IndexPartition
legms::create_partition_on_axes(
  Context ctx,
  Runtime* runtime,
  IndexSpace is,
  const std::vector<AxisPartition<int>>& parts) {

  assert(has_unique_values(parts));
  assert(
    std::all_of(
      parts.begin(),
      parts.end(),
      [nd=static_cast<int>(is.get_dim())](auto& part) {
        // TODO: support negative strides
        return 0 <= part.dim && part.dim < nd && part.stride > 0;
      }));

  switch (is.get_dim()) {
#if LEGMS_MAX_DIM >= 1
  case 1:
    switch (parts.size()) {
#if LEGMS_MAX_DIM >= 1
    case 1:
      return
        ::create_partition_on_axes<1,1>(ctx, runtime, IndexSpaceT<1>(is), parts);
      break;
#endif
    default:
      assert(false);
      break;
    }
    break;
#endif
#if LEGMS_MAX_DIM >= 2
  case 2:
    switch (parts.size()) {
#if LEGMS_MAX_DIM >= 1
    case 1:
      return
        ::create_partition_on_axes<2,1>(ctx, runtime, IndexSpaceT<2>(is), parts);
      break;
#endif
#if LEGMS_MAX_DIM >= 2
    case 2:
      return
        ::create_partition_on_axes<2,2>(ctx, runtime, IndexSpaceT<2>(is), parts);
      break;
#endif
    default:
      assert(false);
      break;
    }
    break;
#endif
#if LEGMS_MAX_DIM >= 3
  case 3:
    switch (parts.size()) {
#if LEGMS_MAX_DIM >= 1
    case 1:
      return
        ::create_partition_on_axes<3,1>(ctx, runtime, IndexSpaceT<3>(is), parts);
      break;
#endif
#if LEGMS_MAX_DIM >= 2
    case 2:
      return
        ::create_partition_on_axes<3,2>(ctx, runtime, IndexSpaceT<3>(is), parts);
      break;
#endif
#if LEGMS_MAX_DIM >= 3
    case 3:
      return
        ::create_partition_on_axes<3,3>(ctx, runtime, IndexSpaceT<3>(is), parts);
      break;
#endif
    default:
      assert(false);
      break;
    }
    break;
#endif
#if LEGMS_MAX_DIM >= 4
  case 4:
    switch (parts.size()) {
#if LEGMS_MAX_DIM >= 1
    case 1:
      return
        ::create_partition_on_axes<4,1>(ctx, runtime, IndexSpaceT<4>(is), parts);
      break;
#endif
#if LEGMS_MAX_DIM >= 2
    case 2:
      return
        ::create_partition_on_axes<4,2>(ctx, runtime, IndexSpaceT<4>(is), parts);
      break;
#endif
#if LEGMS_MAX_DIM >= 3
    case 3:
      return
        ::create_partition_on_axes<4,3>(ctx, runtime, IndexSpaceT<4>(is), parts);
      break;
#endif
#if LEGMS_MAX_DIM >= 4
    case 4:
      return
        ::create_partition_on_axes<4,4>(ctx, runtime, IndexSpaceT<4>(is), parts);
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
  return IndexPartition::NO_PART; // keep compiler happy
}

void
legms::register_tasks(Runtime* runtime) {
  TableReadTask::register_task(runtime);
  TreeIndexSpace::register_tasks(runtime);
  ProjectedIndexPartitionTask::register_task(runtime);
  Table::register_tasks(runtime);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
