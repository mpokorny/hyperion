#include "utility.h"

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

  case ALLOC_FLD(casacore::DataType::TpArrayBool)
    break;

  case ALLOC_FLD(casacore::DataType::TpChar)
    break;

  case ALLOC_FLD(casacore::DataType::TpArrayChar)
    break;

  case ALLOC_FLD(casacore::DataType::TpUChar)
    break;

  case ALLOC_FLD(casacore::DataType::TpArrayUChar)
    break;

  case ALLOC_FLD(casacore::DataType::TpShort)
    break;

  case ALLOC_FLD(casacore::DataType::TpArrayShort)
    break;

  case ALLOC_FLD(casacore::DataType::TpUShort)
    break;

  case ALLOC_FLD(casacore::DataType::TpArrayUShort)
    break;

  case ALLOC_FLD(casacore::DataType::TpInt)
    break;

  case ALLOC_FLD(casacore::DataType::TpArrayInt)
    break;

  case ALLOC_FLD(casacore::DataType::TpUInt)
    break;

  case ALLOC_FLD(casacore::DataType::TpArrayUInt)
    break;

  // case ALLOC_FLD(casacore::DataType::TpInt64)
  //   break;

  // case ALLOC_FLD(casacore::DataType::TpArrayInt64)
  //   break;

  case ALLOC_FLD(casacore::DataType::TpFloat)
    break;

  case ALLOC_FLD(casacore::DataType::TpArrayFloat)
    break;

  case ALLOC_FLD(casacore::DataType::TpDouble)
    break;

  case ALLOC_FLD(casacore::DataType::TpArrayDouble)
    break;

  case ALLOC_FLD(casacore::DataType::TpComplex)
    break;

  case ALLOC_FLD(casacore::DataType::TpArrayComplex)
    break;

  case ALLOC_FLD(casacore::DataType::TpDComplex)
    break;

  case ALLOC_FLD(casacore::DataType::TpArrayDComplex)
    break;

  case ALLOC_FLD(casacore::DataType::TpString)
    break;

  case ALLOC_FLD(casacore::DataType::TpArrayString)
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
  Context ctx,
  Runtime *runtime) {

  const ProjectedIndexPartitionTask::args* targs =
    static_cast<const ProjectedIndexPartitionTask::args*>(task->args);
  Legion::Rect<PRJDIM> bounds = targs->bounds;

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
      ctx,
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
  case 1:
    switch (targs->prjdim) {
    case 1:
      pipt_impl<1, 1>(task, regions, ctx, runtime);
      break;
    case 2:
      pipt_impl<1, 2>(task, regions, ctx, runtime);
      break;
    case 3:
      pipt_impl<1, 3>(task, regions, ctx, runtime);
      break;
    default:
      assert(false);
      break;
    }
    break;
  case 2:
    switch (targs->prjdim) {
    case 1:
      pipt_impl<2, 1>(task, regions, ctx, runtime);
      break;
    case 2:
      pipt_impl<2, 2>(task, regions, ctx, runtime);
      break;
    case 3:
      pipt_impl<2, 3>(task, regions, ctx, runtime);
      break;
    default:
      assert(false);
      break;
    }
    break;
  case 3:
    switch (targs->prjdim) {
    case 1:
      pipt_impl<3, 1>(task, regions, ctx, runtime);
      break;
    case 2:
      pipt_impl<3, 2>(task, regions, ctx, runtime);
      break;
    case 3:
      pipt_impl<3, 3>(task, regions, ctx, runtime);
      break;
    default:
      assert(false);
      break;
    }
    break;
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
  runtime->register_task_variant<base_impl>(registrar);
}

TaskID ProjectedIndexPartitionTask::TASK_ID;

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
