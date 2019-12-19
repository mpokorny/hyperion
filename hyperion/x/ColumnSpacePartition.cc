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
#include <hyperion/utility.h>
#include <hyperion/x/ColumnSpacePartition.h>

using namespace hyperion::x;

using namespace Legion;

void
ColumnSpacePartition::destroy(
  Context ctx,
  Runtime* rt,
  bool destroy_column_space,
  bool destroy_column_space_index_space) {

  if (column_ip != IndexPartition::NO_PART) {
    rt->destroy_index_space(
      ctx,
      rt->get_index_partition_color_space_name(column_ip));
    rt->destroy_index_partition(ctx, column_ip);
    column_ip = IndexPartition::NO_PART;
  }
  if (destroy_column_space)
    column_space.destroy(ctx, rt, destroy_column_space_index_space);
}

TaskID ColumnSpacePartition::create_task_id;

const char* ColumnSpacePartition::create_task_name =
  "x::ColumnSpacePartition::create_task";

struct CreateTaskArgs {
  std::array<hyperion::AxisPartition, ColumnSpace::MAX_DIM> partition;
  size_t partition_dim;
  IndexSpace column_space_is;
};

ColumnSpacePartition::create_result_t
ColumnSpacePartition::create_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *rt) {

  const CreateTaskArgs* args = static_cast<const CreateTaskArgs*>(task->args);
  assert(regions.size() == 1);
  std::vector<hyperion::AxisPartition> partition;
  partition.reserve(args->partition_dim);
  std::copy(
    args->partition.begin(),
    args->partition.begin() + args->partition_dim,
    std::back_inserter(partition));
  return create(ctx, rt, args->column_space_is, partition, regions[0]);
}

Future /* ColumnSpacePartition */
ColumnSpacePartition::create(
  Context ctx,
  Runtime* rt,
  const ColumnSpace& column_space,
  const std::vector<hyperion::AxisPartition>& partition) {

  CreateTaskArgs args;
  TaskLauncher task(create_task_id, TaskArgument(&args, sizeof(args)));
  {
    RegionRequirement req(
      column_space.metadata_lr,
      READ_ONLY,
      EXCLUSIVE,
      column_space.metadata_lr);
    req.add_field(ColumnSpace::AXIS_VECTOR_FID);
    req.add_field(ColumnSpace::AXIS_SET_UID_FID);
    args.column_space_is = column_space.column_is;
    args.partition_dim = partition.size();
    assert(partition.size() <= args.partition.size());
    std::copy(partition.begin(), partition.end(), args.partition.begin());
    task.add_region_requirement(req);
  }
  return rt->execute_task(ctx, task);
}

template <int IS_DIM, int PART_DIM>
static IndexPartitionT<IS_DIM>
create_partition_on_axes(
  Context ctx,
  Runtime* rt,
  IndexSpaceT<IS_DIM> is,
  const std::vector<hyperion::AxisPartition>& parts) {

  Rect<IS_DIM> is_rect = rt->get_index_space_domain(is).bounds;

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

  IndexSpaceT<PART_DIM> cs = rt->create_index_space(ctx, cs_rect);
  IndexPartitionT<IS_DIM> result =
    rt->create_partition_by_restriction(
      ctx,
      is,
      cs,
      transform,
      extent,
      DISJOINT_COMPLETE_KIND);
  return result;
}

ColumnSpacePartition::create_result_t
ColumnSpacePartition::create(
  Context ctx,
  Runtime *rt,
  const IndexSpace& column_space_is,
  const std::vector<hyperion::AxisPartition>& partition,
  const PhysicalRegion& column_space_metadata_pr) {

  const ColumnSpace::AxisVectorAccessor<READ_ONLY>
    av(column_space_metadata_pr, ColumnSpace::AXIS_VECTOR_FID);
  const ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
    uid(column_space_metadata_pr, ColumnSpace::AXIS_SET_UID_FID);

  if (std::any_of(
        partition.begin(),
        partition.end(),
        [auid=uid[0]](auto& p) { return p.axes_uid != auid; }))
    return ColumnSpacePartition();

  std::vector<int> ds =
    hyperion::map(partition, [](const auto& p) { return p.dim; });
  std::vector<int> axes = ColumnSpace::from_axis_vector(av[0]);
  auto dm = hyperion::dimensions_map(ds, axes);
  std::vector<hyperion::AxisPartition> iparts;
  iparts.reserve(dm.size());
  for (size_t i = 0; i < dm.size(); ++i) {
    auto& part = partition[i];
    iparts.push_back(
      hyperion::AxisPartition{part.axes_uid, dm[i], part.stride, part.offset,
          part.lo, part.hi});
  }

  if (!hyperion::has_unique_values(iparts) ||
      std::any_of(
        iparts.begin(),
        iparts.end(),
        [nd=static_cast<int>(column_space_is.get_dim())](auto& part) {
          // TODO: support negative strides
          return part.dim < 0 || nd <= part.dim || part.stride <= 0;
        }))
    return ColumnSpacePartition();

  IndexPartition column_ip;
  switch (iparts.size() * LEGION_MAX_DIM + column_space_is.get_dim()) {
#define CP(PDIM, IDIM)                        \
  case (PDIM * LEGION_MAX_DIM + IDIM): {      \
    column_ip =                               \
      create_partition_on_axes<IDIM,PDIM>(    \
        ctx,                                  \
        rt,                                   \
        IndexSpaceT<IDIM>(column_space_is), iparts); \
    break;                                    \
  }
  HYPERION_FOREACH_MN(CP);
#undef CP
  default:
    assert(false);
    break;
  }

  return
    ColumnSpacePartition(
      ColumnSpace(
        column_space_is,
        column_space_metadata_pr.get_logical_region()),
      column_ip);
}

TaskID ColumnSpacePartition::project_onto_task_id;

const char* ColumnSpacePartition::project_onto_task_name =
  "x::ColumnSpacePartition::project_onto_task";

struct ProjectOntoTaskArgs {
  IndexPartition csp_column_ip;
  IndexSpace tgt_cs_column_is;
};

ColumnSpacePartition::project_onto_result_t
ColumnSpacePartition::project_onto_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *rt) {

  const ProjectOntoTaskArgs* args =
    static_cast<const ProjectOntoTaskArgs*>(task->args);
  assert(regions.size() == 2);
  return
    project_onto(
      ctx,
      rt,
      args->csp_column_ip,
      args->tgt_cs_column_is,
      regions[0],
      regions[1]);
}

Future /* ColumnSpacePartition */
ColumnSpacePartition::project_onto(
  Context ctx,
  Runtime *rt,
  const ColumnSpace& tgt_column_space) const {

  ProjectOntoTaskArgs args;
  args.csp_column_ip = column_ip;
  args.tgt_cs_column_is = column_space.column_is;
  TaskLauncher task(project_onto_task_id, TaskArgument(&args, sizeof(args)));
  for (auto& cs : {&column_space, &tgt_column_space}) {
    RegionRequirement
      req(cs->metadata_lr, READ_ONLY, EXCLUSIVE, cs->metadata_lr);
    req.add_field(ColumnSpace::AXIS_SET_UID_FID);
    req.add_field(ColumnSpace::AXIS_VECTOR_FID);
    task.add_region_requirement(req);
  }
  return rt->execute_task(ctx, task);
}

ColumnSpacePartition::project_onto_result_t
project_onto(
  Context ctx,
  Runtime* rt,
  const IndexPartition& csp_column_ip,
  const IndexSpace& tgt_cs_column_is,
  const PhysicalRegion& csp_cs_metadata_pr,
  const PhysicalRegion& tgt_cs_metadata_pr) {

  {
    const ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
      uid(csp_cs_metadata_pr, ColumnSpace::AXIS_SET_UID_FID);
    const ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
      tgt_uid(tgt_cs_metadata_pr, ColumnSpace::AXIS_SET_UID_FID);
    if (uid[0] != tgt_uid[0] || tgt_cs_column_is == IndexSpace::NO_SPACE)
      return ColumnSpacePartition();
  }
  std::vector<int> dmap;
  {
    const ColumnSpace::AxisVectorAccessor<READ_ONLY>
      av(csp_cs_metadata_pr, ColumnSpace::AXIS_VECTOR_FID);
    const ColumnSpace::AxisVectorAccessor<READ_ONLY>
      tgt_av(tgt_cs_metadata_pr, ColumnSpace::AXIS_VECTOR_FID);
    auto axes = ColumnSpace::from_axis_vector(av[0]);
    auto tgt_axes = ColumnSpace::from_axis_vector(tgt_av[0]);
    dmap = hyperion::dimensions_map(tgt_axes, axes);
  }

  IndexPartition tgt_column_ip =
    hyperion::projected_index_partition(
      ctx,
      rt,
      csp_column_ip,
      tgt_cs_column_is,
      dmap);

  return
    ColumnSpacePartition(
      ColumnSpace(tgt_cs_column_is, tgt_cs_metadata_pr.get_logical_region()),
      tgt_column_ip);
}

void
ColumnSpacePartition::preregister_tasks() {
  {
    // create_task
    create_task_id = Runtime::generate_static_task_id();
    TaskVariantRegistrar registrar(create_task_id, create_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    Runtime::preregister_task_variant<create_result_t, create_task>(
      registrar,
      create_task_name);
  }
  {
    // project_onto_task
    project_onto_task_id = Runtime::generate_static_task_id();
    TaskVariantRegistrar registrar(project_onto_task_id, project_onto_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    Runtime::preregister_task_variant<project_onto_result_t, project_onto_task>(
      registrar,
      project_onto_task_name);
  }
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
