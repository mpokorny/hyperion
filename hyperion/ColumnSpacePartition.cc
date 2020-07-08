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
#include <hyperion/ColumnSpacePartition.h>

using namespace hyperion;

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

int
ColumnSpacePartition::color_dim(Legion::Runtime* rt) const {
  return rt->get_index_partition_color_space(column_ip).get_dim();
}

TaskID ColumnSpacePartition::create_task_ap_id;

const char* ColumnSpacePartition::create_task_ap_name =
  "x::ColumnSpacePartition::create_task_ap";

struct CreateTaskAPArgs {
  std::array<AxisPartition, ColumnSpace::MAX_DIM> partition;
  size_t partition_dim;
  IndexSpace column_space_is;
};

ColumnSpacePartition
ColumnSpacePartition::create_task_ap(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *rt) {

  const CreateTaskAPArgs* args =
    static_cast<const CreateTaskAPArgs*>(task->args);
  assert(regions.size() == 1);
  std::vector<AxisPartition> partition;
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
  const std::vector<AxisPartition>& partition) {

  CreateTaskAPArgs args;
  args.column_space_is = column_space.column_is;
  args.partition_dim = partition.size();
  assert(partition.size() <= args.partition.size());
  std::copy(partition.begin(), partition.end(), args.partition.begin());
  TaskLauncher task(create_task_ap_id, TaskArgument(&args, sizeof(args)));
  {
    RegionRequirement req(
      column_space.metadata_lr,
      READ_ONLY,
      EXCLUSIVE,
      column_space.metadata_lr);
    req.add_field(ColumnSpace::AXIS_VECTOR_FID);
    req.add_field(ColumnSpace::AXIS_SET_UID_FID);
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
  const std::vector<AxisPartition>& parts) {

  // partition color space
  Rect<PART_DIM> cs_rect;
  for (auto n = 0; n < PART_DIM; ++n) {
    const auto& part = parts[n];
    coord_t m;
    if (part.stride > 0)
      m = ((part.limits[1] - part.limits[0] /*+1*/) + part.stride /*-1*/)
        / part.stride;
    else
      m = 1;
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
    if (part.dim >= 0)
      transform[part.dim][n] = part.stride;
  }

  // partition extent
  Rect<IS_DIM> extent = rt->get_index_space_domain(is).bounds;
  bool aliased = false;
  bool complete = true;
  for (auto n = 0; n < PART_DIM; ++n) {
    const auto& part = parts[n];
    if (part.dim >= 0) {
      auto min = extent.lo[part.dim];
      auto max = extent.hi[part.dim];
      extent.lo[part.dim] = part.offset + part.extent[0];
      complete = complete && extent.lo[part.dim] <= min;
      extent.hi[part.dim] = part.offset + part.extent[1];
      complete =
        complete
        && ((extent.hi[part.dim] + cs_rect.hi[part.dim] * part.stride)
            >= max);
      if (cs_rect.hi[part.dim] > 0) {
        auto next_lo = extent.lo[part.dim] + part.stride;
        aliased = aliased || next_lo <= extent.hi[part.dim];
        complete = complete && next_lo == extent.hi[part.dim] + 1;
      }
    } else {
      aliased = true;
    }
  }

  IndexSpaceT<PART_DIM> cs = rt->create_index_space(ctx, cs_rect);
  IndexPartitionT<IS_DIM> result =
    rt->create_partition_by_restriction(
      ctx,
      is,
      cs,
      transform,
      extent,
      (aliased
       ? (complete ? ALIASED_COMPLETE_KIND : ALIASED_INCOMPLETE_KIND)
       : (complete ? DISJOINT_COMPLETE_KIND : DISJOINT_INCOMPLETE_KIND)));
  return result;
}

ColumnSpacePartition
ColumnSpacePartition::create(
  Context ctx,
  Runtime *rt,
  const IndexSpace& column_space_is,
  const std::vector<AxisPartition>& partition,
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

  std::vector<int> ds = map(partition, [](const auto& p) { return p.dim; });
  if (!has_unique_values(ds))
    return ColumnSpacePartition();

  std::vector<int> axes = ColumnSpace::from_axis_vector(av[0]);
  auto dm = dimensions_map(ds, axes);
  std::vector<AxisPartition> iparts;
  iparts.reserve(dm.size());
  for (size_t i = 0; i < dm.size(); ++i) {
    auto& part = partition[i];
    iparts.push_back(
      AxisPartition{part.axes_uid, dm[i], part.stride, part.offset,
          part.extent, part.limits});
  }

  if (std::any_of(
        iparts.begin(),
        iparts.end(),
        [nd=static_cast<int>(column_space_is.get_dim())](auto& part) {
          // TODO: support negative strides
          return nd <= part.dim || part.stride <= 0;
        }))
    return ColumnSpacePartition();

  // if iparts has no elements, make up a partition with a single element that
  // spans the entire index space
  if (iparts.size() == 0)
    iparts.push_back(
      AxisPartition{uid[0], 0, 0, 0,
                    {0, std::numeric_limits<coord_t>::max()}, {0, 0}});

  IndexPartition column_ip;
  switch (iparts.size() * LEGION_MAX_DIM + column_space_is.get_dim()) {
#define CP(PDIM, IDIM)                        \
  case (PDIM * LEGION_MAX_DIM + IDIM): {      \
    column_ip =                               \
      create_partition_on_axes<IDIM,PDIM>(    \
        ctx,                                  \
        rt,                                   \
        IndexSpaceT<IDIM>(column_space_is),   \
        iparts);                              \
    break;                                    \
  }
  HYPERION_FOREACH_NN(CP);
#undef CP
  default:
    assert(false);
    break;
  }

  ColumnSpacePartition result(
    ColumnSpace(
      column_space_is,
      column_space_metadata_pr.get_logical_region()),
    column_ip,
    {});
  AxisPartition no_part;
  no_part.stride = 0;
  auto e =
    std::copy(partition.begin(), partition.end(), result.partition.begin());
  std::fill(e, result.partition.end(), no_part);
  return result;
}

TaskID ColumnSpacePartition::create_task_bs_id;

const char* ColumnSpacePartition::create_task_bs_name =
  "x::ColumnSpacePartition::create_task_bs";

struct CreateTaskBSArgs {
  hyperion::string axes_uid;
  std::array<std::pair<int, coord_t>, ColumnSpace::MAX_DIM> block_sizes;
  size_t partition_dim;
  IndexSpace column_space_is;
};

ColumnSpacePartition
ColumnSpacePartition::create_task_bs(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *rt) {

  const CreateTaskBSArgs* args =
    static_cast<const CreateTaskBSArgs*>(task->args);
  assert(regions.size() == 1);
  std::vector<std::pair<int, coord_t>> block_sizes;
  block_sizes.reserve(args->partition_dim);
  std::copy(
    args->block_sizes.begin(),
    args->block_sizes.begin() + args->partition_dim,
    std::back_inserter(block_sizes));
  return
    create(
      ctx,
      rt,
      args->column_space_is,
      args->axes_uid,
      block_sizes,
      regions[0]);
}

Legion::Future /* ColumnSpacePartition */
ColumnSpacePartition::create(
  Legion::Context ctx,
  Legion::Runtime *rt,
  const ColumnSpace& column_space,
  const std::string& block_axes_uid,
  const std::vector<std::pair<int, Legion::coord_t>>& block_sizes) {

  CreateTaskBSArgs args;
  args.column_space_is = column_space.column_is;
  args.partition_dim = block_sizes.size();
  assert(block_sizes.size() <= args.block_sizes.size());
  std::copy(block_sizes.begin(), block_sizes.end(), args.block_sizes.begin());
  args.axes_uid = block_axes_uid;
  TaskLauncher task(create_task_bs_id, TaskArgument(&args, sizeof(args)));
  {
    RegionRequirement req(
      column_space.metadata_lr,
      READ_ONLY,
      EXCLUSIVE,
      column_space.metadata_lr);
    req.add_field(ColumnSpace::AXIS_VECTOR_FID);
    req.add_field(ColumnSpace::AXIS_SET_UID_FID);
    task.add_region_requirement(req);
  }
  return rt->execute_task(ctx, task);
}

ColumnSpacePartition
ColumnSpacePartition::create(
  Context ctx,
  Runtime *rt,
  const IndexSpace& column_space_is,
  const std::string& block_axes_uid,
  const std::vector<std::pair<int, coord_t>>& block_sizes,
  const PhysicalRegion& column_space_metadata_pr) {

  const ColumnSpace::AxisVectorAccessor<READ_ONLY>
    av(column_space_metadata_pr, ColumnSpace::AXIS_VECTOR_FID);
  std::vector<int> axes = ColumnSpace::from_axis_vector(av[0]);
  std::vector<int> ds;
  ds.reserve(block_sizes.size());
  for (auto& d_sz : block_sizes)
    ds.push_back(std::get<0>(d_sz));
  auto dm = dimensions_map(ds, axes);
  assert(std::none_of(dm.begin(), dm.end(), [](auto& d) { return d < 0; }));
  auto cs_domain = rt->get_index_space_domain(column_space_is);
  auto cs_lo = cs_domain.lo();
  auto cs_hi = cs_domain.hi();

  std::vector<AxisPartition> partition;
  partition.reserve(block_sizes.size());
  for (size_t i = 0; i < block_sizes.size(); ++i) {
    auto& sz = std::get<1>(block_sizes[i]);
    partition.push_back(
        AxisPartition{
          block_axes_uid,
          axes[dm[i]],
          sz,
          0,
          {0, sz - 1},
          {cs_lo[dm[i]], cs_hi[dm[i]]}});
  }
  return
    create(ctx, rt, column_space_is, partition, column_space_metadata_pr);
}

Future /* ColumnSpacePartition */
ColumnSpacePartition::project_onto(
  Context ctx,
  Runtime *rt,
  const ColumnSpace& tgt_column_space) const {

  std::vector<decltype(partition)::value_type> ps;
  auto cd = color_dim(rt);
  ps.reserve(cd);
  std::copy(partition.begin(), partition.begin() + cd, std::back_inserter(ps));
  return create(ctx, rt, tgt_column_space, ps);
}

ColumnSpacePartition
ColumnSpacePartition::project_onto(
  Context ctx,
  Runtime *rt,
  const IndexSpace& tgt_column_is,
  const PhysicalRegion& tgt_column_md) const {

  std::vector<AxisPartition> ps;
  auto cd = color_dim(rt);
  ps.reserve(cd);
  std::copy(partition.begin(), partition.begin() + cd, std::back_inserter(ps));
  return
    ColumnSpacePartition::create(ctx, rt, tgt_column_is, ps, tgt_column_md);
}

void
ColumnSpacePartition::preregister_tasks() {
  {
    // create_task_ap
    create_task_ap_id = Runtime::generate_static_task_id();
    TaskVariantRegistrar registrar(create_task_ap_id, create_task_ap_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    Runtime::preregister_task_variant<ColumnSpacePartition, create_task_ap>(
      registrar,
      create_task_ap_name);
  }
  {
    // create_task_bs
    create_task_bs_id = Runtime::generate_static_task_id();
    TaskVariantRegistrar registrar(create_task_bs_id, create_task_bs_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    Runtime::preregister_task_variant<ColumnSpacePartition, create_task_bs>(
      registrar,
      create_task_bs_name);
  }
}

bool
ColumnSpacePartition::operator<(const ColumnSpacePartition& rhs) const {
  if (column_space < rhs.column_space)
    return true;
  if (column_space == rhs.column_space) {
    if (column_ip < rhs.column_ip)
      return true;
    if (column_ip == rhs.column_ip) {
      for (size_t i = 0; i < ColumnSpace::MAX_DIM; ++i)
        if (partition[i] < rhs.partition[i])
          return true;;
    }
  }
  return false;
}

bool
ColumnSpacePartition::operator==(const ColumnSpacePartition& rhs) const {
  return
    (column_space == rhs.column_space)
    && (column_ip == rhs.column_ip)
    && (partition == rhs.partition);
}

bool
ColumnSpacePartition::operator!=(const ColumnSpacePartition& rhs) const {
  return !operator==(rhs);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
