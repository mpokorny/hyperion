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
#include <hyperion/x/ColumnSpace.h>

using namespace hyperion::x;

using namespace Legion;

ColumnSpace::ColumnSpace(
  const Legion::IndexSpace& column_is_,
  const Legion::LogicalRegion& metadata_lr_)
  : column_is(column_is_)
  , metadata_lr(metadata_lr_) {
}

bool
ColumnSpace::is_valid() const {
  return column_is != IndexSpace::NO_SPACE
    && metadata_lr != LogicalRegion::NO_REGION;
}

bool
ColumnSpace::operator<(const ColumnSpace& rhs) const {
  return column_is < rhs.column_is
    || (column_is == rhs.column_is
        && metadata_lr < rhs.metadata_lr);
}

struct InitTaskArgs {
  ColumnSpace::AXIS_VECTOR_TYPE axes;
  ColumnSpace::AXIS_SET_UID_TYPE axis_set_uid;
};

TaskID ColumnSpace::init_task_id;

const char* ColumnSpace::init_task_name = "ColumnSpace::init_task";

void
ColumnSpace::preregister_tasks() {
  init_task_id = Runtime::generate_static_task_id();
  TaskVariantRegistrar registrar(init_task_id, init_task_name, false);
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  registrar.set_idempotent();
  registrar.set_leaf();
  Runtime::preregister_task_variant<init_task>(registrar, init_task_name);
}

void
ColumnSpace::init_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context,
  Runtime *) {

  const InitTaskArgs* args = static_cast<const InitTaskArgs*>(task->args);
  assert(regions.size() == 1);
  {
    const AxisVectorAccessor<WRITE_ONLY> av(regions[0], AXIS_VECTOR_FID);
    av[0] = args->axes;
  }
  {
    const AxisSetUIDAccessor<WRITE_ONLY> as(regions[0], AXIS_SET_UID_FID);
    as[0] = args->axis_set_uid;
  }
}

void
ColumnSpace::destroy(Context ctx, Runtime* rt, bool destroy_index_space) {
  if (metadata_lr != LogicalRegion::NO_REGION) {
    if (destroy_index_space && column_is != IndexSpace::NO_SPACE)
      rt->destroy_index_space(ctx, column_is);
    rt->destroy_index_space(ctx, metadata_lr.get_index_space());
    rt->destroy_field_space(ctx, metadata_lr.get_field_space());
    rt->destroy_logical_region(ctx, metadata_lr);
    metadata_lr = LogicalRegion::NO_REGION;
  }
}

ColumnSpace
ColumnSpace::create(
  Context ctx,
  Runtime* rt,
  const std::vector<int>& axes,
  const std::string& axis_set_uid,
  const IndexSpace& column_is) {

  assert(axes.size() <= MAX_DIM);

  LogicalRegion metadata_lr;
  {
    IndexSpace is = rt->create_index_space(ctx, Rect<1>(0, 0));
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    fa.allocate_field(sizeof(AXIS_VECTOR_TYPE), AXIS_VECTOR_FID);
    fa.allocate_field(sizeof(AXIS_SET_UID_TYPE), AXIS_SET_UID_FID);
    metadata_lr = rt->create_logical_region(ctx, is, fs);
  }
  {
    RegionRequirement req(metadata_lr, WRITE_ONLY, EXCLUSIVE, metadata_lr);
    req.add_field(AXIS_VECTOR_FID);
    req.add_field(AXIS_SET_UID_FID);
    InitTaskArgs args;
    args.axis_set_uid = axis_set_uid;
    args.axes = to_axis_vector(axes);
    TaskLauncher init(init_task_id, TaskArgument(&args, sizeof(args)));
    init.add_region_requirement(req);
    init.enable_inlining = true;
    rt->execute_task(ctx, init);
  }
  return ColumnSpace(column_is, metadata_lr);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
