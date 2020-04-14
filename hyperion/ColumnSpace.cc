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
#include <hyperion/ColumnSpace.h>
#include <hyperion/Column.h>
#include <hyperion/Table.h>

using namespace hyperion;

using namespace Legion;

ColumnSpace::ColumnSpace(
  const IndexSpace& column_is_,
  const LogicalRegion& metadata_lr_)
  : column_is(column_is_)
  , metadata_lr(metadata_lr_) {
}

ColumnSpace
ColumnSpace::clone(Context ctx, Runtime* rt) const {
  auto cis =
    ((column_is != IndexSpace::NO_SPACE)
     ? rt->union_index_spaces(ctx, {column_is})
     : IndexSpace::NO_SPACE);
  LogicalRegion mlr;
  if (metadata_lr != LogicalRegion::NO_REGION) {
    {
      IndexSpace is = rt->create_index_space(ctx, Rect<1>(0, 0));
      FieldSpace fs = rt->create_field_space(ctx);
      FieldAllocator fa = rt->create_field_allocator(ctx, fs);
      fa.allocate_field(sizeof(AXIS_VECTOR_TYPE), AXIS_VECTOR_FID);
      fa.allocate_field(sizeof(AXIS_SET_UID_TYPE), AXIS_SET_UID_FID);
      fa.allocate_field(sizeof(INDEX_FLAG_TYPE), INDEX_FLAG_FID);
      mlr = rt->create_logical_region(ctx, is, fs);
    }
    CopyLauncher copy;
    RegionRequirement src_req(metadata_lr, READ_ONLY, EXCLUSIVE, metadata_lr);
    RegionRequirement dst_req(mlr, WRITE_ONLY, EXCLUSIVE, mlr);
    copy.add_copy_requirements(src_req, dst_req);
    copy.add_copy_requirements(src_req, dst_req);
    copy.add_copy_requirements(src_req, dst_req);
    copy.add_src_field(0, AXIS_VECTOR_FID);
    copy.add_dst_field(0, AXIS_VECTOR_FID);
    copy.add_src_field(1, AXIS_SET_UID_FID);
    copy.add_dst_field(1, AXIS_SET_UID_FID);
    copy.add_src_field(2, INDEX_FLAG_FID);
    copy.add_dst_field(2, INDEX_FLAG_FID);
    rt->issue_copy_operation(ctx, copy);
  }
  return ColumnSpace(cis, mlr);
}

ColumnSpace
ColumnSpace::clone(
  Context ctx,
  Runtime* rt,
  const IndexSpace& column_is,
  const PhysicalRegion& metadata_pr) {

  return
    ColumnSpace(column_is, metadata_pr.get_logical_region()).clone(ctx, rt);
}

bool
ColumnSpace::is_valid() const {
  return metadata_lr != LogicalRegion::NO_REGION;
}

bool
ColumnSpace::is_empty() const {
  return column_is == IndexSpace::NO_SPACE;
}

std::vector<int>
ColumnSpace::axes(Context ctx, Runtime* rt) const {
  RegionRequirement req(metadata_lr, READ_ONLY, EXCLUSIVE, metadata_lr);
  req.add_field(AXIS_VECTOR_FID);
  auto pr = rt->map_region(ctx, req);
  auto result = from_axis_vector(axes(pr));
  rt->unmap_region(ctx, pr);
  return result;
}

ColumnSpace::AXIS_VECTOR_TYPE
ColumnSpace::axes(PhysicalRegion pr) {
  const AxisVectorAccessor<READ_ONLY> ax(pr, AXIS_VECTOR_FID);
  return ax[0];
}

std::string
ColumnSpace::axes_uid(Context ctx, Runtime* rt) const {
  RegionRequirement req(metadata_lr, READ_ONLY, EXCLUSIVE, metadata_lr);
  req.add_field(AXIS_SET_UID_FID);
  auto pr = rt->map_region(ctx, req);
  auto result = axes_uid(pr);
  rt->unmap_region(ctx, pr);
  return result;
}

ColumnSpace::AXIS_SET_UID_TYPE
ColumnSpace::axes_uid(PhysicalRegion pr) {
  const AxisSetUIDAccessor<READ_ONLY> au(pr, AXIS_SET_UID_FID);
  return au[0];
}

bool
ColumnSpace::is_index(Context ctx, Runtime* rt) const {
  RegionRequirement req(metadata_lr, READ_ONLY, EXCLUSIVE, metadata_lr);
  req.add_field(INDEX_FLAG_FID);
  auto pr = rt->map_region(ctx, req);
  auto result = is_index(pr);
  rt->unmap_region(ctx, pr);
  return result;

}

ColumnSpace::INDEX_FLAG_TYPE
ColumnSpace::is_index(PhysicalRegion pr) {
  const IndexFlagAccessor<READ_ONLY> ifl(pr, INDEX_FLAG_FID);
  return ifl[0];
}

bool
ColumnSpace::operator<(const ColumnSpace& rhs) const {
  return column_is < rhs.column_is
    || (column_is == rhs.column_is
        && metadata_lr < rhs.metadata_lr);
}

bool
ColumnSpace::operator==(const ColumnSpace& rhs) const {
  return column_is == rhs.column_is
    && metadata_lr == rhs.metadata_lr;
}

bool
ColumnSpace::operator!=(const ColumnSpace& rhs) const {
  return !operator==(rhs);
}

struct InitTaskArgs {
  ColumnSpace::AXIS_VECTOR_TYPE axes;
  ColumnSpace::AXIS_SET_UID_TYPE axis_set_uid;
  bool is_index;
};

TaskID ColumnSpace::init_task_id;

const char* ColumnSpace::init_task_name = "x::ColumnSpace::init_task";

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
  {
    const IndexFlagAccessor<WRITE_ONLY> ifl(regions[0], INDEX_FLAG_FID);
    ifl[0] = args->is_index;
  }
}

void
ColumnSpace::destroy(Context ctx, Runtime* rt, bool destroy_index_space) {
  if (metadata_lr != LogicalRegion::NO_REGION) {
    if (destroy_index_space && column_is != IndexSpace::NO_SPACE)
      rt->destroy_index_space(ctx, column_is);
    auto is = metadata_lr.get_index_space();
    auto fs = metadata_lr.get_field_space();
    rt->destroy_logical_region(ctx, metadata_lr);
    rt->destroy_index_space(ctx, is);
    rt->destroy_field_space(ctx, fs);
    metadata_lr = LogicalRegion::NO_REGION;
  }
}

ColumnSpace
ColumnSpace::create(
  Context ctx,
  Runtime* rt,
  const std::vector<int>& axes,
  const std::string& axis_set_uid,
  const IndexSpace& column_is,
  bool is_index) {

  assert(axes.size() <= MAX_DIM);

  LogicalRegion metadata_lr;
  {
    IndexSpace is = rt->create_index_space(ctx, Rect<1>(0, 0));
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    fa.allocate_field(sizeof(AXIS_VECTOR_TYPE), AXIS_VECTOR_FID);
    fa.allocate_field(sizeof(AXIS_SET_UID_TYPE), AXIS_SET_UID_FID);
    fa.allocate_field(sizeof(INDEX_FLAG_TYPE), INDEX_FLAG_FID);
    metadata_lr = rt->create_logical_region(ctx, is, fs);
  }
  {
    RegionRequirement req(metadata_lr, WRITE_ONLY, EXCLUSIVE, metadata_lr);
    req.add_field(AXIS_VECTOR_FID);
    req.add_field(AXIS_SET_UID_FID);
    req.add_field(INDEX_FLAG_FID);
    InitTaskArgs args;
    args.axis_set_uid = axis_set_uid;
    args.axes = to_axis_vector(axes);
    args.is_index = is_index;
    TaskLauncher init(init_task_id, TaskArgument(&args, sizeof(args)));
    init.add_region_requirement(req);
    init.enable_inlining = true;
    rt->execute_task(ctx, init);
  }
  return ColumnSpace(column_is, metadata_lr);
}

RegionRequirement
ColumnSpace::requirements(
  PrivilegeMode privilege,
  CoherenceProperty coherence) const {

  RegionRequirement req(metadata_lr, privilege, coherence, metadata_lr);
  req.add_field(AXIS_VECTOR_FID);
  req.add_field(AXIS_SET_UID_FID);
  req.add_field(INDEX_FLAG_FID);
  return req;
}

struct ReindexedTaskArgs {
  std::array<int, ColumnSpace::MAX_DIM> index_axes;
  unsigned element_rank;
  bool allow_rows;
  IndexSpace column_is;
};

TaskID ColumnSpace::reindexed_task_id;

const char* ColumnSpace::reindexed_task_name = "x::ColumnSpace::reindexed_task";

ColumnSpace::reindexed_result_t
ColumnSpace::reindexed_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  const ReindexedTaskArgs *args =
    static_cast<const ReindexedTaskArgs*>(task->args);
  std::vector<std::pair<int, LogicalRegion>> index_column_lrs;
  index_column_lrs.reserve(regions.size() - 1);
  size_t rg = 0;
  while (++rg < regions.size()) {
    index_column_lrs.emplace_back(
      args->index_axes[rg - 1],
      task->regions[rg].region);
    rt->unmap_region(ctx, regions[rg]);
  }
  return reindexed(
    ctx,
    rt,
    args->element_rank,
    index_column_lrs,
    args->allow_rows,
    args->column_is,
    regions[0]);
}

Future
ColumnSpace::reindexed(
  Context ctx,
  Runtime* rt,
  unsigned element_rank,
  const std::vector<std::pair<int, LogicalRegion>>& index_columns,
  bool allow_rows) const {

  assert(!is_empty());

  ReindexedTaskArgs args;
  for (size_t i = 0; i < index_columns.size(); ++i)
    args.index_axes[i] = std::get<0>(index_columns[i]);
  args.allow_rows = allow_rows;
  args.element_rank = element_rank;
  args.column_is = column_is;
  TaskLauncher task(reindexed_task_id, TaskArgument(&args, sizeof(args)));
  {
    RegionRequirement req(metadata_lr, READ_ONLY, EXCLUSIVE, metadata_lr);
    req.add_field(AXIS_VECTOR_FID);
    req.add_field(AXIS_SET_UID_FID);
    req.add_field(INDEX_FLAG_FID);
    task.add_region_requirement(req);
  }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
  for (auto& [d, lr] : index_columns) {
#pragma GCC diagnostic pop
    RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
    req.add_field(Column::COLUMN_INDEX_ROWS_FID, false);
    task.add_region_requirement(req);
  }
  return rt->execute_task(ctx, task);
}

template <unsigned OLDDIM, unsigned NEWDIM>
static std::tuple<ColumnSpace, LogicalRegion, LayoutConstraintID>
compute_reindexed(
  Context ctx,
  Runtime *rt,
  const std::vector<std::pair<int, LogicalRegion>>& index_column_lrs,
  bool allow_rows,
  const IndexSpace& column_is,
  const std::vector<int>& current_axes,
  const std::string& axes_set_uid,
  const IndexPartition& column_ip) {

  Rect<OLDDIM> col_rect = rt->get_index_space_domain(column_is);
  // we use the name "rows_is" for the index space at or above the "ROW" axis
  IndexSpace rows_is =
    rt->create_index_space(ctx, rt->get_index_partition_color_space(column_ip));
  // logical region over rows_is with a field for the rectangle in the new
  // column index space for each value in row_is
  auto row_map_fs = rt->create_field_space(ctx);
  LayoutConstraintID row_map_lc_id;
  {
    auto fa = rt->create_field_allocator(ctx, row_map_fs);
    fa.allocate_field(
      sizeof(Rect<NEWDIM>),
      ColumnSpace::REINDEXED_ROW_RECTS_FID);

    LayoutConstraintRegistrar lc(row_map_fs);
    add_row_major_order_constraint(lc, rows_is.get_dim())
      .add_constraint(MemoryConstraint(Memory::Kind::GLOBAL_MEM));
    row_map_lc_id = rt->register_layout(lc);
  }
  // row_map_lr is a mapping from current column row index to a rectangle in
  // the new (reindexed) column index space
  auto row_map_lr = rt->create_logical_region(ctx, rows_is, row_map_fs);

  // initialize row_map_lr values to empty rectangles
  Rect<NEWDIM> empty;
  empty.lo[0] = 0;
  empty.hi[0] = -1;
  assert(empty.empty());
  rt->fill_field(
    ctx,
    row_map_lr,
    row_map_lr,
    ColumnSpace::REINDEXED_ROW_RECTS_FID,
    empty);

  // compute new index space rectangle for each row in column (i.e, compute
  // values of row_map_lr)
  {
    std::vector<LogicalRegion> ixlrs;
    ixlrs.reserve(index_column_lrs.size());
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
    for (auto& [d, lr] : index_column_lrs)
#pragma GCC diagnostic pop
      ixlrs.push_back(lr);
    ColumnSpace::compute_row_mapping(
      ctx,
      rt,
      allow_rows,
      column_ip,
      ixlrs,
      row_map_lr);
  }

  // create the new index space via create_partition_by_image_range based on
  // row_map_lr; this index space should be exact (i.e, appropriately sparse
  // or dense), but we start with the bounding index space first
  Rect<NEWDIM> new_bounds;
  std::vector<int> new_axes(NEWDIM);
  {
    // start with axes in the original rows
    auto rowmax = (unsigned)rows_is.get_dim() - 1;
    assert(current_axes[rowmax] == 0);
    unsigned i = 0; // index in new_bounds
    unsigned j = 0; // index in current_axes
    while (i < rowmax) {
      new_bounds.lo[i] = col_rect.lo[j];
      new_bounds.hi[i] = col_rect.hi[j];
      new_axes[i] = current_axes[j];
      ++i; ++j;
    }
    // append new index axes
    for (auto& [d, lr] : index_column_lrs) {
      Rect<1> ix_domain = rt->get_index_space_domain(lr.get_index_space());
      new_bounds.lo[i] = ix_domain.lo[0];
      new_bounds.hi[i] = ix_domain.hi[0];
      new_axes[i] = d;
      ++i;
    }
    // append row axis, if allowed
    if (allow_rows) {
      new_bounds.lo[i] = col_rect.lo[j];
      new_bounds.hi[i] = col_rect.hi[j];
      new_axes[i] = 0;
      ++i;
    }
    ++j;
    // append remaining (element-level) axes
    while (i < NEWDIM) {
      assert(j < OLDDIM);
      new_bounds.lo[i] = col_rect.lo[j];
      new_bounds.hi[i] = col_rect.hi[j];
      new_axes[i] = current_axes[j];
      ++i; ++j;
    }
  }
  auto new_bounds_is = rt->create_index_space(ctx, new_bounds);

  // now reduce the bounding index space to the exact, possibly sparse index
  // space of the reindexed column

  // to do this, we need a logical partition of row_map_lr, which will
  // comprise a single index subspace
  IndexSpaceT<1> all_rows_cs = rt->create_index_space(ctx, Rect<1>(0, 0));
  auto all_rows_ip =
    rt->create_equal_partition(ctx, rows_is, all_rows_cs);
  auto all_rows_row_map_lp =
    rt->get_logical_partition(ctx, row_map_lr, all_rows_ip);
  // those rows in rows_is that are mapped to an empty rectangle correspond to
  // indexes in rows_is that are not present in the exact cross-product index
  // space, and the create_partition_by_image_range function will leave those
  // indexes out of the resulting partition, leaving the index space we're
  // looking for
  IndexPartitionT<NEWDIM> new_bounds_ip(
    rt->create_partition_by_image_range(
      ctx,
      new_bounds_is,
      all_rows_row_map_lp,
      row_map_lr,
      ColumnSpace::REINDEXED_ROW_RECTS_FID,
      all_rows_cs));

  IndexSpace new_is =
    rt->create_index_space(
      ctx,
      rt->get_index_space_domain(
        rt->get_index_subspace(new_bounds_ip, 0)));

  rt->destroy_index_partition(ctx, new_bounds_ip);
  rt->destroy_logical_partition(ctx, all_rows_row_map_lp);
  rt->destroy_index_partition(ctx, all_rows_ip);
  rt->destroy_index_space(ctx, all_rows_cs);
  rt->destroy_index_space(ctx, new_bounds_is);

  return
    std::make_tuple(
      ColumnSpace::create(ctx, rt, new_axes, axes_set_uid, new_is, false),
      row_map_lr,
      row_map_lc_id);
}

ColumnSpace::reindexed_result_t
ColumnSpace::reindexed(
  Context ctx,
  Runtime* rt,
  unsigned element_rank,
  const std::vector<std::pair<int, LogicalRegion>>& index_column_lrs,
  bool allow_rows,
  const IndexSpace& column_is,
  const PhysicalRegion& metadata_pr) {

  assert(column_is != IndexSpace::NO_SPACE);

  const AxisVectorAccessor<READ_ONLY> avs(metadata_pr, AXIS_VECTOR_FID);
  const AxisSetUIDAccessor<READ_ONLY> auid(metadata_pr, AXIS_SET_UID_FID);
  auto current_axes = from_axis_vector(avs[0]);

  //assert((size_t)column_is.get_dim() - element_rank == current_axes.size());
  assert(current_axes.size() >= element_rank + 1);
  assert(current_axes[current_axes.size() - element_rank - 1] == 0);

  std::vector<std::pair<int, coord_t>> parts;
  for (size_t i = 0; i < current_axes.size() - element_rank; ++i)
    parts.emplace_back(current_axes[i], 1);

  ColumnSpacePartition partition =
    ColumnSpacePartition::create(
      ctx,
      rt,
      column_is,
      auid[0],
      parts,
      metadata_pr);

  unsigned olddim = (unsigned)column_is.get_dim();
  unsigned newdim =
    current_axes.size() - 1
    + index_column_lrs.size()
    + (allow_rows ? 1 : 0);
  reindexed_result_t result;
  switch (olddim * LEGION_MAX_DIM + newdim) {
#define COMPUTE_REINDEXED(OLDDIM, NEWDIM)       \
    case (OLDDIM * LEGION_MAX_DIM + NEWDIM): {  \
      result =                                  \
        compute_reindexed<OLDDIM, NEWDIM>(      \
          ctx,                                  \
          rt,                                   \
          index_column_lrs,                     \
          allow_rows,                           \
          column_is,                            \
          current_axes,                         \
          auid[0],                              \
          partition.column_ip);                 \
      break;                                    \
    }
    HYPERION_FOREACH_MN(COMPUTE_REINDEXED);
#undef REINDEX_COLUMN
  default:
    assert(false);
    break;
  }
  partition.destroy(ctx, rt);
  return result;
}

TaskID ColumnSpace::compute_row_mapping_task_id;

const char* ColumnSpace::compute_row_mapping_task_name =
  "x::ColumnSpace::compute_row_mapping_task";

struct ComputeRowMappingTaskArgs {
  unsigned row_dim;
  bool allow_rows;
  IndexPartition row_partition;
};

static std::vector<DomainPoint>
intersection(
  const std::vector<DomainPoint>& first,
  const std::vector<DomainPoint>& second) {

  std::vector<DomainPoint> result;
  std::set_intersection(
    first.begin(),
    first.end(),
    second.begin(),
    second.end(),
    std::back_inserter(result));
  return result;
}

void
ColumnSpace::compute_row_mapping_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *rt) {

  const ComputeRowMappingTaskArgs* args =
    static_cast<const ComputeRowMappingTaskArgs*>(task->args);

  typedef const FieldAccessor<
    READ_ONLY,
    Column::COLUMN_INDEX_ROWS_TYPE,
    1,
    coord_t,
    AffineAccessor<Column::COLUMN_INDEX_ROWS_TYPE, 1, coord_t>,
    false> rows_acc_t;

  auto ixdim = regions.size() - 1;

  unsigned r1 = args->row_dim - 1; // rank above ROW index
  std::vector<DomainPoint> common_rows;
  {
    rows_acc_t rows(regions[0], Column::COLUMN_INDEX_ROWS_FID);
    common_rows = rows[task->index_point[0 + r1]];
  }
  for (size_t i = 1; i < ixdim; ++i) {
    rows_acc_t rows(regions[i], Column::COLUMN_INDEX_ROWS_FID);
    common_rows = intersection(common_rows, rows[task->index_point[i + r1]]);
  }

  if (common_rows.size() > 0) {
    assert((int)args->row_dim == common_rows[0].get_dim());
    std::vector<DomainPoint> here_rows;
    int r2 = (int)r1 - (args->allow_rows ? 1 : 0);
    std::copy_if(
      common_rows.begin(),
      common_rows.end(),
      std::back_inserter(here_rows),
      [task, &r2](auto& pt) {
        bool match = true;
        for (int i = 0; match && i < r2; ++i)
          match = pt[i] == task->index_point[i];
        return match;
      });
    if ((args->allow_rows || here_rows.size() == 1)) {
      auto rectdim =
        ixdim + (args->allow_rows ? 1 : 0) /* + args->row_dim */ - 1
        + args->row_partition.get_dim() /* - args->row_dim */;
      switch (args->row_dim * LEGION_MAX_DIM + rectdim) {
#define WRITE_RECTS(ROWDIM, RECTDIM)                                    \
        case (ROWDIM * LEGION_MAX_DIM + RECTDIM): {                     \
          const FieldAccessor<                                          \
            WRITE_DISCARD, \
            Rect<RECTDIM>, \
            ROWDIM> rects(regions.back(), REINDEXED_ROW_RECTS_FID); \
                                                                    \
          for (size_t i = 0; i < here_rows.size(); ++i) {             \
            Domain row_d =                                              \
              rt->get_index_space_domain(                               \
                rt->get_index_subspace(                                 \
                  args->row_partition,                                  \
                  here_rows[i]));                                       \
            Rect<RECTDIM> row_rect;                                     \
            size_t j = 0;                                               \
            for (; j < r1 + ixdim; ++j) {                               \
              row_rect.lo[j] = task->index_point[j];                    \
              row_rect.hi[j] = task->index_point[j];                    \
            }                                                           \
            if (args->allow_rows) {                                     \
              row_rect.lo[j] = i;                                       \
              row_rect.hi[j] = i;                                       \
              ++j;                                                      \
            }                                                           \
            size_t k = j;                                               \
            for (; j < RECTDIM; ++j) {                                  \
              row_rect.lo[j] = row_d.lo()[j - k + ROWDIM];              \
              row_rect.hi[j] = row_d.hi()[j - k + ROWDIM];              \
            }                                                           \
            rects[here_rows[i]] = row_rect;                             \
          }                                                             \
          break;                                                        \
        }
        HYPERION_FOREACH_MN(WRITE_RECTS);
#undef WRITE_RECTS
      default:
        assert(false);
        break;
      }
    }
  }
}

void
ColumnSpace::compute_row_mapping(
  Context ctx,
  Runtime* rt,
  bool allow_rows,
  IndexPartition row_partition,
  const std::vector<LogicalRegion>& index_column_lrs,
  const LogicalRegion& row_map_lr) {

  ComputeRowMappingTaskArgs args;
  args.allow_rows = allow_rows;
  args.row_partition = row_partition;

  Domain bounds;
  auto row_domain = rt->get_index_partition_color_space(row_partition);
  auto row_domain_lo = row_domain.lo();
  auto row_domain_hi = row_domain.hi();
  args.row_dim = row_domain.get_dim();
  unsigned r1 = args.row_dim - 1;
  switch (r1 + index_column_lrs.size()) {
#define INIT_BOUNDS(DIM)                                        \
    case DIM: {                                                 \
      Rect<DIM> rect;                                           \
      size_t i = 0;                                             \
      while (i < r1) {                                          \
        rect.lo[i] = row_domain_lo[i];                          \
        rect.hi[i] = row_domain_hi[i];                          \
        ++i;                                                    \
      }                                                         \
      while (i < DIM) {                                                 \
        IndexSpaceT<1> cis(index_column_lrs[i - r1].get_index_space());   \
        Rect<1> dom = rt->get_index_space_domain(cis).bounds;       \
        rect.lo[i] = dom.lo[0];                                     \
        rect.hi[i] = dom.hi[0];                                     \
        ++i;                                                        \
      }                                                             \
      bounds = rect;                                                \
      break;                                                        \
    }
    HYPERION_FOREACH_N(INIT_BOUNDS);
#undef INIT_BOUNDS
    default:
      assert(false);
      break;
  }
  IndexTaskLauncher task(
    compute_row_mapping_task_id,
    bounds,
    TaskArgument(&args, sizeof(args)),
    ArgumentMap());

  for (auto& lr : index_column_lrs) {
    RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
    req.add_field(Column::COLUMN_INDEX_ROWS_FID);
    task.add_region_requirement(req);
  }
  {
    RegionRequirement req(row_map_lr, WRITE_ONLY, SIMULTANEOUS, row_map_lr);
    req.add_field(REINDEXED_ROW_RECTS_FID);
    task.add_region_requirement(req);
  }
  rt->execute_index_space(ctx, task);
}

void
ColumnSpace::preregister_tasks() {
  {
    // init_task
    init_task_id = Runtime::generate_static_task_id();
    TaskVariantRegistrar registrar(init_task_id, init_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    registrar.set_leaf();
    Runtime::preregister_task_variant<init_task>(registrar, init_task_name);
  }
  {
    // reindexed_task
    reindexed_task_id = Runtime::generate_static_task_id();
    TaskVariantRegistrar registrar(reindexed_task_id, reindexed_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    Runtime::preregister_task_variant<
      reindexed_result_t,
      reindexed_task>(
        registrar,
        reindexed_task_name);
  }
  {
    // compute_row_mapping_task
    compute_row_mapping_task_id = Runtime::generate_static_task_id();
    TaskVariantRegistrar
      registrar(compute_row_mapping_task_id, compute_row_mapping_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    registrar.set_leaf();
    Runtime::preregister_task_variant<compute_row_mapping_task>(
        registrar,
        compute_row_mapping_task_name);
  }
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
