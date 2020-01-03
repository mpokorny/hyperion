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
#include <hyperion/ColumnPartition.h>

using namespace hyperion;
using namespace Legion;

template <int IS_DIM, int PART_DIM>
static IndexPartitionT<IS_DIM>
create_partition_on_axes(
  Context ctx,
  Runtime* rt,
  IndexSpaceT<IS_DIM> is,
  const std::vector<AxisPartition>& parts) {

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
    extent.lo[part.dim] = part.offset + part.extent[0];
    extent.hi[part.dim] = part.offset + part.extent[1];
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

ColumnPartition::ColumnPartition(
  LogicalRegion axes_uid,
  LogicalRegion axes,
  IndexPartition index_partition_)
  : axes_uid_lr(axes_uid)
  , axes_lr(axes)
  , index_partition(index_partition_) {
}

ColumnPartition
ColumnPartition::create(
  Context ctx,
  Runtime* rt,
  const std::string& axes_uid,
  const std::vector<int>& axes,
  const IndexPartition& ip) {

  LogicalRegion axuid;
  {
    IndexSpace is = rt->create_index_space(ctx, Rect<1>(0, 0));
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    fa.allocate_field(sizeof(hyperion::string), AXES_UID_FID);
    axuid = rt->create_logical_region(ctx, is, fs);
    {
      RegionRequirement req(axuid, WRITE_ONLY, EXCLUSIVE, axuid);
      req.add_field(AXES_UID_FID);
      PhysicalRegion pr = rt->map_region(ctx, req);
      const AxesUidAccessor<WRITE_ONLY> au(pr, AXES_UID_FID);
      au[0] = axes_uid;
      rt->unmap_region(ctx, pr);
    }
  }
  LogicalRegion axs;
  {
    Rect<1> rect(0, axes.size() - 1);
    IndexSpace is = rt->create_index_space(ctx, rect);
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    fa.allocate_field(sizeof(int), AXES_FID);
    axs = rt->create_logical_region(ctx, is, fs);
    {
      RegionRequirement req(axs, WRITE_ONLY, EXCLUSIVE, axs);
      req.add_field(AXES_FID);
      PhysicalRegion pr = rt->map_region(ctx, req);
      const AxesAccessor<WRITE_ONLY> ax(pr, AXES_FID);
      for (PointInRectIterator<1> pir(rect); pir(); pir++)
        ax[*pir] = axes[pir[0]];
      rt->unmap_region(ctx, pr);
    }
  }
  return ColumnPartition(axuid, axs, ip);
}

ColumnPartition
ColumnPartition::create(
  Context ctx,
  Runtime* rt,
  const std::string& axes_uid,
  const std::vector<int>& axes,
  IndexSpace is,
  const std::vector<AxisPartition>& parts) {

  assert(has_unique_values(parts));
  assert(
    std::all_of(
      parts.begin(),
      parts.end(),
      [nd=static_cast<int>(is.get_dim())](auto& part) {
        // TODO: support negative strides
        return 0 <= part.dim && part.dim < nd && part.stride > 0;
      }));

  IndexPartition ip;
  switch (parts.size() * LEGION_MAX_DIM + is.get_dim()) {
#define CP(PDIM, IDIM)                          \
    case (PDIM * LEGION_MAX_DIM + IDIM): {      \
      ip =                                      \
        create_partition_on_axes<IDIM,PDIM>(    \
          ctx,                                  \
          rt,                                   \
          IndexSpaceT<IDIM>(is), parts);        \
      break;                                    \
    }
    HYPERION_FOREACH_MN(CP);
#undef CP
  default:
    assert(false);
    break;
  }
  return create(ctx, rt, axes_uid, axes, ip);
}

void
ColumnPartition::destroy(Context ctx, Runtime* rt, bool destroy_color_space) {
  if (axes_uid_lr != LogicalRegion::NO_REGION) {
    assert(axes_lr != LogicalRegion::NO_REGION);
    std::vector<LogicalRegion*> lrs{&axes_uid_lr, &axes_lr};
    for (auto lr : lrs)
      rt->destroy_field_space(ctx, lr->get_field_space());
    for (auto lr : lrs)
      rt->destroy_index_space(ctx, lr->get_index_space());
    for (auto lr : lrs) {
      rt->destroy_logical_region(ctx, *lr);
      *lr = LogicalRegion::NO_REGION;
    }
  }
  if (index_partition != IndexPartition::NO_PART) {
    if (destroy_color_space) {
      IndexSpace cs = rt->get_index_partition_color_space_name(index_partition);
      rt->destroy_index_space(ctx, cs);
    }
    rt->destroy_index_partition(ctx, index_partition);
    index_partition = IndexPartition::NO_PART;
  }
}

std::string
ColumnPartition::axes_uid(Context ctx, Runtime* rt) const  {
  RegionRequirement req(axes_uid_lr, READ_ONLY, EXCLUSIVE, axes_uid_lr);
  req.add_field(AXES_UID_FID);
  auto pr = rt->map_region(ctx, req);
  std::string result(axes_uid(pr));
  rt->unmap_region(ctx, pr);
  return result;
};

const char*
ColumnPartition::axes_uid(const PhysicalRegion& pr) {
  const AxesUidAccessor<READ_ONLY> axes_uid(pr, AXES_UID_FID);
  return axes_uid[0].val;
}

std::vector<int>
ColumnPartition::axes(Context ctx, Runtime* rt) const {
  RegionRequirement req(axes_lr, READ_ONLY, EXCLUSIVE, axes_lr);
  req.add_field(AXES_FID);
  auto pr = rt->map_region(ctx, req);
  IndexSpaceT<1> is(axes_lr.get_index_space());
  DomainT<1> dom = rt->get_index_space_domain(is);
  std::vector<int> result(Domain(dom).hi()[0] + 1);
  const AxesAccessor<READ_ONLY> ax(pr, AXES_FID);
  for (PointInDomainIterator<1> pid(dom); pid(); pid++)
    result[pid[0]] = ax[*pid];
  rt->unmap_region(ctx, pr);
  return result;
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
