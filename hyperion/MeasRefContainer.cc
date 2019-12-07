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
#include <hyperion/MeasRefContainer.h>

using namespace hyperion;
using namespace Legion;

MeasRefContainer::MeasRefContainer() {}

MeasRefContainer::MeasRefContainer(Legion::LogicalRegion meas_refs)
  : lr(meas_refs) {
}

MeasRefContainer
MeasRefContainer::create(
  Context ctx,
  Runtime* rt,
  const std::unordered_map<std::string, MeasRef>& owned,
  const MeasRefContainer& borrowed) {

  MeasRefContainer result;
  size_t bn =
    (borrowed.lr != LogicalRegion::NO_REGION)
    ? rt->get_index_space_domain(borrowed.lr.get_index_space()).get_volume()
    : 0;
  size_t n = owned.size() + bn;
  if (n > 0) {
    std::optional<PhysicalRegion> pr;
    if (bn > 0) {
      RegionRequirement req(borrowed.lr, READ_ONLY, EXCLUSIVE, borrowed.lr);
      req.add_field(NAME_FID);
      req.add_field(MEAS_REF_FID);
      pr = rt->map_region(ctx, req);
    }
    result = create(ctx, rt, owned, pr);
    if (pr)
      rt->unmap_region(ctx, pr.value());
  }
  return result;
}

MeasRefContainer
MeasRefContainer::create(
  Context ctx,
  Runtime* rt,
  const std::unordered_map<std::string, MeasRef>& owned) {

  return create(ctx, rt, owned, MeasRefContainer());
}

MeasRefContainer
MeasRefContainer::create(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const std::unordered_map<std::string, MeasRef>& owned,
  const std::optional<Legion::PhysicalRegion>& borrowed_pr) {

  std::unordered_map<std::string, std::tuple<bool, MeasRef>> merged_mrs;
  if (borrowed_pr) {
    size_t bn =
      rt->get_index_space_domain(
        borrowed_pr.value().get_logical_region().get_index_space())
      .get_volume();

    const NameAccessor<READ_ONLY> nm(borrowed_pr.value(), NAME_FID);
    const MeasRefAccessor<READ_ONLY> mr(borrowed_pr.value(), MEAS_REF_FID);
    for (size_t i = 0; i < bn; ++i)
      merged_mrs[nm[i]] = std::make_tuple(false, mr[i]);
  }
  for (auto& [nm, mr] : owned)
    merged_mrs[nm] = std::make_tuple(true, mr);

  assert(merged_mrs.size() > 0);
  IndexSpace is =
    rt->create_index_space(ctx, Rect<1>(0, merged_mrs.size() - 1));
  FieldSpace fs = rt->create_field_space(ctx);
  {
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    fa.allocate_field(sizeof(bool), OWNED_FID);
    fa.allocate_field(sizeof(hyperion::string), NAME_FID);
    fa.allocate_field(sizeof(MeasRef), MEAS_REF_FID);
  }
  LogicalRegion lr = rt->create_logical_region(ctx, is, fs);
  {
    RegionRequirement req(lr, WRITE_ONLY, EXCLUSIVE, lr);
    req.add_field(OWNED_FID);
    req.add_field(NAME_FID);
    req.add_field(MEAS_REF_FID);
    auto pr = rt->map_region(ctx, req);
    const OwnedAccessor<WRITE_ONLY> os(pr, OWNED_FID);
    const NameAccessor<WRITE_ONLY> nms(pr, NAME_FID);
    const MeasRefAccessor<WRITE_ONLY> mrs(pr, MEAS_REF_FID);
    size_t i = 0;
    for (auto& [nm, omr] : merged_mrs) {
      auto& [owned, mr] = omr;
      os[i] = owned;
      nms[i] = nm;
      mrs[i] = mr;
      ++i;
    }
    rt->unmap_region(ctx, pr);
  }
  return MeasRefContainer(lr);

}

size_t
MeasRefContainer::size(Legion::Runtime* rt) const {
  if (lr == LogicalRegion::NO_REGION)
    return 0;
  else
    return
      rt->get_index_space_domain(lr.get_index_space()).get_volume();
}

std::vector<
  std::tuple<
    RegionRequirement,
    std::optional<RegionRequirement>>>
MeasRefContainer::component_requirements(
  Context ctx,
  Runtime* rt,
  legion_privilege_mode_t mode) const {

  std::vector<
    std::tuple<
      RegionRequirement,
      std::optional<RegionRequirement>>> result;
  if (lr != LogicalRegion::NO_REGION) {
    RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
    req.add_field(MEAS_REF_FID);
    auto pr = rt->map_region(ctx, req);
    const MeasRefAccessor<READ_ONLY> mrs(pr, MEAS_REF_FID);
    for (PointInDomainIterator<1> pid(
           rt->get_index_space_domain(lr.get_index_space()));
         pid();
         pid++)
      result.push_back(mrs[*pid].requirements(mode));
    rt->unmap_region(ctx, pr);
  }
  return result;
}

std::optional<RegionRequirement>
MeasRefContainer::requirements(
  Context ctx,
  Runtime* rt,
  legion_privilege_mode_t mode) const {

  std::optional<RegionRequirement> result;
  if (lr != LogicalRegion::NO_REGION) {
    RegionRequirement req(lr, mode, EXCLUSIVE, lr);
    req.add_field(MeasRefContainer::OWNED_FID);
    req.add_field(MeasRefContainer::NAME_FID);
    req.add_field(MeasRefContainer::MEAS_REF_FID);
    result = req;
  }
  return result;
}

std::tuple<MeasRef, bool>
MeasRefContainer::lookup(
  Context ctx,
  Runtime* rt,
  const std::string& name) {

  if (lr != LogicalRegion::NO_REGION) {
    RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
    req.add_field(OWNED_FID);
    req.add_field(NAME_FID);
    req.add_field(MEAS_REF_FID);
    auto pr = rt->map_region(ctx, req);
    auto result = lookup(rt, name, pr);
    rt->unmap_region(ctx, pr);
    return result;
  }
  return std::make_tuple(MeasRef(), false);
}

std::tuple<MeasRef, bool>
MeasRefContainer::lookup(
  Legion::Runtime* rt,
  const std::string& name,
  Legion::PhysicalRegion& pr) {

  const OwnedAccessor<READ_ONLY> o(pr, OWNED_FID);
  const NameAccessor<READ_ONLY> nm(pr, NAME_FID);
  const MeasRefAccessor<READ_ONLY> mr(pr, MEAS_REF_FID);
  for (PointInDomainIterator<1> pid(
         rt->get_index_space_domain(pr.get_logical_region().get_index_space()));
       pid();
       pid++)
    if (nm[*pid] == name)
      return std::make_tuple(mr[*pid], o[*pid]);
  return std::make_tuple(MeasRef(), false);
}

void
MeasRefContainer::destroy(Context ctx, Runtime* rt) {
  if (lr != LogicalRegion::NO_REGION) {
    RegionRequirement req(lr, READ_WRITE, EXCLUSIVE, lr);
    req.add_field(OWNED_FID);
    req.add_field(MEAS_REF_FID);
    auto pr = rt->map_region(ctx, req);
    const OwnedAccessor<READ_ONLY> o(pr, OWNED_FID);
    const MeasRefAccessor<READ_WRITE> mr(pr, MEAS_REF_FID);
    for (PointInDomainIterator<1> pid(
           rt->get_index_space_domain(lr.get_index_space()));
         pid();
         pid++)
      if (o[*pid])
        mr[*pid].destroy(ctx, rt);
    rt->unmap_region(ctx, pr);
    rt->destroy_field_space(ctx, lr.get_field_space());
    rt->destroy_index_space(ctx, lr.get_index_space());
    rt->destroy_logical_region(ctx, lr);
  }
}

std::unordered_map<std::string, const MeasRef*>
MeasRefContainer::get_mr_ptrs(
  Legion::Runtime* rt,
  Legion::PhysicalRegion pr) {

  std::unordered_map<std::string, const MeasRef*> result;
  const MeasRefAccessor<READ_ONLY> mr(pr, MEAS_REF_FID);
  const NameAccessor<READ_ONLY> nm(pr, NAME_FID);
  for (PointInDomainIterator<1> pid(
         rt->get_index_space_domain(pr.get_logical_region().get_index_space()));
       pid();
       pid++)
    result.emplace(nm[*pid], mr.ptr(*pid));
  return result;
}

std::tuple<MeasRefDict, std::optional<PhysicalRegion>>
MeasRefContainer::with_measure_references_dictionary_prologue(
  Context ctx,
  Runtime* rt) const {

  std::unordered_map<std::string, const MeasRef*> refs;
  std::optional<PhysicalRegion> pr;
  if (lr != LogicalRegion::NO_REGION) {
    RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
    req.add_field(MEAS_REF_FID);
    req.add_field(NAME_FID);
    pr = rt->map_region(ctx, req);
    refs = get_mr_ptrs(rt, pr.value());
  }
  return std::make_tuple(MeasRefDict(ctx, rt, refs), pr);
}

void
MeasRefContainer::with_measure_references_dictionary_epilogue(
  Context ctx,
  Runtime* rt,
  const std::optional<PhysicalRegion>& pr) const {

  if (pr)
    rt->unmap_region(ctx, pr.value());
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
