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
  const std::vector<MeasRef>& owned,
  const MeasRefContainer& borrowed) {

  size_t bn;
  if (borrowed.lr == LogicalRegion::NO_REGION)
    bn = 0;
  else
    bn = rt->get_index_space_domain(borrowed.lr.get_index_space()).get_volume();
  size_t n = owned.size() + bn;
  if (n == 0)
    return MeasRefContainer();

  IndexSpace is = rt->create_index_space(ctx, Rect<1>(0, n - 1));
  FieldSpace fs = rt->create_field_space(ctx);
  {
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    fa.allocate_field(sizeof(bool), OWNED_FID);
    fa.allocate_field(sizeof(MeasRef), MEAS_REF_FID);
  }
  LogicalRegion lr = rt->create_logical_region(ctx, is, fs);
  {
    RegionRequirement req(lr, WRITE_ONLY, EXCLUSIVE, lr);
    req.add_field(OWNED_FID);
    req.add_field(MEAS_REF_FID);
    auto pr = rt->map_region(ctx, req);
    const OwnedAccessor<WRITE_ONLY> o(pr, OWNED_FID);
    const MeasRefAccessor<WRITE_ONLY> mr(pr, MEAS_REF_FID);
    size_t i = 0;
    while  (i < owned.size()) {
      o[i] = true;
      mr[i] = owned[i];
      ++i;
    }
    if (bn > 0) {
      RegionRequirement
        breq(borrowed.lr, READ_ONLY, EXCLUSIVE, borrowed.lr);
      breq.add_field(MEAS_REF_FID);
      auto bpr = rt->map_region(ctx, breq);
      const MeasRefAccessor<READ_ONLY> bmr(bpr, MEAS_REF_FID);
      for (size_t j = 0; j < bn; ++j) {
        o[i] = false;
        mr[i] = bmr[j];
        ++i;
      }
      rt->unmap_region(ctx, bpr);
    }
    rt->unmap_region(ctx, pr);
  }
  return MeasRefContainer(lr);

}

MeasRefContainer
MeasRefContainer::create(
  Context ctx,
  Runtime* rt,
  const std::vector<MeasRef>& owned) {

  return create(ctx, rt, owned, MeasRefContainer());
}

void
MeasRefContainer::add_prefix_to_owned(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const std::string& prefix) const {

  if (lr != LogicalRegion::NO_REGION) {
    std::string pre = prefix;
    if (pre.back() != '/')
      pre.push_back('/');
    RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
    req.add_field(OWNED_FID);
    req.add_field(MEAS_REF_FID);
    auto pr = rt->map_region(ctx, req);
    const OwnedAccessor<READ_ONLY> owned(pr, OWNED_FID);
    const MeasRefAccessor<READ_ONLY> mrefs(pr, MEAS_REF_FID);
    for (PointInDomainIterator<1> pid(
           rt->get_index_space_domain(lr.get_index_space()));
         pid();
         pid++) {
      if (owned[*pid]) {
        auto& mr = mrefs[*pid];
        RegionRequirement
          r(mr.name_lr, READ_WRITE, EXCLUSIVE, mr.name_lr);
        r.add_field(MeasRef::NAME_FID);
        auto nm_pr = rt->map_region(ctx, r);
        const MeasRef::NameAccessor<READ_WRITE> nm(nm_pr, MeasRef::NAME_FID);
        nm[0] = pre + std::string(nm[0]);
        rt->unmap_region(ctx, nm_pr);
      }
    }
    rt->unmap_region(ctx, pr);
  }
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
    RegionRequirement,
    std::optional<RegionRequirement>>>
MeasRefContainer::component_requirements(
  Context ctx,
  Runtime* rt,
  legion_privilege_mode_t mode) const {

  std::vector<
    std::tuple<
      RegionRequirement,
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
         pid++) {
      const MeasRef& mr = mrs[*pid];
      std::tuple<
        RegionRequirement,
        RegionRequirement,
        std::optional<RegionRequirement>> reqs;
      {
        RegionRequirement
          req(mr.name_lr, mode, EXCLUSIVE, mr.name_lr);
        req.add_field(MeasRef::NAME_FID);
        std::get<0>(reqs) = req;
      }
      {
        RegionRequirement
          req(mr.metadata_lr, mode, EXCLUSIVE, mr.metadata_lr);
        req.add_field(MeasRef::MEASURE_CLASS_FID);
        req.add_field(MeasRef::REF_TYPE_FID);
        req.add_field(MeasRef::NUM_VALUES_FID);
        std::get<1>(reqs) = req;
      }
      if (mr.values_lr != LogicalRegion::NO_REGION) {
        RegionRequirement
          req(mr.values_lr, mode, EXCLUSIVE, mr.values_lr);
        req.add_field(0);
        std::get<2>(reqs) = req;
      }
      result.push_back(reqs);
    }
    rt->unmap_region(ctx, pr);
  }
  return result;
}

void
MeasRefContainer::destroy(Context ctx, Runtime* rt) {
  if (lr != LogicalRegion::NO_REGION) {
    RegionRequirement req(lr, READ_WRITE, EXCLUSIVE, lr);
    req.add_field(OWNED_FID);
    req.add_field(MEAS_REF_FID);
    auto pr = rt->map_region(ctx, req);
    const OwnedAccessor<READ_WRITE> o(pr, OWNED_FID);
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

MeasRefDict
MeasRefContainer::make_dict(
  Context ctx,
  Runtime* rt,
  const std::vector<PhysicalRegion>::iterator& begin_pr,
  const std::vector<PhysicalRegion>::iterator& end_pr) {

  std::vector<std::tuple<std::string, MeasRef::DataRegions>> mr_prs;

  std::vector<PhysicalRegion>::iterator pr = begin_pr;
  const MeasRefContainer::MeasRefAccessor<READ_ONLY>
    mrs(*pr, MeasRefContainer::MEAS_REF_FID);
  for (PointInRectIterator<1> pir(
         rt->get_index_space_domain(
           pr->get_logical_region().get_index_space()));
       pir();
       pir++) {
    ++pr;
    assert(pr != end_pr);
    const MeasRef::NameAccessor<READ_ONLY> names(*pr, MeasRef::NAME_FID);
    std::string name = names[0];
    ++pr;
    std::optional<PhysicalRegion> values_pr;
    if (mrs[*pir].values_lr != LogicalRegion::NO_REGION)
      values_pr = *pr++;
    mr_prs.emplace_back(name, MeasRef::DataRegions{*pr, values_pr});
  }
  assert(++pr == end_pr);
  return MeasRefDict(ctx, rt, mr_prs);
}

std::vector<const MeasRef*>
MeasRefContainer::get_mr_ptrs(
  Legion::Runtime* rt,
  Legion::PhysicalRegion pr,
  bool owned_only) {

  std::vector<const MeasRef*> result;
  const OwnedAccessor<READ_ONLY> owned(pr, OWNED_FID);
  const MeasRefAccessor<READ_ONLY> mr(pr, MEAS_REF_FID);
  for (PointInDomainIterator<1> pid(
         rt->get_index_space_domain(pr.get_logical_region().get_index_space()));
       pid();
       pid++)
    if (!owned_only || owned[*pid])
      result.push_back(mr.ptr(*pid));
  return result;
}

std::tuple<MeasRefDict, std::optional<PhysicalRegion>>
MeasRefContainer::with_measure_references_dictionary_prologue(
  Context ctx,
  Runtime* rt,
  bool owned_only) const {

  std::vector<const MeasRef*> refs;
  std::optional<PhysicalRegion> pr;
  if (lr != LogicalRegion::NO_REGION) {
    RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
    req.add_field(OWNED_FID);
    req.add_field(MEAS_REF_FID);
    pr = rt->map_region(ctx, req);
    refs = get_mr_ptrs(rt, pr.value(), owned_only);
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

std::vector<MeasRef>
MeasRefContainer::clone_meas_refs(
  Context ctx,
  Runtime* rt,
  PhysicalRegion pr) {
  return
    with_measure_references_dictionary(
      ctx,
      rt,
      pr,
      false,
      [](Context c, Runtime* r, MeasRefDict* dict) {
        std::vector<MeasRef> result;
        for (auto& nm : dict->names()) {
          auto ref = dict->get(nm).value();
          if (false) assert(false);
#define ADD_REF(M)                                                    \
          else if (MeasRefDict::holds<M>(ref))                        \
            result.push_back(                                         \
              MeasRef::create(c, r, nm, *MeasRefDict::get<M>(ref)));
          HYPERION_FOREACH_MCLASS(ADD_REF)
#undef ADD_REF
          else
            assert(false);
        }
        return result;
      });
}


// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
