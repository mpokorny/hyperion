#include <legms/MeasRefContainer.h>

using namespace legms;
using namespace Legion;

#ifdef LEGMS_USE_CASACORE

MeasRefContainer::MeasRefContainer() {}

MeasRefContainer::MeasRefContainer(Legion::LogicalRegion meas_refs)
  : meas_refs_lr(meas_refs) {
}

MeasRefContainer
MeasRefContainer::create(
  Context ctx,
  Runtime* rt,
  const std::vector<MeasRef>& owned,
  const MeasRefContainer& borrowed) {

  size_t bn;
  if (borrowed.meas_refs_lr == LogicalRegion::NO_REGION)
    bn = 0;
  else
    bn =
      rt->get_index_space_domain(borrowed.meas_refs_lr.get_index_space())
      .get_volume();
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
        breq(borrowed.meas_refs_lr, READ_ONLY, EXCLUSIVE, borrowed.meas_refs_lr);
      breq.add_field(MEAS_REF_FID);
      auto bpr = rt->map_region(ctx, breq);
      const MeasRefAccessor<READ_ONLY> bmr(bpr, MEAS_REF_FID);
      for (size_t j = 0; j < bn; ++j, ++i) {
        o[i] = false;
        mr[i] = bmr[j];
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

  if (meas_refs_lr != LogicalRegion::NO_REGION) {
    std::string pre = prefix;
    if (pre.back() != '/')
      pre.push_back('/');
    RegionRequirement
      own_req(meas_refs_lr, READ_ONLY, EXCLUSIVE, meas_refs_lr);
    own_req.add_field(OWNED_FID);
    auto own_pr = rt->map_region(ctx, own_req);
    RegionRequirement
      mrefs_req(meas_refs_lr, READ_ONLY, EXCLUSIVE, meas_refs_lr);
    mrefs_req.add_field(MEAS_REF_FID);
    auto mrefs_pr = rt->map_region(ctx, mrefs_req);
    const OwnedAccessor<READ_ONLY> owned(own_pr, OWNED_FID);
    const MeasRefAccessor<READ_ONLY> mrefs(mrefs_pr, MEAS_REF_FID);
    for (PointInDomainIterator<1> pid(
           rt->get_index_space_domain(meas_refs_lr.get_index_space()));
         pid();
         pid++) {
      if (owned[*pid]) {
        auto mr = mrefs[*pid];
        RegionRequirement
          r(mr.name_region, READ_WRITE, EXCLUSIVE, mr.name_region);
        r.add_field(MeasRef::NAME_FID);
        auto nm_pr = rt->map_region(ctx, r);
        const MeasRef::NameAccessor<READ_WRITE> nm(nm_pr, MeasRef::NAME_FID);
        nm[0] = pre + std::string(nm[0]);
        rt->unmap_region(ctx, nm_pr);
      }
    }
    rt->unmap_region(ctx, mrefs_pr);
    rt->unmap_region(ctx, own_pr);
  }
}

void
MeasRefContainer::destroy(Context ctx, Runtime* rt) {
  if (meas_refs_lr != LogicalRegion::NO_REGION) {
    RegionRequirement req(meas_refs_lr, READ_WRITE, EXCLUSIVE, meas_refs_lr);
    req.add_field(OWNED_FID);
    req.add_field(MEAS_REF_FID);
    auto pr = rt->map_region(ctx, req);
    const OwnedAccessor<READ_WRITE> o(pr, OWNED_FID);
    const MeasRefAccessor<READ_WRITE> mr(pr, MEAS_REF_FID);
    for (PointInDomainIterator<1> pid(
           rt->get_index_space_domain(meas_refs_lr.get_index_space()));
         pid();
         pid++)
      if (o[*pid])
        mr[*pid].destroy(ctx, rt);
    rt->unmap_region(ctx, pr);
    rt->destroy_field_space(ctx, meas_refs_lr.get_field_space());
    rt->destroy_index_space(ctx, meas_refs_lr.get_index_space());
    rt->destroy_logical_region(ctx, meas_refs_lr);
  }
}


std::tuple<MeasRefDict, std::optional<PhysicalRegion>>
MeasRefContainer::with_measure_references_dictionary_prologue(
  Context ctx,
  Runtime* rt) const {

  std::vector<const MeasRef*> refs;
  std::optional<PhysicalRegion> pr;
  if (meas_refs_lr != LogicalRegion::NO_REGION) {
    RegionRequirement req(meas_refs_lr, READ_ONLY, EXCLUSIVE, meas_refs_lr);
    req.add_field(MEAS_REF_FID);
    pr = rt->map_region(ctx, req);
    const MeasRefAccessor<READ_ONLY> mr(pr.value(), MEAS_REF_FID);
    for (PointInDomainIterator<1> pid(
           rt->get_index_space_domain(meas_refs_lr.get_index_space()));
         pid();
         pid++)
      refs.push_back(mr.ptr(*pid));
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

#endif // LEGMS_USE_CASACORE

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
