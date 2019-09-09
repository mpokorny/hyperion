#include "Keywords.h"

using namespace legms;
using namespace Legion;

Keywords::Keywords() {}

Keywords::Keywords(pair<LogicalRegion> regions)
  : type_tags_lr(regions.type_tags)
  , values_lr(regions.values) {
}

bool
Keywords::is_empty() const {
  return values_lr == LogicalRegion::NO_REGION;
}

std::vector<std::string>
Keywords::keys(Runtime* rt) const {

  auto fs = type_tags_lr.get_field_space();
  std::vector<FieldID> fids;
  rt->get_field_space_fields(fs, fids);
  std::vector<std::string> result(fids.size());
  for (auto& fid : fids) {
    const char* fname;
    rt->retrieve_name(fs, fid, fname);
    result[fid] = fname;
  }
  return result;
}

std::optional<FieldID>
Keywords::find_keyword(Runtime* rt, const std::string& name) const {

  auto fs = type_tags_lr.get_field_space();
  std::vector<FieldID> fids;
  rt->get_field_space_fields(fs, fids);
  auto f =
    std::find_if(
      fids.begin(),
      fids.end(),
      [&name, &fs, rt](auto fid) {
        const char* fname;
        rt->retrieve_name(fs, fid, fname);
        return name == fname;
      });
  return ((f != fids.end()) ? std::make_optional(*f - 1) : std::nullopt);
}

std::vector<legms::TypeTag>
Keywords::value_types(
  Context ctx,
  Runtime* rt,
  const std::vector<FieldID>& fids) const {

  // allow runtime to catch erroneous FieldIDs
  RegionRequirement req(type_tags_lr, READ_ONLY, EXCLUSIVE, type_tags_lr);
  for (auto& fid : fids)
    req.add_field(fid);
  auto pr = rt->map_region(ctx, req);
  std::vector<legms::TypeTag> result;
  result.reserve(fids.size());
  for (auto& fid : fids)
    result.push_back(value_type(pr, fid));
  rt->unmap_region(ctx, pr);
  return result;
}

Keywords
Keywords::create(
  Context ctx,
  Runtime* rt,
  const kw_desc_t& kws,
  const std::string& name_prefix) {

  LogicalRegion tts, vals;
  if (kws.size() > 0) {
    auto is = rt->create_index_space(ctx, Rect<1>(0, 0));
    auto tt_fs = rt->create_field_space(ctx);
    auto tt_fa = rt->create_field_allocator(ctx, tt_fs);
    auto val_fs = rt->create_field_space(ctx);
    auto val_fa = rt->create_field_allocator(ctx, val_fs);
    for (size_t i = 0; i < kws.size(); ++i) {
      auto& [nm, dt] = kws[i];
      tt_fa.allocate_field(sizeof(legms::TypeTag), i);
      rt->attach_name(tt_fs, i, nm.c_str());
      add_field(dt, val_fa, i);
    }
    tts = rt->create_logical_region(ctx, is, tt_fs);
    {
      std::string tts_name = "kws/type_tags";
      if (name_prefix.size() > 0)
        tts_name =
          ((name_prefix.back() != '/') ? (name_prefix + "/") : name_prefix)
          + tts_name;
      rt->attach_name(tts, tts_name.c_str());
    }
    vals = rt->create_logical_region(ctx, is, val_fs);
    {
      std::string vals_name = "kws/values";
      if (name_prefix.size() > 0)
        vals_name =
          ((name_prefix.back() != '/') ? (name_prefix + "/") : name_prefix)
          + vals_name;
      rt->attach_name(vals, vals_name.c_str());
    }
    RegionRequirement req(tts, WRITE_ONLY, EXCLUSIVE, tts);
    for (size_t i = 0; i < kws.size(); ++i)
      req.add_field(i);
    auto pr = rt->map_region(ctx, req);
    for (size_t i = 0; i < kws.size(); ++i) {
      const TypeTagAccessor<WRITE_ONLY> dt(pr, i);
      dt[0] = std::get<1>(kws[i]);
    }
    rt->unmap_region(ctx, pr);
  }
  return Keywords(Keywords::pair<LogicalRegion>{tts, vals});
}

void
Keywords::destroy(Context ctx, Runtime* rt) {
  if (type_tags_lr != LogicalRegion::NO_REGION) {
    assert(values_lr != LogicalRegion::NO_REGION);
    rt->destroy_field_space(ctx, type_tags_lr.get_field_space());
    rt->destroy_field_space(ctx, values_lr.get_field_space());
    // type_tags and values share one IndexSpace
    rt->destroy_index_space(ctx, type_tags_lr.get_index_space());
    rt->destroy_logical_region(ctx, type_tags_lr);
    rt->destroy_logical_region(ctx, values_lr);
    type_tags_lr = LogicalRegion::NO_REGION;
    values_lr = LogicalRegion::NO_REGION;
  }
}

legms::TypeTag
Keywords::value_type(const PhysicalRegion& tt, FieldID fid) {

  const TypeTagAccessor<READ_ONLY> dt(tt, fid);
  return dt[0];
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
