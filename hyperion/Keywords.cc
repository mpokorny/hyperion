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
#include <hyperion/Keywords.h>

using namespace hyperion;
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

size_t
Keywords::size(Runtime* rt) const {
  if (is_empty())
    return 0;
  std::vector<FieldID> fids;
  rt->get_field_space_fields(type_tags_lr.get_field_space(), fids);
  return fids.size();
}

std::vector<std::string>
Keywords::keys(Runtime* rt) const {
  std::vector<std::string> result;
  if (!is_empty()) {
    std::vector<FieldID> fids;
    auto fs = type_tags_lr.get_field_space();
    rt->get_field_space_fields(fs, fids);
    result.resize(fids.size());
    for (auto& fid : fids) {
      const char* fname;
      rt->retrieve_name(fs, fid, fname);
      result[fid] = fname;
    }
  }
  return result;
}

CXX_OPTIONAL_NAMESPACE::optional<FieldID>
Keywords::find_keyword(Runtime* rt, const std::string& name) const {

  if (is_empty())
    return CXX_OPTIONAL_NAMESPACE::nullopt;
  std::vector<FieldID> fids;
  auto fs = type_tags_lr.get_field_space();
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
  return
    (f != fids.end())
    ? CXX_OPTIONAL_NAMESPACE::optional<FieldID>(*f)
    : CXX_OPTIONAL_NAMESPACE::nullopt;
}

std::vector<CXX_OPTIONAL_NAMESPACE::optional<hyperion::TypeTag>>
Keywords::value_types(
  Context ctx,
  Runtime* rt,
  const std::vector<FieldID>& fids) const {

  std::vector<CXX_OPTIONAL_NAMESPACE::optional<hyperion::TypeTag>>
    result(fids.size());
  if (!is_empty()) {
    auto n = size(rt);
    RegionRequirement req(type_tags_lr, READ_ONLY, EXCLUSIVE, type_tags_lr);
    for (auto& fid : fids)
      if (/*0 <= fid && */fid < n)
        req.add_field(fid);
    auto pr = rt->map_region(ctx, req);
    for (size_t i = 0; i < fids.size(); ++i)
      if (/*0 <= fids[i] && */fids[i] < n)
        result[i] = value_type(pr, fids[i]);
    rt->unmap_region(ctx, pr);
  }
  return result;
}

Keywords
Keywords::clone(Legion::Context ctx, Legion::Runtime* rt) const {
  Keywords result;
  if (!is_empty()) {
    std::vector<FieldID> fids;
    auto fs = type_tags_lr.get_field_space();
    rt->get_field_space_fields(fs, fids);
    auto reqs = requirements(rt, fids, READ_ONLY).value();
    auto prs =
      reqs.map(
        [&ctx, rt](const RegionRequirement& r){
          return rt->map_region(ctx, r);
        });
    result = clone(ctx, rt, prs);
    prs.map(
      [&ctx, rt](const PhysicalRegion& p) {
        rt->unmap_region(ctx, p);
        return 0;
      });
  }
  return result;
}

Keywords
Keywords::clone(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const pair<Legion::PhysicalRegion>& prs) {

  std::vector<FieldID> fids;
  auto fs = prs.type_tags.get_logical_region().get_field_space();
  rt->get_field_space_fields(fs, fids);
  kw_desc_t kws;
  for (auto fid : fids) {
    const char* fname;
    rt->retrieve_name(fs, fid, fname);
    const TypeTagAccessor<READ_ONLY> type_tags(prs.type_tags, fid);
    kws.emplace_back(fname, type_tags[0]);
  }
  Keywords result = create(ctx, rt, kws);
  auto res_reqs = result.requirements(rt, fids, WRITE_ONLY).value();
  auto res_prs =
    res_reqs.map(
      [&ctx, rt](const RegionRequirement& r){
        return rt->map_region(ctx, r);
      });
  for (size_t i = 0; i < kws.size(); ++i) {
    switch (std::get<1>(kws[i])) {
#define CPY(DT)                                             \
      case DT:                                              \
        Keywords::write<WRITE_ONLY>(                        \
          res_prs,                                          \
          i,                                                \
          Keywords::read<DataType<DT>::ValueType>(prs, i)   \
          .value());                                        \
        break;
      HYPERION_FOREACH_DATATYPE(CPY);
#undef CPY
      default:
        assert(false);
    }
  }
  res_prs.map(
    [&ctx, rt](const PhysicalRegion& p) {
      rt->unmap_region(ctx, p);
      return 0;
    });
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
#if HAVE_CXX17
      auto& [nm, dt] = kws[i];
#else // !HAVE_CXX17
      auto& nm = std::get<0>(kws[i]);
      auto& dt = std::get<1>(kws[i]);
#endif // HAVE_CXX17
      tt_fa.allocate_field(sizeof(hyperion::TypeTag), i);
      rt->attach_name(tt_fs, i, nm.c_str());
      add_field(dt, val_fa, i);
    }
    tts = rt->create_logical_region(ctx, is, tt_fs);
    {
      std::string tts_name = add_name_prefix(name_prefix, "kws/type_tags");
      rt->attach_name(tts, tts_name.c_str());
    }
    vals = rt->create_logical_region(ctx, is, val_fs);
    {
      std::string vals_name = add_name_prefix(name_prefix, "kws/values");
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

Keywords::pair<Legion::RegionRequirement>
Keywords::requirements(
  Runtime *rt,
  const Keywords::pair<PhysicalRegion>& prs,
  PrivilegeMode mode,
  bool mapped) {

  auto ttlr = prs.type_tags.get_logical_region();
  auto vllr = prs.values.get_logical_region();
  Legion::RegionRequirement tt(ttlr, READ_ONLY, EXCLUSIVE, ttlr);
  Legion::RegionRequirement vl(vllr, mode, EXCLUSIVE, vllr);
  std::vector<FieldID> fids;
  rt->get_field_space_fields(ttlr.get_field_space(), fids);
  for (auto& fid : fids) {
    tt.add_field(fid, mapped);
    vl.add_field(fid, mapped);
  }
  return pair<RegionRequirement>{tt, vl};
}

void
Keywords::destroy(Context ctx, Runtime* rt) {
  if (type_tags_lr != LogicalRegion::NO_REGION) {
    assert(values_lr != LogicalRegion::NO_REGION);
    auto tt_fs = type_tags_lr.get_field_space();
    auto v_fs = values_lr.get_field_space();
    // type_tags and values share one IndexSpace
    auto is = type_tags_lr.get_index_space();
    rt->destroy_logical_region(ctx, type_tags_lr);
    rt->destroy_logical_region(ctx, values_lr);
    rt->destroy_field_space(ctx, tt_fs);
    rt->destroy_field_space(ctx, v_fs);
    rt->destroy_index_space(ctx, is);
    type_tags_lr = LogicalRegion::NO_REGION;
    values_lr = LogicalRegion::NO_REGION;
  }
}

hyperion::TypeTag
Keywords::value_type(const PhysicalRegion& tt, FieldID fid) {

  const TypeTagAccessor<READ_ONLY> dt(tt, fid);
  return dt[0];
}

std::unordered_map<
  std::string, std::
  tuple<hyperion::TypeTag, CXX_ANY_NAMESPACE::any>>
Keywords::to_map(Legion::Context ctx, Legion::Runtime *rt) const {
  std::unordered_map<
    std::string,
    std::tuple<hyperion::TypeTag, CXX_ANY_NAMESPACE::any>> result;
  if (!is_empty()) {
    std::vector<FieldID> fids;
    auto fs = type_tags_lr.get_field_space();
    rt->get_field_space_fields(fs, fids);
    auto reqs = requirements(rt, fids, READ_ONLY).value();
    auto prs =
      reqs.map(
        [&ctx, rt](const RegionRequirement& r){
          return rt->map_region(ctx, r);
        });
    result = to_map(rt, prs);
    prs.map(
      [&ctx, rt](const PhysicalRegion& p) {
        rt->unmap_region(ctx, p);
        return 0;
      });
  }
  return result;
}

std::unordered_map<
  std::string,
  std::tuple<hyperion::TypeTag, CXX_ANY_NAMESPACE::any>>
Keywords::to_map(Legion::Runtime *rt, const pair<Legion::PhysicalRegion>& prs) {

  std::unordered_map<
    std::string,
    std::tuple<hyperion::TypeTag, CXX_ANY_NAMESPACE::any>> result;
  std::vector<FieldID> fids;
  auto fs = prs.type_tags.get_logical_region().get_field_space();
  rt->get_field_space_fields(fs, fids);
  for (auto& fid : fids) {
    const char* fname;
    rt->retrieve_name(fs, fid, fname);
    const TypeTagAccessor<READ_ONLY> tts(prs.type_tags, fid);
    switch (tts[0]) {
#define VAL(DT)                                                   \
      case DT: {                                                  \
        const ValueAccessor<READ_ONLY, DataType<DT>::ValueType>   \
          vals(prs.values, fid);                                  \
        result[fname] = {DT, vals[0]};                            \
        break;                                                    \
      }
      HYPERION_FOREACH_DATATYPE(VAL)
      default:
        assert(false);
        break;
    }
  }
  return result;
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
