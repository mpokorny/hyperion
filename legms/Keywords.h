#ifndef LEGMS_WITH_KEYWORDS_H_
#define LEGMS_WITH_KEYWORDS_H_

#pragma GCC visibility push(default)
#include <algorithm>
#include <optional>
#include <string>
#include <tuple>
#include <vector>
#pragma GCC visibility pop

#include "legms.h"
#include "utility.h"

namespace legms {

struct LEGMS_API Keywords {

  Legion::LogicalRegion type_tags;

  Legion::LogicalRegion values;

  template <typename T>
  struct pair {
    T type_tags;
    T values;

    template <typename F>
    pair<std::invoke_result_t<F,T>>
    map(F f) {
      return pair{f(type_tags), f(values)};
    }
  };

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using TypeTagAccessor =
    FieldAccessor<
    MODE,
    TypeTag,
    1,
    coord_t,
    Legion::AffineAccessor<TypeTag, 1, coord_t>,
    CHECK_BOUNDS>;

  template <legion_privilege_mode_t MODE, typename FT, bool CHECK_BOUNDS=false>
  using ValueAccessor =
    FieldAccessor<
    MODE,
    FT,
    1,
    coord_t,
    Legion::AffineAccessor<FT, 1, coord_t>,
    CHECK_BOUNDS>;

  Keywords(pair<Legion::LogicalRegion> regions)
    : type_tags(regions.type_tags)
    , values(regions.values) {}

  std::vector<std::string>
  keys() const {

    auto fs = type_tags.get_field_space();
    std::unordered_map<Legion::FieldID> fids;
    rt->get_field_space_fields(fs, fids);
    std::vector<std::string> result(fids.size());
    std::for_each(
      fids.begin(),
      fids.end(),
      [&fs, rt](auto fid) {
        const char* fname;
        rt->retrieve_name(fs, fid, fname);
        result[fid] = fname;
      });
    return result;
  }

  std::optional<Legion::FieldID>
  find_keyword(Legion::Runtime* rt, const std::string& name) const {

    auto fs = type_tags.get_field_space();
    std::vector<Legion::FieldID> fids;
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
    return ((f != fids.end()) ? (*f - 1) : std::nullopt);
  }

  TypeTag
  value_type(
    Legion::Context ctx,
    Legion::Runtime* rt,
    Legion::FieldID fid) const {

    RegionRequirement req(type_tags, READ_ONLY, EXCLUSIVE, type_tags);
    auto pr = rt->map_region(ctx, req);
    auto result = value_type(pr, fid);
    rt->unmap_region(ctx, pr);
    return result;
  }

  template <legion_privilege_mode_t MODE, template <typename> C>
  pair<Legion::RegionRequirement>
  requirements(const C<FieldID>& fids) const {
    RegionRequirement tt(type_tags, READ_ONLY, EXCLUSIVE, type_tags);
    RegionRequirement v(values, MODE, EXCLUSIVE, values);
    std::for_each(
      fids.begin(),
      fids.end(),
      [&tt, &v](auto fid) {
        tt.add_field(fid);
        v.add_field(fid);
      });
    return pair{tt, v};
  }

  template <typename T>
  bool
  write(
    Legion::FieldID fid,
    const T& val,
    Legion::Context ctx,
    Legion::Runtime* rt) const {

    auto reqs = requirements<WRITE_ONLY>(std::vector<FieldID>{fid});
    auto prs = reqs.map([&ctx, rt](auto& r){ return rt->map_region(ctx, r); });
    bool result = write(prs, fid, val);
    prs.map([&ctx, rt](auto& p) { rt->unmap_region(ctx, p); });
    return result;
  }

  template <typename T>
  std::optional<T>
  read(Legion::FieldID fid, Legion::Context ctx, Legion::Runtime* rt) const {

    auto reqs = requirements<READ_ONLY>(std::vector<FieldID>{fid});
    auto prs = reqs.map([&ctx, rt](auto& r){ return rt->map_region(ctx, r); });
    auto result = read(prs, fid);
    prs.map([&ctx, rt](auto& p) { rt->unmap_region(ctx, p); });
    return result;
  }

  static Keywords
  make(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<std::tuple<std::string, TypeTag>>& kws) {

    LogicalRegion tts, vals;
    if (kws.size() > 0) {
      auto is = rt->create_index_space(ctx, Legion::Rect<1>(0, 0));
      auto tt_fs = rt->create_field_space(ctx);
      auto tt_fa = rt->create_field_allocator(ctx, tt_fs);
      auto val_fs = rt->create_field_space(ctx);
      auto val_fa = rt->create_field_allocator(ctx, val_fs);
      for (size_t i = 0; i < kws.size(); ++i) {
        auto& [nm, dt] = kws[i];
        tt_fa.allocate_field(sizeof(TypeTag), i);
        rt->attach_name(tt_fs, i, nm.c_str());
        add_field(dt, val_fa, i);
      }
      tts = rt->create_logical_region(ctx, is, tt_fs);
      vals = rt->create_logical_region(ctx, is, val_fs);
      RegionRequirement req(tts, WRITE_ONLY, EXCLUSIVE, tts);
      for (size_t i = 0; i < kws.size(); ++i)
        req.add_field(i);
      auto pr = rt->map_region(ctx, req);
      for (size_t i = 0; i < kws.size(), ++i) {
        const TypeTagAccessor<WRITE_ONLY> dt(pr, i);
        dt[0] = std::get<1>(kws[i]);
      }
      rt->unmap_region(ctx, pr);
    }
    return Keywords(tts, vals);
  }

  static TypeTag
  value_type(const Legion::PhysicalRegion& tt, Legion::FieldID fid) {

    const TypeTagAccessor<READ_ONLY> dt(tt, fid);
    return dt[0];
  }

  template <int MODE, typename T>
  static bool
  write(
    const pair<Legion::PhysicalRegion>& prs,
    Legion::FieldID fid,
    const T& t) {

    const TypeTagAccessor<READ_ONLY> dt(prs.type_tags, fid);
    const ValueAccessor<MODE, T> val(prs.values, fid);
    if (dt[0] == ValueType<T>::DataType) {
      val[0] = t;
      return true;
    }
    return false;
  }

  template <typename T>
  static std::optional<T>
  read(const pair<Legion::PhysicalRegion>& prs, Legion::FieldID fid) {

    const TypeTagAccessor<READ_ONLY> dt(prs.type_tags, fid);
    const ValueAccessor<READ_ONLY, T> val(prs.values, fid + 1);
    return ((dt[0] == ValueType<T>::DataType) ? val[0] : std::nullopt);
  }
};

} // end namespace legms

#endif // LEGMS_WITH_KEYWORDS_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
