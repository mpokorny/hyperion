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
    legms::TypeTag,
    1,
    coord_t,
    Legion::AffineAccessor<legms::TypeTag, 1, coord_t>,
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

  typedef std::vector<std::tuple<std::string, legms::TypeTag>> kw_desc_t;

  Keywords(pair<Legion::LogicalRegion> regions);

  std::vector<std::string>
  keys(Legion::Runtime* rt) const;

  std::optional<Legion::FieldID>
  find_keyword(Legion::Runtime* rt, const std::string& name) const;

  std::vector<legms::TypeTag>
  value_types(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<Legion::FieldID>& fids) const;

  template <legion_privilege_mode_t MODE, template <typename> C>
  pair<Legion::RegionRequirement>
  requirements(const C<FieldID>& fids) const {

    // allow runtime to catch erroneous FieldIDs
    RegionRequirement tt(type_tags, READ_ONLY, EXCLUSIVE, type_tags);
    RegionRequirement v(values, MODE, EXCLUSIVE, values);
    std::for_each(
      std::begin(fids),
      std::end(fids),
      [&tt, &v](auto fid) {
        tt.add_field(fid);
        v.add_field(fid);
      });
    return pair{tt, v};
  }

  template <typename T>
  bool
  write(
    Legion::Context ctx,
    Legion::Runtime* rt,
    Legion::FieldID fid,
    const T& t) const {

    bool result = false;
    auto reqs = requirements<WRITE_ONLY>(std::vector<FieldID>{fid});
    if (reqs) {
      auto prs =
        reqs.value().map([&ctx, rt](auto& r){ return rt->map_region(ctx, r); });
      result = write(prs, fid, t);
      prs.map([&ctx, rt](auto& p) { rt->unmap_region(ctx, p); });
    }
    return result;
  }

  template <typename T>
  std::optional<T>
  read(Legion::Context ctx, Legion::Runtime* rt, Legion::FieldID fid) const {

    bool result = false;
    auto reqs = requirements<READ_ONLY>(std::vector<FieldID>{fid});
    if (reqs) {
      auto prs =
        reqs.value().map([&ctx, rt](auto& r){ return rt->map_region(ctx, r); });
      result = read(prs, fid);
      prs.map([&ctx, rt](auto& p) { rt->unmap_region(ctx, p); });
    }
    return result;
  }

  static Keywords
  create(Legion::Context ctx, Legion::Runtime* rt, const kw_desc_t& kws);

  void
  destroy(Legion::Context ctx, Legion::Runtime* rt);

  static legms::TypeTag
  value_type(const Legion::PhysicalRegion& tt, Legion::FieldID fid);

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
