#ifndef LEGMS_WITH_KEYWORDS_H_
#define LEGMS_WITH_KEYWORDS_H_

#pragma GCC visibility push(default)
#include <algorithm>
#include <optional>
#include <string>
#include <tuple>
#include <vector>
#pragma GCC visibility pop

#include <legms/legms.h>
#include <legms/utility.h>

namespace legms {

struct LEGMS_API Keywords {

  Legion::LogicalRegion type_tags_lr;

  Legion::LogicalRegion values_lr;

  template <typename T>
  struct pair {
    T type_tags;
    T values;

    template <typename F>
    pair<std::invoke_result_t<F,T>>
    map(F f) {
      return pair<std::invoke_result_t<F,T>>{f(type_tags), f(values)};
    }
  };

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using TypeTagAccessor =
    Legion::FieldAccessor<
    MODE,
    legms::TypeTag,
    1,
    coord_t,
    Legion::AffineAccessor<legms::TypeTag, 1, coord_t>,
    CHECK_BOUNDS>;

  template <legion_privilege_mode_t MODE, typename FT, bool CHECK_BOUNDS=false>
  using ValueAccessor =
    Legion::FieldAccessor<
    MODE,
    FT,
    1,
    coord_t,
    Legion::AffineAccessor<FT, 1, coord_t>,
    CHECK_BOUNDS>;

  typedef std::vector<std::tuple<std::string, legms::TypeTag>> kw_desc_t;

  Keywords();

  Keywords(pair<Legion::LogicalRegion> regions);

  bool
  is_empty() const;

  size_t
  size(Legion::Runtime* rt) const;

  std::vector<std::string>
  keys(Legion::Runtime* rt) const;

  std::optional<Legion::FieldID>
  find_keyword(Legion::Runtime* rt, const std::string& name) const;

  std::vector<std::optional<legms::TypeTag>>
  value_types(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<Legion::FieldID>& fids) const;

  template <template <typename> typename C>
  std::optional<pair<Legion::RegionRequirement>>
  requirements(
    Legion::Runtime* rt,
    const C<Legion::FieldID>& fids,
    legion_privilege_mode_t mode) const {

    std::optional<pair<Legion::RegionRequirement>> result;
    auto n = size(rt);
    if (
      std::all_of(
        fids.begin(),
        fids.end(),
        [&n](auto& fid) { return 0 <= fid && fid < n;} )) {
      Legion::RegionRequirement
        tt(type_tags_lr, READ_ONLY, EXCLUSIVE, type_tags_lr);
      Legion::RegionRequirement v(values_lr, mode, EXCLUSIVE, values_lr);
      std::for_each(
        std::begin(fids),
        std::end(fids),
        [&tt, &v](auto fid) {
          tt.add_field(fid);
          v.add_field(fid);
        });
      result = pair<Legion::RegionRequirement>{tt, v};
    }
    return result;
  }

  template <typename T>
  bool
  write(
    Legion::Context ctx,
    Legion::Runtime* rt,
    Legion::FieldID fid,
    const T& t) const {

    bool result = false;
    auto reqs =
      requirements(rt, std::vector<Legion::FieldID>{fid}, WRITE_ONLY);
    if (reqs) {
      auto prs =
        reqs.value().map(
          [&ctx, rt](const Legion::RegionRequirement& r){
            return rt->map_region(ctx, r);
          });
      result = write<WRITE_ONLY>(prs, fid, t);
      prs.map(
        [&ctx, rt](const Legion::PhysicalRegion& p) {
          rt->unmap_region(ctx, p);
          return 0;
        });
    }
    return result;
  }

  template <typename T>
  std::optional<T>
  read(Legion::Context ctx, Legion::Runtime* rt, Legion::FieldID fid) const {

    std::optional<T> result;
    auto reqs =
      requirements(rt, std::vector<Legion::FieldID>{fid}, READ_ONLY);
    if (reqs) {
      auto prs =
        reqs.value().map(
          [&ctx, rt](const Legion::RegionRequirement& r){
            return rt->map_region(ctx, r);
          });
      result = read<T>(prs, fid);
      prs.map(
        [&ctx, rt](const Legion::PhysicalRegion& p) {
          rt->unmap_region(ctx, p);
          return 0;
        });
    }
    return result;
  }

  static Keywords
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const kw_desc_t& kws,
    const std::string& name_prefix = "");

  void
  destroy(Legion::Context ctx, Legion::Runtime* rt);

  static legms::TypeTag
  value_type(const Legion::PhysicalRegion& tt, Legion::FieldID fid);

  template <legion_privilege_mode_t MODE, typename T>
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
    const ValueAccessor<READ_ONLY, T> val(prs.values, fid);
    return ((dt[0] == ValueType<T>::DataType) ? val[0] : std::optional<T>());
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
