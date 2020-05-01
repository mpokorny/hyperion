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
#ifndef HYPERION_WITH_KEYWORDS_H_
#define HYPERION_WITH_KEYWORDS_H_

#include <hyperion/hyperion.h>
#include <hyperion/utility.h>

#pragma GCC visibility push(default)
# include <algorithm>
# include <any>
# include <optional>
# include <string>
# include <tuple>
# include <unordered_map>
# include <vector>
#pragma GCC visibility pop

namespace hyperion {

class  HYPERION_API Keywords {
public:

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
    hyperion::TypeTag,
    1,
    coord_t,
    Legion::AffineAccessor<hyperion::TypeTag, 1, coord_t>,
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

  typedef std::vector<std::tuple<std::string, hyperion::TypeTag>> kw_desc_t;

  Keywords();

  Keywords(pair<Legion::LogicalRegion> regions);

  bool
  is_empty() const;

  uint_least8_t
  num_regions() const {
    return (is_empty() ? 0 : 2);
  }

  size_t
  size(Legion::Runtime* rt) const;

  std::vector<std::string>
  keys(Legion::Runtime* rt) const;

  std::optional<Legion::FieldID>
  find_keyword(Legion::Runtime* rt, const std::string& name) const;

  std::vector<std::optional<hyperion::TypeTag>>
  value_types(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<Legion::FieldID>& fids) const;

  Keywords
  clone(Legion::Context ctx, Legion::Runtime* rt) const;

  static Keywords
  clone(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const pair<Legion::PhysicalRegion>& prs);

  template <template <typename> typename C>
  std::optional<pair<Legion::RegionRequirement>>
  requirements(
    Legion::Runtime* rt,
    const C<Legion::FieldID>& fids,
    Legion::PrivilegeMode mode,
    bool mapped = true) const {

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
        [&mapped, &tt, &v](auto fid) {
          tt.add_field(fid, mapped);
          v.add_field(fid, mapped);
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
      requirements<std::vector>(
        rt,
        std::vector<Legion::FieldID>{fid},
        READ_ONLY);
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

  static hyperion::TypeTag
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

  std::unordered_map<std::string, std::tuple<hyperion::TypeTag, std::any>>
  to_map(Legion::Context ctx, Legion::Runtime *rt) const;

  static std::unordered_map<std::string, std::tuple<hyperion::TypeTag, std::any>>
  to_map(Legion::Runtime *rt, const pair<Legion::PhysicalRegion>& prs);
};

} // end namespace hyperion

#endif // HYPERION_WITH_KEYWORDS_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
