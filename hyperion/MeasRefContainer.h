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
#ifndef HYPERION_MEAS_REF_CONTAINER_H_
#define HYPERION_MEAS_REF_CONTAINER_H_

#include <hyperion/hyperion.h>
#include <hyperion/MeasRef.h>
#include <hyperion/MeasRefDict.h>

#pragma GCC visibility push(default)
# include <array>
# include <numeric>
# include <optional>
# include <tuple>
# include <unordered_map>
# include <vector>
#pragma GCC visibility pop

namespace hyperion {

class HYPERION_API MeasRefContainer {
public:

  static const constexpr Legion::FieldID OWNED_FID = 0;
  static const constexpr Legion::FieldID NAME_FID = 1;
  static const constexpr Legion::FieldID MEAS_REF_FID = 2;
  Legion::LogicalRegion lr;

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using OwnedAccessor =
    Legion::FieldAccessor<
    MODE,
    bool,
    1,
    Legion::coord_t,
    Legion::AffineAccessor<bool, 1, Legion::coord_t>,
    CHECK_BOUNDS>;

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using NameAccessor =
    Legion::FieldAccessor<
    MODE,
    hyperion::string,
    1,
    Legion::coord_t,
    Legion::AffineAccessor<hyperion::string, 1, Legion::coord_t>,
    CHECK_BOUNDS>;

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using MeasRefAccessor =
    Legion::FieldAccessor<
    MODE,
    MeasRef,
    1,
    Legion::coord_t,
    Legion::AffineAccessor<MeasRef, 1, Legion::coord_t>,
    CHECK_BOUNDS>;

  MeasRefContainer();

  MeasRefContainer(Legion::LogicalRegion meas_refs);

  std::tuple<MeasRef, bool>
  lookup(Legion::Context ctx, Legion::Runtime* rt, const std::string& name);

  static std::tuple<MeasRef, bool>
  lookup(
    Legion::Runtime* rt,
    const std::string& name,
    Legion::PhysicalRegion& pr);

  static MeasRefContainer
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::unordered_map<std::string, MeasRef>& owned,
    const MeasRefContainer& borrowed);

  static MeasRefContainer
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::unordered_map<std::string, MeasRef>& owned);

  static MeasRefContainer
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::unordered_map<std::string, MeasRef>& owned,
    const std::optional<Legion::PhysicalRegion>& borrowed_pr);

  size_t
  size(Legion::Runtime* rt) const;

  std::vector<
    std::tuple<
      Legion::RegionRequirement,
      std::optional<Legion::RegionRequirement>>>
  component_requirements(
    Legion::Context ctx,
    Legion::Runtime* rt,
    legion_privilege_mode_t mode = READ_ONLY) const;

  std::optional<Legion::RegionRequirement>
  requirements(
    Legion::Context ctx,
    Legion::Runtime* rt,
    legion_privilege_mode_t mode) const;

  template <
    typename FN,
    std::enable_if_t<
      !std::is_void_v<
       std::invoke_result_t<FN, Legion::Context, Legion::Runtime*, MeasRefDict*>>,
      int> = 0>
  std::invoke_result_t<FN, Legion::Context, Legion::Runtime*, MeasRefDict*>
  with_measure_references_dictionary(
    Legion::Context ctx,
    Legion::Runtime* rt,
    FN fn) const {

    auto [dict, pr] = with_measure_references_dictionary_prologue(ctx, rt);
    auto result = fn(ctx, rt, &dict);
    with_measure_references_dictionary_epilogue(ctx, rt, pr);
    return result;
  }

  template <
    typename FN,
    std::enable_if_t<
      std::is_void_v<
        std::invoke_result_t<FN, Legion::Context, Legion::Runtime*, MeasRefDict*>>,
      int> = 0>
  void
  with_measure_references_dictionary(
    Legion::Context ctx,
    Legion::Runtime* rt,
    FN fn) const {

    auto [dict, pr] = with_measure_references_dictionary_prologue(ctx, rt);
    fn(ctx, rt, &dict);
    with_measure_references_dictionary_epilogue(ctx, rt, pr);
  }

  template <
    typename FN,
    std::enable_if_t<
      !std::is_void_v<
       std::invoke_result_t<FN, Legion::Context, Legion::Runtime*, MeasRefDict*>>,
      int> = 0>
  static std::invoke_result_t<FN, Legion::Context, Legion::Runtime*, MeasRefDict*>
  with_measure_references_dictionary(
    Legion::Context ctx,
    Legion::Runtime* rt,
    Legion::PhysicalRegion pr,
    FN fn) {

    auto mrs = get_mr_ptrs(rt, pr);
    auto dict = MeasRefDict(ctx, rt, mrs);
    return fn(ctx, rt, &dict);
  }

  template <
    typename FN,
    std::enable_if_t<
      std::is_void_v<
        std::invoke_result_t<FN, Legion::Context, Legion::Runtime*, MeasRefDict*>>,
      int> = 0>
  static void
  with_measure_references_dictionary(
    Legion::Context ctx,
    Legion::Runtime* rt,
    Legion::PhysicalRegion pr,
    FN fn) {

    auto mrs = get_mr_ptrs(rt, pr);
    auto dict = MeasRefDict(ctx, rt, mrs);
    fn(ctx, rt, &dict);
    return;
  }

  void
  destroy(Legion::Context ctx, Legion::Runtime* rt);

private:

  static std::unordered_map<std::string, const MeasRef*>
  get_mr_ptrs(Legion::Runtime* rt, Legion::PhysicalRegion pr);

  std::tuple<MeasRefDict, std::optional<Legion::PhysicalRegion>>
  with_measure_references_dictionary_prologue(
    Legion::Context ctx,
    Legion::Runtime* rt) const;

  void
  with_measure_references_dictionary_epilogue(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::optional<Legion::PhysicalRegion>& pr) const;
};

} // end namespace hyperion

#endif // HYPERION_MEAS_REF_CONTAINER_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
