#ifndef LEGMS_MEAS_REF_CONTAINER_H_
#define LEGMS_MEAS_REF_CONTAINER_H_

#pragma GCC visibility push(default)
#include <array>
#include <numeric>
#include <optional>
#include <tuple>
#include <vector>
#pragma GCC visibility pop

#include <legms/legms.h>

#ifdef LEGMS_USE_CASACORE
#include <legms/MeasRef.h>
#include <legms/MeasRefDict.h>

namespace legms {

class LEGMS_API MeasRefContainer {
public:

  static const constexpr Legion::FieldID OWNED_FID = 0;
  static const constexpr Legion::FieldID MEAS_REF_FID = 1;
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

  static MeasRefContainer
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<MeasRef>& owned,
    const MeasRefContainer& borrowed);

  static MeasRefContainer
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<MeasRef>& owned);

  void
  add_prefix_to_owned(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::string& prefix) const;

  size_t
  size(Legion::Runtime* rt) const;

  std::vector<Legion::RegionRequirement>
  component_requirements(
    Legion::Context ctx,
    Legion::Runtime* rt,
    legion_privilege_mode_t mode = READ_ONLY) const;

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

  static std::vector<const MeasRef*>
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

} // end namespace legms

#endif // LEGMS_USE_CASACORE
#endif // LEGMS_MEAS_REF_CONTAINER_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
