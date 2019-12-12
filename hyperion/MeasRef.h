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
#ifndef HYPERION_MEAS_REF_H_
#define HYPERION_MEAS_REF_H_

#include <hyperion/hyperion.h>
#include <hyperion/utility.h>
#include <hyperion/Measures.h>

#pragma GCC visibility push(default)
# include <array>
# include <memory>
# include <optional>
# include <vector>
# include <casacore/measures/Measures.h>
#pragma GCC visibility pop

namespace hyperion {

class HYPERION_API MeasRef {
public:

  enum ArrayComponent {
    VALUE = 0,
    OFFSET,
    EPOCH,
    POSITION,
    DIRECTION,
    RADIAL_VELOCITY,
    // NOT SUPPORTED: COMET,
    NUM_COMPONENTS
  };

  typedef DataType<HYPERION_TYPE_DOUBLE>::ValueType VALUE_TYPE;

  typedef unsigned MEASURE_CLASS_TYPE;
  typedef unsigned REF_TYPE_TYPE;
  typedef unsigned NUM_VALUES_TYPE;

  static const constexpr Legion::FieldID MEASURE_CLASS_FID = 0;
  static const constexpr Legion::FieldID REF_TYPE_FID = 1;
  static const constexpr Legion::FieldID NUM_VALUES_FID = 2;

  Legion::LogicalRegion metadata_lr;
  Legion::LogicalRegion values_lr;

  struct DataRegions {
    Legion::PhysicalRegion metadata;
    Legion::PhysicalRegion values;
  };

  template <legion_privilege_mode_t MODE, int N, bool CHECK_BOUNDS=false>
  using ValueAccessor =
    Legion::FieldAccessor<
    MODE,
    VALUE_TYPE,
    N,
    Legion::coord_t,
    Legion::AffineAccessor<VALUE_TYPE, N, Legion::coord_t>,
    CHECK_BOUNDS>;

  template <legion_privilege_mode_t MODE, int N, bool CHECK_BOUNDS=false>
  using RefTypeAccessor =
    Legion::FieldAccessor<
    MODE,
    REF_TYPE_TYPE,
    N,
    Legion::coord_t,
    Legion::AffineAccessor<REF_TYPE_TYPE, N, Legion::coord_t>,
    CHECK_BOUNDS>;

  template <legion_privilege_mode_t MODE, int N, bool CHECK_BOUNDS=false>
  using MeasureClassAccessor =
    Legion::FieldAccessor<
    MODE,
    MEASURE_CLASS_TYPE,
    N,
    Legion::coord_t,
    Legion::AffineAccessor<MEASURE_CLASS_TYPE, N, Legion::coord_t>,
    CHECK_BOUNDS>;

  template <legion_privilege_mode_t MODE, int N, bool CHECK_BOUNDS=false>
  using NumValuesAccessor =
    Legion::FieldAccessor<
    MODE,
    NUM_VALUES_TYPE,
    N,
    Legion::coord_t,
    Legion::AffineAccessor<NUM_VALUES_TYPE, N, Legion::coord_t>,
    CHECK_BOUNDS>;

  MeasRef() {}

  MeasRef(
    Legion::LogicalRegion metadata_lr_,
    Legion::LogicalRegion values_lr_)
    : metadata_lr(metadata_lr_)
    , values_lr(values_lr_) {
    assert(metadata_lr != Legion::LogicalRegion::NO_REGION);
  }

  MeasRef(
    Legion::LogicalRegion metadata_lr_,
    std::optional<Legion::LogicalRegion> values_lr_)
    : MeasRef(
        metadata_lr_,
        values_lr_.value_or(Legion::LogicalRegion::NO_REGION)) {
  }

  bool
  operator==(const MeasRef& rhs) const {
    return metadata_lr == rhs.metadata_lr && values_lr == rhs.values_lr;
  }

  bool
  operator!=(const MeasRef& rhs) const {
    return !operator==(rhs);
  }

  std::tuple<
    Legion::RegionRequirement,
    std::optional<Legion::RegionRequirement>>
  requirements(legion_privilege_mode_t mode) const;

  bool
  is_empty() const {
    return metadata_lr == Legion::LogicalRegion::NO_REGION;
  }

  MClass
  mclass(Legion::Context ctx, Legion::Runtime* rt) const;

  static MClass
  mclass(Legion::PhysicalRegion pr);

  bool
  equiv(Legion::Context ctx, Legion::Runtime* rt, const MeasRef& other) const;

  static bool
  equiv(Legion::Runtime* rt, const DataRegions& x, const DataRegions& y);

  MeasRef
  clone(Legion::Context ctx, Legion::Runtime* rt) const;

  static MeasRef
  clone(Legion::Context ctx, Legion::Runtime* rt, const DataRegions& drs);

  template <typename Ms>
  static MeasRef
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    std::vector<casacore::MeasRef<Ms>>& meas_ref) {
    if (false) assert(false);
#define CREATE(M)                                                       \
    else if (typeid(MClassT<M>::type).hash_code() == typeid(Ms).hash_code()) { \
      std::vector<casacore::MRBase*> mrbs;                              \
      mrbs.reserve(meas_ref.size());                                    \
      std::transform(                                                   \
        meas_ref.begin(),                                               \
        meas_ref.end(),                                                 \
        std::back_inserter(mrbs),                                       \
        [](auto& mr) { return &mr; });                                  \
      return create(ctx, rt, mrbs, M);                                  \
    }
    HYPERION_FOREACH_MCLASS(CREATE)
#undef CM
    else assert(false);
  }

  template <typename Ms>
  static MeasRef
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const casacore::MeasRef<Ms>& meas_ref) {
    std::vector<casacore::MeasRef<Ms>> mrs{meas_ref};
    return create<Ms>(ctx, rt, mrs);
  }

  static std::array<Legion::LogicalRegion, 2>
  create_regions(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const IndexTreeL& metadata_tree,
    const IndexTreeL& value_tree);

  std::vector<std::unique_ptr<casacore::MRBase>>
  make(Legion::Context ctx, Legion::Runtime* rt) const;

  template <typename Ms>
  std::vector<std::shared_ptr<typename casacore::MeasRef<Ms>>>
  make(Legion::Context ctx, Legion::Runtime* rt) const {

    std::vector<std::shared_ptr<typename casacore::MeasRef<Ms>>> result;
    auto mrbs = make(ctx, rt);
    result.reserve(mrbs.size());
    for (auto&& mrb : mrbs) {
      auto mr =
        std::dynamic_pointer_cast<typename casacore::MeasRef<Ms>>(
          std::shared_ptr<casacore::MRBase>(std::move(mrb)));
      if (mr)
        result.push_back(mr);
    }
    return result;
  }

  static std::vector<std::unique_ptr<casacore::MRBase>>
  make(Legion::Runtime* rt, DataRegions prs);

  template <typename Ms>
  static std::vector<std::shared_ptr<typename casacore::MeasRef<Ms>>>
  make(Legion::Runtime* rt, DataRegions prs) {

    std::vector<std::shared_ptr<typename casacore::MeasRef<Ms>>> result;
    auto mrbs = make(rt, prs);
    result.reserve(mrbs.size());
    for (auto&& mrb : mrbs) {
      auto mr =
        std::dynamic_pointer_cast<typename casacore::MeasRef<Ms>>(
          std::shared_ptr<casacore::MRBase>(std::move(mrb)));
      if (mr)
        result.push_back(mr);
    }
    return result;
  }

  void
  destroy(Legion::Context ctx, Legion::Runtime* rt);

private:

  static MeasRef
  create(
    Legion::Context ctx,
    Legion::Runtime *rt,
    const std::vector<casacore::MRBase*>& mrbs,
    MClass klass);
};

} // end namespace hyperion

#endif // HYPERION_MEAS_REF_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
