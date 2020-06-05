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
#ifndef HYPERION_MEAS_REF_H_
#define HYPERION_MEAS_REF_H_

#include <hyperion/hyperion.h>
#include <hyperion/utility.h>
#include <hyperion/Measures.h>

#include <array>
#include <memory>
#include CXX_OPTIONAL_HEADER
#include <unordered_map>
#include <vector>

#include <casacore/measures/Measures.h>

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

  typedef unsigned M_CODE_TYPE;

  static const constexpr Legion::FieldID M_CODE_FID = 0;

  Legion::LogicalRegion metadata_lr;
  Legion::LogicalRegion values_lr;
  Legion::LogicalRegion index_lr;

  struct DataRegions {
    Legion::PhysicalRegion metadata;
    Legion::PhysicalRegion values;
    CXX_OPTIONAL_NAMESPACE::optional<Legion::PhysicalRegion> index;
  };

  template <legion_privilege_mode_t MODE, int N, bool CHECK_BOUNDS=false>
  using ValueAccessor =
    Legion::FieldAccessor<
      MODE,
      VALUE_TYPE,
      N,
      Legion::coord_t,
      Legion::GenericAccessor<VALUE_TYPE, N, Legion::coord_t>,
      CHECK_BOUNDS>;

  template <legion_privilege_mode_t MODE, int N, bool CHECK_BOUNDS=false>
  using RefTypeAccessor =
    Legion::FieldAccessor<
      MODE,
      REF_TYPE_TYPE,
      N,
      Legion::coord_t,
      Legion::GenericAccessor<REF_TYPE_TYPE, N, Legion::coord_t>,
      CHECK_BOUNDS>;

  template <legion_privilege_mode_t MODE, int N, bool CHECK_BOUNDS=false>
  using MeasureClassAccessor =
    Legion::FieldAccessor<
      MODE,
      MEASURE_CLASS_TYPE,
      N,
      Legion::coord_t,
      Legion::GenericAccessor<MEASURE_CLASS_TYPE, N, Legion::coord_t>,
      CHECK_BOUNDS>;

  template <legion_privilege_mode_t MODE, int N, bool CHECK_BOUNDS=false>
  using NumValuesAccessor =
    Legion::FieldAccessor<
      MODE,
      NUM_VALUES_TYPE,
      N,
      Legion::coord_t,
      Legion::GenericAccessor<NUM_VALUES_TYPE, N, Legion::coord_t>,
      CHECK_BOUNDS>;

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using MCodeAccessor =
    Legion::FieldAccessor<
      MODE,
      M_CODE_TYPE,
      1,
      Legion::coord_t,
      Legion::GenericAccessor<M_CODE_TYPE, 1, Legion::coord_t>,
      CHECK_BOUNDS>;

  MeasRef() {}

  MeasRef(
    Legion::LogicalRegion metadata_lr_,
    Legion::LogicalRegion values_lr_,
    Legion::LogicalRegion index_lr_)
    : metadata_lr(metadata_lr_)
    , values_lr(values_lr_)
    , index_lr(index_lr_) {
    assert(metadata_lr != Legion::LogicalRegion::NO_REGION);
  }

  // MeasRef(
  //   Legion::LogicalRegion metadata_lr_,
  //   CXX_OPTIONAL_NAMESPACE::optional<Legion::LogicalRegion> values_lr_)
  //   : MeasRef(
  //       metadata_lr_,
  //       values_lr_.value_or(Legion::LogicalRegion::NO_REGION)) {
  // }

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
    Legion::RegionRequirement,
    CXX_OPTIONAL_NAMESPACE::optional<Legion::RegionRequirement>>
    requirements(Legion::PrivilegeMode mode, bool mapped = true) const;

  static std::tuple<
    Legion::RegionRequirement,
    Legion::RegionRequirement,
    CXX_OPTIONAL_NAMESPACE::optional<Legion::RegionRequirement>>
  requirements(
    const DataRegions& drs,
    Legion::PrivilegeMode mode,
    bool mapped = true);

  bool
  is_empty() const {
    return metadata_lr == Legion::LogicalRegion::NO_REGION;
  }

  uint_least8_t
  num_regions() const {
    unsigned result = 0;
    for (auto lr : {metadata_lr, values_lr, index_lr})
      if (lr != Legion::LogicalRegion::NO_REGION)
        ++result;
    return result;
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
    const std::vector<std::tuple<casacore::MeasRef<Ms>, unsigned>>& meas_refs,
    bool no_index=false) {

    if (false) assert(false);
#if HAVE_CXX17
#define CREATE(MC)                                                      \
    else if (typeid(MClassT<MC>::type).hash_code() == typeid(Ms).hash_code()) { \
      std::vector<std::tuple<casacore::MRBase*, unsigned>> mrbs;        \
      mrbs.reserve(meas_refs.size());                                   \
      for (auto& [mr, c] : meas_refs)                                   \
        mrbs.emplace_back(                                              \
          const_cast<casacore::MRBase*>(                                \
            static_cast<const casacore::MRBase*>(&mr)),                 \
          c);                                                           \
      return create(ctx, rt, mrbs, MC, no_index);                       \
    }
#else // !HAVE_CXX17
#define CREATE(MC)                                                      \
    else if (typeid(MClassT<MC>::type).hash_code() == typeid(Ms).hash_code()) { \
      std::vector<std::tuple<casacore::MRBase*, unsigned>> mrbs;        \
      mrbs.reserve(meas_refs.size());                                   \
      for (auto& mr_c : meas_refs) {                                    \
        auto& mr = std::get<0>(mr_c);                                   \
        auto& c = std::get<1>(mr_c);                                    \
        mrbs.emplace_back(                                              \
          const_cast<casacore::MRBase*>(                                \
            static_cast<const casacore::MRBase*>(&mr)),                 \
          c);                                                           \
      }                                                                 \
      return create(ctx, rt, mrbs, MC, no_index);                       \
    }
#endif // HAVE_CXX17
    HYPERION_FOREACH_MCLASS(CREATE)
#undef CREATE
    else {
      assert(false);
      return MeasRef();
    }
  }

  template <typename Ms>
  static MeasRef
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const casacore::MeasRef<Ms>& meas_ref) {
    std::vector<std::tuple<casacore::MeasRef<Ms>, unsigned>> mrs{{meas_ref, 0}};
    return create<Ms>(ctx, rt, mrs, true);
  }

  static MeasRef
  create(
    Legion::Context ctx,
    Legion::Runtime *rt,
    const std::vector<std::tuple<casacore::MRBase*, unsigned>>& mrbs,
    MClass klass,
    bool no_index=false);

  static std::array<Legion::LogicalRegion, 3>
  create_regions(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const IndexTreeL& metadata_tree,
    const IndexTreeL& value_tree,
    bool no_index=false);

  std::tuple<
    std::vector<std::unique_ptr<casacore::MRBase>>,
    std::unordered_map<unsigned, unsigned>>
  make(Legion::Context ctx, Legion::Runtime* rt) const;

  template <typename Ms>
  std::tuple<
    std::vector<std::shared_ptr<typename casacore::MeasRef<Ms>>>,
    std::unordered_map<unsigned, unsigned>>
  make(Legion::Context ctx, Legion::Runtime* rt) const {

    auto mrbs = make(ctx, rt);
    std::vector<std::shared_ptr<typename casacore::MeasRef<Ms>>> tmrs;
    tmrs.reserve(std::get<0>(mrbs).size());
    for (auto&& mrb : std::get<0>(mrbs)) {
      auto mr =
        std::dynamic_pointer_cast<typename casacore::MeasRef<Ms>>(
          std::shared_ptr<casacore::MRBase>(std::move(mrb)));
      if (mr)
        tmrs.push_back(mr);
      else
        return
          std::make_tuple(
            std::vector<std::shared_ptr<typename casacore::MeasRef<Ms>>>(),
            std::unordered_map<unsigned, unsigned>());
    }
    return std::make_tuple(tmrs, std::get<1>(mrbs));
  }

  static std::tuple<
    std::vector<std::unique_ptr<casacore::MRBase>>,
    std::unordered_map<unsigned, unsigned>>
  make(Legion::Runtime* rt, const DataRegions& prs);

  template <typename Ms>
  static std::tuple<
    std::vector<std::shared_ptr<typename casacore::MeasRef<Ms>>>,
    std::unordered_map<unsigned, unsigned>>
  make(Legion::Runtime* rt, const DataRegions& prs) {

    auto mrbs = make(rt, prs);
    std::vector<std::shared_ptr<typename casacore::MeasRef<Ms>>> tmrs;
    tmrs.reserve(std::get<0>(mrbs).size());
    for (auto& mrb : std::get<0>(mrbs)) {
      auto mr =
        std::dynamic_pointer_cast<typename casacore::MeasRef<Ms>>(
          std::shared_ptr<casacore::MRBase>(std::move(mrb)));
      assert(mr);
      tmrs.push_back(mr);
    }
    return std::make_tuple(tmrs, std::get<1>(mrbs));
  }

  void
  destroy(Legion::Context ctx, Legion::Runtime* rt);

};

} // end namespace hyperion

#endif // HYPERION_MEAS_REF_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
