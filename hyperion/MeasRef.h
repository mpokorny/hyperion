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

  typedef hyperion::string NAME_TYPE;

  static const constexpr Legion::FieldID MEASURE_CLASS_FID = 0;
  static const constexpr Legion::FieldID REF_TYPE_FID = 1;
  static const constexpr Legion::FieldID NUM_VALUES_FID = 2;

  static const constexpr Legion::FieldID NAME_FID = 0;

  Legion::LogicalRegion name_region;
  Legion::LogicalRegion value_region;
  Legion::LogicalRegion metadata_region;

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using NameAccessor =
    Legion::FieldAccessor<
    MODE,
    NAME_TYPE,
    1,
    Legion::coord_t,
    Legion::AffineAccessor<NAME_TYPE, 1, Legion::coord_t>,
    CHECK_BOUNDS>;

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
    Legion::LogicalRegion name_region_,
    Legion::LogicalRegion value_region_,
    Legion::LogicalRegion metadata_region_)
    : name_region(name_region_)
    , value_region(value_region_)
    , metadata_region(metadata_region_) {
    assert(name_region != Legion::LogicalRegion::NO_REGION);
    assert(metadata_region != Legion::LogicalRegion::NO_REGION);
  }

  std::string
  name(Legion::Context ctx, Legion::Runtime* rt) const;

  static const NAME_TYPE&
  name(Legion::PhysicalRegion pr) {
    const NameAccessor<READ_ONLY> nm(pr, NAME_FID);
    return nm[0];
  }

  static std::string::size_type
  find_tag(const std::string& name);

  MClass
  mclass(Legion::Context ctx, Legion::Runtime* rt) const;

  static MClass
  mclass(Legion::PhysicalRegion pr);

  bool
  equiv(Legion::Context ctx, Legion::Runtime* rt, const MeasRef& other) const;

  static bool
  equiv(
    Legion::Runtime* rt,
    const std::tuple<
      Legion::PhysicalRegion,
      std::optional<Legion::PhysicalRegion>>& x,
    const std::tuple<
      Legion::PhysicalRegion,
      std::optional<Legion::PhysicalRegion>>& y);

  template <typename Ms>
  static MeasRef
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::string& name,
    casacore::MeasRef<Ms> meas_ref) {
    if (false) assert(false);
#define CREATE(M)                                                       \
    else if (typeid(MClassT<M>::type).hash_code() == typeid(Ms).hash_code()) \
      return create(ctx, rt, name, &meas_ref, M);
    HYPERION_FOREACH_MCLASS(CREATE)
#undef CM
    else assert(false);
  }

  static std::array<Legion::LogicalRegion, 3>
  create_regions(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::string& name,
    const IndexTreeL& metadata_tree,
    const std::optional<IndexTreeL>& value_tree);

  std::unique_ptr<casacore::MRBase>
  make(Legion::Context ctx, Legion::Runtime* rt) const;

  template <typename Ms>
  std::optional<std::shared_ptr<typename casacore::MeasRef<Ms>>>
  make(Legion::Context ctx, Legion::Runtime* rt) const {
    std::optional<std::shared_ptr<typename casacore::MeasRef<Ms>>> result;
    std::shared_ptr<casacore::MRBase> mrb = make(ctx, rt);
    auto mr = std::dynamic_pointer_cast<typename casacore::MeasRef<Ms>>(mrb);
    if (mr)
      result = mr;
    return result;
  }

  void
  destroy(Legion::Context ctx, Legion::Runtime* rt);

private:

  static MeasRef
  create(
    Legion::Context ctx,
    Legion::Runtime *rt,
    const std::string& name,
    casacore::MRBase* mr,
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
