#ifndef LEGMS_MEAS_REF_H_
#define LEGMS_MEAS_REF_H_

#include <legms/legms.h>
#include <legms/utility.h>
#include <legms/Measures.h>

#pragma GCC visibility push(default)
#include <array>
#include <memory>
#include <optional>
#pragma GCC visibility pop

#ifdef LEGMS_USE_CASACORE
#pragma GCC visibility push(default)
#include <casacore/measures/Measures.h>
#pragma GCC visibility pop

namespace legms {

class LEGMS_API MeasRef {
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

  typedef DataType<LEGMS_TYPE_DOUBLE>::ValueType VALUE_TYPE;

  typedef unsigned MEASURE_CLASS_TYPE;
  typedef unsigned REF_TYPE_TYPE;
  typedef unsigned NUM_VALUES_TYPE;

  typedef legms::string NAME_TYPE;

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
    LEGMS_FOREACH_MCLASS(CREATE)
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

} // end namespace legms

#endif // LEGMS_USE_CASACORE
#endif // LEGMS_MEAS_REF_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
