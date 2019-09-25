#ifndef LEGMS_MEASURES_H_
#define LEGMS_MEASURES_H_

#include "legms.h"
#include "utility.h"

#include <memory>

#ifdef LEGMS_USE_CASACORE
#include <casacore/measures/Measures.h>

namespace legms {

class LEGMS_API MeasureRegion {
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

  enum MClass {
    M_BASELINE,
    M_DIRECTION,
    M_DOPPLER,
    M_EARTH_MAGNETIC,
    M_EPOCH,
    M_FREQUENCY,
    M_POSITION,
    M_RADIAL_VELOCITY,
    M_UVW,
    M_NUM_CLASSES,
    M_NONE = M_NUM_CLASSES
  };

  typedef casacore::Double VALUE_TYPE;
  typedef unsigned MEASURE_CLASS_TYPE;
  typedef unsigned REF_TYPE_TYPE;
  typedef unsigned NUM_VALUES_TYPE;

  static const constexpr Legion::FieldID MEASURE_CLASS_FID = 0;
  static const constexpr Legion::FieldID REF_TYPE_FID = 1;
  static const constexpr Legion::FieldID NUM_VALUES_FID = 2;

  Legion::LogicalRegion value_region;
  Legion::LogicalRegion metadata_region;

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

  MeasureRegion(
    Legion::LogicalRegion value_region_,
    Legion::LogicalRegion metadata_region_)
    : value_region(value_region_)
    , metadata_region(metadata_region_) {
  }

  static MeasureRegion
  create_like(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const casacore::Measure& measure,
    bool with_reference = true);

  static MeasureRegion
  create_from(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const casacore::Measure& measure,
    bool with_reference = true);

  std::unique_ptr<casacore::Measure>
  make(Legion::Context ctx, Legion::Runtime* rt) const;

  template <typename M>
  std::unique_ptr<M>
  make(Legion::Context ctx, Legion::Runtime* rt) const {
    std::unique_ptr<casacore::Measure> measure = make(ctx, rt);
    std::unique_ptr<M> result;
    if (measure)
      result.reset(dynamic_cast<M*>(measure.release()));
    return result;
  }

  void
  destroy(Legion::Context ctx, Legion::Runtime* rt);
};

} // end namespace legms

#endif // LEGMS_USE_CASACORE
#endif // LEGMS_MEASURES_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
