#ifndef LEGMS_MEASURES_H_
#define LEGMS_MEASURES_H_

#include "legms.h"
#include "utility.h"

#include <memory>

#ifdef LEGMS_USE_CASACORE
#include <casacore/measures/Measures.h>

#include <casacore/measures/Measures/MBaseline.h>
#include <casacore/measures/Measures/MDirection.h>
#include <casacore/measures/Measures/MDoppler.h>
#include <casacore/measures/Measures/MEarthMagnetic.h>
#include <casacore/measures/Measures/MEpoch.h>
#include <casacore/measures/Measures/MFrequency.h>
#include <casacore/measures/Measures/MPosition.h>
#include <casacore/measures/Measures/MRadialVelocity.h>
#include <casacore/measures/Measures/Muvw.h>

namespace legms {

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

template <MClass k>
struct MClassT {
  // typedef casacore::... type;
  // std::string name;
};
template <>
struct MClassT<MClass::M_BASELINE> {
  typedef casacore::MBaseline type;
  static const std::string name;
};

template <>
struct MClassT<MClass::M_DIRECTION> {
  typedef casacore::MDirection type;
  static const std::string name;
};

template <>
struct MClassT<MClass::M_DOPPLER> {
  typedef casacore::MDoppler type;
  static const std::string name;
};

template <>
struct MClassT<MClass::M_EARTH_MAGNETIC> {
  typedef casacore::MEarthMagnetic type;
  static const std::string name;
};
template <>
struct MClassT<MClass::M_EPOCH> {
  typedef casacore::MEpoch type;
  static const std::string name;
};
template <>
struct MClassT<MClass::M_FREQUENCY> {
  typedef casacore::MFrequency type;
  static const std::string name;
};
template <>
struct MClassT<MClass::M_POSITION> {
  typedef casacore::MPosition type;
  static const std::string name;
};
template <>
struct MClassT<MClass::M_RADIAL_VELOCITY> {
  typedef casacore::MRadialVelocity type;
  static const std::string name;
};
template <>
struct MClassT<MClass::M_UVW> {
  typedef casacore::Muvw type;
  static const std::string name;
};

#define FOREACH_MCLASS(__func__)                \
  __func__(MClass::M_BASELINE)                  \
  __func__(MClass::M_DIRECTION)                 \
  __func__(MClass::M_DOPPLER)                   \
  __func__(MClass::M_EARTH_MAGNETIC)            \
  __func__(MClass::M_EPOCH)                     \
  __func__(MClass::M_FREQUENCY)                 \
  __func__(MClass::M_POSITION)                  \
  __func__(MClass::M_RADIAL_VELOCITY)           \
  __func__(MClass::M_UVW)

} // end namespace legms

#endif // LEGMS_USE_CASACORE
#endif // LEGMS_MEASURES_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
