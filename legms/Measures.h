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
#ifndef LEGMS_MEASURES_H_
#define LEGMS_MEASURES_H_

#include <legms/legms.h>
#include <legms/utility.h>

#pragma GCC visibility push(default)
#include <memory>
#pragma GCC visibility pop

#ifdef LEGMS_USE_CASACORE
#pragma GCC visibility push(default)
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
#include <casacore/measures/Measures/MeasureHolder.h>
#pragma GCC visibility pop

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
  // static bool
  // holds(const casacore::MeasureHolder& mh);
  // static const type&
  // get(const casacore::MeasureHolder& mh);
};
template <>
struct LEGMS_API MClassT<MClass::M_BASELINE> {
  typedef casacore::MBaseline type;
  static const std::string name;
  static bool
  holds(const casacore::MeasureHolder& mh) {
    return mh.isMBaseline();
  }
  static const type&
  get(const casacore::MeasureHolder& mh) {
    return mh.asMBaseline();
  }
};

template <>
struct LEGMS_API MClassT<MClass::M_DIRECTION> {
  typedef casacore::MDirection type;
  static const std::string name;
  static bool
  holds(const casacore::MeasureHolder& mh) {
    return mh.isMDirection();
  }
  static const type&
  get(const casacore::MeasureHolder& mh) {
    return mh.asMDirection();
  }
};

template <>
struct LEGMS_API MClassT<MClass::M_DOPPLER> {
  typedef casacore::MDoppler type;
  static const std::string name;
  static bool
  holds(const casacore::MeasureHolder& mh) {
    return mh.isMDoppler();
  }
  static const type&
  get(const casacore::MeasureHolder& mh) {
    return mh.asMDoppler();
  }
};

template <>
struct LEGMS_API MClassT<MClass::M_EARTH_MAGNETIC> {
  typedef casacore::MEarthMagnetic type;
  static const std::string name;
  static bool
  holds(const casacore::MeasureHolder& mh) {
    return mh.isMEarthMagnetic();
  }
  static const type&
  get(const casacore::MeasureHolder& mh) {
    return mh.asMEarthMagnetic();
  }
};
template <>
struct LEGMS_API MClassT<MClass::M_EPOCH> {
  typedef casacore::MEpoch type;
  static const std::string name;
  static bool
  holds(const casacore::MeasureHolder& mh) {
    return mh.isMEpoch();
  }
  static const type&
  get(const casacore::MeasureHolder& mh) {
    return mh.asMEpoch();
  }
};
template <>
struct LEGMS_API MClassT<MClass::M_FREQUENCY> {
  typedef casacore::MFrequency type;
  static const std::string name;
  static bool
  holds(const casacore::MeasureHolder& mh) {
    return mh.isMFrequency();
  }
  static const type&
  get(const casacore::MeasureHolder& mh) {
    return mh.asMFrequency();
  }
};
template <>
struct LEGMS_API MClassT<MClass::M_POSITION> {
  typedef casacore::MPosition type;
  static const std::string name;
  static bool
  holds(const casacore::MeasureHolder& mh) {
    return mh.isMPosition();
  }
  static const type&
  get(const casacore::MeasureHolder& mh) {
    return mh.asMPosition();
  }
};
template <>
struct LEGMS_API MClassT<MClass::M_RADIAL_VELOCITY> {
  typedef casacore::MRadialVelocity type;
  static const std::string name;
  static bool
  holds(const casacore::MeasureHolder& mh) {
    return mh.isMRadialVelocity();
  }
  static const type&
  get(const casacore::MeasureHolder& mh) {
    return mh.asMRadialVelocity();
  }
};
template <>
struct LEGMS_API MClassT<MClass::M_UVW> {
  typedef casacore::Muvw type;
  static const std::string name;
  static bool
  holds(const casacore::MeasureHolder& mh) {
    return mh.isMuvw();
  }
  static const type&
  get(const casacore::MeasureHolder& mh) {
    return mh.asMuvw();
  }
};

#define LEGMS_FOREACH_MCLASS(__func__)          \
  __func__(::legms::MClass::M_BASELINE)         \
  __func__(::legms::MClass::M_DIRECTION)        \
  __func__(::legms::MClass::M_DOPPLER)          \
  __func__(::legms::MClass::M_EARTH_MAGNETIC)   \
  __func__(::legms::MClass::M_EPOCH)            \
  __func__(::legms::MClass::M_FREQUENCY)        \
  __func__(::legms::MClass::M_POSITION)         \
  __func__(::legms::MClass::M_RADIAL_VELOCITY)  \
  __func__(::legms::MClass::M_UVW)

} // end namespace legms

#endif // LEGMS_USE_CASACORE
#endif // LEGMS_MEASURES_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
