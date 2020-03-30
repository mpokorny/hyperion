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
#ifndef HYPERION_MEASURES_H_
#define HYPERION_MEASURES_H_

#include <hyperion/hyperion.h>
#include <hyperion/utility.h>

#pragma GCC visibility push(default)
# include <memory>
# include <optional>
# include <tuple>
# include <vector>

# include <experimental/mdspan>

# include <casacore/tables/Tables.h>
# include <casacore/measures/Measures.h>
# include <casacore/measures/Measures/MBaseline.h>
# include <casacore/measures/Measures/MDirection.h>
# include <casacore/measures/Measures/MDoppler.h>
# include <casacore/measures/Measures/MEarthMagnetic.h>
# include <casacore/measures/Measures/MEpoch.h>
# include <casacore/measures/Measures/MFrequency.h>
# include <casacore/measures/Measures/MPosition.h>
# include <casacore/measures/Measures/MRadialVelocity.h>
# include <casacore/measures/Measures/Muvw.h>
# include <casacore/measures/Measures/MeasureHolder.h>
#pragma GCC visibility pop

namespace hyperion {

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

template <typename M>
struct MClassTBase {

  typedef M type;

  template <hyperion::TypeTag DT>
  static M
  load(
    const std::experimental::mdspan<
      typename DataType<DT>::ValueType,
      std::experimental::dynamic_extent>& vs,
    const char* units,
    const casacore::MeasRef<M>& mr) {

    casacore::Vector<typename DataType<DT>::ValueType> cv(vs.extent(0));
    for (size_t i = 0; i < vs.extent(0); ++i)
      cv[i] = vs(i);
    return M(M::MVType(casacore::Quantum(cv, units)), mr);
  }

  template <hyperion::TypeTag DT>
  static void
  store(
    const M& meas,
    const char *units,
    std::experimental::mdspan<
      typename DataType<DT>::ValueType,
      std::experimental::dynamic_extent>& vs) {

    auto p = meas.get(units).getValue();
    assert(p.size() == vs.extent(0));
    for (size_t i = 0; i < vs.extent(0); ++i)
      vs(i) = p[i];
  }
};

template <>
struct HYPERION_API MClassT<MClass::M_BASELINE>
  : public MClassTBase<casacore::MBaseline> {

  static constexpr const unsigned mrank = 1;

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
struct HYPERION_API MClassT<MClass::M_DIRECTION>
  : public MClassTBase<casacore::MDirection> {

  static constexpr const unsigned mrank = 1;

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
struct HYPERION_API MClassT<MClass::M_DOPPLER>
  : public MClassTBase<casacore::MDoppler> {

  static constexpr const unsigned mrank = 0;

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
struct HYPERION_API MClassT<MClass::M_EARTH_MAGNETIC>
  : public MClassTBase<casacore::MEarthMagnetic> {

  static constexpr const unsigned mrank = 1;

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
struct HYPERION_API MClassT<MClass::M_EPOCH>
  : public MClassTBase<casacore::MEpoch> {

  static constexpr const unsigned mrank = 0;

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
struct HYPERION_API MClassT<MClass::M_FREQUENCY>
  : public MClassTBase<casacore::MFrequency> {

  static constexpr const unsigned mrank = 0;

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
struct HYPERION_API MClassT<MClass::M_POSITION>
  : public MClassTBase<casacore::MPosition> {

  static constexpr const unsigned mrank = 1;

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
struct HYPERION_API MClassT<MClass::M_RADIAL_VELOCITY>
  : public MClassTBase<casacore::MRadialVelocity> {

  static constexpr const unsigned mrank = 0;

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
struct HYPERION_API MClassT<MClass::M_UVW>
  : public MClassTBase<casacore::Muvw> {

  static constexpr const unsigned mrank = 1;

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

#define HYPERION_FOREACH_MCLASS(__func__)          \
  __func__(::hyperion::MClass::M_BASELINE)         \
  __func__(::hyperion::MClass::M_DIRECTION)        \
  __func__(::hyperion::MClass::M_DOPPLER)          \
  __func__(::hyperion::MClass::M_EARTH_MAGNETIC)   \
  __func__(::hyperion::MClass::M_EPOCH)            \
  __func__(::hyperion::MClass::M_FREQUENCY)        \
  __func__(::hyperion::MClass::M_POSITION)         \
  __func__(::hyperion::MClass::M_RADIAL_VELOCITY)  \
  __func__(::hyperion::MClass::M_UVW)

std::optional<
  std::tuple<
    hyperion::MClass,
    std::vector<std::tuple<std::unique_ptr<casacore::MRBase>, unsigned>>,
    std::optional<std::string>>>
get_meas_refs(const casacore::Table& table, const std::string& colname);

class MeasRef;

std::tuple<std::string, MeasRef>
create_named_meas_refs(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const std::tuple<
    hyperion::MClass,
    std::vector<std::tuple<casacore::MRBase*, unsigned>>>& mrs);

} // end namespace hyperion

#endif // HYPERION_MEASURES_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
