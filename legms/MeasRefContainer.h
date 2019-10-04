#ifndef LEGMS_MEAS_REF_CONTAINER_H_
#define LEGMS_MEAS_REF_CONTAINER_H_

#pragma GCC visibility push(default)
#include <array>
#include <numeric>
#pragma GCC visibility pop

#include <legms/legms.h>

#ifdef LEGMS_USE_CASACORE
#include <legms/MeasRef.h>

namespace legms {

class MeasRefContainer {
public:

  unsigned num_meas_refs;
  std::array<MeasRef, LEGMS_MAX_NUM_MS_MEASURES> meas_refs;

  MeasRefContainer()
    : num_meas_refs(0) {
  }

  template <template <typename> typename C>
  MeasRefContainer(const C<const MeasRef>& mrs) {

    num_meas_refs =
      std::accumulate(
        mrs.begin(),
        mrs.end(),
        0,
        [this](unsigned i, const MeasRef& mr) {
          assert(i < LEGMS_MAX_NUM_MS_MEASURES);
          meas_refs[i] = mr;
          return i + 1;
        });
  }

  template <typename MRIter>
  MeasRefContainer(MRIter begin, MRIter end) {

    num_meas_refs =
      std::accumulate(
        begin,
        end,
        0,
        [this](unsigned i, const MeasRef& mr) {
          assert(i < LEGMS_MAX_NUM_MS_MEASURES);
          meas_refs[i] = mr;
          return i + 1;
        });
  }

  MeasRefContainer(const MeasRefContainer& other)
    : num_meas_refs(other.num_meas_refs)
    , meas_refs(other.meas_refs) {
  }

  MeasRefContainer(MeasRefContainer&& other)
    : num_meas_refs(std::move(other).num_meas_refs)
    , meas_refs(std::move(other).meas_refs) {
  }

  MeasRefContainer&
  operator=(const MeasRefContainer& other) {
    num_meas_refs = other.num_meas_refs;
    meas_refs = other.meas_refs;
    return *this;
  }

  MeasRefContainer&
  operator=(MeasRefContainer&& other) {
    num_meas_refs = std::move(other).num_meas_refs;
    meas_refs = std::move(other).meas_refs;
    return *this;
  }
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
