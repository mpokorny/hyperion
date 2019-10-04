#ifndef LEGMS_MEAS_REF_CONTAINER_H_
#define LEGMS_MEAS_REF_CONTAINER_H_

#pragma GCC visibility push(default)
#include <array>
#include <numeric>
#include <tuple>
#pragma GCC visibility pop

#include <legms/legms.h>

#ifdef LEGMS_USE_CASACORE
#include <legms/MeasRef.h>

namespace legms {

class MeasRefContainer {
public:

  unsigned num_meas_refs;
  std::array<std::tuple<bool, MeasRef>, LEGMS_MAX_NUM_MS_MEASURES> meas_refs;

  MeasRefContainer()
    : num_meas_refs(0) {
  }

  template <template <typename> typename C>
  MeasRefContainer(const C<MeasRef>& owned) {

    num_meas_refs =
      std::accumulate(
        owned.begin(),
        owned.end(),
        0,
        [this](unsigned i, const MeasRef& mr) {
          assert(i < LEGMS_MAX_NUM_MS_MEASURES);
          meas_refs[i] = std::make_tuple(true, mr);
          return i + 1;
        });
  }

  template <typename MRIter>
  MeasRefContainer(MRIter begin_owned, MRIter end_owned) {

    num_meas_refs =
      std::accumulate(
        begin_owned,
        end_owned,
        0,
        [this](unsigned i, const MeasRef& mr) {
          assert(i < LEGMS_MAX_NUM_MS_MEASURES);
          meas_refs[i] = std::make_tuple(true, mr);
          return i + 1;
        });
  }

  template <template <typename> typename C>
  MeasRefContainer(
    const C<MeasRef> owned,
    const MeasRefContainer& borrowed) {

    num_meas_refs =
      std::accumulate(
        owned.begin(),
        owned.end(),
        0,
        [this](unsigned i, const MeasRef& mr) {
          assert(i < LEGMS_MAX_NUM_MS_MEASURES);
          meas_refs[i] = std::make_tuple(true, mr);
          return i + 1;
        });
    num_meas_refs =
      std::accumulate(
        borrowed.meas_refs.begin(),
        borrowed.meas_refs.end(),
        num_meas_refs,
        [this](unsigned i, auto& bmr) {
          assert(i < LEGMS_MAX_NUM_MS_MEASURES);
          meas_refs[i] = std::make_tuple(false, std::get<1>(bmr));
          return i + 1;
        });
  }

protected:

  std::vector<MeasRef*>
  owned_meas_ref() const {
    std::vector<MeasRef*> result;
    for (auto& [is_owned, mr] : meas_refs)
      if (is_owned)
        result.push_back(const_cast<MeasRef*>(&mr));
    return result;
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
