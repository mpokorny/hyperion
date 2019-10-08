#ifndef LEGMS_MEAS_REF_DICT_H_
#define LEGMS_MEAS_REF_DICT_H_

#include "legms.h"
#include "utility.h"
#include "MeasRef.h"

#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <variant>

#ifdef LEGMS_USE_CASACORE

namespace legms {

class LEGMS_API MeasRefDict {
  // this class is meant to be instantiated and destroyed within a single task,
  // instances should not escape their enclosing context
public:

  typedef std::variant<
    casacore::MeasRef<casacore::MBaseline>,
    casacore::MeasRef<casacore::MDirection>,
    casacore::MeasRef<casacore::MDoppler>,
    casacore::MeasRef<casacore::MEarthMagnetic>,
    casacore::MeasRef<casacore::MEpoch>,
    casacore::MeasRef<casacore::MFrequency>,
    casacore::MeasRef<casacore::MPosition>,
    casacore::MeasRef<casacore::MRadialVelocity>,
    casacore::MeasRef<casacore::Muvw>> Ref;

  template <template <typename> typename C>
  MeasRefDict(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const C<const MeasRef*>& refs)
    : m_ctx(ctx)
    , m_rt(rt) {

    for (auto& mr : refs)
      m_meas_refs[mr->name(ctx, rt)] = mr;
  }

  template <typename MRIter>
  MeasRefDict(
    Legion::Context ctx,
    Legion::Runtime* rt,
    MRIter begin,
    MRIter end)
    : m_ctx(ctx)
    , m_rt(rt) {

    std::transform(
      begin,
      end,
      std::inserter(m_meas_refs, m_meas_refs.end()),
      [&ctx, rt](const MeasRef* mr) {
        return std::make_pair(mr->name(ctx, rt), mr);
      });
  }

  std::unordered_set<std::string>
  names() const;

  std::optional<const Ref*>
  get(const std::string& name) const;

private:

  Legion::Context m_ctx;

  Legion::Runtime* m_rt;

  // use pointers to MeasRef, rather than MeasRef values, even though values
  // would be cheap to copy, to emphasize that MeasRefDict instances depend on
  // the context in which they are created
  std::unordered_map<std::string, const MeasRef*> m_meas_refs;

  // save unique_ptr to Refs to provide stability for pointers returned from
  // get()
  mutable std::unordered_map<std::string, std::unique_ptr<const Ref>> m_refs;
};

} // end namespace legms

#endif // LEGMS_USE_CASACORE
#endif // LEGMS_MEAS_REF_DICT_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
