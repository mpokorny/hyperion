#include "MeasRefDict.h"
#include <algorithm>

using namespace legms;

std::unordered_set<std::string>
MeasRefDict::names() const {

  std::unordered_set<std::string> result;
  std::transform(
    m_meas_refs.begin(),
    m_meas_refs.end(),
    std::inserter(result, result.end()),
    [](auto& nm_r) { return std::get<0>(nm_r); });
  return result;
}

std::optional<const MeasRefDict::Ref*>
MeasRefDict::get(const std::string& name) const {
  std::optional<const Ref*> result;
  if (m_meas_refs.count(name) > 0) {
    if (m_refs.count(name) == 0) {
      const auto& mr = m_meas_refs.at(name);
      switch (mr->mclass(m_ctx, m_rt)) {
#define MK(M)                                                 \
        case M:                                               \
          m_refs.insert(                                      \
            std::make_pair(                                   \
              name,                                           \
              std::make_unique<Ref>(                          \
                *mr->make<MClassT<M>::type>(m_ctx, m_rt))));  \
          break;
        LEGMS_FOREACH_MCLASS(MK)
#undef MK
      default:
          assert(false);
        break;
      }
    }
    result = m_refs.at(name).get();
  }
  return result;
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
