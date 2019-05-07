#ifndef LEGMS_WITH_KEYWORDS_H_
#define LEGMS_WITH_KEYWORDS_H_

#include <algorithm>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <casacore/casa/Utilities/DataType.h>

#include "legms.h"

namespace legms {

class WithKeywords {
public:

  WithKeywords(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::unordered_map<std::string, casacore::DataType>& kws)
    : m_context(ctx)
    , m_runtime(runtime) {

    std::transform(
      kws.begin(),
      kws.end(),
      std::back_inserter(m_keywords),
      [](auto& nm_dt) { return std::get<0>(nm_dt); });

    if (kws.size() > 0) {
      auto is = m_runtime->create_index_space(m_context, Legion::Rect<1>(0, 0));
      auto fs = m_runtime->create_field_space(m_context);
      auto fa = m_runtime->create_field_allocator(m_context, fs);
      std::for_each(
        kws.begin(),
        kws.end(),
        [&](auto& nm_dt) {
          auto& [nm, dt] = nm_dt;
          auto fid = add_field(dt, fa);
          m_runtime->attach_name(fs, fid, nm.c_str());
        });
      m_keywords_region = m_runtime->create_logical_region(m_context, is, fs);
      m_runtime->destroy_field_space(m_context, fs);
      m_runtime->destroy_index_space(m_context, is);
    }
    else {
      m_keywords_region = Legion::LogicalRegion::NO_REGION;
    }
  }

  WithKeywords(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    Legion::LogicalRegion region)
    : m_context(ctx)
    , m_runtime(runtime)
    , m_keywords_region(region) {
  }

  virtual ~WithKeywords() {
    if (m_keywords_region != Legion::LogicalRegion::NO_REGION)
      m_runtime->destroy_logical_region(m_context, m_keywords_region);
  }

  const std::vector<std::string>&
  keywords() const {
    return m_keywords;
  }

  Legion::LogicalRegion
  keywords_region() const {
    return m_keywords_region;
  }

private:

  Legion::Context m_context;

  Legion::Runtime* m_runtime;

  std::vector<std::string> m_keywords;

  Legion::LogicalRegion m_keywords_region;
};

} // end namespace legms

#endif // LEGMS_WITH_KEYWORDS_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
