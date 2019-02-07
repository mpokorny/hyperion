#ifndef LEGMS_MS_WITH_KEYWORDS_H_
#define LEGMS_MS_WITH_KEYWORDS_H_

#include <algorithm>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <casacore/casa/Utilities/DataType.h>

#include "legion.h"
#include "utility.h"

namespace legms {
namespace ms {

class WithKeywords {
public:

  WithKeywords(
    const std::unordered_map<std::string, casacore::DataType>& kws)
    : m_keywords(kws) {
  }

  std::vector<std::string>
  keywords() const {
    std::vector<std::string> result;
    std::transform(
      m_keywords.begin(),
      m_keywords.end(),
      std::back_inserter(result),
      [](auto& nm_dt) { return std::get<0>(nm_dt); });
    return result;
  }

  Legion::LogicalRegion
  keywords_region(Legion::Context ctx, Legion::Runtime* runtime) const {
    auto is = runtime->create_index_space(ctx, Legion::Rect<1>(0, 0));
    auto fs = runtime->create_field_space(ctx);
    auto fa = runtime->create_field_allocator(ctx, fs);
    std::for_each(
      m_keywords.begin(),
      m_keywords.end(),
      [&](auto& nm_dt) {
        auto& [nm, dt] = nm_dt;
        auto fid = legms::add_field(dt, fa);
        runtime->attach_name(fs, fid, nm.c_str());
      });
    Legion::LogicalRegion result = runtime->create_logical_region(ctx, is, fs);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
    return result;
  }

private:

  std::unordered_map<std::string, casacore::DataType> m_keywords;

};


} // end namespace ms
} // end namespace legms

#endif // LEGMS_MS_WITH_KEYWORDS_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
