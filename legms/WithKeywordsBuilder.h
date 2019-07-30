#ifndef LEGMS_WITH_KEYWORDS_BUILDER_H_
#define LEGMS_WITH_KEYWORDS_BUILDER_H_

#pragma GCC visibility push(default)
#include <string>
#include <tuple>
#include <vector>
#pragma GCC visibility pop

#include "legms.h"
#include "utility.h"

namespace legms {

class LEGMS_API WithKeywordsBuilder {
public:

  WithKeywordsBuilder() {}

  void
  add_keyword(const std::string& name, TypeTag datatype) {
    m_keywords.emplace_back(name, datatype);
  }

  const std::vector<std::tuple<std::string, TypeTag>>&
  keywords() const {
    return m_keywords;
  }

private:

  std::vector<std::tuple<std::string, TypeTag>> m_keywords;

};

} // end namespace legms

#endif // LEGMS_WITH_KEYWORDS_BUILDER_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
