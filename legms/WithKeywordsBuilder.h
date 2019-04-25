#ifndef LEGMS_WITH_KEYWORDS_BUILDER_H_
#define LEGMS_WITH_KEYWORDS_BUILDER_H_

#include <string>
#include <tuple>
#include <vector>

#include <casacore/casa/Utilities/DataType.h>

namespace legms {

class WithKeywordsBuilder {
public:

  WithKeywordsBuilder() {}

  void
  add_keyword(const std::string& name, casacore::DataType datatype) {
    m_keywords.emplace_back(name, datatype);
  }

  const std::vector<std::tuple<std::string, casacore::DataType>>&
  keywords() const {
    return m_keywords;
  }

private:

  std::vector<std::tuple<std::string, casacore::DataType>> m_keywords;

};

} // end namespace legms

#endif // LEGMS_WITH_KEYWORDS_BUILDER_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
