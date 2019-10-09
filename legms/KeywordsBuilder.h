#ifndef LEGMS_KEYWORDS_BUILDER_H_
#define LEGMS_KEYWORDS_BUILDER_H_

#pragma GCC visibility push(default)
#include <string>
#include <tuple>
#include <vector>
#pragma GCC visibility pop

#include <legms/legms.h>
#include <legms/utility.h>
#include <legms/Keywords.h>

namespace legms {

class LEGMS_API KeywordsBuilder {
public:

  KeywordsBuilder() {}

  void
  add_keyword(const std::string& name, TypeTag datatype) {
    m_keywords.emplace_back(name, datatype);
  }

  const Keywords::kw_desc_t&
  keywords() const {
    return m_keywords;
  }

private:

  Keywords::kw_desc_t m_keywords;

};

} // end namespace legms

#endif // LEGMS_KEYWORDS_BUILDER_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
