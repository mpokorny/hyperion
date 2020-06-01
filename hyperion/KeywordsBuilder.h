/*
 * Copyright 2020 Associated Universities, Inc. Washington DC, USA.
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
#ifndef HYPERION_KEYWORDS_BUILDER_H_
#define HYPERION_KEYWORDS_BUILDER_H_

#include <hyperion/hyperion.h>
#include <hyperion/utility.h>
#include <hyperion/Keywords.h>

#include <string>
#include <tuple>
#include <vector>

namespace hyperion {

class HYPERION_API KeywordsBuilder {
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

} // end namespace hyperion

#endif // HYPERION_KEYWORDS_BUILDER_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
