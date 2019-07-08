#ifndef LEGMS_WITH_KEYWORDS_H_
#define LEGMS_WITH_KEYWORDS_H_

#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

#include "legms.h"
#include "utility.h"

namespace legms {

class WithKeywords {
public:

  typedef std::vector<std::tuple<std::string, TypeTag>> kw_desc_t;

  WithKeywords(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const kw_desc_t& kws)
    : m_context(ctx)
    , m_runtime(runtime) {

    std::vector<TypeTag> datatypes;
    if (kws.size() > 0) {
      auto is = m_runtime->create_index_space(m_context, Legion::Rect<1>(0, 0));
      auto fs = m_runtime->create_field_space(m_context);
      auto fa = m_runtime->create_field_allocator(m_context, fs);
      for (size_t i = 0; i < kws.size(); ++i) {
        auto& [nm, dt] = kws[i];
        add_field(dt, fa, i);
        m_runtime->attach_name(fs, i, nm.c_str());
        datatypes.push_back(dt);
      }
      m_keywords_region = m_runtime->create_logical_region(m_context, is, fs);
      // TODO: keep?
      // m_runtime->destroy_field_space(m_context, fs);
      // m_runtime->destroy_index_space(m_context, is);
    }
    else {
      m_keywords_region = Legion::LogicalRegion::NO_REGION;
    }
    init(datatypes);
  }

  WithKeywords(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    Legion::LogicalRegion region,
    const std::vector<TypeTag>& datatypes)
    : m_context(ctx)
    , m_runtime(runtime)
    , m_keywords_region(region) {

    init(datatypes);
  }

  virtual ~WithKeywords() {
    // TODO: keep?
    // if (m_keywords_region != Legion::LogicalRegion::NO_REGION)
    //   m_runtime->destroy_logical_region(m_context, m_keywords_region);
  }

  const kw_desc_t&
  keywords() const {
    return m_keywords;
  }

  Legion::LogicalRegion
  keywords_region() const {
    return m_keywords_region;
  }

  std::vector<TypeTag>
  keyword_datatypes() const {
    std::vector<TypeTag> result;
    std::transform(
      m_keywords.begin(),
      m_keywords.end(),
      std::back_inserter(result),
      [](auto& kw) { return std::get<1>(kw); });
    return result;
  }

  size_t
  num_keywords() const {
    return m_keywords.size();
  }

  Legion::Context&
  context() const {
    return m_context;
  }

  Legion::Runtime*
  runtime() const {
    return m_runtime;
  }

private:

  void
  init(const std::vector<TypeTag>& datatypes) {
    if (m_keywords_region != Legion::LogicalRegion::NO_REGION) {
      Legion::FieldSpace fs = m_keywords_region.get_field_space();
      for (size_t i = 0; i < datatypes.size(); ++i) {
        const char* name;
        m_runtime->retrieve_name(fs, i, name);
        m_keywords.emplace_back(name, datatypes[i]);
      }
    }
  }
protected:

  mutable Legion::Context m_context;

  mutable Legion::Runtime* m_runtime;

private:

  kw_desc_t m_keywords;

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
