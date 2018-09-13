#ifndef LEGMS_MS_TABLE_H_
#define LEGMS_MS_TABLE_H_

#include <algorithm>
#include <cassert>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <casacore/casa/Utilities/DataType.h>
#include "legion.h"

#include "utility.h"
#include "WithKeywords.h"
#include "TableBuilder.h"
#include "Column.h"
#include "IndexTree.h"

namespace legms {
namespace ms {

class Table
  : public WithKeywords {
public:

  Table(const TableBuilder& builder)
    : WithKeywords(builder.keywords())
    , m_name(builder.name()) {

    assert(builder.m_columns.size() > 0);
    std::transform(
      builder.m_columns.begin(),
      builder.m_columns.end(),
      std::back_inserter(m_columns),
      [](auto& cb) {
        return std::make_pair(cb.name, std::make_shared(cb));
      });
  }

  const std::string&
  name() const {
    return m_name;
  }

  unsigned
  row_rank() const {
    return std::get<1>(*m_columns.begin())->row_rank();
  }

  unsigned
  rank() const {
    return std::get<1>(*max_rank_column())->rank();
  }

  std::unordered_set<std::string>
  column_names() const {
    std::unordered_set<std::string> result;
    std::transform(
      m_columns.begin(),
      m_columns.end(),
      std::back_inserter(result),
      [](auto& col) {
        return col.name();
      });
    return result;
  }

  std::vector<std::shared_ptr<Column>>
  columns(const std::vector<std::string>& names) const {
    std::vector<std::shared_ptr<Column>> result;
    std::accumulate(
      names.begin(),
      names.end(),
      std::back_inserter(result),
      [this](auto& nm) {
        return m_columns[nm];
      });
    return result;
  }

  std::optional<Legion::IndexSpace>
  index_space(Legion::Context ctx, Legion::Runtime* runtime) const {
    return std::get<1>(*max_rank_column())->index_space(ctx, runtime);
  }

  std::vector<Legion::FieldID>
  add_column_fields(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    Legion::FieldSpace fs,
    Legion::FieldAllocator fa,
    const std::vector<std::string>& colnames) const {

    std::vector<Legion::FieldID> result;
    std::transform(
      colnames.begin(),
      colnames.end(),
      std::back_inserter(result),
      [this, &ctx, runtime, &fs, &fa](auto& colname) {
        auto col =
          std::find_if(
            m_columns.begin(),
            m_columns.end(),
            [&colname](auto& col) {
              return col.name() == colname;
            });
        assert(col != m_columns.end());
        return col->add_field(ctx, runtime, fs, fa);
      });
    return result;
  }

protected:

  std::unordered_map<std::string, std::shared_ptr<Column>>::const_iterator
  max_rank_column() const {
    auto result = m_columns.begin();
    for (auto e = result; e != m_columns.end(); ++e) {
      if (std::get<1>(*e)->rank() > std::get<1>(*result)->rank())
        result = e;
    }
    return result;
  }

  std::string m_name;

  std::unordered_map<std::string, std::shared_ptr<Column>> m_columns;
};

} // end namespace ms
} // end namespace legms

#endif // LEGMS_MS_TABLE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
