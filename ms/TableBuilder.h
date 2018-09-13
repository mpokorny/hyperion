#ifndef LEGMS_MS_TABLE_BUILDER_H_
#define LEGMS_MS_TABLE_BUILDER_H_

#include <algorithm>
#include <cassert>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <casacore/casa/Utilities/DataType.h>

#include "legion.h"

#include "WithKeywordsBuilder.h"
#include "ColumnBuilder.h"
#include "IndexTree.h"

namespace legms {
namespace ms {

class Table;

class TableBuilder
  : public WithKeywordsBuilder {

  friend class Table;

public:

  TableBuilder(const std::string& name, IndexTreeL row_index_shape)
    : WithKeywordsBuilder()
    , m_name(name)
    , m_row_index_shape(row_index_shape) {
  }

  const std::string&
  name() const {
    return m_name;
  }

  template <typename ColGen>
  void
  add_column(ColGen generator) {
    add_column(generator(m_row_index_shape));
  }

  void
  add_column(std::unique_ptr<ColumnBuilder>&& col) {
    assert(col->row_index_shape() == m_row_index_shape);
    assert(m_columns.size() == 0
           || (std::get<1>(*m_columns.begin())->row_index_tree()
               == col->row_index_tree()));
    assert(m_columns.count(col->name()) == 0);
    m_columns[col->name()] = std::move(col);
  }

  void
  add_row(const std::unordered_map<std::string, std::any>& args) {
    std::for_each(
      m_columns.begin(),
      m_columns.end(),
      [&args](auto& nm_col) {
        const std::string& nm = std::get<0>(nm_col);
        auto nm_arg = args.find(nm);
        std::any arg = ((nm_arg != args.end()) ? nm_arg->second : std::any());
        std::get<1>(nm_col)->add_row(arg);
      });
  }

  std::unordered_set<std::string>
  column_names() const {
    std::unordered_set<std::string> result;
    std::transform(
      m_columns.begin(),
      m_columns.end(),
      std::back_inserter(result),
      [](auto& nm_col) {
        return std::get<0>(nm_col);
      });
    return result;
  }

protected:

  std::string m_name;

  std::unordered_map<std::string, std::unique_ptr<ColumnBuilder>> m_columns;

  IndexTreeL m_row_index_shape;
};


} // end namespace ms
} // end namespace legms

#endif // LEGMS_MS_TABLE_BUILDER_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
