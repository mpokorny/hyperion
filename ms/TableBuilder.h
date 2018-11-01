#ifndef LEGMS_MS_TABLE_BUILDER_H_
#define LEGMS_MS_TABLE_BUILDER_H_

#include <algorithm>
#include <cassert>
#include <experimental/filesystem>
#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "legion.h"

#include <casacore/casa/aipstype.h>
#include <casacore/casa/BasicSL/String.h>
#include <casacore/tables/Tables.h>

#include "WithKeywordsBuilder.h"
#include "ColumnBuilder.h"
#include "Column.h"
#include "ColumnHint.h"
#include "IndexTree.h"

namespace legms {
namespace ms {

class Table;

class TableBuilder
  : public WithKeywordsBuilder {

  friend class Table;

public:

  TableBuilder(const std::string& name, IndexTreeL row_index_pattern)
    : WithKeywordsBuilder()
    , m_name(name)
    , m_row_index_pattern(row_index_pattern) {
  }

  const std::string&
  name() const {
    return m_name;
  }

  template <typename T>
  void
  add_scalar_column(const std::string& name) {

    add_column(ScalarColumnBuilder::generator<T>(name)(m_row_index_pattern));
  }

  template <int DIM, typename T>
  void
  add_array_column(
    const std::string& name,
    std::function<std::array<size_t, DIM>(const std::any&)> row_dimensions) {

    add_column(
      ArrayColumnBuilder<DIM>::template generator<T>(name, row_dimensions)(
        m_row_index_pattern));
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

  void
  add_row() {
    add_row(std::unordered_map<std::string, std::any>());
  }

  std::unordered_set<std::string>
  column_names() const {
    std::unordered_set<std::string> result;
    std::transform(
      m_columns.begin(),
      m_columns.end(),
      std::inserter(result, result.end()),
      [](auto& nm_col) {
        return std::get<0>(nm_col);
      });
    return result;
  }

  std::vector<Column::Generator>
  column_generators() const {
    std::vector<Column::Generator> result;
    transform(
      m_columns.begin(),
      m_columns.end(),
      std::back_inserter(result),
      [](auto& cb) {
        return
          [cb](Legion::Context ctx, Legion::Runtime *runtime) {
            return std::make_shared<Column>(ctx, runtime, *cb.second);
          };
      });
    return result;
  }

  static ColumnHint
  inferred_column_hint(const casacore::ColumnDesc& coldesc);

  static TableBuilder
  from_casacore_table(
    const std::experimental::filesystem::path& path,
    const std::unordered_set<std::string>& column_selections,
    const std::unordered_map<std::string, ColumnHint>& column_hints);

  static const std::unordered_map<std::string, ColumnHint>&
  ms_column_hints(const std::string& table);

protected:

  static const std::unordered_map<
  std::string,
  std::unordered_map<std::string, ColumnHint>> m_ms_column_hints;

  static const std::unordered_map<std::string, ColumnHint> m_empty_column_hints;

  void
  add_column(std::unique_ptr<ColumnBuilder>&& col) {
    assert(col->row_index_pattern() == m_row_index_pattern);
    assert(m_columns.count(col->name()) == 0);
    if (m_columns.size() > 0) {
      auto h = std::min(m_columns[m_max_rank_column]->rank(), col->rank()) - 1;
      assert(m_columns[m_max_rank_column]->index_tree().pruned(h)
             == col->index_tree().pruned(h));
      if (col->rank() > m_columns[m_max_rank_column]->rank())
        m_max_rank_column = col->name();
    } else {
      m_max_rank_column = col->name();
    }
    m_columns[col->name()] = std::move(col);
  }

  std::string m_name;

  std::unordered_map<std::string, std::shared_ptr<ColumnBuilder>> m_columns;

  IndexTreeL m_row_index_pattern;

  std::string m_max_rank_column;
};


} // end namespace ms
} // end namespace legms

#endif // LEGMS_MS_TABLE_BUILDER_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
