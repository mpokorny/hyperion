#ifndef LEGMS_MS_TABLE_H_
#define LEGMS_MS_TABLE_H_

#include <cassert>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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

  Table(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    const std::vector<Column::Generator>& column_generators,
    const std::unordered_map<std::string, casacore::DataType>& kws =
    std::unordered_map<std::string, casacore::DataType>());

  template <typename GeneratorIter>
  Table(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    GeneratorIter generator_first,
    GeneratorIter generator_last,
    const std::unordered_map<std::string, casacore::DataType>& kws =
    std::unordered_map<std::string, casacore::DataType>())
    : WithKeywords(kws)
    , m_name(name)
    , m_context(ctx)
    , m_runtime(runtime) {

    std::transform(
      generator_first,
      generator_last,
      std::inserter(m_columns, m_columns.end()),
      [&ctx, runtime](auto gen) {
        auto col = gen(ctx, runtime);
        return std::make_pair(col->name(), col);
      });

    if (m_columns.size() > 0) {
      auto row_index_pattern = (*m_columns.begin()).second->row_index_pattern();
      auto num_rows = (*m_columns.begin()).second->num_rows();
      assert(
        std::all_of(
          m_columns.begin(),
          m_columns.end(),
          [&row_index_pattern, &num_rows](auto& nc) {
            return row_index_pattern == nc.second->row_index_pattern()
              && num_rows == nc.second->num_rows();
          }));
    }
  }

  virtual ~Table() {
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

  const IndexTreeL&
  row_index_pattern() const {
    return std::get<1>(*m_columns.begin())->row_index_pattern();
  }

  bool
  is_empty() const {
    return
      m_columns.size() == 0
      || std::get<1>(*m_columns.begin())->index_tree() == IndexTreeL();
  }

  Column::row_number_t
  num_rows() const {
    if (m_columns.size() == 0)
      return 0;
    return std::get<1>(*m_columns.begin())->num_rows();
  }

  Column::row_number_t
  row_number(const std::vector<Legion::coord_t>& index) const {
    return row_number(row_index_pattern(), index.begin(), index.end());
  }

  std::unordered_set<std::string>
  column_names() const;

  std::shared_ptr<Column>
  column(const std::string& name) const {
    return m_columns.at(name);
  }

  Legion::IndexSpace
  index_space() const;

  std::vector<Legion::LogicalRegion>
  logical_regions(const std::vector<std::string>& colnames) const;

  template <typename Iter>
  std::vector<Legion::IndexPartition>
  index_partitions(
    const Legion::IndexPartition& ipart,
    Iter colnames_iter,
    Iter colnames_iter_end) const {

    assert(
      m_runtime->get_parent_index_space(m_context, ipart) == index_space());

    std::vector<Legion::IndexPartition> result;
    std::transform(
      colnames_iter,
      colnames_iter_end,
      std::back_inserter(result),
      [this, &ipart](auto& colname) {
        return column(colname)->projected_index_partition(ipart);
      });
    return result;
  }

  std::vector<Legion::IndexPartition>
  index_partitions(
    const Legion::IndexPartition& ipart,
    const std::vector<std::string>& colnames) const {

    return index_partitions(ipart, colnames.begin(), colnames.end());
  }

  std::vector<Legion::IndexPartition>
  row_block_index_partitions(
    const std::optional<Legion::IndexPartition>& ipart,
    const std::vector<std::string>& colnames,
    size_t block_size) const;

  template <typename Iter>
  static Column::row_number_t
  row_number(const IndexTreeL& row_pattern, Iter index, Iter index_end) {

    if (row_pattern == IndexTreeL())
      return 0;
    assert(index != index_end);

    int lo, hi;
    std::tie(lo, hi) = row_pattern.index_range();
    if (*index < lo)
      return 0;
    Column::row_number_t result =
      (*index - lo) / (hi - lo + 1) * row_pattern.size();
    auto i0 = (*index - lo) % (hi - lo + 1);
    auto ch = row_pattern.children().begin();
    auto ch_end = row_pattern.children().end();
    while (ch != ch_end) {
      auto& [b0, bn, t] = *ch;
      if (i0 >= b0 + bn) {
        result += bn * t.size();
        ++ch;
      } else {
        if (i0 >= b0)
          result += (i0 - b0) * t.size() + row_number(t, index + 1, index_end);
        break;
      }
    }
    return result;
  }

  Legion::IndexPartition
  row_partition(
    const std::vector<std::vector<Column::row_number_t>>& rowp,
    bool include_unselected = false,
    bool sorted_selections = false) const;

  Legion::Context
  context() const {
    return m_context;
  }

  Legion::Runtime*
  runtime() const {
    return m_runtime;
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

  Legion::Context m_context;

  Legion::Runtime* m_runtime;
};

} // end namespace ms
} // end namespace legms

#endif // LEGMS_MS_TABLE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
