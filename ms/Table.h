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
    const TableBuilder& builder);

  Table(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    const std::unordered_set<std::shared_ptr<Column>>& columns,
    const std::unordered_map<std::string, casacore::DataType>& kws =
      std::unordered_map<std::string, casacore::DataType>());

  virtual ~Table() {
    std::for_each(
      m_logical_regions.begin(),
      m_logical_regions.end(),
      [this](auto& nm_lr_fid) {
        m_runtime->destroy_logical_region(
          m_context,
          std::get<0>(std::get<1>(nm_lr_fid)));
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

  size_t
  num_rows() const {
    if (m_columns.size() == 0)
      return 0;
    return std::get<1>(*m_columns.begin())->num_rows();
  }

  size_t
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

  std::vector<std::tuple<Legion::LogicalRegion, Legion::FieldID>>
  logical_regions(const std::vector<std::string>& colnames) const;

  std::vector<Legion::IndexPartition>
  index_partitions(
    const Legion::IndexPartition& ipart,
    const std::vector<std::string>& colnames) const;

  std::tuple<std::vector<Legion::IndexPartition>, Legion::IndexPartition>
  row_block_index_partitions(
    const std::optional<Legion::IndexPartition>& ipart,
    const std::vector<std::string>& colnames,
    size_t block_size) const;

  template <typename IndexIter, typename IndexIterEnd>
  static size_t
  row_number(
    const IndexTreeL& row_pattern,
    IndexIter index,
    IndexIterEnd index_end) {

    if (row_pattern == IndexTreeL())
      return 0;
    assert(index != index_end);

    int lo, hi;
    std::tie(lo, hi) = row_pattern.index_range();
    if (*index < lo)
      return 0;
    size_t result = (*index - lo) / (hi - lo + 1) * row_pattern.size();
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

  static void
  initialize_projections(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    Legion::LogicalRegion lr,
    Legion::LogicalPartition lp);

  std::string m_name;

  std::unordered_map<std::string, std::shared_ptr<Column>> m_columns;

  Legion::Context m_context;

  Legion::Runtime* m_runtime;

  mutable std::unordered_map<
    std::string,
    std::tuple<Legion::LogicalRegion, Legion::FieldID>> m_logical_regions;
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
