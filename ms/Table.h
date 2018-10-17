#ifndef LEGMS_MS_TABLE_H_
#define LEGMS_MS_TABLE_H_

#include <cassert>
#include <memory>
#include <mutex>
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

    assert(m_columns.size() > 0);
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

  virtual ~Table() {
    std::lock_guard<decltype(m_logical_regions_mutex)>
      lock(m_logical_regions_mutex);
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

  template <typename Iter>
  std::vector<Legion::IndexPartition>
  index_partitions(
    const Legion::IndexPartition& ipart,
    Iter colnames_iter,
    Iter colnames_iter_end) const {

    auto is = index_space();

    assert(m_runtime->get_parent_index_space(m_context, ipart) == is);

    std::set<unsigned> ranks;
    std::transform(
      colnames_iter,
      colnames_iter_end,
      std::inserter(ranks, ranks.end()),
      [this](auto& colname) { return column(colname)->rank(); });
    auto fs = m_runtime->create_field_space(m_context);
    {
      auto fa = m_runtime->create_field_allocator(m_context, fs);
      std::for_each(
        ranks.begin(),
        ranks.end(),
        [&fa](auto r) {
          switch (r) {
          case 1:
            fa.allocate_field(sizeof(Legion::Point<1>), 1);
            break;
          case 2:
            fa.allocate_field(sizeof(Legion::Point<2>), 2);
            break;
          case 3:
            fa.allocate_field(sizeof(Legion::Point<3>), 3);
            break;
          default:
            assert(false);
            break;
          }
        });
    }
    auto proj_lr = m_runtime->create_logical_region(m_context, is, fs);
    auto proj_lp = m_runtime->get_logical_partition(m_context, proj_lr, ipart);
    initialize_projections(m_context, m_runtime, proj_lr, proj_lp);
    m_runtime->destroy_field_space(m_context, fs);

    unsigned  reg_rank = is.get_dim();
    auto color_space =
      m_runtime->get_index_partition_color_space_name(m_context, ipart);
    std::vector<Legion::IndexPartition> result;
    std::transform(
      colnames_iter,
      colnames_iter_end,
      std::back_inserter(result),
      [&, this](auto& colname) {
        auto col = column(colname);
        auto rank = col->rank();
        if (rank < reg_rank)
          return m_runtime->create_partition_by_image(
            m_context,
            col->index_space(),
            proj_lp,
            proj_lr,
            rank,
            color_space);
        else
          return ipart;
      });
    //runtime->destroy_logical_partition(ctx, proj_lp);
    m_runtime->destroy_logical_region(m_context, proj_lr);
    m_runtime->destroy_index_space(m_context, color_space);
    return result;
  }

  std::vector<Legion::IndexPartition>
  index_partitions(
    const Legion::IndexPartition& ipart,
    const std::vector<std::string>& colnames) const {

    return index_partitions(ipart, colnames.begin(), colnames.end());
  }

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

  mutable std::mutex m_logical_regions_mutex;

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
