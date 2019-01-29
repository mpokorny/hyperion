#ifndef LEGMS_MS_TABLE_H_
#define LEGMS_MS_TABLE_H_

#include <algorithm>
#include <cassert>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "legion.h"

#include "utility.h"
#include "WithKeywords.h"
#include "TableBuilder.h"
#include "Column.h"
#include "IndexTree.h"
#include "MSTable.h"
#include "ColumnPartition.h"

namespace legms {
namespace ms {

class Table
  : public WithKeywords {
public:

  Table(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    unsigned full_rank,
    const std::vector<int>& row_axes,
    const std::unordered_map<std::string, casacore::DataType>& kws =
    std::unordered_map<std::string, casacore::DataType>())
    : WithKeywords(kws)
    , m_name(name)
    , m_full_rank(full_rank)
    , m_row_axes(row_axes)
    , m_context(ctx)
    , m_runtime(runtime) {

    assert(
      std::all_of(
        m_row_axes.begin(),
        m_row_axes.end(),
        [this](auto& i) {
          return 0 <= i && static_cast<unsigned>(i) < m_full_rank;
        }));
  };

  template <typename D>
  Table(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    unsigned full_rank,
    const std::vector<D>& row_axes,
    const std::unordered_map<std::string, casacore::DataType>& kws =
    std::unordered_map<std::string, casacore::DataType>())
    : WithKeywords(kws)
    , m_name(name)
    , m_full_rank(full_rank)
    , m_context(ctx)
    , m_runtime(runtime) {

    std::transform(
      row_axes.begin(),
      row_axes.end(),
      std::back_inserter(m_row_axes),
      [](auto& d) { return static_cast<int>(d); });

    assert(
      std::all_of(
        m_row_axes.begin(),
        m_row_axes.end(),
        [this](auto& i) {
          return 0 <= i && static_cast<unsigned>(i) < m_full_rank; }));
  };

  virtual ~Table() {
  }

  const std::string&
  name() const {
    return m_name;
  }

  unsigned
  full_rank() const {
    return m_full_rank;
  }

  unsigned
  row_rank() const {
    return m_row_axes.size();
  }

  const std::vector<int>&
  row_axes() const {
    return m_row_axes;
  }

  const IndexTreeL&
  row_index_pattern() const {
    return column(min_rank_column_name())->row_index_pattern();
  }

  bool
  is_empty() const {
    return column(min_rank_column_name())->index_tree() == IndexTreeL();
  }

  virtual std::unordered_set<std::string>
  column_names() const = 0;

  virtual std::shared_ptr<Column>
  column(const std::string& name) const = 0;

  virtual const std::string&
  min_rank_column_name() const = 0;

  virtual const std::string&
  max_rank_column_name() const = 0;

  Column::row_number_t
  num_rows() const {
    return column(min_rank_column_name())->num_rows();
  }

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

  virtual std::unique_ptr<ColumnPartition>
  row_partition(
    const std::vector<std::vector<Column::row_number_t>>& rowp,
    bool include_unselected = false,
    bool sorted_selections = false) const = 0;

  Legion::Context&
  context() const {
    return m_context;
  }

  Legion::Runtime*
  runtime() const {
    return m_runtime;
  }

  static std::unique_ptr<Table>
  from_ms(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::experimental::filesystem::path& path,
    const std::unordered_set<std::string>& column_selections);

private:

  std::string m_name;

  unsigned m_full_rank;

protected:

  std::vector<int> m_row_axes;

  mutable Legion::Context m_context;

  mutable Legion::Runtime* m_runtime;

  static std::optional<Legion::coord_t>
  find_color(
    const std::vector<std::vector<Column::row_number_t>>& rowp,
    Column::row_number_t rn,
    bool sorted_selections);
};

template <typename D>
class TableT
  : public Table {
public:

  TableT(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    const std::vector<D>& row_axes,
    const std::vector<typename ColumnT<D>::Generator>& column_generators,
    const std::unordered_map<std::string, casacore::DataType>& kws =
    std::unordered_map<std::string, casacore::DataType>())
    : TableT(
      ctx,
      runtime,
      name,
      row_axes,
      column_generators.begin(),
      column_generators.end(),
      kws) {}

  template <typename GeneratorIter>
  TableT(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    const std::vector<D>& row_axes,
    GeneratorIter generator_first,
    GeneratorIter generator_last,
    const std::unordered_map<std::string, casacore::DataType>& kws =
    std::unordered_map<std::string, casacore::DataType>())
    : Table(ctx, runtime, name, static_cast<int>(D::last) + 1, row_axes, kws) {

    std::transform(
      generator_first,
      generator_last,
      std::inserter(m_columns, m_columns.end()),
      [&ctx, runtime](auto gen) {
        auto col = gen(ctx, runtime);
        return std::make_pair(col->name(), col);
      });

    assert(m_columns.size() > 0);

    auto col0 = (*m_columns.begin()).second;
    auto row_index_pattern = col0->row_index_pattern();
    auto num_rows = col0->num_rows();
    assert(
      std::all_of(
        m_columns.begin(),
        m_columns.end(),
        [&row_index_pattern, &num_rows, &row_axes](auto& nc) {
          auto cax = nc.second->axes();
          auto mm =
            std::mismatch(
              cax.begin(),
              cax.end(),
              row_axes.begin(),
              row_axes.end());
          return row_index_pattern == nc.second->row_index_pattern()
            && mm.second == row_axes.end()
            && num_rows == nc.second->num_rows();
        }));

    std::tie(std::ignore, m_min_rank_colname, m_max_rank_colname) =
      std::accumulate(
        m_columns.begin(),
        m_columns.end(),
        std::make_tuple(col0->rank(), col0->name(), col0->name()),
        [](auto &acc, auto& nc) {
          auto& [mrank, mincol, maxcol] = acc;
          auto& [name, col] = nc;
          if (col->rank() < mrank)
            return std::make_tuple(col->rank(), name, maxcol);
          if (col->rank() > mrank)
            return std::make_tuple(col->rank(), mincol, name);
          return acc;
        });
  }

  virtual ~TableT() {
  }

  std::vector<D>
  row_axes() const {
    std::vector<D> result;
    result.reserve(row_rank());
    std::transform(
      m_row_axes.begin(),
      m_row_axes.end(),
      std::back_inserter(result),
      [](auto& d) { return static_cast<D>(d); });
    return result;
  }

  Column::row_number_t
  row_number(const std::vector<Legion::coord_t>& index) const {
    return row_number(row_index_pattern(), index.begin(), index.end());
  }

  std::unordered_set<std::string>
  column_names() const override {
    std::unordered_set<std::string> result;
    std::transform(
      m_columns.begin(),
      m_columns.end(),
      std::inserter(result, result.end()),
      [](auto& col) {
        return col.first;
      });
    return result;
  }

  std::shared_ptr<Column>
  column(const std::string& name) const override {
    return m_columns.at(name);
  }

  std::shared_ptr<ColumnT<D>>
  columnT(const std::string& name) const {
    return m_columns.at(name);
  }

  std::unique_ptr<ColumnPartition>
  row_partition(
    const std::vector<std::vector<Column::row_number_t>>& rowp,
    bool include_unselected = false,
    bool sorted_selections = false) const override {

    auto rn_lr = column(min_rank_column_name())->logical_region();
    auto unselected_color = (include_unselected ? rowp.size() : -1);
    Legion::LogicalRegion color_lr;
    switch (rn_lr.get_dim()) {
    case 1:
      color_lr =
        row_colors<1>(
          m_context,
          m_runtime,
          rn_lr,
          rowp,
          unselected_color,
          sorted_selections);
      break;
    case 2:
      color_lr =
        row_colors<2>(
          m_context,
          m_runtime,
          rn_lr,
          rowp,
          unselected_color,
          sorted_selections);
      break;
    case 3:
      color_lr =
        row_colors<3>(
          m_context,
          m_runtime,
          rn_lr,
          rowp,
          unselected_color,
          sorted_selections);
      break;
    default:
      assert(false);
      break;
    }

    auto color_space =
      m_runtime->create_index_space(
        m_context,
        Legion::Rect<1>(0, rowp.size() - (include_unselected ? 0 : 1)));
    auto ipart =
      m_runtime->create_partition_by_field(
        m_context,
        color_lr,
        color_lr,
        0,
        color_space);
    m_runtime->destroy_index_space(m_context, color_space);
    m_runtime->destroy_logical_region(m_context, color_lr);
    return
      std::make_unique<ColumnPartitionT<D>>(
        m_context,
        m_runtime,
        ipart,
        columnT(min_rank_column_name())->axes());
  }

protected:

  const std::string&
  min_rank_column_name() const override {
    return m_min_rank_colname;
  }

  const std::string&
  max_rank_column_name() const override {
    return m_max_rank_colname;
  }

  template <int DIM>
  static Legion::LogicalRegion
  row_colors(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    Legion::LogicalRegion rn_lr,
    const std::vector<std::vector<Column::row_number_t>>& rowp,
    Legion::coord_t unselected_color,
    bool sorted_selections) {

    auto color_fs = runtime->create_field_space(ctx);
    auto color_fa = runtime->create_field_allocator(ctx, color_fs);
    color_fa.allocate_field(sizeof(Legion::Point<1>), 0);
    auto result =
      runtime->create_logical_region(ctx, rn_lr.get_index_space(), color_fs);
    auto color_task = Legion::InlineLauncher(
      Legion::RegionRequirement(result, WRITE_DISCARD, EXCLUSIVE, result));
    color_task.add_field(0);
    auto color_pr = runtime->map_region(ctx, color_task);
    const Legion::FieldAccessor<
      WRITE_DISCARD,
      Legion::Point<1>,
      DIM,
      Legion::coord_t,
      Legion::AffineAccessor<Legion::Point<1>, DIM, Legion::coord_t>,
      true> colors(color_pr, 0);

    auto rn_task = Legion::InlineLauncher(
      Legion::RegionRequirement(rn_lr, READ_ONLY, EXCLUSIVE, rn_lr));
    rn_task.add_field(Column::row_number_fid);
    auto rn_pr = runtime->map_region(ctx, rn_task);
    const Legion::FieldAccessor<
      READ_ONLY,
      Column::row_number_t,
      DIM,
      Legion::coord_t,
      Legion::AffineAccessor<Column::row_number_t, DIM, Legion::coord_t>,
      true> rns(rn_pr, Column::row_number_fid);

    Legion::DomainT<DIM> domain =
      runtime->get_index_space_domain(ctx, rn_lr.get_index_space());
    Legion::PointInDomainIterator<DIM> pid(domain, false);
    assert(pid());
    auto prev_row_number = rns[*pid];
    auto prev_color =
      find_color(rowp, prev_row_number, sorted_selections).
      value_or(unselected_color);
    colors[*pid] = prev_color;
    pid++;
    while (pid()) {
      auto row_number = rns[*pid];
      if (row_number != prev_row_number) {
        prev_row_number = rns[*pid];
        prev_color =
          find_color(rowp, prev_row_number, sorted_selections).
          value_or(unselected_color);
      }
      colors[*pid] = prev_color;
      pid++;
    }
    runtime->unmap_region(ctx, rn_pr);
    runtime->unmap_region(ctx, color_pr);
    return result;
  }

  std::unordered_map<std::string, std::shared_ptr<ColumnT<D>>> m_columns;

  std::string m_min_rank_colname;

  std::string m_max_rank_colname;
};


template <MSTables T>
static std::unique_ptr<TableT<typename MSTable<T>::Axes>>
from_ms(
  Legion::Context ctx,
  Legion::Runtime* runtime,
  const std::experimental::filesystem::path& path,
  const std::unordered_set<std::string>& column_selections) {

  auto builder = TableBuilder::from_ms<T>(path, column_selections);
  return
    std::make_unique<TableT<typename MSTable<T>::Axes>>(
      ctx,
      runtime,
      builder.name(),
      builder.row_axes(),
      builder.column_generators(),
      builder.keywords());
}

} // end namespace ms
} // end namespace legms

#endif // LEGMS_MS_TABLE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
