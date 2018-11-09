#include <algorithm>
#include <memory>
#include "Table.h"

using namespace legms;
using namespace legms::ms;
using namespace Legion;
using namespace std;

Table::Table(
  Context ctx,
  Runtime* runtime,
  const std::string& name,
  const std::vector<Column::Generator>& column_generators,
  const std::unordered_map<std::string, casacore::DataType>& kws)
  : Table(
    ctx,
    runtime,
    name,
    column_generators.begin(),
    column_generators.end(),
    kws) {
}

unordered_set<string>
Table::column_names() const {
  unordered_set<string> result;
  transform(
    m_columns.begin(),
    m_columns.end(),
    inserter(result, result.end()),
    [](auto& col) {
      return col.first;
    });
  return result;
}

IndexSpace
Table::index_space() const {
  return std::get<1>(*max_rank_column())->index_space();
}

vector<LogicalRegion>
Table::logical_regions(const vector<string>& colnames) const {

  vector<LogicalRegion> result;
  transform(
    colnames.begin(),
    colnames.end(),
    back_inserter(result),
    [this](auto& colname) {
      return column(colname)->logical_region();
    });
  return result;
}

std::vector<IndexPartition>
Table::row_block_index_partitions(
  const std::optional<IndexPartition>& ipart,
  const vector<std::string>& colnames,
  size_t block_size) const {

  IndexPartition block_partition;
  {
    auto nr = num_rows();
    std::vector<std::vector<Column::row_number_t>>
      rowp((nr + block_size - 1) / block_size);
    for (Column::row_number_t i = 0; i < nr; ++i)
      rowp[i / block_size].push_back(i);
    block_partition = row_partition(rowp, false, true);
  }

  vector<IndexPartition> result;
  if (ipart) {
    auto num_blocks = (num_rows() + block_size - 1) / block_size;
    map<IndexSpace, IndexPartition> subspace_partitions;
    for (size_t i = 0; i < num_blocks; ++i)
      subspace_partitions[
        m_runtime->get_index_subspace(m_context, block_partition, i)] =
        IndexPartition::NO_PART;
    m_runtime->create_cross_product_partitions(
      m_context,
      block_partition,
      ipart.value(),
      subspace_partitions);

    IndexSpace full_color_space;
    {
      Domain d =
        m_runtime->get_index_partition_color_space(m_context, ipart.value());
      std::vector<DomainPoint> pts;
      switch (d.get_dim()) {
      case 1:
        for (size_t i = 0; i < num_blocks; ++i)
          for (PointInDomainIterator<1> pid(d); pid(); pid++)
            pts.push_back(Point<2>(i, pid[0]));
        break;
      case 2:
        for (size_t i = 0; i < num_blocks; ++i)
          for (PointInDomainIterator<2> pid(d); pid(); pid++)
            pts.push_back(Point<3>(i, pid[0], pid[1]));
        break;
      default:
        assert(false);
        break;
      }
      full_color_space = m_runtime->create_index_space(m_context, pts);
    }
    IndexPartition full_partition =
      m_runtime->create_pending_partition(
        m_context,
        index_space(),
        full_color_space);
    std::for_each(
      subspace_partitions.begin(),
      subspace_partitions.end(),
      [this, &full_partition](auto& sp) {
        auto b = m_runtime->get_index_space_color(m_context, std::get<0>(sp));
        Domain d =
          m_runtime->get_index_partition_color_space(
            m_context,
            std::get<1>(sp));
        switch (d.get_dim()) {
        case 1:
          for (PointInDomainIterator<1> pid(d); pid(); pid++)
            m_runtime->create_index_space_union(
              m_context,
              full_partition,
              Point<2>(b, pid[0]),
              {m_runtime->get_index_subspace(
                  m_context,
                  std::get<1>(sp),
                  pid[0])});
          break;
        case 2:
          for (PointInDomainIterator<2> pid(d); pid(); pid++)
            m_runtime->create_index_space_union(
              m_context,
              full_partition,
              Point<3>(b, pid[0], pid[1]),
              {m_runtime->get_index_subspace(
                  m_context,
                  std::get<1>(sp),
                  *pid)});
          break;
        default:
          assert(false);
          break;
        }
      });

    result = index_partitions(full_partition, colnames);
    m_runtime->destroy_index_space(m_context, full_color_space);
    m_runtime->destroy_index_partition(m_context, full_partition);
  } else {
    result = index_partitions(block_partition, colnames);
  }
  m_runtime->destroy_index_partition(m_context, block_partition);
  return result;
}

optional<coord_t>
find_color(
  const std::vector<std::vector<Column::row_number_t>>& rowp,
  Column::row_number_t rn,
  bool sorted_selections) {

  optional<coord_t> result;
  if (sorted_selections) {
    for (size_t i = 0; !result && i < rowp.size(); ++i) {
      auto rns = rowp[i];
      if (binary_search(rns.begin(), rns.end(), rn))
        result = i;
    }
  } else {
    for (size_t i = 0; !result && i < rowp.size(); ++i) {
      auto rns = rowp[i];
      if (find(rns.begin(), rns.end(), rn) != rns.end())
        result = i;
    }
  }
  return result;
}

template <int DIM>
static LogicalRegion
row_colors(
  Context ctx,
  Runtime* runtime,
  IndexSpace index_space,
  LogicalRegion rn_lr,
  const std::vector<std::vector<Column::row_number_t>>& rowp,
  coord_t unselected_color,
  bool sorted_selections) {

  auto color_fs = runtime->create_field_space(ctx);
  auto color_fa = runtime->create_field_allocator(ctx, color_fs);
  color_fa.allocate_field(sizeof(Point<1>), 0);
  auto result =
    runtime->create_logical_region(ctx, index_space, color_fs);
  auto color_task = InlineLauncher(
    RegionRequirement(result, WRITE_DISCARD, EXCLUSIVE, result));
  color_task.add_field(0);
  auto color_pr = runtime->map_region(ctx, color_task);
  const FieldAccessor<
    WRITE_DISCARD,
    Point<1>,
    DIM,
    coord_t,
    AffineAccessor<Point<1>, DIM, coord_t>,
    true> colors(color_pr, 0);

  auto rn_task = InlineLauncher(
    RegionRequirement(rn_lr, READ_ONLY, EXCLUSIVE, rn_lr));
  rn_task.add_field(Column::row_number_fid);
  auto rn_pr = runtime->map_region(ctx, rn_task);
  const FieldAccessor<
    READ_ONLY,
    Column::row_number_t,
    DIM,
    coord_t,
    AffineAccessor<Column::row_number_t, DIM, coord_t>,
    true> rns(rn_pr, Column::row_number_fid);

  DomainT<DIM> domain = runtime->get_index_space_domain(ctx, index_space);
  PointInDomainIterator<DIM> pid(domain, false);
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

IndexPartition
Table::row_partition(
  const std::vector<std::vector<Column::row_number_t>>& rowp,
  bool include_unselected,
  bool sorted_selections) const {

  auto rn_lr = std::get<1>(*max_rank_column())->logical_region();
  auto unselected_color = (include_unselected ? rowp.size() : -1);
  LogicalRegion color_lr;
  switch (rank()) {
  case 1:
    color_lr =
      row_colors<1>(
        m_context,
        m_runtime,
        index_space(),
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
        index_space(),
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
        index_space(),
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
      Rect<1>(0, rowp.size() - (include_unselected ? 0 : 1)));
  auto result =
    m_runtime->create_partition_by_field(
      m_context,
      color_lr,
      color_lr,
      0,
      color_space);
  m_runtime->destroy_index_space(m_context, color_space);
  m_runtime->destroy_logical_region(m_context, color_lr);
  return result;
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
