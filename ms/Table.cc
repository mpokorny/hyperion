#include <algorithm>
#include "Table.h"
#include "FillProjectionsTask.h"

using namespace legms::ms;
using namespace Legion;
using namespace std;

Table::Table(
  Legion::Context ctx,
  Legion::Runtime* runtime,
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

vector<tuple<LogicalRegion, FieldID>>
Table::logical_regions(const vector<string>& colnames) const {

  std::lock_guard<decltype(m_logical_regions_mutex)>
    lock(m_logical_regions_mutex);

  vector<tuple<LogicalRegion, FieldID>> result;
  transform(
    colnames.begin(),
    colnames.end(),
    back_inserter(result),
    [this](auto& colname) {
      if (m_logical_regions.count(colname) == 0) {
        auto fs = m_runtime->create_field_space(m_context);
        auto fa = m_runtime->create_field_allocator(m_context, fs);
        auto col = column(colname);
        auto fid = col->add_field(fs, fa);
        m_logical_regions[colname] =
          make_tuple(
            m_runtime->create_logical_region(m_context, col->index_space(), fs),
            move(fid));
      }
      return m_logical_regions[colname];
    });
  return result;
}

std::tuple<std::vector<IndexPartition>, IndexPartition>
Table::row_block_index_partitions(
  const std::optional<IndexPartition>& ipart,
  const vector<std::string>& colnames,
  size_t block_size) const {

  FieldSpace block_fs = m_runtime->create_field_space(m_context);
  auto fa = m_runtime->create_field_allocator(m_context, block_fs);
  auto block_fid = fa.allocate_field(sizeof(Point<1>));
  auto is = index_space();
  LogicalRegion block_lr =
    m_runtime->create_logical_region(m_context, is, block_fs);
  InlineLauncher block_launcher(
    RegionRequirement(
      block_lr,
      WRITE_DISCARD,
      EXCLUSIVE,
      block_lr));
  block_launcher.add_field(block_fid);
  PhysicalRegion block_pr = m_runtime->map_region(m_context, block_launcher);
  switch (is.get_dim()) {
  case 1: {
    const FieldAccessor<WRITE_DISCARD, Point<1>, 1> blocks(block_pr, block_fid);
    for (PointInDomainIterator<1> pid(m_runtime->get_index_space_domain(is));
         pid();
         pid++)
      blocks[*pid] = *pid / block_size;
    break;
  }
  case 2: {
    const FieldAccessor<WRITE_DISCARD, Point<1>, 2> blocks(block_pr, block_fid);
    for (PointInDomainIterator<2> pid(m_runtime->get_index_space_domain(is));
         pid();
         pid++) {
      std::array<coord_t, 2> idx{pid[0], pid[1]};
      blocks[*pid] =
        Table::row_number(
          row_index_pattern(), idx.begin(), idx.end()) / block_size;
    }
    break;
  }
  case 3: {
    const FieldAccessor<WRITE_DISCARD, Point<1>, 3> blocks(block_pr, block_fid);
    for (PointInDomainIterator<3> pid(m_runtime->get_index_space_domain(is));
         pid();
         pid++) {
      std::array<coord_t, 3> idx{pid[0], pid[1], pid[2]};
      blocks[*pid] =
        Table::row_number(
          row_index_pattern(), idx.begin(), idx.end()) / block_size;
    }
    break;
  }
  default:
    assert(false);
    break;
  }
  m_runtime->unmap_region(m_context, block_pr);
  auto num_blocks = (num_rows() + block_size - 1) / block_size;
  IndexSpace block_color_space =
    m_runtime->create_index_space(m_context, Rect<1>(0, num_blocks - 1));
  IndexPartition block_partition =
    m_runtime->create_partition_by_field(
      m_context,
      block_lr,
      block_lr,
      block_fid,
      block_color_space);
  std::tuple<vector<IndexPartition>, IndexPartition> result;
  if (ipart) {
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
      m_runtime->create_pending_partition(m_context, is, full_color_space);
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

    auto projected_partitions = index_partitions(full_partition, colnames);
    result = std::make_tuple(projected_partitions, full_partition);
    m_runtime->destroy_index_space(m_context, full_color_space);
  } else {
    result =
      std::make_tuple(
        index_partitions(block_partition, colnames),
        block_partition);
  }
  return result;
}

void
Table::initialize_projections(
  Context ctx,
  Runtime* runtime,
  LogicalRegion lr,
  LogicalPartition lp) {

  auto launch_space =
    runtime->get_index_partition_color_space_name(
      ctx,
      lp.get_index_partition());
  auto reg_rank = lr.get_index_space().get_dim();
  set<FieldID> fids;
  runtime->get_field_space_fields(ctx, lr.get_field_space(), fids);
  switch (reg_rank) {
  case 1:
    break;

  case 2:
    if (fids.count(1) > 0) {
      FillProjectionsTask<1, 2> f1(lr, lp, 1, launch_space);
      f1.dispatch(ctx, runtime);
    }
    break;

  case 3:
    if (fids.count(1) > 0) {
      FillProjectionsTask<1, 3> f1(lr, lp, 1, launch_space);
      f1.dispatch(ctx, runtime);
    }
    if (fids.count(2) > 0) {
      FillProjectionsTask<2, 3> f2(lr, lp, 2, launch_space);
      f2.dispatch(ctx, runtime);
    }
    break;

  default:
    assert(false);
    break;
  }
  runtime->destroy_index_space(ctx, launch_space);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
