#include <algorithm>
#include "Table.h"

using namespace legms::ms;
using namespace Legion;
using namespace std;

Table::Table(const TableBuilder& builder)
  : WithKeywords(builder.keywords())
  , m_name(builder.name()) {

  assert(builder.m_columns.size() > 0);
  transform(
    builder.m_columns.begin(),
    builder.m_columns.end(),
    inserter(m_columns, m_columns.end()),
    [](auto& cb) {
      return make_pair(cb.first, shared_ptr<Column>(new Column(*cb.second)));
    });
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

shared_ptr<Column>
Table::column(FieldID fid) const {
  auto ncp = find_if(
    m_columns.begin(),
    m_columns.end(),
    [&fid](auto& nc) {
      return get<1>(nc)->field_id().value_or(fid - 1) == fid;
    });
  if (ncp != m_columns.end())
    return get<1>(*ncp);
  else
    return shared_ptr<Column>();
}

std::optional<IndexSpace>
Table::index_space(Context ctx, Runtime* runtime) const {
  return std::get<1>(*max_rank_column())->index_space(ctx, runtime);
}

vector<tuple<LogicalRegion, FieldID>>
Table::logical_regions(
  Context ctx,
  Runtime* runtime,
  const vector<string>& colnames) const {

  vector<tuple<LogicalRegion, FieldID>> result;
  transform(
    colnames.begin(),
    colnames.end(),
    back_inserter(result),
    [this, &ctx, runtime](auto& colname) {
      auto fs = runtime->create_field_space(ctx);
      auto fa = runtime->create_field_allocator(ctx, fs);
      auto col = column(colname);
      auto fid = col->add_field(runtime, fs, fa);
      return
        make_tuple(
          runtime->create_logical_region(
            ctx,
            col->index_space(ctx, runtime).value(),
            fs),
          move(fid));
    });
  return result;
}

vector<IndexPartition>
Table::index_partitions(
  Context ctx,
  Runtime* runtime,
  const IndexPartition& ipart,
  const vector<string>& colnames) const {

  auto is = index_space(ctx, runtime).value();

  assert(runtime->get_parent_index_space(ctx, ipart) == is);

  set<unsigned> ranks;
  transform(
    colnames.begin(),
    colnames.end(),
    inserter(ranks, ranks.end()),
    [this](auto& colname) { return column(colname)->rank(); });
  auto fs = runtime->create_field_space(ctx);
  {
    auto fa = runtime->create_field_allocator(ctx, fs);
    for_each(
      ranks.begin(),
      ranks.end(),
      [&fa](auto r) {
        switch (r) {
        case 1:
          fa.allocate_field(sizeof(Point<1>), 1);
          break;
        case 2:
          fa.allocate_field(sizeof(Point<2>), 2);
          break;
        case 3:
          fa.allocate_field(sizeof(Point<3>), 3);
          break;
        default:
          assert(false);
          break;
        }
      });
  }
  auto proj_lr = runtime->create_logical_region(ctx, is, fs);
  auto proj_lp = runtime->get_logical_partition(ctx, proj_lr, ipart);
  initialize_projections(ctx, runtime, proj_lr, proj_lp);
  runtime->destroy_field_space(ctx, fs);

  unsigned  reg_rank = is.get_dim();
  auto color_space =
    runtime->get_index_partition_color_space_name(ctx, ipart);
  vector<IndexPartition> result;
  transform(
    colnames.begin(),
    colnames.end(),
    back_inserter(result),
    [&, this](auto& colname) {
      auto col = column(colname);
      auto rank = col->rank();
      if (rank < reg_rank)
        return runtime->create_partition_by_image(
          ctx,
          col->index_space(ctx, runtime).value(),
          proj_lp,
          proj_lr,
          rank,
          color_space);
      else
        return ipart;
    });
  //runtime->destroy_logical_partition(ctx, proj_lp);
  runtime->destroy_logical_region(ctx, proj_lr);
  runtime->destroy_index_space(ctx, color_space);
  return result;
}

std::tuple<std::vector<IndexPartition>, IndexPartition>
Table::row_block_index_partitions(
  Context ctx,
  Runtime* runtime,
  const std::optional<IndexPartition>& ipart,
  const vector<std::string>& colnames,
  size_t block_size) const {

  FieldSpace block_fs = runtime->create_field_space(ctx);
  auto fa = runtime->create_field_allocator(ctx, block_fs);
  auto block_fid = fa.allocate_field(sizeof(Point<1>));
  auto is = index_space(ctx, runtime).value();
  LogicalRegion block_lr = runtime->create_logical_region(ctx, is, block_fs);
  InlineLauncher block_launcher(
    RegionRequirement(
      block_lr,
      WRITE_DISCARD,
      EXCLUSIVE,
      block_lr));
  block_launcher.add_field(block_fid);
  PhysicalRegion block_pr = runtime->map_region(ctx, block_launcher);
  switch (is.get_dim()) {
  case 1: {
    const FieldAccessor<WRITE_DISCARD, Point<1>, 1> blocks(block_pr, block_fid);
    for (PointInDomainIterator<1> pid(runtime->get_index_space_domain(is));
         pid();
         pid++)
      blocks[*pid] = *pid / block_size;
    break;
  }
  case 2: {
    const FieldAccessor<WRITE_DISCARD, Point<1>, 2> blocks(block_pr, block_fid);
    for (PointInDomainIterator<2> pid(runtime->get_index_space_domain(is));
         pid();
         pid++) {
      std::array<coord_t, 2> idx{pid[0], pid[1]};
      blocks[*pid] =
        Table::row_number(
          row_index_shape(), idx.begin(), idx.end()) / block_size;
    }
    break;
  }
  case 3: {
    const FieldAccessor<WRITE_DISCARD, Point<1>, 3> blocks(block_pr, block_fid);
    for (PointInDomainIterator<3> pid(runtime->get_index_space_domain(is));
         pid();
         pid++) {
      std::array<coord_t, 3> idx{pid[0], pid[1], pid[2]};
      blocks[*pid] =
        Table::row_number(
          row_index_shape(), idx.begin(), idx.end()) / block_size;
    }
    break;
  }
  default:
    assert(false);
    break;
  }
  runtime->unmap_region(ctx, block_pr);
  auto num_blocks = (num_rows() + block_size - 1) / block_size;
  IndexSpace block_color_space =
    runtime->create_index_space(ctx, Rect<1>(0, num_blocks - 1));
  IndexPartition block_partition =
    runtime->create_partition_by_field(
      ctx,
      block_lr,
      block_lr,
      block_fid,
      block_color_space);
  std::tuple<vector<IndexPartition>, IndexPartition> result;
  if (ipart) {
    map<IndexSpace, IndexPartition> all_partitions;
    all_partitions[block_color_space] = IndexPartition::NO_PART;
    runtime->create_cross_product_partitions(
      ctx,
      block_partition,
      ipart.value(),
      all_partitions);

    auto projected_partitions =
      index_partitions(
        ctx,
        runtime,
        all_partitions[block_color_space],
        colnames);
    result =
      std::make_tuple(
        projected_partitions,
        all_partitions[block_color_space]);
  } else {
    result =
      std::make_tuple(
        index_partitions(ctx, runtime, block_partition, colnames),
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
  //runtime->destroy_index_space(ctx, launch_space);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
