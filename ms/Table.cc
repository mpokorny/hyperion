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
      return make_tuple(
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
  auto proj_lr = runtime->create_logical_region(ctx, is, fs);
  auto proj_lp = runtime->get_logical_partition(ctx, proj_lr, ipart);
  initialize_projections(ctx, runtime, proj_lr, proj_lp);

  auto reg_rank = runtime->get_index_space_depth(ctx, is);
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
  auto reg_rank = runtime->get_index_space_depth(ctx, lr.get_index_space());
  set<FieldID> fids;
  runtime->get_field_space_fields(ctx, lr.get_field_space(), fids);
  switch (reg_rank) {
  case 1:
    break;

  case 2:
    if (fids.count(1) > 0) {
      FillProjectionsTask<1, 2> f1(lr, lp, launch_space);
      f1.dispatch(ctx, runtime);
    }
    break;

  case 3:
    if (fids.count(1) > 0) {
      FillProjectionsTask<1, 3> f1(lr, lp, launch_space);
      f1.dispatch(ctx, runtime);
    }
    if (fids.count(2) > 0) {
      FillProjectionsTask<2, 3> f2(lr, lp, launch_space);
      f2.dispatch(ctx, runtime);
    }
    break;

  default:
    assert(false);
    break;
  }
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
