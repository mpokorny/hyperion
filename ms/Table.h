#ifndef LEGMS_MS_TABLE_H_
#define LEGMS_MS_TABLE_H_

#include <algorithm>
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

template <int PROJDIM, int REGIONDIM>
class FillProjectionsTask
  : public Legion::IndexTaskLauncher {

public:

  static_assert(PROJDIM <= REGIONDIM);

  static Legion::TaskID TASK_ID;
  constexpr static const char * const TASK_NAME =
    "fill_projections"; // #PROJDIM "-" #REGIONDIM;

  typedef Legion::FieldAccessor<
    WRITE_DISCARD,
    Legion::Point<PROJDIM>,
    REGIONDIM,
    Legion::coord_t,
    Legion::AffineAccessor<Legion::Point<PROJDIM>, REGIONDIM, Legion::coord_t>,
    false> WDProjectionAccessor;

  FillProjectionsTask(
    Legion::LogicalRegion lr,
    Legion::LogicalPartition lp,
    Legion::IndexSpace launch_space)
    : Legion::IndexTaskLauncher(
      TASK_ID,
      launch_space,
      Legion::TaskArgument(),
      Legion::ArgumentMap()) {

    add_region_requirement(
      Legion::RegionRequirement(lp, 0, WRITE_DISCARD, EXCLUSIVE, lr));
  }

  void
  dispatch(Legion::Context ctx, Legion::Runtime *runtime) {
    runtime->execute_index_space(ctx, *this);
  }

  static void
  base_impl(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *runtime) {

    const WDProjectionAccessor projections(regions[0], PROJDIM);
    Legion::DomainT<REGIONDIM> domain =
      runtime->get_index_space_domain(
        ctx,
        task->regions[0].region.get_index_space());
    for (Legion::PointInDomainIterator<REGIONDIM> pid(domain); pid(); pid++) {
      Legion::coord_t pt[PROJDIM];
      for (size_t i = 0; i < PROJDIM; ++i)
        pt[i] = pid[i];
      projections[*pid] = Legion::Point<PROJDIM>(pt);
    }
  }

  static void
  register_task(Legion::Runtime* runtime, Legion::TaskID tid) {
    TASK_ID = tid;
    Legion::TaskVariantRegistrar registrar(TASK_ID, TASK_NAME);
    registrar.add_constraint(
      Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    registrar.set_leaf();
    runtime->register_task_variant<base_impl>(registrar);
  }
};

template <int REGIONDIM>
class FillProjectionsTask<1, REGIONDIM>
  : public Legion::IndexTaskLauncher {

public:

  static_assert(1 <= REGIONDIM);

  static Legion::TaskID TASK_ID;
  constexpr static const char * const TASK_NAME =
    "fill_projections"; // #PROJDIM "-" #REGIONDIM;

  typedef Legion::FieldAccessor<
    WRITE_DISCARD,
    Legion::Point<1>,
    REGIONDIM,
    Legion::coord_t,
    Legion::AffineAccessor<Legion::Point<1>, REGIONDIM, Legion::coord_t>,
    false> WDProjectionAccessor;

  FillProjectionsTask(
    Legion::LogicalRegion lr,
    Legion::LogicalPartition lp,
    Legion::IndexSpace launch_space)
    : Legion::IndexTaskLauncher(
      TASK_ID,
      launch_space,
      Legion::TaskArgument(),
      Legion::ArgumentMap()) {

    add_region_requirement(
      Legion::RegionRequirement(lp, 0, WRITE_DISCARD, EXCLUSIVE, lr));
  }

  void
  dispatch(Legion::Context ctx, Legion::Runtime *runtime) {
    runtime->execute_index_space(ctx, *this);
  }

  static void
  base_impl(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *runtime) {

    const WDProjectionAccessor projections(regions[0], 1);
    Legion::DomainT<REGIONDIM> domain =
      runtime->get_index_space_domain(
        ctx,
        task->regions[0].region.get_index_space());
    for (Legion::PointInDomainIterator<REGIONDIM> pid(domain); pid(); pid++)
      projections[*pid] = Legion::Point<1>(pid[0]);
  }

  static void
  register_task(Legion::Runtime* runtime, Legion::TaskID tid) {
    TASK_ID = tid;
    Legion::TaskVariantRegistrar registrar(TASK_ID, TASK_NAME);
    registrar.add_constraint(
      Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    registrar.set_leaf();
    runtime->register_task_variant<base_impl>(registrar);
  }
};

class FillProjectionsTasks {
public:

  static const int MAX_DIM = 3;

  static const int num_tasks = MAX_DIM * (MAX_DIM + 1) / 2;

  template <int PROJDIM, int REGIONDIM>
  static Legion::TaskID
  task_id(Legion::Runtime* runtime) {
    static_assert(REGIONDIM <= MAX_DIM);
    static_assert(PROJDIM <= REGIONDIM);
    return
      runtime->generate_library_task_ids(
        "legms::FillProjectionsTasks",
        num_tasks)
      + ((REGIONDIM - 1) * REGIONDIM) / 2 + PROJDIM - 1;
  }

  static void
  register_tasks(Legion::Runtime* runtime) {
    switch (MAX_DIM) {
    case 3:
      reg_tasks3<3>(runtime);
    case 2:
      reg_tasks2<2>(runtime);
    case 1:
      reg_tasks1<1>(runtime);
      break;
    }
  }

private:

#define REG_TASKS(n) \
  template <int PROJDIM> \
  static void reg_tasks ## n (Legion::Runtime *runtime) {         \
    FillProjectionsTask<PROJDIM, n>::register_task(                   \
      runtime,                                                    \
      task_id<PROJDIM, n>(runtime));                              \
    reg_tasks##n<PROJDIM - 1>(runtime);                           \
  }

  REG_TASKS(3)
  REG_TASKS(2)
  REG_TASKS(1)
#undef REG_TASKS
};

template <> inline
void FillProjectionsTasks::reg_tasks3<0>(Legion::Runtime*) {}
template <> inline
void FillProjectionsTasks::reg_tasks2<0>(Legion::Runtime*) {}
template <> inline
void FillProjectionsTasks::reg_tasks1<0>(Legion::Runtime*) {}

template <int PROJDIM, int REGIONDIM>
Legion::TaskID FillProjectionsTask<PROJDIM, REGIONDIM>::TASK_ID;

template <int REGIONDIM>
Legion::TaskID FillProjectionsTask<1, REGIONDIM>::TASK_ID;

class Table
  : public WithKeywords {
public:

  Table(const TableBuilder& builder)
    : WithKeywords(builder.keywords())
    , m_name(builder.name()) {

    assert(builder.m_columns.size() > 0);
    std::transform(
      builder.m_columns.begin(),
      builder.m_columns.end(),
      std::inserter(m_columns, m_columns.end()),
      [](auto& cb) {
        return std::make_pair(
          cb.first,
          std::shared_ptr<Column>(new Column(*cb.second)));
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
  row_index_shape() const {
    return std::get<1>(*m_columns.begin())->row_index_shape();
  }

  size_t
  row_number(const std::vector<Legion::coord_t>& index) const {
    return row_number(row_index_shape(), index.begin(), index.end());
  }

  std::unordered_set<std::string>
  column_names() const {
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
  column(const std::string& name) const {
    return m_columns.at(name);
  }

  std::shared_ptr<Column>
  column(Legion::FieldID fid) const {
    auto ncp = std::find_if(
      m_columns.begin(),
      m_columns.end(),
      [&fid](auto& nc) {
        return std::get<1>(nc)->field_id().value_or(fid - 1) == fid;
      });
    if (ncp != m_columns.end())
      return std::get<1>(*ncp);
    else
      return std::shared_ptr<Column>();
  }

  std::optional<Legion::IndexSpace>
  index_space(Legion::Context ctx, Legion::Runtime* runtime) const {
    return std::get<1>(*max_rank_column())->index_space(ctx, runtime);
  }

  std::vector<std::tuple<Legion::LogicalRegion, Legion::FieldID>>
  logical_regions(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::vector<std::string>& colnames) const {

    std::vector<std::tuple<Legion::LogicalRegion, Legion::FieldID>> result;
    std::transform(
      colnames.begin(),
      colnames.end(),
      std::back_inserter(result),
      [this, &ctx, runtime](auto& colname) {
        auto fs = runtime->create_field_space(ctx);
        auto fa = runtime->create_field_allocator(ctx, fs);
        auto col = column(colname);
        auto fid = col->add_field(runtime, fs, fa);
        return std::make_tuple(
          runtime->create_logical_region(
            ctx,
            col->index_space(ctx, runtime).value(),
            fs),
          std::move(fid));
      });
    return result;
  }

  std::vector<Legion::IndexPartition>
  index_partitions(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const Legion::IndexPartition& ipart,
    const std::vector<std::string>& colnames) const {

    auto is = index_space(ctx, runtime).value();

    assert(runtime->get_parent_index_space(ctx, ipart) == is);

    std::set<unsigned> ranks;
    std::transform(
      colnames.begin(),
      colnames.end(),
      std::inserter(ranks, ranks.end()),
      [this](auto& colname) { return column(colname)->rank(); });
    auto fs = runtime->create_field_space(ctx);
    auto fa = runtime->create_field_allocator(ctx, fs);
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
    auto proj_lr = runtime->create_logical_region(ctx, is, fs);
    auto proj_lp = runtime->get_logical_partition(ctx, proj_lr, ipart);
    initialize_projections(ctx, runtime, proj_lr, proj_lp);

    auto reg_rank = runtime->get_index_space_depth(ctx, is);
    auto color_space =
      runtime->get_index_partition_color_space_name(ctx, ipart);
    std::vector<Legion::IndexPartition> result;
    std::transform(
      colnames.begin(),
      colnames.end(),
      std::back_inserter(result),
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

  template <typename IndexIter, typename IndexIterEnd>
  static size_t
  row_number(
    const IndexTreeL& row_shape,
    IndexIter index,
    IndexIterEnd index_end) {

    if (row_shape == IndexTreeL())
      return 0;
    assert(index != index_end);

    int lo, hi;
    std::tie(lo, hi) = row_shape.index_range();
    if (*index < lo)
      return 0;
    size_t result = (*index - lo) / (hi - lo + 1) * row_shape.size();
    auto i0 = (*index - lo) % (hi - lo + 1);
    auto ch = row_shape.children().begin();
    auto ch_end = row_shape.children().end();
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
    Legion::LogicalPartition lp) {

    auto launch_space =
      runtime->get_index_partition_color_space_name(
        ctx,
        lp.get_index_partition());
    auto reg_rank = runtime->get_index_space_depth(ctx, lr.get_index_space());
    std::set<Legion::FieldID> fids;
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

  std::string m_name;

  std::unordered_map<std::string, std::shared_ptr<Column>> m_columns;
};

} // end namespace ms
} // end namespace legms

#endif // LEGMS_MS_TABLE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:

