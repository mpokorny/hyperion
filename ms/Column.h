#ifndef LEGMS_MS_COLUMN_H_
#define LEGMS_MS_COLUMN_H_

#include <casacore/casa/Utilities/DataType.h>
#include "legion.h"

#include "utility.h"
#include "tree_index_space.h"
#include "WithKeywords.h"
#include "IndexTree.h"
#include "ColumnBuilder.h"

namespace legms {
namespace ms {

#define MAX_DIM 8

template <int PROJDIM, int REGIONDIM>
class FillProjectionsLauncher
  : public Legion::TaskLauncher {

public:

  static_assert(REGIONDIM <= MAX_DIM);
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

  FillProjectionsLauncher(
    const std::array<int, PROJDIM>* projected,
    Legion::IndexSpaceT<REGIONDIM> is,
    Legion::Context ctx,
    Legion::Runtime* runtime)
    : Legion::TaskLauncher(
      TASK_ID,
      TaskArgument(projected, sizeof(*projected))) {
    // 'projected' is used as TaskArgument, since that value is not copied
    // before launching the task, the user must ensure that it remains valid
    // until then
    Legion::FieldSpace fs = runtime->create_field_space(ctx);
    {
      auto fa = runtime->create_field_allocator(ctx, fs);
      auto fid = fa.allocate_field(sizeof(Legion::Point<PROJDIM>));
      runtime->attach_name(fs, fid, projections_field);
    }
    // user must destroy this logical region
    Legion::LogicalRegionT<REGIONDIM> m_lr =
      runtime->create_logical_region(ctx, is, fs);

    add_region_requirement(
      Legion::RegionRequirement(m_lr, 0, WRITE_DISCARD, EXCLUSIVE, m_lr));
  }

  Legion::LogicalRegionT<REGIONDIM>
  logical_region() const {
    return m_lr;
  }

  void
  dispatch(Legion::Context ctx, Legion::Runtime *runtime) {
    runtime->execute_task(ctx, *this);
  }

  static void
  base_impl(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *runtime) {

    const std::array<int, PROJDIM>* projected =
      static_cast<const std::array<int, PROJDIM>*>(task->args);

    auto lr = regions[0].get_logical_region();
    std::vector<Legion::FieldID> fids;
    runtime->get_field_space_fields(lr.get_field_space(), fids);

    const WDProjectionAccessor projections(regions[0], fids[0]);
    Legion::DomainT<REGIONDIM> domain =
      runtime->get_index_space_domain(ctx, lr.get_index_space());
    for (Legion::PointInDomainIterator<REGIONDIM> pid(domain); pid(); pid++) {
      Legion::coord_t pt[PROJDIM];
      for (size_t i = 0; i < PROJDIM; ++i)
        pt[i] = pid[(*projected)[i]];
      projections[*pid] = Legion::Point(pt);
    }
  }

  static void
  register_task(Legion::Runtime* runtime, Legion::TaskID tid) {
    TASK_ID = tid;
    Legion::TaskVariantRegistrar registrar(TASK_ID, TASK_NAME);
    registrar.add_constraint(
      Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    registrar.set_leaf();
    runtime->register_task_variant<base_impl>(registrar, TASK_NAME);
  }

  static constexpr const char *projections_field = "index_projections";

private:

  Legion::LogicalRegionT<REGIONDIM> m_lr;
};

class FillProjections {
public:

  static const int num_tasks = MAX_DIM * (MAX_DIM + 1) / 2;

  template <int PROJDIM, int REGIONDIM>
  static Legion::TaskID
  task_id(Legion::Runtime* runtime) {
    static_assert(REGIONDIM <= MAX_DIM);
    static_assert(PROJDIM <= REGIONDIM);
    return
      runtime->generate_library_task_ids("legms::FillProjections", num_tasks) +
      ((REGIONDIM - 1) * REGIONDIM) / 2 + PROJDIM - 1;
  }

  static void
  register_tasks(Legion::Runtime* runtime) {
    switch (MAX_DIM) {
    case 8:
      reg_tasks8<8>(runtime);
      break;
    case 7:
      reg_tasks7<7>(runtime);
      break;
    case 6:
      reg_tasks6<6>(runtime);
      break;
    case 5:
      reg_tasks5<5>(runtime);
      break;
    case 4:
      reg_tasks4<4>(runtime);
      break;
    case 3:
      reg_tasks3<3>(runtime);
      break;
    case 2:
      reg_tasks2<2>(runtime);
      break;
    case 1:
      reg_tasks1<1>(runtime);
      break;
    }
  }

private:

#define REG_TASKS(n) \
  template <int PROJDIM> \
  static void reg_tasks ## n (Legion::Runtime *runtime) {         \
    FillProjectionsLauncher<PROJDIM, n>::register_task(           \
      runtime,                                                    \
      task_id<PROJDIM, n>(runtime));                              \
    reg_tasks##n<PROJDIM - 1>(runtime);                           \
  }

  REG_TASKS(8)
  REG_TASKS(7)
  REG_TASKS(6)
  REG_TASKS(5)
  REG_TASKS(4)
  REG_TASKS(3)
  REG_TASKS(2)
  REG_TASKS(1)
#undef REG_TASKS
};

template <>
void FillProjections::reg_tasks8<0>(Legion::Runtime*) {}
template <>
void FillProjections::reg_tasks7<0>(Legion::Runtime*) {}
template <>
void FillProjections::reg_tasks6<0>(Legion::Runtime*) {}
template <>
void FillProjections::reg_tasks5<0>(Legion::Runtime*) {}
template <>
void FillProjections::reg_tasks4<0>(Legion::Runtime*) {}
template <>
void FillProjections::reg_tasks3<0>(Legion::Runtime*) {}
template <>
void FillProjections::reg_tasks2<0>(Legion::Runtime*) {}
template <>
void FillProjections::reg_tasks1<0>(Legion::Runtime*) {}

class Column
  : public WithKeywords {
public:

  Column(const ColumnBuilder& builder)
    : WithKeywords(builder.keywords())
    , m_name(builder.name())
    , m_datatype(builder.datatype())
    , m_row_rank(builder.row_rank())
    , m_rank(builder.rank())
    , m_row_index_shape(builder.row_index_shape())
    , m_index_tree(builder.index_tree()) {
  }

  const std::string&
  name() const {
    return m_name;
  }

  casacore::DataType
  datatype() const {
    return m_datatype;
  }

  const IndexTreeL
  row_index_tree() const {
    return m_index_tree.pruned(m_row_rank - 1);
  }

  const IndexTreeL&
  index_tree() const {
    return m_index_tree;
  }

  const IndexTreeL&
  row_index_shape() const {
    return m_row_index_shape;
  }

  unsigned
  row_rank() const {
    return m_row_rank;
  }

  unsigned
  rank() const {
    return m_rank;
  }

  std::optional<Legion::IndexSpace>
  index_space(Legion::Context ctx, Legion::Runtime* runtime) const {
    if (!m_index_space)
      m_index_space = legms::tree_index_space(m_index_tree, ctx, runtime);
    return m_index_space;
  }

  Legion::FieldID
  add_field(
    Legion::Runtime *runtime,
    Legion::FieldSpace fs,
    Legion::FieldAllocator fa,
    Legion::FieldID field_id = AUTO_GENERATE_ID) const {

    Legion::FieldID result = legms::add_field(m_datatype, fa, field_id);
    runtime->attach_name(fs, result, name().c_str());
    return result;
  }

  template <int DIM>
  std::optional<Legion::LogicalRegion>
  index_projections(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::array<int, DIM>& projected) const {

    assert(DIM <= m_rank);

    assert(
      std::all_of(
        projected.begin(),
        projected.end(),
        [this](auto& d) { return 0 <= d && d < m_rank; }));

    if (m_index_tree == IndexTreeL())
      return std::nullopt;

    Legion::LogicalRegion result;
    switch (m_rank) {
    case 1: {
      FillProjectionsLauncher<DIM, 1> fill_projections(
        &projected,
        index_space(ctx, runtime).value(),
        ctx,
        runtime);
      fill_projections.dispatch(ctx, runtime);
      result = fill_projections.logical_region();
      break;
    }
    case 2: {
      FillProjectionsLauncher<DIM, 2> fill_projections(
        &projected,
        index_space(ctx, runtime).value(),
        ctx,
        runtime);
      fill_projections.dispatch(ctx, runtime);
      result = fill_projections.logical_region();
      break;
    }
    case 3: {
      FillProjectionsLauncher<DIM, 3> fill_projections(
        &projected,
        index_space(ctx, runtime).value(),
        ctx,
        runtime);
      fill_projections.dispatch(ctx, runtime);
      result = fill_projections.logical_region();
      break;
    }
    case 4: {
      FillProjectionsLauncher<DIM, 4> fill_projections(
        &projected,
        index_space(ctx, runtime).value(),
        ctx,
        runtime);
      fill_projections.dispatch(ctx, runtime);
      result = fill_projections.logical_region();
      break;
    }
    case 5: {
      FillProjectionsLauncher<DIM, 5> fill_projections(
        &projected,
        index_space(ctx, runtime).value(),
        ctx,
        runtime);
      fill_projections.dispatch(ctx, runtime);
      result = fill_projections.logical_region();
      break;
    }
    case 6: {
      FillProjectionsLauncher<DIM, 6> fill_projections(
        &projected,
        index_space(ctx, runtime).value(),
        ctx,
        runtime);
      fill_projections.dispatch(ctx, runtime);
      result = fill_projections.logical_region();
      break;
    }
    case 7: {
      FillProjectionsLauncher<DIM, 7> fill_projections(
        &projected,
        index_space(ctx, runtime).value(),
        ctx,
        runtime);
      fill_projections.dispatch(ctx, runtime);
      result = fill_projections.logical_region();
      break;
    }
    case 8: {
      FillProjectionsLauncher<DIM, 8> fill_projections(
        &projected,
        index_space(ctx, runtime).value(),
        ctx,
        runtime);
      fill_projections.dispatch(ctx, runtime);
      result = fill_projections.logical_region();
      break;
    }
    }

    return result;
  }

private:

  std::string m_name;

  casacore::DataType m_datatype;

  unsigned m_row_rank;

  unsigned m_rank;

  mutable std::optional<Legion::IndexSpace> m_index_space;

  IndexTreeL m_row_index_shape;

  IndexTreeL m_index_tree;
};

} // end namespace ms
} // end namespace legms

#endif // LEGMS_MS_COLUMN_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
