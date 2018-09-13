#ifndef LEGMS_MS_MAIN_H_
#define LEGMS_MS_MAIN_H_

#include <algorithm>
#include <cassert>
#include <functional>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <casacore/casa/Utilities/DataType.h>
#include <casacore/tables/Tables.h>
#include "legion.h"

#include "utility.h"

namespace legms {
namespace ms {

typedef IndexTree<coord_t> IndexTreeL;

#define MAX_DIM 8

template <int PROJDIM, int REGIONDIM>
class FillProjectionsLauncher
  : public Legion::TaskLauncher {

public:

  static_assert(REGIONDIM <= MAX_DIM);
  static_assert(PROJDIM <= REGIONDIM);

  static Legion::TaskID TASK_ID;
  constexpr static const char * const TASK_NAME =
    "fill_projections" #PROJDIM "-" #REGIONDIM;

  typedef Legion::FieldAccessor<
    WRITE_DISCARD,
    Legion::Point<PROJDIM>,
    REGIONDIM,
    coord_t,
    Legion::Realm::AffineAccessor<Legion::Point<PROJDIM>, REGIONDIM, coord_t>,
    false> WDProjectionAccessor;

  FillProjectionsLauncher(
    const std::array<int, PROJDIM>* projected,
    Legion::IndexSpaceT<REGIONDIM> is,
    Legion::Context ctx,
    Legion::Runtime* runtime)
    : Legion::TaskLauncher(
      TASK_ID,
      TaskArgument(sizeof(*projected), projected)) {
    // 'projected' is used as TaskArgument, since that value is not copied
    // before launching the task, the user must ensure that it remains valid
    // until then
    Legion::FieldSpace fs = runtime->create_field_space(ctx);
    {
      auto fa = runtime->create_field_allocator(ctx, fs);
      auto fid = fa.allocate_field(sizeof(Legion::Point<PROJDIM>));
      runtime->attach_name(ctx, fs, fid, projections_field);
    }
    // user must destroy this logical region
    Legion::LogicalRegionT<COLDIM> m_lr =
      runtime->create_logical_region(ctx, is, fs);

    add_region_requirement(
      RegionRequirement(m_lr, 0, WRITE_DISCARD, EXCLUSIVE, m_lr));
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
    const std::vector<PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *runtime) {

    const std::array<int, PROJDIM>* projected =
      static_cast<const std::array<int, PROJDIM>*>(task->args);

    auto lr = regions[0].get_logical_region();
    std::vector<Legion::FieldID> fids;
    runtime->get_field_space_fields(lr.get_field_space(), fids);

    const WDProjectionAccessor projections(regions[0], fids[0]);
    Legion::Domain domain =
      runtime->get_index_space_domain(ctx, lr.get_index_space());
    for (Legion::PointInDomainIterator pid(domain); pid(); pid++) {
      coord_t pt[PROJDIM];
      for (size_t i = 0; i < PROJDIM; ++i)
        pt[i] = pid[projected[i]];
      projections[*pid] = Legion::Point<PROJDIM>(pt);
    }
  }

  static void
  register_task(Legion::Runtime* runtime, Legion::TaskID tid) {
    TASK_ID = tid;
    TaskVariantRegistrar registrar(TASK_ID, TASK_NAME);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    runtime->register_task_variant<base_impl>(registrar, TASK_NAME);
  }

  static const char *projections_field = "index_projections";

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
  template <int PROJDIM>\
  reg_tasks##n(Legion::Runtime *runtime) {\
    FillProjectionsLauncher<n, PROJDIM>::register_task(\
      runtime,\
      task_id<n, PROJDIM>());\
    reg_tasks##n<PROJDIM - 1>(runtime);\
  }\
  template <>\
  reg_tasks##n<0>(Legion::Runtime* runtime) {\
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

class WithKeywordsBuilder {
public:

  WithKeywordsBuilder() {}

  void
  add_keyword(const std::string& name, casacore::DataType datatype) {
    m_keywords.emplace(name, datatype);
  }

  const std::unordered_set<std::tuple<std::string, casacore::DataType>>&
  keywords() const {
    return m_keywords;
  }

private:

  std::unordered_set<std::tuple<std::string, casacore::DataType>> m_keywords;

};

class ColumnBuilder
  : public WithKeywordsBuilder {
public:

  ColumnBuilder(
    const std::string& name,
    casacore::DataType datatype,
    unsigned row_rank,
    unsigned element_rank,
    const IndexTreeL& row_index_shape,
    const IndexTreeL& index_tree)
    : WithKeywordsBuilder()
    , m_name(name)
    , m_datatype(datatype)
    , m_row_rank(row_rank)
    , m_rank(row_rank + element_rank)
    , m_row_index_shape(row_index_shape)
    , m_row_index_iterator(row_index_shape)
    , m_index_tree(index_tree) {

    assert(index_tree == IndexTreeL()
           || index_tree.rank().value_or(0) == m_rank);
    assert(row_index_shape.rank().value_or(0) == row_rank);
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
    return m_index_tree.pruned_to(m_row_rank - 1);
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
  rank() const {
    return m_rank;
  }

  unsigned
  row_rank() const {
    return m_row_rank;
  }

  virtual void
  add_row(const std::any&) = 0;

protected:

  void
  set_next_row(const IndexTreeL& element_tree) {
    std::vector<size_t> row_index = *m_row_index_iterator;
    ++m_row_index_iterator;
    IndexTreeL result =
      std::accumulate(
        row_index.rend(),
        row_index.rbegin(),
        element_tree,
        [](const auto& t, const auto& i) {
          return IndexTreeL({{i, 1, t}});
        });
    m_index_tree = std::move(m_index_tree.merged_with(result));
  }

private:

  std::string m_name;

  casacore::DataType m_datatype;

  unsigned m_row_rank;

  unsigned m_rank;

  IndexTreeL m_row_index_shape;

  IndexTreeIterator<coord_t> m_row_index_iterator;

  IndexTreeL m_index_tree;
};

template <int ROWDIM>
class ScalarColumnBuilder
  : public ColumnBuilder {
public:

  ScalarColumnBuilder(
    const std::string& name,
    casacore::DataType datatype,
    const IndexTreeL& row_index_shape,
    const IndexTreeL& index_tree)
    : ColumnBuilder(name, datatype, ROWDIM, 0, row_index_shape, index_tree) {
  }

  void
  add_row(const std::any&) override {
    set_next_row(IndexTreeL());
  }
};

template <int ROWDIM, int ARRAYDIM>
class ArrayColumnBuilder
  : public ColumnBuilder {
public:

  ArrayColumnBuilder(
    const std::string& name,
    casacore::DataType datatype,
    const IndexTreeL& row_index_shape,
    const IndexTreeL& index_tree)
    : ColumnBuilder(
      name,
      datatype,
      ROWDIM,
      ARRAYDIM,
      row_index_shape,
      index_tree) {
  }

  void
  add_row(const std::any& args) override {
    auto ary = row_dimensions(args);
    auto a = ary.rbegin();
    IndexTreeL t =
      std::accumulate(
        a + 1,
        ary.rend(),
        *a,
        [](const auto& t, const auto& d) {
          return IndexTreeL({{d, t}});
        });
    set_next_row(t);
  }

protected:

  virtual std::array<size_t, ARRAYDIM>
  row_dimensions(const std::any&) = 0;
};

class WithKeywords {
public:

  WithKeywords(
    const std::unordered_set<std::tuple<std::string, casacore::DataType>>& kw)
    : m_keywords(kw.begin(), kw.end()) {
  }

  std::vector<std::string>
  keywords() const {
    std::vector<std::string> result;
    std::transform(
      m_keywords.begin(),
      m_keywords.end(),
      std::back_inserter(result),
      [](auto& nm_dt) { return std::get<0>(nm_dt); });
    return result;
  }

  Legion::LogicalRegion
  keywords_region(Legion::Context ctx, Legion::Runtime* runtime) const {
    auto is = runtime->create_index_space(ctx, Legion::Rect<1>(0, 0));
    auto fs = runtime->create_field_space(ctx);
    auto fa = runtime->create_field_allocator(ctx, fs);
    std::for_each(
      m_keywords.begin(),
      m_keywords.end(),
      [&](auto& nm_dt) {
        auto& [nm, dt] = nm_dt;
        auto fid = legms::ms::add_field(dt, fa);
        runtime->attach_name(fs, fid, nm);
      });
    return runtime->create_logical_region(ctx, is, fs);
  }

private:

  std::vector<std::tuple<std::string, casacore::DataType>> m_keywords;
};

class Column
  : public WithKeywords {
public:

  ColumnBase(const ColumnBuilder& builder)
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
    return m_index_tree.pruned_to(m_row_rank - 1);
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
  rank() const {
    return m_rank;
  }

  std::optional<Legion::IndexSpace>
  index_space(Legion::Context ctx, Legion::Runtime* runtime) const {
    if (!m_index_space)
      m_index_space = tree_index_space(m_index_tree, ctx, runtime);
    return m_index_space;
  }

  Legion::FieldID
  add_field(
    Legion::Runtime *runtime;
    Legion::FieldSpace fs,
    Legion::FieldAllocator fa,
    Legion::FieldID field_id = AUTO_GENERATE_ID) const {

    Legion::FieldID result =
      legms::ms::add_field(m_datatype, fa, field_id);
    runtime->attach_name(fs, result, name());
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
    case 1:
      FillProjectionsLauncher<DIM, 1> fill_projections(
        &projected,
        index_space().value(),
        ctx,
        runtime);
      fill_projections.dispatch(ctx, runtime);
      result = fill_projections.logical_region();
      break;

    case 2:
      FillProjectionsLauncher<DIM, 2> fill_projections(
        &projected,
        index_space().value(),
        ctx,
        runtime);
      fill_projections.dispatch(ctx, runtime);
      result = fill_projections.logical_region();
      break;

    case 3:
      FillProjectionsLauncher<DIM, 3> fill_projections(
        &projected,
        index_space().value(),
        ctx,
        runtime);
      fill_projections.dispatch(ctx, runtime);
      result = fill_projections.logical_region();
      break;

    case 4:
      FillProjectionsLauncher<DIM, 4> fill_projections(
        &projected,
        index_space().value(),
        ctx,
        runtime);
      fill_projections.dispatch(ctx, runtime);
      result = fill_projections.logical_region();
      break;

    case 5:
      FillProjectionsLauncher<DIM, 5> fill_projections(
        &projected,
        index_space().value(),
        ctx,
        runtime);
      fill_projections.dispatch(ctx, runtime);
      result = fill_projections.logical_region();
      break;
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

template <int ROWDIM>
class TableBuilder
  : public WithKeywordsBuilder {
public:

  TableBuilder(IndexTreeL row_index_shape)
    : WithKeywordsBuilder()
    , m_row_index_shape(row_index_shape) {

    assert(row_index_shape.rank().value_or(0) == ROWDIM);
  }

  template <typename ColGen>
  void
  add_column(ColGen generator) {
    add_column(generator(m_row_index_shape));
  }

  void
  add_column(const ColumnBuilder& col) {
    assert(validate_column_builder(col));
    m_columns.insert(col);
  }

  void
  add_column(ColumnBuilder&& col) {
    assert(validate_column_builder(col));
    m_columns.emplace(std::move(col));
  }

  void
  add_row(const std::unordered_map<std::string, std::any>& args) {
    std::for_each(
      m_columns.begin(),
      m_columns.end(),
      [&args](ColumnBuilder& col) {
        auto nm_arg = args.find(col.name());
        std::any arg = ((nm_arg != args.end()) ? nm_arg->second : std::any());
        col.add_row(arg);
      });
  }

  std::unordered_set<std::string>
  column_names() const {
    std::unordered_set<std::string> result;
    std::transform(
      m_columns.begin(),
      m_columns.end(),
      std::inserter(result),
      [](auto& col) {
        return col.name();
      });
    return result;
  }

  bool
  validate_column_builder(const ColumnBuilder& col) const {
    assert(col.row_index_shape() == m_row_index_shape);
    assert(m_columns.size() == 0
           || m_columns.begin()->row_index_tree() == col.row_index_tree());
    assert(column_names().count(col.name()) == 0);
  }

  std::unordered_set<ColumnBuilder> m_columns;

  IndexTreeL m_row_index_shape;
};

template <int ROWDIM>
class Table
  : public WithKeywords {
public:

  Table(const TableBuilder& builder)
    : WithKeywords(builder.keywords()) {

    assert(builder.m_columns.size() > 0);
    std::transform(
      builder.m_columns.begin(),
      builder.m_columns.end(),
      std::inserter(m_columns),
      [](auto& cb) {
        return Column(cb);
      });
  }

  unsigned
  row_rank() const {
    return m_columns.begin()->row_rank();
  }

  unsigned
  rank() const {
    return max_rank_column()->rank();
  }

  std::unordered_set<std::string>
  column_names() const {
    std::unordered_set<std::string> result;
    std::transform(
      m_columns.begin(),
      m_columns.end(),
      std::inserter(result),
      [](auto& col) {
        return col.name();
      });
    return result;
  }

  std::optional<Legion::IndexSpace>
  index_space(Legion::Context ctx, Legion::Runtime* runtime) const {
    return max_rank_column()->index_space(ctx, runtime);
  }

  std::vector<Legion::FieldID>
  add_column_fields(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    Legion::FieldSpace fs,
    Legion::FieldAllocator fa,
    const std::vector<std::string>& colnames) const {

    std::vector<Legion::FieldID> result;
    std::transform(
      colnames.start(),
      colnames.end(),
      std::back_inserter(result),
      [&ctx, runtime, &fs, &fa](auto& colname) {
        auto col =
          std::find_if(
            m_columns.start(),
            m_columns.end(),
            [&colname](auto& col) {
              return col.name() == colname;
            });
        assert(col != m_columns.end());
        return col->add_field(ctx, runtime, fs, fa);
      });
    return result;
  }

  typeof(m_columns)::const_iterator_t
  max_rank_column() const {
    typeof(m_columns)::const_iterator_t result = m_columns.begin();
    for (typeof(m_columns)::const_iterator_t e = result;
         e != m_columns.end();
         ++e) {
      if (e->rank() > result->rank())
        result = e;
    }
    return result;
  }

  std::unordered_set<Column> m_columns;
};

} // end namespace ms
} // end namespace legms

#endif // LEGMS_MS_MAIN_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
