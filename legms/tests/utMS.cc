#include <algorithm>
#include <experimental/filesystem>
#include <map>
#include <memory>
#include <vector>

#include "legms.h"
#include "utility.h"
#include "IndexTree.h"
#include "Column.h"
#include "Table.h"
#include "TableBuilder.h"
#include "TableReadTask.h"

#include "TestSuiteDriver.h"
#include "TestRecorder.h"
#include "TestExpression.h"

namespace fs = std::experimental::filesystem;

using namespace legms;
using namespace Legion;

enum {
  MS_TEST_SUITE,
  VERIFY_COLUMN_TASK,
};

template <typename T, int DIM>
using RO = FieldAccessor<READ_ONLY, T, DIM, coord_t, AffineAccessor<T, DIM, coord_t>>;

#define TE(f) testing::TestEval([&](){ return f; }, #f)

struct VerifyColumnTaskArgs {
  legms::TypeTag tag;
  char table[160];
  char column[32];
};

void
verify_scalar_column(
  const casacore::Table& tb,
  const VerifyColumnTaskArgs *targs,
  const std::vector<PhysicalRegion>& regions,
  Context context,
  Runtime* runtime) {

  testing::TestLog<READ_WRITE> log(regions[1], regions[2], context, runtime);
  testing::TestRecorder<READ_WRITE> recorder(log);

  DomainT<1> col_dom(regions[0].get_bounds<1,coord_t>());

#define CMP(TAG)                                                    \
  case (TAG): {                                                     \
    auto scol =                                                     \
      casacore::ScalarColumn<DataType<TAG>::CasacoreType>(          \
        tb,                                                         \
        casacore::String(targs->column));                           \
    recorder.assert_true(                                           \
      std::string("verify bounds, column ") + targs->column,        \
      TE(Domain(col_dom)) == Domain(Rect<1>(0, scol.nrow() - 1)));  \
    casacore::Vector<DataType<TAG>::CasacoreType> ary =             \
      scol.getColumn();                                             \
    const RO<DataType<TAG>::ValueType, 1>                           \
      col(regions[0], Column::value_fid);                           \
    PointInDomainIterator<1> pid(col_dom);                          \
    recorder.expect_true(                                           \
      std::string("verify values, column ") + targs->column,        \
      testing::TestEval(                                            \
      [&pid, &col, &ary, targs]() {                                 \
      bool result = true;                                           \
      for (; result && pid(); pid++) {                              \
        DataType<TAG>::ValueType a;                                 \
        DataType<TAG>::from_casacore(a, ary[pid[0]]);               \
        result = DataType<TAG>::equiv(a, col[*pid]);                \
      }                                                             \
      return result;                                                \
    }));                                                            \
    break;                                                          \
  }

  switch (targs->tag) {
    LEGMS_FOREACH_DATATYPE(CMP);
  }
#undef CMP
}

template <int DIM>
void
verify_array_column(
  const casacore::Table& tb,
  const VerifyColumnTaskArgs *targs,
  const std::vector<PhysicalRegion>& regions,
  Context context,
  Runtime* runtime) {

  testing::TestLog<READ_WRITE> log(regions[1], regions[2], context, runtime);
  testing::TestRecorder<READ_WRITE> recorder(log);

  DomainT<DIM> col_dom(regions[0].get_bounds<DIM,coord_t>());

#define CMP(TAG)                                                        \
  case (TAG): {                                                         \
    auto acol =                                                         \
      casacore::ArrayColumn<DataType<TAG>::CasacoreType>(               \
        tb,                                                             \
        casacore::String(targs->column));                               \
    recorder.assert_true(                                               \
      std::string("verify rank, column ") + targs->column,              \
      TE(acol.ndim(0)) == DIM - 1);                                     \
    recorder.assert_true(                                               \
      std::string("verify nrows, column ") + targs->column,             \
      TE(Domain(col_dom).hi()[0]) == acol.nrow() - 1);                  \
    {                                                                   \
      PointInDomainIterator<DIM> pid(col_dom, false);                   \
      recorder.assert_true(                                             \
        std::string("verify bounds, column ") + targs->column,          \
        testing::TestEval(                                              \
          [&pid, &acol]() {                                             \
            bool result = true;                                         \
            while (result && pid()) {                                   \
              auto last_p = *pid;                                       \
              while (result && pid()) {                                 \
                pid++;                                                  \
                if (!pid() || pid[0] != last_p[0]) {                    \
                  casacore::IPosition shp(acol.shape(last_p[0]));       \
                  Point<DIM> cpt;                                       \
                  cpt[0] = last_p[0];                                   \
                  for (size_t i = 0; i < DIM - 1; ++i)                  \
                    cpt[i + 1] = shp[DIM - 2 - i] - 1;                  \
                  result = cpt == last_p;                               \
                }                                                       \
                if (pid())                                              \
                  last_p = *pid;                                        \
              }                                                         \
            }                                                           \
            return result;                                              \
          }));                                                          \
    }                                                                   \
    {                                                                   \
      const RO<DataType<TAG>::ValueType, DIM>                           \
        col(regions[0], Column::value_fid);                             \
      PointInDomainIterator<DIM> pid(col_dom, false);                   \
      recorder.assert_true(                                             \
        std::string("verify values, column ") + targs->column,          \
        testing::TestEval(                                              \
          [&pid, &acol, &col]() {                                       \
            bool result = true;                                         \
            casacore::Array<DataType<TAG>::CasacoreType> ary;           \
            casacore::IPosition ipos(DIM - 1);                          \
            while (result && pid()) {                                   \
              auto row = pid[0];                                        \
              acol.get(row, ary, true);                                 \
              while (result && pid()) {                                 \
                for (size_t i = 0; i < DIM - 1; ++i)                    \
                  ipos[DIM - 2 - i] = pid[i + 1];                       \
                DataType<TAG>::ValueType a;                             \
                DataType<TAG>::from_casacore(a, ary(ipos));             \
                result = DataType<TAG>::equiv(a, col[*pid]);            \
                pid++;                                                  \
                if (pid() && pid[0] != row) {                           \
                  row = pid[0];                                         \
                  acol.get(row, ary, true);                             \
                }                                                       \
              }                                                         \
            }                                                           \
            return result;                                              \
          }));                                                          \
    }                                                                   \
    break;                                                              \
}

  switch (targs->tag) {
    LEGMS_FOREACH_DATATYPE(CMP);
  }
#undef CMP
}

void
verify_column_task(
  const Task* task,
  const std::vector<PhysicalRegion>& region,
  Context context,
  Runtime* runtime) {

  const VerifyColumnTaskArgs *args =
    static_cast<const VerifyColumnTaskArgs*>(task->args);

  casacore::Table tb(
    casacore::String(args->table),
    casacore::TableLock::PermanentLockingWait);

  auto cdesc = tb.tableDesc()[casacore::String(args->column)];
  if (cdesc.isScalar()) {
    verify_scalar_column(tb, args, region, context, runtime);
  } else {
#define VERIFY_ARRAY(N) \
    case (N): verify_array_column<N>(tb, args, region, context, runtime); break;

    switch (cdesc.ndim() + 1) {
      LEGMS_FOREACH_N(VERIFY_ARRAY);
    }
#undef VERIFY_ARRAY
  }
}

void
read_full_ms(
  testing::TestLog<READ_WRITE>& log,
  Context context,
  Runtime* runtime) {

  testing::TestRecorder<READ_WRITE> recorder(log);

  static const std::string t0_path("data/t0.ms");
  std::unique_ptr<const Table> table =
    Table::from_ms(context, runtime, t0_path, {"*"});
  recorder.assert_true(
    "t0.ms MAIN table successfully read",
    bool(table));
  recorder.expect_true(
    "main table name is 'MAIN'",
    TE(table->name()) == "MAIN");
  recorder.expect_true(
    "main table is not empty",
    TE(!table->is_empty()));

  std::vector<std::string> expected_columns{
    "UVW",
    "FLAG",
    //"FLAG_CATEGORY",
    "WEIGHT",
    "SIGMA",
    "ANTENNA1",
    "ANTENNA2",
    "ARRAY_ID",
    "DATA_DESC_ID",
    "EXPOSURE",
    "FEED1",
    "FEED2",
    "FIELD_ID",
    "FLAG_ROW",
    "INTERVAL",
    "OBSERVATION_ID",
    "PROCESSOR_ID",
    "SCAN_NUMBER",
    "STATE_ID",
    "TIME",
    "TIME_CENTROID",
    "DATA",
    //"WEIGHT_SPECTRUM"
  };
  recorder.assert_true(
    "table has expected columns",
    TE(std::set<std::string>(
         table->column_names().begin(),
         table->column_names().end()) ==
       std::set<std::string>(
         expected_columns.begin(),
         expected_columns.end())));

  //
  // read MS table columns to initialize the Column LogicalRegions
  //
  {
    TableReadTask table_read_task(
      t0_path,
      table.get(),
      expected_columns.begin(),
      expected_columns.end(),
      2000);
    table_read_task.dispatch();
  }

  // compare column LogicalRegions to values read using casacore functions
  // directly
  IndexSpace col_is(
    runtime->create_index_space(
      context,
      Rect<1>(0, expected_columns.size() - 1)));
  auto remaining_log =
    log.get_log_references_by_state({testing::TestState::UNKNOWN})[0];
  IndexPartition col_log_ip =
    runtime->create_equal_partition(
      context,
      remaining_log.log_region().get_index_space(),
      col_is);
  LogicalPartitionT<1> verify_col_logs(
    runtime->get_logical_partition(
      context,
      remaining_log.log_region(),
      col_log_ip));
  VerifyColumnTaskArgs args;
  std::strncpy(args.table, t0_path.c_str(), sizeof(args.table));
  args.table[sizeof(args.table) - 1] = '\0';
  // can't use IndexTaskLauncher here since column LogicalRegions are not
  // sub-regions of a common LogicalPartition
  TaskLauncher verify_task(
    VERIFY_COLUMN_TASK,
    TaskArgument(&args, sizeof(args)));
  for (size_t i = 0; i < expected_columns.size(); ++i) {
    auto col = table->column(expected_columns[i]);
    if (col->logical_region() != LogicalRegion::NO_REGION) {
      args.tag = col->datatype();
      std::strncpy(args.column, col->name().c_str(), sizeof(args.column));
      args.column[sizeof(args.column) - 1] = '\0';
      verify_task.region_requirements.clear();
      verify_task.add_region_requirement(
        RegionRequirement(
          col->logical_region(),
          READ_ONLY,
          EXCLUSIVE,
          col->logical_region()));
      verify_task.add_field(0, Column::value_fid);
      auto log_reqs =
        remaining_log.requirements<READ_WRITE>(
          runtime->get_logical_subregion_by_color(verify_col_logs, Point<1>(i)),
          log.log_reference().log_region());
      std::for_each(
        log_reqs.begin(),
        log_reqs.end(),
        [&verify_task](auto& req) {
          verify_task.add_region_requirement(req);
        });
      runtime->execute_task(context, verify_task);
    }
  }
  runtime->destroy_logical_partition(context, verify_col_logs);
  runtime->destroy_index_partition(context, col_log_ip);
  runtime->destroy_index_space(context, col_is);
}

void
ms_test_suite(
  const Task*,
  const std::vector<PhysicalRegion>& regions,
  Context context,
  Runtime* runtime) {

  register_tasks(runtime);

  testing::TestLog<READ_WRITE> log(regions[0], regions[1], context, runtime);

  read_full_ms(log, context, runtime);
}

int
main(int argc, char** argv) {

  TaskVariantRegistrar registrar(VERIFY_COLUMN_TASK, "verify_column_task");
  registrar.add_constraint(ProcessorConstraint(Processor::IO_PROC));
  Runtime::preregister_task_variant<verify_column_task>(
    registrar,
    "verify_column_task");

  testing::TestSuiteDriver driver =
    testing::TestSuiteDriver::make<ms_test_suite>(
      MS_TEST_SUITE,
      "ms_test_suite");

  return driver.start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
