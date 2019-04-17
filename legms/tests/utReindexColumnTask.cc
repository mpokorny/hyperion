#include "testing/TestSuiteDriver.h"
#include "testing/TestRecorder.h"

#include <algorithm>
#include <memory>
#include <ostream>
#include <vector>
#include "utility.h"
#include "Table.h"
#include "Column.h"

#include "legion.h"

using namespace legms;
using namespace Legion;

enum {
  REINDEX_COLUMN_TASK_TEST_SUITE,
};

enum struct Table0Axes {
  index = -1,
  ROW = 0,
  X,
  Y
};

std::ostream&
operator<<(std::ostream& stream, const Table0Axes& ax) {
  switch (ax) {
  case Table0Axes::index:
    stream << "Table0Axes::index";
    break;
  case Table0Axes::ROW:
    stream << "Table0Axes::ROW";
    break;
  case Table0Axes::X:
    stream << "Table0Axes::X";
    break;
  case Table0Axes::Y:
    stream << "Table0Axes::Y";
    break;
  }
  return stream;
}

std::ostream&
operator<<(std::ostream& stream, const std::vector<Table0Axes>& axs) {
  stream << "[";
  const char* sep = "";
  std::for_each(
    axs.begin(),
    axs.end(),
    [&stream, &sep](auto& ax) {
      stream << sep << ax;
      sep = ",";
    });
  stream << "]";
  return stream;
}

#define TABLE0_NUM_X 4
#define TABLE0_NUM_Y 3
#define TABLE0_NUM_ROWS (TABLE0_NUM_X * TABLE0_NUM_Y)
unsigned table0_x[TABLE0_NUM_ROWS] {
                   0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
unsigned table0_y[TABLE0_NUM_ROWS] {
                   0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
unsigned table0_z[TABLE0_NUM_ROWS] {
                   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

ColumnT<Table0Axes>::Generator
table0_col(const std::string& name) {
  return
    [=](Context context, Runtime* runtime) {
      return
        std::make_unique<ColumnT<Table0Axes>>(
          context,
          runtime,
          name,
          ValueType<unsigned>::DataType,
          std::vector<Table0Axes>{Table0Axes::ROW},
          IndexTreeL(TABLE0_NUM_ROWS));
    };
}

PhysicalRegion
attach_table0_col(
  const ColumnT<Table0Axes>* col,
  unsigned *base,
  Context context,
  Runtime* runtime) {

  const Memory local_sysmem =
    Machine::MemoryQuery(Machine::get_machine())
    .has_affinity_to(runtime->get_executing_processor(context))
    .only_kind(Memory::SYSTEM_MEM)
    .first();

  AttachLauncher
    task(EXTERNAL_INSTANCE, col->logical_region(), col->logical_region());
  task.attach_array_soa(
    base,
    true,
    {Column::value_fid},
    local_sysmem);
  return runtime->attach_external_resource(context, task);
}

#define TE(f) testing::TestEval([&](){ return f; }, #f)

void
reindex_column_task_test_suite(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context context,
  Runtime* runtime) {

  register_tasks(runtime);

  testing::TestRecorder<WRITE_DISCARD> recorder(
    testing::TestLog<WRITE_DISCARD>(regions[0], regions[1], context, runtime));

  TableT<Table0Axes>
    table0(
      context,
      runtime,
      "table0",
      {"ROW"},
      {table0_col("X"),
       table0_col("Y"),
       table0_col("Z")});
  auto col_x =
    attach_table0_col(table0.columnT("X").get(), table0_x, context, runtime);
  auto col_y =
    attach_table0_col(table0.columnT("Y").get(), table0_y, context, runtime);
  auto col_z =
    attach_table0_col(table0.columnT("Z").get(), table0_y, context, runtime);

  IndexColumnTask icx(table0.columnT("X"), static_cast<int>(Table0Axes::X));
  IndexColumnTask icy(table0.columnT("Y"), static_cast<int>(Table0Axes::Y));
  std::vector<Future> icfs {
    icx.dispatch(context, runtime),
    icy.dispatch(context, runtime)};
  std::vector<std::shared_ptr<Column>> ics;
  std::for_each(
    icfs.begin(),
    icfs.end(),
    [&context, &runtime, &ics](Future& f) {
      auto col =
        f.get_result<ColumnGenArgs>().operator()<Table0Axes>(context, runtime);
      ics.push_back(std::move(col));
    });

  ReindexColumnTask rcz(
    table0.columnT("Z"),
    0,
    ics,
    {static_cast<int>(Table0Axes::X), static_cast<int>(Table0Axes::Y)},
    false);
  Future fz = rcz.dispatch(context, runtime);
  auto cz =
    fz.get_result<ColumnGenArgs>().operator()<Table0Axes>(context, runtime);
  recorder.assert_true(
    "Reindexed column index space rank is 2",
    TE(cz->rank()) == 2);
  recorder.expect_true(
    "Reindexed column index space dimensions are X and Y",
    TE(cz->axesT()) == std::vector<Table0Axes>{ Table0Axes::X, Table0Axes::Y });

  runtime->detach_external_resource(context, col_x);
  runtime->detach_external_resource(context, col_y);
  runtime->detach_external_resource(context, col_z);
}

int
main(int argc, char* argv[]) {

  testing::TestSuiteDriver driver =
    testing::TestSuiteDriver::make<reindex_column_task_test_suite>(
      REINDEX_COLUMN_TASK_TEST_SUITE,
      "reindex_column_task_test_suite");

  return driver.start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End: