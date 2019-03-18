#include "testing/TestSuiteDriver.h"
#include "testing/TestRecorder.h"

#include <memory>

#include "utility.h"
#include "Table.h"
#include "Column.h"

#include "legion.h"

using namespace legms;
using namespace Legion;

enum {
  INDEX_COLUMN_TASK_TEST_SUITE,
};

enum struct Table0Axes {
  index = -1,
  ROW = 0,
  X,
  Y
};

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

//#define TE(f) testing::TestEval([&](){ return f; })

void
index_column_task_test_suite(
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
      {table0_col("X"), table0_col("Y"), table0_col("Z")});
  auto col_x =
    attach_table0_col(table0.columnT("X").get(), table0_x, context, runtime);
  auto col_y =
    attach_table0_col(table0.columnT("Y").get(), table0_y, context, runtime);
  IndexColumnTask icx(table0.columnT("X"));
  IndexColumnTask icy(table0.columnT("Y"));
  Future fx = icx.dispatch(context, runtime);
  Future fy = icy.dispatch(context, runtime);

  auto cx =
    fx.get_result<ColumnGenArgs>().operator()<Table0Axes>(context, runtime);
  recorder.assert_true(
    "IndexColumnTask X result has one axis",
    cx->axesT().size() == 1);
  recorder.expect_true(
    "IndexColumnTask X result axis is 'index'",
    cx->axesT()[0] == Table0Axes::index);
  recorder.expect_true(
    "IndexColumnTask X result has one-dimensional IndexSpace",
    cx->index_space().get_dim() == 1);
  Domain xd = runtime->get_index_space_domain(cx->index_space());
  recorder.expect_true(
    "IndexColumnTask X result IndexSpace has range [0,3]",
    (xd.lo()[0] == 0)
    && (xd.hi()[0] == TABLE0_NUM_X - 1));

  auto cy =
    fy.get_result<ColumnGenArgs>().operator()<Table0Axes>(context, runtime);
  recorder.assert_true(
    "IndexColumnTask Y result has one axis",
    cy->axesT().size() == 1);
  recorder.expect_true(
    "IndexColumnTask Y result axis is 'index'",
    cy->axesT()[0] == Table0Axes::index);
  recorder.expect_true(
    "IndexColumnTask Y result has one-dimensional IndexSpace",
    cy->index_space().get_dim() == 1);
  Domain yd = runtime->get_index_space_domain(cy->index_space());
  recorder.expect_true(
    "IndexColumnTask Y result IndexSpace has range [0,2]",
    (yd.lo()[0] == 0)
    && (yd.hi()[0] == TABLE0_NUM_Y - 1));

  runtime->detach_external_resource(context, col_x);
  runtime->detach_external_resource(context, col_y);
}

int
main(int argc, char* argv[]) {

  testing::TestSuiteDriver driver =
    testing::TestSuiteDriver::make<index_column_task_test_suite>(
      INDEX_COLUMN_TASK_TEST_SUITE,
      "index_column_task_test_suite");

#if 0
  {
    TaskVariantRegistrar registrar(TEST_LOG_SUBTASK_ID, "subtask_suite");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<test_log_subtask>(
      registrar,
      "subtask_suite");
  }
#endif

  return driver.start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
