#include "testing/TestSuiteDriver.h"
#include "testing/TestRecorder.h"

#include <memory>
#include <ostream>
#include <vector>

#include "utility.h"
#include "Table.h"
#include "Column.h"

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

template <>
struct legms::AxesUID<Table0Axes> {
  static constexpr const char* id = "Table0Axes";
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

#define TABLE0_NUM_X 4
#define TABLE0_NUM_Y 3
#define TABLE0_NUM_ROWS (TABLE0_NUM_X * TABLE0_NUM_Y)
unsigned table0_x[TABLE0_NUM_ROWS] {
                   0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
unsigned table0_y[TABLE0_NUM_ROWS] {
                   0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};

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
      {static_cast<int>(Table0Axes::ROW)},
      {table0_col("X"),
       table0_col("Y")});
  auto col_x =
    attach_table0_col(table0.columnT("X").get(), table0_x, context, runtime);
  auto col_y =
    attach_table0_col(table0.columnT("Y").get(), table0_y, context, runtime);
  IndexColumnTask icx(table0.columnT("X"), static_cast<int>(Table0Axes::X));
  IndexColumnTask icy(table0.columnT("Y"), static_cast<int>(Table0Axes::Y));
  Future fx = icx.dispatch(context, runtime);
  Future fy = icy.dispatch(context, runtime);

  auto cx =
    fx.get_result<ColumnGenArgs>().operator()<Table0Axes>(context, runtime);
  recorder.assert_true(
    "IndexColumnTask X result has one axis",
    TE(cx->axesT().size()) == 1u);
  recorder.expect_true(
    "IndexColumnTask X result axis is 'Table0Axes::X'",
    TE(cx->axesT()[0]) == Table0Axes::X);
  recorder.assert_true(
    "IndexColumnTask X result has one-dimensional IndexSpace",
    TE(cx->index_space().get_dim()) == 1);
  Domain xd = runtime->get_index_space_domain(cx->index_space());
  recorder.expect_true(
    "IndexColumnTask X result IndexSpace has expected range",
    TE(xd.lo()[0] == 0) && (TE(xd.hi()[0]) == TABLE0_NUM_X - 1));

  auto cy =
    fy.get_result<ColumnGenArgs>().operator()<Table0Axes>(context, runtime);
  recorder.assert_true(
    "IndexColumnTask Y result has one axis",
    TE(cy->axesT().size()) == 1u);
  recorder.expect_true(
    "IndexColumnTask Y result axis is 'Table0Axes::Y'",
    TE(cy->axesT()[0]) == Table0Axes::Y);
  recorder.assert_true(
    "IndexColumnTask Y result has one-dimensional IndexSpace",
    TE(cy->index_space().get_dim()) == 1);
  Domain yd = runtime->get_index_space_domain(cy->index_space());
  recorder.expect_true(
    "IndexColumnTask Y result IndexSpace has expected range",
    TE(yd.lo()[0] == 0) && (TE(yd.hi()[0]) == TABLE0_NUM_Y - 1));

  runtime->detach_external_resource(context, col_x);
  runtime->detach_external_resource(context, col_y);
}

int
main(int argc, char* argv[]) {

  testing::TestSuiteDriver driver =
    testing::TestSuiteDriver::make<index_column_task_test_suite>(
      INDEX_COLUMN_TASK_TEST_SUITE,
      "index_column_task_test_suite");

  return driver.start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
