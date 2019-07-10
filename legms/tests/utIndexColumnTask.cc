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
  ROW = 0,
  X,
  Y
};

template <>
struct legms::Axes<Table0Axes> {
  static const constexpr char* uid = "Table0Axes";
  static const std::vector<std::string> names;
  static const unsigned num_axes = 3;
#ifdef LEGMS_USE_HDF5
  static const hid_t h5_datatype;
#endif
};

const std::vector<std::string>
legms::Axes<Table0Axes>::names{"ROW", "X", "Y"};

#ifdef LEGMS_USE_HDF5
hid_t
h5_dt() {
  hid_t result = H5Tenum_create(H5T_NATIVE_UCHAR);
  Table0Axes a = Table0Axes::ROW;
  herr_t err = H5Tenum_insert(result, "ROW", &a);
  assert(err >= 0);
  a = Table0Axes::X;
  err = H5Tenum_insert(result, "X", &a);
  a = Table0Axes::Y;
  err = H5Tenum_insert(result, "Y", &a);
  return result;
}

const hid_t
legms::Axes<Table0Axes>::h5_datatype = h5_dt();
#endif

std::ostream&
operator<<(std::ostream& stream, const Table0Axes& ax) {
  switch (ax) {
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
#define OX 22
#define TABLE0_NUM_Y 3
#define OY 30
#define TABLE0_NUM_ROWS (TABLE0_NUM_X * TABLE0_NUM_Y)
unsigned table0_x[TABLE0_NUM_ROWS] {
                   OX + 0, OX + 0, OX + 0,
                     OX + 1, OX + 1, OX + 1,
                     OX + 2, OX + 2, OX + 2,
                     OX + 3, OX + 3, OX + 3};
unsigned table0_y[TABLE0_NUM_ROWS] {
                   OY + 0, OY + 1, OY + 2,
                     OY + 0, OY + 1, OY + 2,
                     OY + 0, OY + 1, OY + 2,
                     OY + 0, OY + 1, OY + 2};

Column::Generator
table0_col(const std::string& name) {
  return
    [=](Context context, Runtime* runtime) {
      return
        std::make_unique<Column>(
          context,
          runtime,
          name,
          std::vector<Table0Axes>{Table0Axes::ROW},
          ValueType<unsigned>::DataType,
          IndexTreeL(TABLE0_NUM_ROWS));
    };
}

PhysicalRegion
attach_table0_col(
  const Column* col,
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
  const Task*,
  const std::vector<PhysicalRegion>& regions,
  Context context,
  Runtime* runtime) {

  register_tasks(runtime);
  AxesRegistrar::register_axes<Table0Axes>();

  testing::TestRecorder<WRITE_DISCARD> recorder(
    testing::TestLog<WRITE_DISCARD>(regions[0], regions[1], context, runtime));

  Table
    table0(
      context,
      runtime,
      "table0",
      std::vector<Table0Axes>{Table0Axes::ROW},
      {table0_col("X"),
       table0_col("Y")});
  auto col_x =
    attach_table0_col(table0.column("X").get(), table0_x, context, runtime);
  auto col_y =
    attach_table0_col(table0.column("Y").get(), table0_y, context, runtime);
  IndexColumnTask icx(table0.column("X"), static_cast<int>(Table0Axes::X));
  IndexColumnTask icy(table0.column("Y"), static_cast<int>(Table0Axes::Y));
  Future fx = icx.dispatch(context, runtime);
  Future fy = icy.dispatch(context, runtime);

  auto cx =
    fx.get_result<ColumnGenArgs>().operator()(context, runtime);
  recorder.assert_true(
    "IndexColumnTask X result has one axis",
    TE(cx->axes().size()) == 1u);
  recorder.expect_true(
    "IndexColumnTask X result axis is 'Table0Axes::X'",
    TE(cx->axes()[0]) == static_cast<int>(Table0Axes::X));
  recorder.assert_true(
    "IndexColumnTask X result has one-dimensional IndexSpace",
    TE(cx->index_space().get_dim()) == 1);
  Domain xd = runtime->get_index_space_domain(cx->index_space());
  recorder.expect_true(
    "IndexColumnTask X result IndexSpace has expected range",
    TE(xd.lo()[0] == 0) && (TE(xd.hi()[0]) == TABLE0_NUM_X - 1));
  recorder.expect_true(
    "IndexColumnTask X result has expected values",
    testing::TestEval(
      [&context, runtime, &cx]() {
        RegionRequirement
          req(cx->logical_region(), READ_ONLY, EXCLUSIVE, cx->logical_region());
        req.add_field(Column::value_fid);
        PhysicalRegion pr = runtime->map_region(context, req);
        const FieldAccessor<
          READ_ONLY, unsigned, 1, coord_t,
          AffineAccessor<unsigned, 1, coord_t>, true>
          x(pr, Column::value_fid);
        bool result =
          x[0] == OX && x[1] == OX + 1 && x[2] == OX + 2 && x[3] == OX + 3;
        runtime->unmap_region(context, pr);
        return result;
      }));
  recorder.expect_true(
    "IndexColumnTask X result has expected index groups",
    testing::TestEval(
      [&context, runtime, &cx]() {
        RegionRequirement
          req(cx->logical_region(), READ_ONLY, EXCLUSIVE, cx->logical_region());
        req.add_field(IndexColumnTask::indices_fid);
        PhysicalRegion pr = runtime->map_region(context, req);
        const FieldAccessor<
          READ_ONLY, std::vector<DomainPoint>, 1, coord_t,
          AffineAccessor<std::vector<DomainPoint>, 1, coord_t>, true>
          x(pr, IndexColumnTask::indices_fid);
        bool result =
          (x[0] ==
           std::vector<DomainPoint>{Point<1>(0), Point<1>(1), Point<1>(2)})
          && (x[1] ==
              std::vector<DomainPoint>{Point<1>(3), Point<1>(4), Point<1>(5)})
          && (x[2] ==
              std::vector<DomainPoint>{Point<1>(6), Point<1>(7), Point<1>(8)})
          && (x[3] ==
              std::vector<DomainPoint>{Point<1>(9), Point<1>(10), Point<1>(11)});
        runtime->unmap_region(context, pr);
        return result;
      }));

  auto cy =
    fy.get_result<ColumnGenArgs>().operator()(context, runtime);
  recorder.assert_true(
    "IndexColumnTask Y result has one axis",
    TE(cy->axes().size()) == 1u);
  recorder.expect_true(
    "IndexColumnTask Y result axis is 'Table0Axes::Y'",
    TE(cy->axes()[0]) == static_cast<int>(Table0Axes::Y));
  recorder.assert_true(
    "IndexColumnTask Y result has one-dimensional IndexSpace",
    TE(cy->index_space().get_dim()) == 1);
  Domain yd = runtime->get_index_space_domain(cy->index_space());
  recorder.expect_true(
    "IndexColumnTask Y result IndexSpace has expected range",
    TE(yd.lo()[0] == 0) && (TE(yd.hi()[0]) == TABLE0_NUM_Y - 1));
  recorder.expect_true(
    "IndexColumnTask Y result has expected values",
    testing::TestEval(
      [&context, runtime, &cy]() {
        RegionRequirement
          req(cy->logical_region(), READ_ONLY, EXCLUSIVE, cy->logical_region());
        req.add_field(Column::value_fid);
        PhysicalRegion pr = runtime->map_region(context, req);
        const FieldAccessor<
          READ_ONLY, unsigned, 1, coord_t,
          AffineAccessor<unsigned, 1, coord_t>, true>
          y(pr, Column::value_fid);
        bool result = y[0] == OY && y[1] == OY + 1 && y[2] == OY + 2;
        runtime->unmap_region(context, pr);
        return result;
      }));
  recorder.expect_true(
    "IndexColumnTask Y result has expected index groups",
    testing::TestEval(
      [&context, runtime, &cy]() {
        RegionRequirement
          req(cy->logical_region(), READ_ONLY, EXCLUSIVE, cy->logical_region());
        req.add_field(IndexColumnTask::indices_fid);
        PhysicalRegion pr = runtime->map_region(context, req);
        const FieldAccessor<
          READ_ONLY, std::vector<DomainPoint>, 1, coord_t,
          AffineAccessor<std::vector<DomainPoint>, 1, coord_t>, true>
          y(pr, IndexColumnTask::indices_fid);
        bool result =
          (y[0] ==
           std::vector<DomainPoint>{
            Point<1>(0), Point<1>(3), Point<1>(6), Point<1>(9)})
          && (y[1] ==
              std::vector<DomainPoint>{
                Point<1>(1), Point<1>(4), Point<1>(7), Point<1>(10)})
          && (y[2] ==
              std::vector<DomainPoint>{
                Point<1>(2), Point<1>(5), Point<1>(8), Point<1>(11)});
        runtime->unmap_region(context, pr);
        return result;
      }));

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
