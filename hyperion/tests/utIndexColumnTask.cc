/*
 * Copyright 2019 Associated Universities, Inc. Washington DC, USA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <hyperion/testing/TestSuiteDriver.h>
#include <hyperion/testing/TestRecorder.h>

#include <memory>
#include <ostream>
#include <vector>

#include <hyperion/utility.h>
#include <hyperion/Table.h>
#include <hyperion/Column.h>

#ifdef HYPERION_USE_CASACORE
# include <hyperion/MeasRef.h>
#endif

using namespace hyperion;
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
struct hyperion::Axes<Table0Axes> {
  static const constexpr char* uid = "Table0Axes";
  static const std::vector<std::string> names;
  static const unsigned num_axes = 3;
#ifdef HYPERION_USE_HDF5
  static const hid_t h5_datatype;
#endif
};

const std::vector<std::string>
hyperion::Axes<Table0Axes>::names{"ROW", "X", "Y"};

#ifdef HYPERION_USE_HDF5
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
hyperion::Axes<Table0Axes>::h5_datatype = h5_dt();
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
    [=](Context ctx, Runtime* rt, const std::string& name_prefix) {
      return
        Column::create(
          ctx,
          rt,
          name,
          std::vector<Table0Axes>{Table0Axes::ROW},
          ValueType<unsigned>::DataType,
          IndexTreeL(TABLE0_NUM_ROWS),
#ifdef HYPERION_USE_CASACORE
          MeasRef(),
          std::nullopt,
#endif
          {},
          name_prefix);
    };
}

PhysicalRegion
attach_table0_col(
  Context context,
  Runtime* runtime,
  const Column& col,
  unsigned *base) {

  const Memory local_sysmem =
    Machine::MemoryQuery(Machine::get_machine())
    .has_affinity_to(runtime->get_executing_processor(context))
    .only_kind(Memory::SYSTEM_MEM)
    .first();

  AttachLauncher task(EXTERNAL_INSTANCE, col.values_lr, col.values_lr);
  task.attach_array_soa(
    base,
    true,
    {Column::VALUE_FID},
    local_sysmem);
  return runtime->attach_external_resource(context, task);
}

#define TE(f) testing::TestEval([&](){ return f; }, #f)

void
index_column_task_test_suite(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  register_tasks(ctx, rt);

  testing::TestRecorder<WRITE_DISCARD> recorder(
    testing::TestLog<WRITE_DISCARD>(
      task->regions[0].region,
      regions[0],
      task->regions[1].region,
      regions[1],
      ctx,
      rt));

  Table table0 =
    Table::create(
      ctx,
      rt,
      "table0",
      std::vector<Table0Axes>{Table0Axes::ROW},
      std::vector<Column::Generator>{
        table0_col("X"),
        table0_col("Y")});
  auto col_x =
    attach_table0_col(ctx, rt, table0.column(ctx, rt, "X"), table0_x);
  auto col_y =
    attach_table0_col(ctx, rt, table0.column(ctx, rt, "Y"), table0_y);
  IndexColumnTask icx(table0.column(ctx, rt, "X"));
  IndexColumnTask icy(table0.column(ctx, rt, "Y"));
  Future fx = icx.dispatch(ctx, rt);
  Future fy = icy.dispatch(ctx, rt);

  {
    auto cx = fx.get_result<LogicalRegion>();
    recorder.assert_false(
      "IndexColumnTask X result is not empty",
      TE(cx == LogicalRegion::NO_REGION));
    recorder.assert_true(
      "IndexColumnTask X result has one-dimensional IndexSpace",
      TE(cx.get_index_space().get_dim()) == 1);
    Domain xd = rt->get_index_space_domain(cx.get_index_space());
    recorder.expect_true(
      "IndexColumnTask X result IndexSpace has expected range",
      TE(xd.lo()[0] == 0) && (TE(xd.hi()[0]) == TABLE0_NUM_X - 1));
    recorder.expect_true(
      "IndexColumnTask X result has expected values",
      testing::TestEval(
        [&ctx, rt, &cx]() {
          RegionRequirement req(cx, READ_ONLY, EXCLUSIVE, cx);
          req.add_field(IndexColumnTask::VALUE_FID);
          PhysicalRegion pr = rt->map_region(ctx, req);
          const FieldAccessor<
            READ_ONLY, unsigned, 1, coord_t,
            AffineAccessor<unsigned, 1, coord_t>, true>
            x(pr, IndexColumnTask::VALUE_FID);
          bool result =
            x[0] == OX && x[1] == OX + 1 && x[2] == OX + 2 && x[3] == OX + 3;
          rt->unmap_region(ctx, pr);
          return result;
        }));
    recorder.expect_true(
      "IndexColumnTask X result has expected index groups",
      testing::TestEval(
        [&ctx, rt, &cx]() {
          RegionRequirement req(cx, READ_ONLY, EXCLUSIVE, cx);
          req.add_field(IndexColumnTask::ROWS_FID);
          PhysicalRegion pr = rt->map_region(ctx, req);
          const FieldAccessor<
            READ_ONLY, std::vector<DomainPoint>, 1, coord_t,
            AffineAccessor<std::vector<DomainPoint>, 1, coord_t>, true>
            x(pr, IndexColumnTask::ROWS_FID);
          bool result =
            (x[0] ==
             std::vector<DomainPoint>{Point<1>(0), Point<1>(1), Point<1>(2)})
            && (x[1] ==
                std::vector<DomainPoint>{Point<1>(3), Point<1>(4), Point<1>(5)})
            && (x[2] ==
                std::vector<DomainPoint>{Point<1>(6), Point<1>(7), Point<1>(8)})
            && (x[3] ==
                std::vector<DomainPoint>{Point<1>(9), Point<1>(10), Point<1>(11)});
          rt->unmap_region(ctx, pr);
          return result;
        }));
    auto is = cx.get_index_space();
    auto fs = cx.get_field_space();
    rt->destroy_logical_region(ctx, cx);
    rt->destroy_field_space(ctx, fs);
    rt->destroy_index_space(ctx, is);
  }
  {
    auto cy = fy.get_result<LogicalRegion>();
    recorder.assert_false(
      "IndexColumnTask Y result is not empty",
      TE(cy == LogicalRegion::NO_REGION));
    recorder.assert_true(
      "IndexColumnTask Y result has one-dimensional IndexSpace",
      TE(cy.get_index_space().get_dim()) == 1);
    Domain yd = rt->get_index_space_domain(cy.get_index_space());
    recorder.expect_true(
      "IndexColumnTask Y result IndexSpace has expected range",
      TE(yd.lo()[0] == 0) && (TE(yd.hi()[0]) == TABLE0_NUM_Y - 1));
    recorder.expect_true(
      "IndexColumnTask Y result has expected values",
      testing::TestEval(
        [&ctx, rt, &cy]() {
          RegionRequirement req(cy, READ_ONLY, EXCLUSIVE, cy);
          req.add_field(IndexColumnTask::VALUE_FID);
          PhysicalRegion pr = rt->map_region(ctx, req);
          const FieldAccessor<
            READ_ONLY, unsigned, 1, coord_t,
            AffineAccessor<unsigned, 1, coord_t>, true>
            y(pr, IndexColumnTask::VALUE_FID);
          bool result = y[0] == OY && y[1] == OY + 1 && y[2] == OY + 2;
          rt->unmap_region(ctx, pr);
          return result;
        }));
    recorder.expect_true(
      "IndexColumnTask Y result has expected index groups",
      testing::TestEval(
        [&ctx, rt, &cy]() {
          RegionRequirement req(cy, READ_ONLY, EXCLUSIVE, cy);
          req.add_field(IndexColumnTask::ROWS_FID);
          PhysicalRegion pr = rt->map_region(ctx, req);
          const FieldAccessor<
            READ_ONLY, std::vector<DomainPoint>, 1, coord_t,
            AffineAccessor<std::vector<DomainPoint>, 1, coord_t>, true>
            y(pr, IndexColumnTask::ROWS_FID);
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
          rt->unmap_region(ctx, pr);
          return result;
        }));
    auto is = cy.get_index_space();
    auto fs = cy.get_field_space();
    rt->destroy_logical_region(ctx, cy);
    rt->destroy_field_space(ctx, fs);
    rt->destroy_index_space(ctx, is);
  }
  rt->detach_external_resource(ctx, col_x);
  rt->detach_external_resource(ctx, col_y);
  table0.destroy(ctx, rt);
}

int
main(int argc, char* argv[]) {

  AxesRegistrar::register_axes<Table0Axes>();

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
