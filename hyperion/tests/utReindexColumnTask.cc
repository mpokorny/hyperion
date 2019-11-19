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

#include <hyperion/utility.h>
#include <hyperion/Table.h>
#include <hyperion/Column.h>

#include <algorithm>
#include <memory>
#include <ostream>
#include <vector>

using namespace hyperion;
using namespace Legion;

enum {
  REINDEX_COLUMN_TASK_TEST_SUITE,
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

Column::Generator
table0_col(
  const std::string& name
#ifdef HYPERION_USE_CASACORE
  , const std::vector<MeasRef>& measures
#endif
  ) {
  return
    [=](Context ctx, Runtime* rt, const std::string& name_prefix
#ifdef HYPERION_USE_CASACORE
        , const MeasRefContainer& table_mr
#endif
      ) {
      return
        Column::create(
          ctx,
          rt,
          name,
          std::vector<Table0Axes>{Table0Axes::ROW},
          ValueType<unsigned>::DataType,
          IndexTreeL(TABLE0_NUM_ROWS),
#ifdef HYPERION_USE_CASACORE
          MeasRefContainer::create(ctx, rt, measures, table_mr),
          true,
#endif
          {},
          name_prefix);
    };
}

PhysicalRegion
attach_table0_col(
  Context ctx,
  Runtime* rt,
  const Column& col,
  unsigned *base) {

  const Memory local_sysmem =
    Machine::MemoryQuery(Machine::get_machine())
    .has_affinity_to(rt->get_executing_processor(ctx))
    .only_kind(Memory::SYSTEM_MEM)
    .first();

  AttachLauncher task(EXTERNAL_INSTANCE, col.values_lr, col.values_lr);
  task.attach_array_soa(
    base,
    true,
    {Column::VALUE_FID},
    local_sysmem);
  PhysicalRegion result = rt->attach_external_resource(ctx, task);
  AcquireLauncher acq(col.values_lr, col.values_lr, result);
  acq.add_field(Column::VALUE_FID);
  rt->issue_acquire(ctx, acq);
  return result;
}

#define TE(f) testing::TestEval([&](){ return f; }, #f)

void
reindex_column_task_test_suite(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  register_tasks(ctx, rt);

  testing::TestRecorder<READ_WRITE> recorder(
    testing::TestLog<READ_WRITE>(
      task->regions[0].region,
      regions[0],
      task->regions[1].region,
      regions[1],
      ctx,
      rt));

#ifdef HYPERION_USE_CASACORE
  casacore::MeasRef<casacore::MEpoch> tai(casacore::MEpoch::TAI);
  casacore::MeasRef<casacore::MEpoch> utc(casacore::MEpoch::UTC);
  auto table0_meas_ref =
    MeasRefContainer::create(
      ctx,
      rt,
      {MeasRef::create(ctx, rt, "EPOCH", tai)});

  casacore::MeasRef<casacore::MDirection>
    direction(casacore::MDirection::J2000);
  casacore::MeasRef<casacore::MFrequency>
    frequency(casacore::MFrequency::GEO);
  std::unordered_map<std::string, std::vector<MeasRef>> col_measures{
    {"X", {MeasRef::create(ctx, rt, "DIRECTION", direction)}},
    {"Y", {}},
    {"Z", {MeasRef::create(ctx, rt, "EPOCH", utc)}}
  };
  std::vector<Column::Generator> column_generators{
    table0_col("X", col_measures["X"]),
    table0_col("Y", col_measures["Y"]),
    table0_col("Z", col_measures["Z"])
  };
#else
  std::vector<Column::Generator> column_generators{
    table0_col("X"),
    table0_col("Y"),
    table0_col("Z")
  };
#endif

  Table table0 =
    Table::create(
      ctx,
      rt,
      "table0",
      std::vector<Table0Axes>{Table0Axes::ROW},
      column_generators
#ifdef HYPERION_USE_CASACORE
      , table0_meas_ref
#endif
      );

  auto col_x =
    attach_table0_col(ctx, rt, table0.column(ctx, rt, "X"), table0_x);
  auto col_y =
    attach_table0_col(ctx, rt, table0.column(ctx, rt, "Y"), table0_y);
  auto col_z =
    attach_table0_col(ctx, rt, table0.column(ctx, rt, "Z"), table0_z);

  IndexColumnTask icx(table0.column(ctx, rt, "X"));
  Future icfx = icx.dispatch(ctx, rt);
  IndexColumnTask icy(table0.column(ctx, rt, "Y"));
  Future icfy = icy.dispatch(ctx, rt);
  std::vector<std::tuple<int, LogicalRegion>> ics{
    {static_cast<int>(Table0Axes::X), icfx.get_result<LogicalRegion>()},
    {static_cast<int>(Table0Axes::Y), icfy.get_result<LogicalRegion>()}
  };

  auto cz = table0.column(ctx, rt, "Z");
  ReindexColumnTask rcz_task(cz, false, cz.axes(ctx, rt), 0, ics, false);
  Future fcz = rcz_task.dispatch(ctx, rt);
  auto rcz = fcz.get_result<Column>();
  recorder.assert_true(
    "Reindexed column index space rank is 2",
    TE(rcz.rank(rt)) == 2);
  recorder.expect_true(
    "Reindexed column index space dimensions are X and Y",
    TE(rcz.axes(ctx, rt)) ==
    map_to_int(std::vector<Table0Axes>{ Table0Axes::X, Table0Axes::Y }));
  {
    RegionRequirement req(rcz.values_lr, READ_ONLY, EXCLUSIVE, rcz.values_lr);
    req.add_field(Column::VALUE_FID);
    PhysicalRegion pr = rt->map_region(ctx, req);
    Rect<2> d(rt->get_index_space_domain(rcz.values_lr.get_index_space()));
    const FieldAccessor<
      READ_ONLY, unsigned, 2, coord_t,
      AffineAccessor<unsigned, 2, coord_t>, false> z(pr, Column::VALUE_FID);
    recorder.expect_true(
      "Reindexed column values are correct",
      testing::TestEval(
        [&d, &z]() {
          bool all_eq = true;
          for (PointInDomainIterator<2> pid(d); all_eq && pid(); pid++)
            all_eq = z[*pid] == pid[0] * TABLE0_NUM_Y + pid[1];
          return all_eq;
        },
        "all(z[x,y] == x * TABLE0_NUM_Y + y)"));
  }

  rt->detach_external_resource(ctx, col_x);
  rt->detach_external_resource(ctx, col_y);
  rt->detach_external_resource(ctx, col_z);
}

int
main(int argc, char* argv[]) {

  AxesRegistrar::register_axes<Table0Axes>();

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
