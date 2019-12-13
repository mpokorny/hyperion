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
  REINDEXED_TEST_SUITE,
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
unsigned table0_z[TABLE0_NUM_ROWS] {
                   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

Column::Generator
table0_col(
  const std::string& name
#ifdef HYPERION_USE_CASACORE
  , const std::optional<MeasRef>& measure
#endif
  ) {
  return [=](Context ctx, Runtime* rt, const std::string& name_prefix) {
#ifdef HYPERION_USE_CASACORE
      MeasRef mr = measure.value_or(MeasRef());
#endif
      return
        Column::create(
          ctx,
          rt,
          name,
          std::vector<Table0Axes>{Table0Axes::ROW},
          ValueType<unsigned>::DataType,
          IndexTreeL(TABLE0_NUM_ROWS),
#ifdef HYPERION_USE_CASACORE
          mr,
          std::nullopt,
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

  AttachLauncher
    task(EXTERNAL_INSTANCE, col.values_lr, col.values_lr);
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
reindexed_test_suite(
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
  casacore::MeasRef<casacore::MEpoch> utc(casacore::MEpoch::UTC);

  casacore::MeasRef<casacore::MDirection>
    direction(casacore::MDirection::J2000);
  casacore::MeasRef<casacore::MFrequency>
    frequency(casacore::MFrequency::GEO);
  auto columnX_direction = MeasRef::create(ctx, rt, direction);
  auto columnZ_epoch = MeasRef::create(ctx, rt, utc);
  std::unordered_map<std::string, std::optional<MeasRef>>
    col_measures{
    {"X", {columnX_direction}},
    {"Y", {std::nullopt}},
    {"Z", {columnZ_epoch}}};
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
      column_generators);

  auto col_x =
    attach_table0_col(ctx, rt, table0.column(ctx, rt, "X"), table0_x);
  auto col_y =
    attach_table0_col(ctx, rt, table0.column(ctx, rt, "Y"), table0_y);
  auto col_z =
    attach_table0_col(ctx, rt, table0.column(ctx, rt, "Z"), table0_z);

  auto f =
    table0.reindexed(
      ctx,
      rt,
      std::vector<Table0Axes>{Table0Axes::X, Table0Axes::Y},
      false);

  auto tb = f.template get_result<Table>();
  recorder.expect_true(
    "Reindexed table is not empty",
    TE(!tb.is_empty(ctx, rt)));

  recorder.expect_true(
    "Reindexed table has ('X', 'Y') index axes",
    testing::TestEval(
      [&ctx, rt, &tb]() {
        auto axes = tb.index_axes(ctx, rt);
        return
          axes ==
          std::vector<int>{
          static_cast<int>(Table0Axes::X),
          static_cast<int>(Table0Axes::Y)};
      }));

  auto colnames = tb.column_names(ctx, rt);
  {
    recorder.assert_true(
      "Reindexed table has 'X' column",
      TE(std::find(colnames.begin(), colnames.end(), "X") != colnames.end()));

    auto cx = tb.column(ctx, rt, "X");

    recorder.expect_true(
      "Reindexed 'X' column has only 'X' axis",
      TE(cx.axes(ctx, rt)
         == map_to_int(std::vector<Table0Axes>{Table0Axes::X})));

    recorder.assert_true(
      "Reindexed 'X' column has expected size",
      testing::TestEval(
        [&cx, &ctx, rt]() {
          auto is = cx.values_lr.get_index_space();
          auto dom = rt->get_index_space_domain(ctx, is);
          Rect<1> r(dom.bounds<1,coord_t>());
          return r == Rect<1>(0, TABLE0_NUM_X - 1);
        }));

    recorder.expect_true(
      "Reindexed 'X' column has expected values",
      testing::TestEval(
        [&cx, &ctx, rt]() {
          RegionRequirement
            req(cx.values_lr, READ_ONLY, EXCLUSIVE, cx.values_lr);
          req.add_field(Column::VALUE_FID);
          PhysicalRegion pr = rt->map_region(ctx, req);
          const FieldAccessor<
            READ_ONLY, unsigned, 1, coord_t,
            AffineAccessor<unsigned, 1, coord_t>, true>
            x(pr, Column::VALUE_FID);
          bool result =
            x[0] == OX && x[1] == OX + 1 && x[2] == OX + 2 && x[3] == OX + 3;
          rt->unmap_region(ctx, pr);
          return result;
        }));
  }
  {
    recorder.assert_true(
      "Reindexed table has 'Y' column",
      TE(std::find(colnames.begin(), colnames.end(), "Y") != colnames.end()));

    auto cy = tb.column(ctx, rt, "Y");

    recorder.expect_true(
      "Reindexed 'Y' column has only 'Y' axis",
      TE(cy.axes(ctx, rt)
         == map_to_int(std::vector<Table0Axes>{Table0Axes::Y})));

    recorder.assert_true(
      "Reindexed 'Y' column has expected size",
      testing::TestEval(
        [&cy, &ctx, rt]() {
          auto is = cy.values_lr.get_index_space();
          auto dom = rt->get_index_space_domain(ctx, is);
          Rect<1> r(dom.bounds<1,coord_t>());
          return r == Rect<1>(0, TABLE0_NUM_Y - 1);
        }));

    recorder.expect_true(
      "Reindexed 'Y' column has expected values",
      testing::TestEval(
        [&cy, &ctx, rt]() {
          RegionRequirement
            req(cy.values_lr, READ_ONLY, EXCLUSIVE, cy.values_lr);
          req.add_field(Column::VALUE_FID);
          PhysicalRegion pr = rt->map_region(ctx, req);
          const FieldAccessor<
            READ_ONLY, unsigned, 1, coord_t,
            AffineAccessor<unsigned, 1, coord_t>, true>
            y(pr, Column::VALUE_FID);
          bool result =
            y[0] == OY && y[1] == OY + 1 && y[2] == OY + 2;
          rt->unmap_region(ctx, pr);
          return result;
        }));
  }
  {
    recorder.assert_true(
      "Reindexed table has 'Z' column",
      TE(std::find(colnames.begin(), colnames.end(), "Z") != colnames.end()));

    auto cz = tb.column(ctx, rt, "Z");

    recorder.expect_true(
      "Reindexed 'Z' column has only ('X', 'Y') axes",
      TE(cz.axes(ctx, rt)
         == map_to_int(std::vector<Table0Axes>{Table0Axes::X, Table0Axes::Y})));

    recorder.expect_true(
      "Reindexed 'Z' column has expected size",
      testing::TestEval(
        [&cz, &ctx, rt]() {
          auto is = cz.values_lr.get_index_space();
          auto dom = rt->get_index_space_domain(ctx, is);
          Rect<2> r(dom.bounds<2,coord_t>());
          return
            r ==
            Rect<2>(
              Point<2>(0, 0),
              Point<2>(TABLE0_NUM_X - 1, TABLE0_NUM_Y - 1));
        }));

    recorder.expect_true(
      "Reindexed 'Z' column has expected values",
      testing::TestEval(
        [&cz, &ctx, rt]() {
          RegionRequirement
            req(cz.values_lr, READ_ONLY, EXCLUSIVE, cz.values_lr);
          req.add_field(Column::VALUE_FID);
          PhysicalRegion pr = rt->map_region(ctx, req);
          const FieldAccessor<
            READ_ONLY, unsigned, 2, coord_t,
            AffineAccessor<unsigned, 2, coord_t>, true>
            z(pr, Column::VALUE_FID);
          bool result = true;
          DomainT<2,coord_t> dom(pr);
          for (PointInDomainIterator<2> pid(dom); pid(); pid++)
            result = result &&
              z[*pid] == table0_z[pid[0] * TABLE0_NUM_Y + pid[1]];
          rt->unmap_region(ctx, pr);
          return result;
        }));
  }
}

int
main(int argc, char* argv[]) {

  AxesRegistrar::register_axes<Table0Axes>();

  testing::TestSuiteDriver driver =
    testing::TestSuiteDriver::make<reindexed_test_suite>(
      REINDEXED_TEST_SUITE,
      "reindexed_test_suite");

  return driver.start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
