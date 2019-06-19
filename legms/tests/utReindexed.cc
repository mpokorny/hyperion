#include "testing/TestSuiteDriver.h"
#include "testing/TestRecorder.h"

#include <algorithm>
#include <memory>
#include <ostream>
#include <vector>

#include "utility.h"
#include "Table.h"
#include "Column.h"

using namespace legms;
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
struct legms::Axes<Table0Axes> {
  static const std::unordered_map<Table0Axes, std::string> names;
};

const std::unordered_map<Table0Axes, std::string>
Axes<Table0Axes>::names{
  {Table0Axes::ROW, "ROW"}, {Table0Axes::X, "X"}, {Table0Axes::Y, "Y"}};

template <>
struct legms::AxesUID<Table0Axes> {
  static constexpr const char* id = "Table0Axes";
};

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

template <>
hid_t TableT<Table0Axes>::m_h5_axes_datatype = h5_dt();

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
reindexed_test_suite(
  const Task*,
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
       table0_col("Y"),
       table0_col("Z")});
  auto col_x =
    attach_table0_col(table0.columnT("X").get(), table0_x, context, runtime);
  auto col_y =
    attach_table0_col(table0.columnT("Y").get(), table0_y, context, runtime);
  auto col_z =
    attach_table0_col(table0.columnT("Z").get(), table0_z, context, runtime);

  auto f = table0.reindexed({Table0Axes::X, Table0Axes::Y}, false);

  auto rt =
    f.get_result<TableGenArgs>().operator()<Table0Axes>(context, runtime);
  recorder.expect_true("Reindexed table is not empty", TE(!rt->is_empty()));

  recorder.expect_true(
    "Reindexed table has ('X', 'Y') index axes",
    testing::TestEval(
      [&rt]() {
        auto axes = rt->index_axes();
        return
          axes ==
          std::vector<int>{
          static_cast<int>(Table0Axes::X),
          static_cast<int>(Table0Axes::Y)};
      }));

  {
    recorder.assert_true(
      "Reindexed table has 'X' column",
      TE(rt->has_column("X")));

    auto rx = rt->columnT("X");

    recorder.expect_true(
      "Reindexed 'X' column has only 'X' axis",
      TE(rx->axes()) == std::vector<int>{static_cast<int>(Table0Axes::X)});

    recorder.assert_true(
      "Reindexed 'X' column has expected size",
      testing::TestEval(
        [&rx, &context, runtime]() {
          auto is = rx->index_space();
          auto dom = runtime->get_index_space_domain(context, is);
          Rect<1> r(dom.bounds<1,coord_t>());
          return r == Rect<1>(0, TABLE0_NUM_X - 1);
        }));

    recorder.expect_true(
      "Reindexed 'X' column has expected values",
      testing::TestEval(
        [&rx, &context, runtime]() {
          RegionRequirement
            req(
              rx->logical_region(),
              READ_ONLY,
              EXCLUSIVE,
              rx->logical_region());
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
  }
  {
    recorder.assert_true(
      "Reindexed table has 'Y' column",
      TE(rt->has_column("Y")));

    auto ry = rt->columnT("Y");

    recorder.expect_true(
      "Reindexed 'Y' column has only 'Y' axis",
      TE(ry->axes()) == std::vector<int>{static_cast<int>(Table0Axes::Y)});

    recorder.assert_true(
      "Reindexed 'Y' column has expected size",
      testing::TestEval(
        [&ry, &context, runtime]() {
          auto is = ry->index_space();
          auto dom = runtime->get_index_space_domain(context, is);
          Rect<1> r(dom.bounds<1,coord_t>());
          return r == Rect<1>(0, TABLE0_NUM_Y - 1);
        }));

    recorder.expect_true(
      "Reindexed 'Y' column has expected values",
      testing::TestEval(
        [&ry, &context, runtime]() {
          RegionRequirement
            req(
              ry->logical_region(),
              READ_ONLY,
              EXCLUSIVE,
              ry->logical_region());
          req.add_field(Column::value_fid);
          PhysicalRegion pr = runtime->map_region(context, req);
          const FieldAccessor<
            READ_ONLY, unsigned, 1, coord_t,
            AffineAccessor<unsigned, 1, coord_t>, true>
            y(pr, Column::value_fid);
          bool result =
            y[0] == OY && y[1] == OY + 1 && y[2] == OY + 2;
          runtime->unmap_region(context, pr);
          return result;
        }));
  }
  {
    recorder.assert_true(
      "Reindexed table has 'Z' column",
      TE(rt->has_column("Z")));

    auto rz = rt->columnT("Z");

    recorder.expect_true(
      "Reindexed 'Z' column has only ('X', 'Y') axes",
      TE(rz->axes()) ==
      std::vector<int>{
        static_cast<int>(Table0Axes::X),
        static_cast<int>(Table0Axes::Y)});

    recorder.expect_true(
      "Reindexed 'Z' column has expected size",
      testing::TestEval(
        [&rz, &context, runtime]() {
          auto is = rz->index_space();
          auto dom = runtime->get_index_space_domain(context, is);
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
        [&rz, &context, runtime]() {
          RegionRequirement
            req(
              rz->logical_region(),
              READ_ONLY,
              EXCLUSIVE,
              rz->logical_region());
          req.add_field(Column::value_fid);
          PhysicalRegion pr = runtime->map_region(context, req);
          const FieldAccessor<
            READ_ONLY, unsigned, 2, coord_t,
            AffineAccessor<unsigned, 2, coord_t>, true>
            z(pr, Column::value_fid);
          bool result = true;
          DomainT<2> dom(pr);
          for (PointInDomainIterator<2> pid(dom); pid(); pid++)
            result = result &&
              z[*pid] == table0_z[pid[0] * TABLE0_NUM_Y + pid[1]];
          runtime->unmap_region(context, pr);
          return result;
        }));
  }
}

int
main(int argc, char* argv[]) {

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
