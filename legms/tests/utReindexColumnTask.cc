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
  REINDEX_COLUMN_TASK_TEST_SUITE,
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
#ifdef USE_HDF5
  static const hid_t h5_datatype;
#endif
};

const std::vector<std::string>
legms::Axes<Table0Axes>::names{"ROW", "X", "Y"};

#ifdef USE_HDF5
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
reindex_column_task_test_suite(
  const Task*,
  const std::vector<PhysicalRegion>& regions,
  Context context,
  Runtime* runtime) {

  register_tasks(runtime);
#ifdef USE_HDF5
  H5DatatypeManager::register_axes_datatype(
    Axes<Table0Axes>::uid,
    Axes<Table0Axes>::h5_datatype);
#endif

  testing::TestRecorder<WRITE_DISCARD> recorder(
    testing::TestLog<WRITE_DISCARD>(regions[0], regions[1], context, runtime));

  Table
    table0(
      context,
      runtime,
      "table0",
      std::vector<Table0Axes>{Table0Axes::ROW},
      {table0_col("X"),
       table0_col("Y"),
       table0_col("Z")});
  auto col_x =
    attach_table0_col(table0.column("X").get(), table0_x, context, runtime);
  auto col_y =
    attach_table0_col(table0.column("Y").get(), table0_y, context, runtime);
  auto col_z =
    attach_table0_col(table0.column("Z").get(), table0_z, context, runtime);

  IndexColumnTask icx(table0.column("X"), static_cast<int>(Table0Axes::X));
  IndexColumnTask icy(table0.column("Y"), static_cast<int>(Table0Axes::Y));
  std::vector<Future> icfs {
    icx.dispatch(context, runtime),
    icy.dispatch(context, runtime)};
  std::vector<std::shared_ptr<Column>> ics;
  std::for_each(
    icfs.begin(),
    icfs.end(),
    [&context, &runtime, &ics](Future& f) {
      auto col =
        f.get_result<ColumnGenArgs>().operator()(context, runtime);
      ics.push_back(std::move(col));
    });

  ReindexColumnTask rcz(table0.column("Z"), 0, ics, false);
  Future fz = rcz.dispatch(context, runtime);
  auto cz =
    fz.get_result<ColumnGenArgs>().operator()(context, runtime);
  recorder.assert_true(
    "Reindexed column index space rank is 2",
    TE(cz->rank()) == 2);
  recorder.expect_true(
    "Reindexed column index space dimensions are X and Y",
    TE(cz->axes()) ==
    map_to_int(std::vector<Table0Axes>{ Table0Axes::X, Table0Axes::Y }));
  {
    RegionRequirement
      req(cz->logical_region(), READ_ONLY, EXCLUSIVE, cz->logical_region());
    req.add_field(Column::value_fid);
    PhysicalRegion pr = runtime->map_region(context, req);
    DomainT<2> d(pr);
    const FieldAccessor<
      READ_ONLY, unsigned, 2, coord_t,
      AffineAccessor<unsigned, 2, coord_t>, false> z(pr, Column::value_fid);
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
