#include "testing/TestSuiteDriver.h"
#include "testing/TestRecorder.h"

#include <algorithm>
#include <array>
#include <memory>
#include <ostream>
#include <vector>

#include "utility.h"
#include "Table.h"
#include "Column.h"

using namespace legms;
using namespace Legion;

enum {
  TABLE_TEST_SUITE,
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

const std::array<DomainPoint, TABLE0_NUM_ROWS> part_cs{
  Point<2>({0, 0}),
    Point<2>({0, 1}),
    Point<2>({0, 2}),
    Point<2>({1, 0}),
    Point<2>({1, 1}),
    Point<2>({1, 1}),
    Point<2>({2, 2}),
    Point<2>({2, 1}),
    Point<2>({2, 2}),
    Point<2>({3, 0}),
    Point<2>({0, 1}),
    Point<2>({3, 2})
    };
#define CS(ROW,XORY) static_cast<unsigned>(part_cs[ROW][XORY])

unsigned table0_x[TABLE0_NUM_ROWS] {
                   OX + CS(0,0), OX + CS(1,0), OX + CS(2,0),
                     OX + CS(3,0), OX + CS(4,0), OX + CS(5,0),
                     OX + CS(6,0), OX + CS(7,0), OX + CS(8,0),
                     OX + CS(9,0), OX + CS(10,0), OX + CS(11,0)};
unsigned table0_y[TABLE0_NUM_ROWS] {
                   OY + CS(0,1), OY + CS(1,1), OY + CS(2,1),
                     OY + CS(3,1), OY + CS(4,1), OY + CS(5,1),
                     OY + CS(6,1), OY + CS(7,1), OY + CS(8,1),
                     OY + CS(9,1), OY + CS(10,1), OY + CS(11,1)};
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

template <typename T, int DIM>
using ROAccessor =
  FieldAccessor<
  READ_ONLY,
  T,
  DIM,
  coord_t,
  AffineAccessor<T, DIM, coord_t>,
  false>;

template <typename F>
bool
cmp_values(
  Context context,
  Runtime* runtime,
  LogicalRegion col_lr,
  LogicalPartition col_lp,
  DomainT<2> colors,
  F cmp) {

  bool result = true;
  for (PointInDomainIterator<2> c(colors); result && c(); c++) {
    LogicalRegion lr =
      runtime->get_logical_subregion_by_color(context, col_lp, *c);
    RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, col_lr);
    req.add_field(Column::value_fid);
    PhysicalRegion pr = runtime->map_region(context, req);
    DomainT<1> rows =
      runtime->get_index_space_domain(context, lr.get_index_space());
    const ROAccessor<unsigned, 1> v(pr, Column::value_fid);
    for (PointInDomainIterator<1> r(rows); result && r(); r++)
      result = cmp(*c, v[*r]);
    runtime->unmap_region(context, pr);
  }
  return result;
}

bool
check_partition(
  Context context,
  Runtime* runtime,
  const Column* column,
  IndexPartition ip) {

  bool result = true;
  LogicalPartition col_lp =
    runtime->get_logical_partition(context, column->logical_region(), ip);
  DomainT<2> colors =
    runtime->get_index_partition_color_space<1,coord_t,2,coord_t>(
      IndexPartitionT<1>(ip));
  if (column->name() == "X")
    result =
      cmp_values(
        context,
        runtime,
        column->logical_region(),
        col_lp,
        colors,
        [](Point<2> c, unsigned v) { return v == OX + c[0]; });
  else if (column->name() == "Y")
    result =
      cmp_values(
        context,
        runtime,
        column->logical_region(),
        col_lp,
        colors,
        [](Point<2> c, unsigned v) { return v == OY + c[1]; });
  else // column->name() == "Z"
    result =
      cmp_values(
        context,
        runtime,
        column->logical_region(),
        col_lp,
        colors,
        [](Point<2> c, unsigned v) {
          return CS(v, 0) == c[0] && CS(v, 1) == c[1];
        });
  runtime->destroy_logical_partition(context, col_lp);
  return result;
}

void
table_test_suite(
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
       table0_col("Y"),
       table0_col("Z")});
  auto col_x =
    attach_table0_col(table0.column("X").get(), table0_x, context, runtime);
  auto col_y =
    attach_table0_col(table0.column("Y").get(), table0_y, context, runtime);
  auto col_z =
    attach_table0_col(table0.column("Z").get(), table0_z, context, runtime);

  std::unordered_map<std::string, PhysicalRegion> cols{
    {"X", col_x},
    {"Y", col_y},
    {"Z", col_z}};

  auto fparts =
    table0.partition_by_value(
      context,
      runtime,
      std::vector<Table0Axes>{Table0Axes::X, Table0Axes::Y});

  recorder.assert_true(
    "IndexPartitions named for all table columns",
    TE(fparts.count("X") == 1
       && fparts.count("Y") == 1
       && fparts.count("Z") == 1));

  std::unordered_map<std::string, IndexPartition> parts;
  std::transform(
    fparts.begin(),
    fparts.end(),
    std::inserter(parts, parts.end()),
    [](auto& c_f) {
      auto& [c, f] = c_f;
      return std::make_pair(c, f.template get_result<IndexPartition>());
    });

  recorder.expect_true(
    "All column IndexPartitions are non-empty",
    TE(
      std::all_of(
        parts.begin(),
        parts.end(),
        [](auto& p) { return p.second != IndexPartition::NO_PART; })));

  recorder.expect_true(
    "All column IndexPartitions are one dimensional",
    TE(
      std::all_of(
        parts.begin(),
        parts.end(),
        [](auto& p) { return p.second.get_dim() == 1; })));

  recorder.expect_true(
    "All column IndexPartitions have a two-dimensional color space",
    TE(
      std::all_of(
        parts.begin(),
        parts.end(),
        [&context, runtime](auto& p) {
          return
            runtime->get_index_partition_color_space(context, p.second)
            .get_dim() == 2;
        })));

  recorder.expect_true(
    "All column IndexPartitions have the same color space",
    TE(
      std::all_of(
        ++parts.begin(),
        parts.end(),
        [cs=runtime->get_index_partition_color_space(
            context,
            parts.begin()->second),
         &context, runtime](auto& p) {
          return
            runtime->get_index_partition_color_space(context, p.second) == cs;
        })));

  recorder.expect_true(
    "Column IndexPartition has expected color space",
    testing::TestEval(
      [&parts, &context, runtime]() {
        auto cs =
          runtime->get_index_partition_color_space(
            IndexPartitionT<2>(parts.begin()->second));
        std::set<Point<2>> part_dom(part_cs.begin(), part_cs.end());
        bool dom_in_cs =
          std::all_of(
            part_dom.begin(),
            part_dom.end(),
            [&cs](auto& p) {
              return cs.contains(p);
            });
        return cs.get_volume() == part_dom.size() && dom_in_cs;
      }));

  recorder.expect_true(
    "All columns partitioned as expected",
    TE(
      std::all_of(
        parts.begin(),
        parts.end(),
        [&table0, &context, runtime](auto& p) {
          return
            check_partition(
              context,
              runtime,
              table0.column(p.first).get(), p.second);
        })));
}

int
main(int argc, char* argv[]) {

  testing::TestSuiteDriver driver =
    testing::TestSuiteDriver::make<table_test_suite>(
      TABLE_TEST_SUITE,
      "table_test_suite");

  return driver.start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
