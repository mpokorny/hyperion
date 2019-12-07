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

#ifdef HYPERION_USE_CASACORE
# include <hyperion/MeasRefContainer.h>
#endif

#include <algorithm>
#include <array>
#include <memory>
#include <ostream>
#include <vector>

using namespace hyperion;
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
table0_col(
  const std::string& name
#ifdef HYPERION_USE_CASACORE
  , const std::unordered_map<std::string, MeasRef>& measures
  , const std::optional<std::string>& meas_name = std::nullopt
#endif
  ) {
  return
    [=]
    (Context ctx, Runtime* rt, const std::string& name_prefix
#ifdef HYPERION_USE_CASACORE
     , const MeasRefContainer& table_mr
#endif
      ) {
#ifdef HYPERION_USE_CASACORE
      MeasRef mr;
      bool own_mr = false;
      if (meas_name) {
        auto mrs =
          MeasRefContainer::create(ctx, rt, measures, table_mr);
        std::tie(mr, own_mr) = mrs.lookup(ctx, rt, meas_name.value());
      }
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
          own_mr,
          meas_name.value_or(""),
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
  PhysicalRegion result = runtime->attach_external_resource(context, task);
  AcquireLauncher acq(col.values_lr, col.values_lr, result);
  acq.add_field(Column::VALUE_FID);
  runtime->issue_acquire(context, acq);
  return result;
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
  PhysicalRegion col_pr,
  LogicalPartition col_lp,
  DomainT<2> colors,
  F cmp) {

  bool result = true;
  for (PointInDomainIterator<2> c(colors); result && c(); c++) {
    LogicalRegion lr =
      runtime->get_logical_subregion_by_color(context, col_lp, *c);
    DomainT<1> rows =
      runtime->get_index_space_domain(context, lr.get_index_space());
    const ROAccessor<unsigned, 1> v(col_pr, Column::VALUE_FID);
    for (PointInDomainIterator<1> r(rows); result && r(); r++)
      result = cmp(*c, v[*r]);
  }
  return result;
}

bool
check_partition(
  Context context,
  Runtime* runtime,
  const std::unordered_map<std::string, PhysicalRegion>& prs,
  const Column& column,
  IndexPartition ip) {

  bool result = true;
  LogicalPartition col_lp =
    runtime->get_logical_partition(context, column.values_lr, ip);
  DomainT<2> colors =
    runtime->get_index_partition_color_space<1,coord_t,2,coord_t>(
      IndexPartitionT<1>(ip));
  auto colname = column.name(context, runtime);
  if (colname == "X")
    result =
      cmp_values(
        context,
        runtime,
        prs.at("X"),
        col_lp,
        colors,
        [](Point<2> c, unsigned v) { return v == OX + c[0]; });
  else if (colname == "Y")
    result =
      cmp_values(
        context,
        runtime,
        prs.at("Y"),
        col_lp,
        colors,
        [](Point<2> c, unsigned v) { return v == OY + c[1]; });
  else // colname == "Z"
    result =
      cmp_values(
        context,
        runtime,
        prs.at("Z"),
        col_lp,
        colors,
        [](Point<2> c, unsigned v) {
          return CS(v, 0) == c[0] && CS(v, 1) == c[1];
        });
  runtime->destroy_logical_partition(context, col_lp);
  return result;
}

#ifdef HYPERION_USE_CASACORE
static bool
verify_mrc_names(
  Context ctx,
  Runtime* rt,
  const MeasRefContainer& mrc,
  std::set<std::string> expected) {

  std::set<std::string> names;
  if (mrc.lr != LogicalRegion::NO_REGION) {
    RegionRequirement req(mrc.lr, READ_ONLY, EXCLUSIVE, mrc.lr);
    req.add_field(MeasRefContainer::NAME_FID);
    auto pr = rt->map_region(ctx, req);
    const MeasRefContainer::NameAccessor<READ_ONLY>
      nms(pr, MeasRefContainer::NAME_FID);
    for (PointInDomainIterator<1>
           pid(rt->get_index_space_domain(mrc.lr.get_index_space()));
         pid();
         pid++)
      names.insert(nms[*pid]);
    rt->unmap_region(ctx, pr);
  }
  return names == expected;
}
#endif // HYPERION_USE_CASACORE

void
table_test_suite(
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
  auto table0_epoch = MeasRef::create(ctx, rt, tai);

  casacore::MeasRef<casacore::MDirection>
    direction(casacore::MDirection::J2000);
  casacore::MeasRef<casacore::MFrequency>
    frequency(casacore::MFrequency::GEO);
  std::unordered_map<std::string, std::unordered_map<std::string, MeasRef>>
    col_measures{
    {"X", {{"DIRECTION", MeasRef::create(ctx, rt, direction)}}},    
    {"Y", {}},
    {"Z", {{"EPOCH", MeasRef::create(ctx, rt, utc)}}}
  };
  std::vector<Column::Generator> column_generators{
    table0_col("X", col_measures["X"], "DIRECTION"),
    table0_col("Y", col_measures["Y"]),
    table0_col("Z", col_measures["Z"], "EPOCH")
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
      , {{"EPOCH", table0_epoch}}
      , MeasRefContainer()
#endif
      );

#ifdef HYPERION_USE_CASACORE
  recorder.expect_true(
    "Create expected table measures using table name prefix",
    testing::TestEval(
      [&table0, &ctx, rt]() {
        return
          verify_mrc_names(ctx, rt, table0.meas_refs, {"EPOCH"});
      }));
  recorder.expect_true(
    "'X' column DIRECTION measure is that defined by the column",
    TE(table0.column(ctx, rt, "X").meas_ref == col_measures["X"]["DIRECTION"]));
  recorder.expect_true(
    "'Y' column has no associated measure",
    TE(table0.column(ctx, rt, "Y").meas_ref.is_empty()));
  recorder.expect_true(
    "'Z' column EPOCH measure is that defined by the column",
    TE(table0.column(ctx, rt, "Z").meas_ref == col_measures["Z"]["EPOCH"]));
#endif

  auto col_x =
    attach_table0_col(ctx, rt, table0.column(ctx, rt, "X"), table0_x);
  auto col_y =
    attach_table0_col(ctx, rt, table0.column(ctx, rt, "Y"), table0_y);
  auto col_z =
    attach_table0_col(ctx, rt, table0.column(ctx, rt, "Z"), table0_z);

  std::unordered_map<std::string, PhysicalRegion> cols{
    {"X", col_x},
    {"Y", col_y},
    {"Z", col_z}};

  auto fparts =
    table0.partition_by_value(
      ctx,
      rt,
      std::vector<Table0Axes>{Table0Axes::X, Table0Axes::Y});

  recorder.assert_true(
    "IndexPartitions named for all table columns",
    TE(fparts.count("X") == 1
       && fparts.count("Y") == 1
       && fparts.count("Z") == 1));

  std::unordered_map<std::string, ColumnPartition> parts;
  for (auto&fp : fparts) {
    auto& [c, f] = fp;
    auto cp = f.template get_result<ColumnPartition>();
    parts.emplace(c, cp);
  }

  recorder.expect_true(
    "All column IndexPartitions are non-empty",
    TE(
      std::all_of(
        parts.begin(),
        parts.end(),
        [](auto& p) {
          return p.second.index_partition != IndexPartition::NO_PART;
        })));

  recorder.expect_true(
    "All column IndexPartitions are one dimensional",
    TE(
      std::all_of(
        parts.begin(),
        parts.end(),
        [](auto& p) {
          return p.second.index_partition.get_dim() == 1;
        })));

  recorder.expect_true(
    "All column IndexPartitions have a two-dimensional color space",
    TE(
      std::all_of(
        parts.begin(),
        parts.end(),
        [&ctx, rt](auto& p) {
          return
            rt->get_index_partition_color_space(
              ctx,
              p.second.index_partition)
            .get_dim() == 2;
        })));

  recorder.expect_true(
    "All column IndexPartitions have the same color space",
    TE(
      std::all_of(
        ++parts.begin(),
        parts.end(),
        [cs=rt->get_index_partition_color_space(
            ctx,
            parts.begin()->second.index_partition),
         &ctx, rt](auto& p) {
          return
            rt->get_index_partition_color_space(
              ctx,
              p.second.index_partition)
            == cs;
        })));

  recorder.expect_true(
    "Column IndexPartition has expected color space",
    testing::TestEval(
      [&parts, &ctx, rt]() {
        auto cs =
          rt->get_index_partition_color_space(
            IndexPartitionT<2>(parts.begin()->second.index_partition));
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
        [&cols, &table0, &ctx, rt](auto& p) {
          return
            check_partition(
              ctx,
              rt,
              cols,
              table0.column(ctx, rt, p.first),
              p.second.index_partition);
        })));

  {
    bool destroy_color_space = false;
    for (auto& p : parts) {
      std::get<1>(p).destroy(ctx, rt, destroy_color_space);
      destroy_color_space = false;
    }
  }

  for (auto& pr : {col_x, col_y, col_z}) {
    ReleaseLauncher rel(pr.get_logical_region(), pr.get_logical_region(), pr);
    rel.add_field(Column::VALUE_FID);
    rt->issue_release(ctx, rel);
    rt->unmap_region(ctx, pr);
  }
}

int
main(int argc, char* argv[]) {

  AxesRegistrar::register_axes<Table0Axes>();

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
