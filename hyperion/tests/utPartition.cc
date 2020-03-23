/*
f * Copyright 2020 Associated Universities, Inc. Washington DC, USA.
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
#include <hyperion/tree_index_space.h>
#include <hyperion/Table.h>
#include <hyperion/ColumnSpacePartition.h>
#include <hyperion/PhysicalTable.h>

#include <algorithm>
#include <array>
#include <memory>
#include <ostream>
#include <vector>

using namespace hyperion;
using namespace Legion;

using namespace std::string_literals;

enum {
  TABLE_TEST_SUITE,
  VERIFY_PARTITIONS_TASK
};

enum struct Table0Axes {
  ROW = 0,
  W
};

enum {
  COL_W,
  COL_X,
  COL_Y,
  COL_Z
};

template <>
struct hyperion::Axes<Table0Axes> {
  static const constexpr char* uid = "Table0Axes";
  static const std::vector<std::string> names;
  static const unsigned num_axes = 2;
#ifdef HYPERION_USE_HDF5
  static const hid_t h5_datatype;
#endif
};

const std::vector<std::string>
hyperion::Axes<Table0Axes>::names{"ROW", "W"};

#ifdef HYPERION_USE_HDF5
hid_t
h5_dt() {
  hid_t result = H5Tenum_create(H5T_NATIVE_UCHAR);
  Table0Axes a = Table0Axes::ROW;
  herr_t err = H5Tenum_insert(result, "ROW", &a);
  assert(err >= 0);
  a = Table0Axes::W;
  err = H5Tenum_insert(result, "W", &a);
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
  case Table0Axes::W:
    stream << "Table0Axes::W";
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
#define TABLE0_ROWS0_NUM_W 2
#define TABLE0_ROWS1_NUM_W 3

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
#define TABLE0_NUM_W_COLS std::max(TABLE0_ROWS0_NUM_W, TABLE0_ROWS1_NUM_W)
unsigned table0_w[TABLE0_NUM_ROWS * TABLE0_NUM_W_COLS];

std::unordered_map<std::string, unsigned*> col_arrays{
  {"W", table0_w},
  {"X", table0_x},
  {"Y", table0_y},
  {"Z", table0_z}
};

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

  AttachLauncher task(EXTERNAL_INSTANCE, col.vlr, col.vlr);
  task.attach_array_soa(base, false, {col.fid}, local_sysmem);
  PhysicalRegion result = runtime->attach_external_resource(context, task);
  AcquireLauncher acq(col.vlr, col.vlr, result);
  acq.add_field(col.fid);
  runtime->issue_acquire(context, acq);
  return result;
}

#define TE(f) testing::TestEval([&](){ return f; }, #f)

bool
verify_xyz(
  const PhysicalColumnTD<HYPERION_TYPE_UINT, 1, 1, Legion::AffineAccessor>& col,
  const DomainT<1>& domain,
  unsigned *ary) {

  bool result = true;
  auto vals = col.accessor<READ_ONLY, true>();
  for (PointInDomainIterator<1> pid(domain); pid() && result; pid++)
    result = ary[pid[0]] == vals[*pid];
  return result;
}

bool
verify_w(
  const PhysicalColumnTD<HYPERION_TYPE_UINT, 1, 2, Legion::AffineAccessor>& col,
  const DomainT<2>& domain,
  unsigned *ary) {

  bool result = true;
  auto vals = col.accessor<READ_ONLY, true>();
  for (PointInDomainIterator<2> pid(domain); pid() && result; pid++)
    result = ary[TABLE0_NUM_W_COLS * pid[0] + pid[1]] == vals[*pid];
  return result;
}

void
verify_partitions_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  testing::TestRecorder<READ_WRITE> recorder(
    testing::TestLog<READ_WRITE>(
      task->regions[0].region,
      regions[0],
      task->regions[1].region,
      regions[1],
      ctx,
      rt));

  auto [pt, rit, pit] =
    PhysicalTable::create(
      rt,
      task->regions.begin() + 2,
      task->regions.end(),
      regions.begin() + 2,
      regions.end()).value();
  assert(rit == task->regions.end());
  assert(pit == regions.end());

  const unsigned *BLOCK_SZ = static_cast<const unsigned*>(task->args);
  ColumnSpacePartition xyz_part =
    task->futures[0].get_result<ColumnSpacePartition>();
  auto colW = pt.column("W").value();
  auto w_part =
    xyz_part.project_onto(
      ctx,
      rt,
      colW->parent().get_index_space(),
      colW->metadata());
  for (PointInRectIterator<1> p(
         rt->get_index_partition_color_space(xyz_part.column_ip));
       p();
       p++) {
    {
      auto xyz_p_is =
        rt->get_index_subspace(xyz_part.column_ip, DomainPoint(*p));
      recorder.expect_true(
        "Row partition " + std::to_string(p[0]) + " has expected row numbers",
        testing::TestEval(
          [&xyz_p_is, p, BLOCK_SZ, rt]() {
            bool result = true;
            for (PointInDomainIterator<1> pid(
                   rt->get_index_space_domain(xyz_p_is));
                 result && pid();
                 pid++) {
              result = pid[0] / *BLOCK_SZ == p[0];
            }
            return result;
          }));
      for (auto& c : {"X"s, "Y"s, "Z"s}) {
        auto col = pt.column(c);
        recorder.expect_true(
          "Column '" + c + "' has expected values in row partition "
          + std::to_string(p[0]),
          TE(verify_xyz(
               *col.value(),
               rt->get_index_space_domain(xyz_p_is),
               col_arrays.at(c))));
      }
    }
    {
      auto w_p_is = rt->get_index_subspace(w_part.column_ip, DomainPoint(*p));
      recorder.expect_true(
        "Projected row partition " + std::to_string(p[0])
        + " has expected row numbers",
        testing::TestEval(
          [&w_p_is, p, BLOCK_SZ, rt]() {
            bool result = true;
            for (PointInDomainIterator<2> pid(
                   rt->get_index_space_domain(w_p_is));
                 result && pid();
                 pid++)
              result = pid[0] / *BLOCK_SZ == p[0];
            return result;
          }));
      auto col = pt.column("W");
      recorder.expect_true(
        "Column 'W' has expected values in row partition "
        + std::to_string(p[0]),
        TE(verify_w(
             *col.value(),
             rt->get_index_space_domain(w_p_is),
             col_arrays.at("W"))));
    }
  }
  xyz_part.destroy(ctx, rt);
  w_part.destroy(ctx, rt);
}

void
table_test_suite(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  testing::TestRecorder<READ_WRITE> recorder(
    testing::TestLog<READ_WRITE>(
      task->regions[0].region,
      regions[0],
      task->regions[1].region,
      regions[1],
      ctx,
      rt));

  std::iota(
    &table0_w[0],
    &table0_w[sizeof(table0_w) / sizeof(table0_w[0])],
    0.0);

  auto xyz_is = rt->create_index_space(ctx, Rect<1>(0, TABLE0_NUM_ROWS - 1));
  auto xyz_space =
    ColumnSpace::create(
      ctx,
      rt,
      std::vector<Table0Axes>{Table0Axes::ROW},
      xyz_is,
      false);

  IndexSpace w_is;
  {
    IndexTreeL wr2(
      {{1, IndexTreeL(TABLE0_ROWS0_NUM_W)},
       {1, IndexTreeL(TABLE0_ROWS1_NUM_W)}});
    IndexTreeL wtree(
      wr2,
      (TABLE0_NUM_ROWS / 2) * (TABLE0_ROWS0_NUM_W + TABLE0_ROWS1_NUM_W));
    w_is = tree_index_space(wtree, ctx, rt);
  }
  auto w_space =
    ColumnSpace::create(
      ctx,
      rt,
      std::vector<Table0Axes>{Table0Axes::ROW, Table0Axes::W},
      w_is,
      false);

#ifdef HYPERION_USE_CASACORE
  casacore::MeasRef<casacore::MEpoch> utc(casacore::MEpoch::UTC);
  casacore::MeasRef<casacore::MDirection>
    direction(casacore::MDirection::J2000);
  std::vector<std::pair<std::string, TableField>> xyz_fields{
    {"X",
     TableField(
       HYPERION_TYPE_UINT,
       COL_X,
       MeasRef(),
       std::nullopt,
       Keywords())},
    {"Y",
     TableField(
       HYPERION_TYPE_UINT,
       COL_Y,
       MeasRef(),
       std::nullopt,
       Keywords())},
    {"Z",
     TableField(
       HYPERION_TYPE_UINT,
       COL_Z,
       MeasRef(),
       std::nullopt,
       Keywords())}
  };
  std::vector<std::pair<std::string, TableField>> w_fields{
    {"W",
     TableField(HYPERION_TYPE_UINT, COL_W, MeasRef(), std::nullopt, Keywords())}
  };
#else
  std::vector<std::pair<std::string, TableField>> xyz_fields{
    {"X",
     TableField(HYPERION_TYPE_UINT, COL_X, Keywords())},
    {"Y",
     TableField(HYPERION_TYPE_UINT, COL_Y, Keywords())},
    {"Z",
     TableField(HYPERION_TYPE_UINT, COL_Z, Keywords())}
  };
  std::vector<std::pair<std::string, TableField>> w_fields{
    {"W",
     TableField(HYPERION_TYPE_UINT, COL_W, Keywords())}
  };
#endif

  auto table0 =
    Table::create(
      ctx,
      rt,
      {{xyz_space, true, xyz_fields}, {w_space, false, w_fields}});

  {
    auto cols =
      Table::column_map(
        table0.columns(ctx, rt).get<Table::columns_result_t>());

    std::unordered_map<std::string, PhysicalRegion> col_prs;
    for (auto& c : {"W"s, "X"s, "Y"s, "Z"s})
      col_prs[c] = attach_table0_col(ctx, rt, cols.at(c), col_arrays.at(c));
    auto [treqs, parts] =
      table0.requirements(ctx, rt);
    rt->unmap_all_regions(ctx);
    {
      unsigned NUM_PARTS = 3;
      unsigned BLOCK_SZ = TABLE0_NUM_ROWS / NUM_PARTS;
      auto xyz_part =
        ColumnSpacePartition::create(
          ctx,
          rt,
          xyz_space,
          std::vector<std::pair<Table0Axes,coord_t>>{
            {Table0Axes::ROW, BLOCK_SZ}});

      TaskLauncher
        pttask(
          VERIFY_PARTITIONS_TASK,
          TaskArgument(&BLOCK_SZ, sizeof(BLOCK_SZ)));
      pttask.add_region_requirement(task->regions[0]);
      pttask.add_region_requirement(task->regions[1]);
      for (auto& r : treqs)
        pttask.add_region_requirement(r);
      pttask.add_future(xyz_part);
      rt->execute_task(ctx, pttask);
    }
    {
      unsigned NUM_PARTS = 2;
      unsigned BLOCK_SZ = TABLE0_NUM_ROWS / NUM_PARTS;
      auto xyz_part =
        ColumnSpacePartition::create(
          ctx,
          rt,
          xyz_space,
          std::vector<std::pair<Table0Axes,coord_t>>{
            {Table0Axes::ROW, BLOCK_SZ}});

      TaskLauncher
        pttask(
          VERIFY_PARTITIONS_TASK,
          TaskArgument(&BLOCK_SZ, sizeof(BLOCK_SZ)));
      pttask.add_region_requirement(task->regions[0]);
      pttask.add_region_requirement(task->regions[1]);
      for (auto& r : treqs)
        pttask.add_region_requirement(r);
      pttask.add_future(xyz_part);
      rt->execute_task(ctx, pttask);
    }
    for (auto& c : {"W"s, "X"s, "Y"s, "Z"s}) {
      auto pr = col_prs.at(c);
      ReleaseLauncher rel(pr.get_logical_region(), pr.get_logical_region(), pr);
      rel.add_field(cols.at(c).fid);
      rt->issue_release(ctx, rel);
      rt->detach_external_resource(ctx, pr);
    }
  }

  table0.destroy(ctx, rt, true, true);
}

int
main(int argc, char* argv[]) {

  AxesRegistrar::register_axes<Table0Axes>();

  {
    TaskVariantRegistrar
      registrar(VERIFY_PARTITIONS_TASK, "verify_partitions_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    Runtime::preregister_task_variant<verify_partitions_task>(
      registrar,
      "verify_partitions_task");
  }

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
