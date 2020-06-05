/*
 * Copyright 2020 Associated Universities, Inc. Washington DC, USA.
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
#include <hyperion/TableMapper.h>

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
  VERIFY_TABLE_COLUMNS_TASK,
  VERIFY_COLUMN_GROUPS_TASK
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

const unsigned group1 = 88;
const unsigned group2 = 42;

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
  [[maybe_unused]] herr_t err = H5Tenum_insert(result, "ROW", &a);
  assert(err >= 0);
  a = Table0Axes::W;
  err = H5Tenum_insert(result, "W", &a);
  assert(err >= 0);
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

  AttachLauncher task(EXTERNAL_INSTANCE, col.region, col.region);
  task.attach_array_soa(base, false, {col.fid}, local_sysmem);
  PhysicalRegion result = runtime->attach_external_resource(context, task);
  AcquireLauncher acq(col.region, col.region, result);
  acq.add_field(col.fid);
  runtime->issue_acquire(context, acq);
  return result;
}

#if HAVE_CXX17
#define TE(f) testing::TestEval([&](){ return f; }, #f)
#else
#define TE(f) testing::TestEval<std::function<bool()>>([&](){ return f; }, #f)
#endif

bool
verify_xyz(
  const PhysicalColumnTD<HYPERION_TYPE_UINT, 1, 1, Legion::AffineAccessor>& col,
  unsigned *ary) {

  bool result = true;
  auto vals = col.accessor<READ_ONLY, true>();
  for (PointInDomainIterator<1> pid(col.domain()); pid() && result; pid++)
    result = ary[pid[0]] == vals[*pid];
  return result;
}

bool
verify_w(
  const PhysicalColumnTD<HYPERION_TYPE_UINT, 1, 2, Legion::AffineAccessor>& col,
  unsigned *ary) {

  bool result = true;
  auto vals = col.accessor<READ_ONLY, true>();
  for (PointInDomainIterator<2> pid(col.domain()); pid() && result; pid++)
    result = ary[TABLE0_NUM_W_COLS * pid[0] + pid[1]] == vals[*pid];
  return result;
}

void
verify_table_columns_task(
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

  const Table::Desc* desc = static_cast<const Table::Desc*>(task->args);

  auto ptcr =
    PhysicalTable::create(
      rt,
      *desc,
      task->regions.begin() + 2,
      task->regions.end(),
      regions.begin() + 2,
      regions.end()).value();
#if HAVE_CXX17
  auto& [pt, rit, pit] = ptcr;
#else // !HAVE_CXX17
  auto& pt = std::get<0>(ptcr);
  auto& rit = std::get<1>(ptcr);
  auto& pit = std::get<2>(ptcr);
#endif // HAVE_CXX17
  assert(rit == task->regions.end());
  assert(pit == regions.end());

  for (auto& cn : {"X"s, "Y"s, "Z"s}) {
    auto opc = pt.column(cn);
    recorder.assert_true(
      "Column '" + cn + "' exists in mapped Table",
      TE((bool)opc));
    recorder.expect_true(
      "Column '" + cn + "' has expected values",
      TE(verify_xyz(*opc.value(), col_arrays.at(cn))));
  }
  {
    auto opc = pt.column("W");
    recorder.assert_true(
      "Column 'W' exists in mapped Table",
      TE((bool)opc));
    recorder.expect_true(
      "Column 'W' has expected values",
      TE(verify_w(*opc.value(), col_arrays.at("W"))));
  }

  // {
  //   constexpr unsigned NUM_PARTS = 3;
  //   constexpr unsigned BLOCK_SZ = TABLE0_NUM_ROWS / NUM_PARTS;
  //   auto xyz_part =
  //     ColumnSpacePartition::create(
  //       ctx,
  //       rt,
  //       xyz_space,
  //       std::vector<std::pair<Table0Axes,coord_t>>{
  //         {Table0Axes::ROW, BLOCK_SZ}})
  //     .get<ColumnSpacePartition>();
  //   auto w_part =
  //     xyz_part.project_onto(ctx, rt, w_space)
  //     .get<ColumnSpacePartition>();

  //   for (unsigned p = 0; p < NUM_PARTS; ++p) {
  //     {
  //       auto xyz_p_is = rt->get_index_subspace(xyz_part.column_ip, p);
  //       recorder.expect_true(
  //         "Row partition " + std::to_string(p) + " has expected row numbers",
  //         testing::TestEval(
  //           [&xyz_p_is, p, rt]() {
  //             bool result = true;
  //             for (PointInDomainIterator<1> pid(
  //                    rt->get_index_space_domain(xyz_p_is));
  //                  result && pid();
  //                  pid++) {
  //               result = pid[0] / BLOCK_SZ == p;
  //             }
  //             return result;
  //           }));
  //       for (auto& c : {"X", "Y", "Z"}) {
  //         std::string cstr(c);
  //         auto col = cols.at(cstr);
  //         auto pr = col_prs.at(cstr);
  //         recorder.expect_true(
  //           "Column '" + cstr + "' has expected values in row partition "
  //           + std::to_string(p),
  //           TE(verify_xyz(ctx, rt, col, xyz_p_is, pr, col_arrays.at(cstr))));
  //       }
  //     }
  //     {
  //       auto w_p_is = rt->get_index_subspace(w_part.column_ip, p);
  //       recorder.expect_true(
  //         "Projected row partition " + std::to_string(p)
  //         + " has expected row numbers",
  //         testing::TestEval(
  //           [&w_p_is, p, rt]() {
  //             bool result = true;
  //             for (PointInDomainIterator<2> pid(
  //                    rt->get_index_space_domain(w_p_is));
  //                  result && pid();
  //                  pid++)
  //               result = pid[0] / BLOCK_SZ == p;
  //             return result;
  //           }));
  //       auto col = cols.at("W");
  //       auto pr = col_prs.at("W");
  //       recorder.expect_true(
  //         "Column 'W' has expected values in row partition "
  //         + std::to_string(p),
  //         TE(verify_w(ctx, rt, col, w_p_is, pr, col_arrays.at("W"))));
  //     }
  //   }
  //   xyz_part.destroy(ctx, rt);
  //   w_part.destroy(ctx, rt);
  // }
  // {
  //   constexpr unsigned BLOCK_SZ = 2;
  //   constexpr unsigned NUM_PARTS =
  //     (TABLE0_NUM_W_COLS + (BLOCK_SZ - 1)) / BLOCK_SZ;
  //   auto w_part =
  //     ColumnSpacePartition::create(
  //       ctx,
  //       rt,
  //       w_space,
  //       std::vector<std::pair<Table0Axes,coord_t>>{
  //         {Table0Axes::W, BLOCK_SZ}})
  //     .get<ColumnSpacePartition>();
  //   auto xyz_part =
  //     w_part.project_onto(ctx, rt, xyz_space)
  //     .get<ColumnSpacePartition>();
  //   for (unsigned p = 0; p < NUM_PARTS; ++p) {
  //     {
  //       auto w_p_is = rt->get_index_subspace(w_part.column_ip, p);
  //       recorder.expect_true(
  //         "w-axis partition " + std::to_string(p)
  //         + " has expected index values",
  //         testing::TestEval(
  //           [&w_p_is, p, rt]() {
  //             bool result = true;
  //             for (PointInDomainIterator<2> pid(
  //                    rt->get_index_space_domain(w_p_is));
  //                  result && pid();
  //                  pid++)
  //               result = pid[1] / BLOCK_SZ == p;
  //             return result;
  //           }));
  //     }
  //     {
  //       auto xyz_p_is = rt->get_index_subspace(xyz_part.column_ip, p);
  //       recorder.expect_true(
  //         "Projected w-axis partition " + std::to_string(p)
  //         + " has expected index values",
  //         testing::TestEval(
  //           [&xyz_p_is, p, rt]() {
  //             std::vector<coord_t> rows(TABLE0_NUM_ROWS);
  //             std::iota(rows.begin(), rows.end(), 0);
  //             auto rows_end = rows.end();
  //             for (PointInDomainIterator<1> pid(
  //                    rt->get_index_space_domain(xyz_p_is));
  //                  pid();
  //                  pid++)
  //               rows_end = std::remove(rows.begin(), rows_end, pid[0]);
  //             rows.erase(rows_end, rows.end());
  //             return rows.size() == 0;
  //           }));
  //     }
  //   }
  // }
}

void
verify_column_groups_task(
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

  const Table::Desc* desc = static_cast<const Table::Desc*>(task->args);

  auto ptcr =
    PhysicalTable::create(
      rt,
      *desc,
      task->regions.begin() + 2,
      task->regions.end(),
      regions.begin() + 2,
      regions.end()).value();
#if HAVE_CXX17
  auto& [pt, rit, pit] = ptcr;
#else // !HAVE_CXX17
  auto& pt = std::get<0>(ptcr);
  auto& rit = std::get<1>(ptcr);
  auto& pit = std::get<2>(ptcr);
#endif // HAVE_CXX17
  assert(rit == task->regions.end());
  assert(pit == regions.end());

  auto prx = pt.column("X").value()->values().value();
  auto pry = pt.column("Y").value()->values().value();
  auto prz = pt.column("Z").value()->values().value();

  recorder.expect_true(
    "Columns X and Y, in a common mapped group, share a PhysicalRegion",
    TE(prx == pry));
  recorder.expect_false(
    "Columns X and Z, in different mapped groups, "
    "have different PhysicalRegions",
    TE(prx == prz));
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

  IndexTreeL wr2(
    {{1, IndexTreeL(TABLE0_ROWS0_NUM_W)},
     {1, IndexTreeL(TABLE0_ROWS1_NUM_W)}});
  IndexTreeL wtree(
    wr2,
    (TABLE0_NUM_ROWS / 2) * (TABLE0_ROWS0_NUM_W + TABLE0_ROWS1_NUM_W));

  Table table0;
  {
    auto xyz_is = rt->create_index_space(ctx, Rect<1>(0, TABLE0_NUM_ROWS - 1));
    auto xyz_space =
      ColumnSpace::create(
        ctx,
        rt,
        std::vector<Table0Axes>{Table0Axes::ROW},
        xyz_is,
        false);

    auto w_is = tree_index_space(wtree, ctx, rt);
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
         Keywords(),
         MeasRef::create(ctx, rt, direction),
         CXX_OPTIONAL_NAMESPACE::nullopt)},
      {"Y",
       TableField(
         HYPERION_TYPE_UINT,
         COL_Y,
         Keywords(),
         MeasRef(),
         CXX_OPTIONAL_NAMESPACE::nullopt)},
      {"Z",
       TableField(
         HYPERION_TYPE_UINT,
         COL_Z,
         Keywords(),
         MeasRef::create(ctx, rt, utc),
         CXX_OPTIONAL_NAMESPACE::nullopt)}
    };
#else
    std::vector<std::pair<std::string, TableField>> xyz_fields{
      {"X", TableField(HYPERION_TYPE_UINT, COL_X)},
      {"Y", TableField(HYPERION_TYPE_UINT, COL_Y)},
      {"Z", TableField(HYPERION_TYPE_UINT, COL_Z)}
    };
#endif
    std::vector<std::pair<std::string, TableField>> w_fields{
      {"W", TableField(HYPERION_TYPE_UINT, COL_W)}
    };

    table0 =
      Table::create(
        ctx,
        rt,
        xyz_space,
        {{xyz_space, xyz_fields}, {w_space, w_fields}});
  }
// #ifdef HYPERION_USE_CASACORE
//   recorder.expect_true(
//     "Create expected table measures using table name prefix",
//     testing::TestEval(
//       [&table0, &ctx, rt]() {
//         return
//           verify_mrc_names(ctx, rt, table0.meas_refs, {"EPOCH"});
//       }));
//   recorder.expect_true(
//     "'X' column DIRECTION measure is that defined by the column",
//     TE(table0.column(ctx, rt, "X").meas_ref == col_measures["X"]["DIRECTION"]));
//   recorder.expect_true(
//     "'Y' column has no associated measure",
//     TE(table0.column(ctx, rt, "Y").meas_ref.is_empty()));
//   recorder.expect_true(
//     "'Z' column EPOCH measure is that defined by the column",
//     TE(table0.column(ctx, rt, "Z").meas_ref == col_measures["Z"]["EPOCH"]));
// #endif

  {
    auto cols = table0.columns();

    std::unordered_map<std::string, PhysicalRegion> col_prs;
    for (auto& c : {"W"s, "X"s, "Y"s, "Z"s})
      col_prs[c] = attach_table0_col(ctx, rt, cols.at(c), col_arrays.at(c));
    {
      auto reqs = table0.requirements(ctx, rt);
#if HAVE_CXX17
      auto& [treqs, tparts, tdesc] = reqs;
#else // !HAVE_CXX17
      auto& treqs = std::get<0>(reqs);
      auto& tdesc = std::get<2>(reqs);
#endif // HAVE_CXX17
      TaskLauncher pttask(
        VERIFY_TABLE_COLUMNS_TASK,
        TaskArgument(&tdesc, sizeof(tdesc)),
        Predicate::TRUE_PRED,
        table_mapper);
      pttask.add_region_requirement(task->regions[0]);
      pttask.add_region_requirement(task->regions[1]);
      for (auto& r : treqs)
        pttask.add_region_requirement(r);
      rt->unmap_all_regions(ctx);
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
  {
    Column::Requirements reqA = Column::default_requirements;
    reqA.values = Column::Req{WRITE_ONLY, EXCLUSIVE, false};
    reqA.tag = TableMapper::to_mapping_tag(group1);
    Column::Requirements reqB = Column::default_requirements;
    reqB.values = Column::Req{WRITE_ONLY, EXCLUSIVE, false};
    reqB.tag = TableMapper::to_mapping_tag(group2);
    auto reqs =
      table0.requirements(
        ctx,
        rt,
        ColumnSpacePartition(),
        {{"X", reqA},
         {"Y", reqA},
         {"Z", reqB},
         {"W", CXX_OPTIONAL_NAMESPACE::nullopt}});
#if HAVE_CXX17
    auto& [treqs, tparts, tdesc] = reqs;
#else // !HAVE_CXX17
    auto& treqs = std::get<0>(reqs);
    auto& tdesc = std::get<2>(reqs);
#endif // HAVE_CXX17
    TaskLauncher vcgtask(
      VERIFY_COLUMN_GROUPS_TASK,
      TaskArgument(&tdesc, sizeof(tdesc)),
      Predicate::TRUE_PRED,
      table_mapper);
    vcgtask.add_region_requirement(task->regions[0]);
    vcgtask.add_region_requirement(task->regions[1]);
    for (auto& r : treqs)
      vcgtask.add_region_requirement(r);
    rt->unmap_all_regions(ctx);
    rt->execute_task(ctx, vcgtask);
  }
  rt->remap_region(ctx, regions[0]);
  rt->remap_region(ctx, regions[1]);
  recorder.update_position();
  // do tests of column removal and addition in this task, since field additions
  // and removals must be lexically scoped, and this task is where we initially
  // added the columns we're about to remove
  {
    auto rm = table0.remove_columns(ctx, rt, {"W"});
    recorder.expect_true(
      "Call to remove 'W' returned TRUE",
      TE(rm));
  }
  {
    auto cols = table0.columns();
    recorder.expect_true(
      "Column 'W' successfully removed from table",
      TE(cols.find("W") == cols.end()));
    recorder.expect_true(
      "Table has three columns after removal of 'W'",
      TE(cols.size() == 3));
    recorder.expect_true(
      "Columns 'X', 'Y', and 'Z' present in table after removal of 'W'",
      TE(cols.find("X") != cols.end()
         && cols.find("Y") != cols.end()
         && cols.find("Z") != cols.end()));
  }
  {
    auto rm = table0.remove_columns(ctx, rt, {"X", "Z"});
    recorder.expect_true(
      "Call to remove 'X' and 'Z' returned TRUE",
      TE(rm));
  }
  {
    auto cols = table0.columns();
    recorder.expect_true(
      "Columns 'X' and 'Z' successfully removed from table",
      TE(cols.find("X") == cols.end()
         && cols.find("Z") == cols.end()));
    recorder.expect_true(
      "Table has one column after removal of 'X' and 'Z",
      TE(cols.size() == 1));
    recorder.expect_true(
      "Column 'Y' present in table after removal of 'X' and 'Z'",
      TE(cols.find("Y") != cols.end()));
  }
  {
    // need a new instance of w_space, as the previous was owned and deleted by
    // the table0
    auto w_is = tree_index_space(wtree, ctx, rt);
    auto w_space =
      ColumnSpace::create(
        ctx,
        rt,
        std::vector<Table0Axes>{Table0Axes::ROW, Table0Axes::W},
        w_is,
        false);
    std::vector<std::pair<std::string, TableField>> w_fields{
      {"W", TableField(HYPERION_TYPE_UINT, COL_W)}
    };
    auto add = table0.add_columns(ctx, rt, {{w_space, w_fields}});
    recorder.expect_true(
      "Call to add 'W' returned TRUE",
      TE(add));
  }
  {
    auto cols = table0.columns();
    recorder.expect_true(
      "Column 'W' successfully added to table",
      TE(cols.find("W") != cols.end()));
  }
  {
    auto cs = table0.columns().at("Y").cs;
    std::vector<std::pair<std::string, TableField>> xz_fields{
      {"X", TableField(HYPERION_TYPE_UINT, COL_X)},
      {"Z", TableField(HYPERION_TYPE_UINT, COL_Z)}
    };
    auto add = table0.add_columns(ctx, rt, {{cs, xz_fields}});
    recorder.expect_true(
      "Call to add 'X' and 'Z' returned TRUE",
      TE(add));
  }
  {
    auto cols = table0.columns();
    recorder.expect_true(
      "Columns 'X' and 'Z' successfully added to table",
      TE(cols.find("X") != cols.end()
         && cols.find("Z") != cols.end()));
    auto cs = cols.at("Y").cs;
    recorder.expect_true(
      "Columns 'X' and 'Z' share 'Y's ColumnSpace",
      TE(cols.at("X").cs == cs && cols.at("Z").cs == cs));
  }

  table0.destroy(ctx, rt);
}

int
main(int argc, char* argv[]) {

  AxesRegistrar::register_axes<Table0Axes>();

  testing::TestSuiteDriver driver =
    testing::TestSuiteDriver::make<table_test_suite>(
      TABLE_TEST_SUITE,
      "table_test_suite");

  {
    TaskVariantRegistrar
      registrar(VERIFY_TABLE_COLUMNS_TASK, "verify_table_columns_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.add_layout_constraint_set(
      TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
      soa_row_major_layout);
    registrar.set_idempotent();
    registrar.set_leaf();
    TableMapper::add_table_layout_constraint(registrar);
    Runtime::preregister_task_variant<verify_table_columns_task>(
      registrar,
      "verify_table_columns_task");
  }
  {
    TaskVariantRegistrar
      registrar(VERIFY_COLUMN_GROUPS_TASK, "verify_column_groups_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.add_layout_constraint_set(
      TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
      soa_row_major_layout);
    registrar.add_layout_constraint_set(
      TableMapper::to_mapping_tag(group1),
      soa_row_major_layout);
    registrar.add_layout_constraint_set(
      TableMapper::to_mapping_tag(group2),
      soa_row_major_layout);
    registrar.set_idempotent();
    registrar.set_leaf();
    TableMapper::add_table_layout_constraint(registrar);
    Runtime::preregister_task_variant<verify_column_groups_task>(
      registrar,
      "verify_column_groups_task");
  }

  return driver.start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
