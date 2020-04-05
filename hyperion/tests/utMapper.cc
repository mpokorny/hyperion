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

#include <hyperion/hyperion.h>
#include <hyperion/Table.h>
#include <hyperion/PhysicalTable.h>
#include <hyperion/DefaultMapper.h>

using namespace hyperion;
using namespace Legion;

enum {
  MAPPER_TEST_SUITE,
  VERIFY_LAYOUTS_TASK
};

enum struct Table0Axes {
  A0 = 0,
  A1
};

enum {
  COL_FOO = 1000,
  COL_C0,
  COL_C1,
  COL_C2,
  COL_C3
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
hyperion::Axes<Table0Axes>::names{"A0", "A1"};

#ifdef HYPERION_USE_HDF5
hid_t
h5_dt() {
  hid_t result = H5Tenum_create(H5T_NATIVE_UCHAR);
  Table0Axes a = Table0Axes::A0;
  herr_t err = H5Tenum_insert(result, "A0", &a);
  assert(err >= 0);
  a = Table0Axes::A1;
  err = H5Tenum_insert(result, "A1", &a);
  return result;
}

const hid_t
hyperion::Axes<Table0Axes>::h5_datatype = h5_dt();
#endif

#define TE(f) testing::TestEval([&](){ return f; }, #f)

constexpr unsigned nx = 3;
constexpr unsigned ny = 2;

// constexpr unsigned
// encode_fxy(unsigned f, unsigned x, unsigned y) {
//   return (f << 8) | (x << 4) | (y << 0);
// }

// constexpr std::tuple<unsigned, unsigned, unsigned>
// decode_fxy(unsigned fxy) {
//   return {(fxy >> 8) & 0xF, (fxy >> 4) & 0xF, (fxy >> 0) & 0xF};
// }

// unsigned c0[nx * ny]{
//   encode_fxy(0, 0, 0), encode_fxy(0, 0, 1),
//   encode_fxy(0, 1, 0), encode_fxy(0, 1, 1),
//   encode_fxy(0, 2, 0), encode_fxy(0, 2, 1)};
// unsigned c1[nx * ny]{
//   encode_fxy(1, 0, 0), encode_fxy(1, 0, 1),
//   encode_fxy(1, 1, 0), encode_fxy(1, 1, 1),
//   encode_fxy(1, 2, 0), encode_fxy(1, 2, 1)};
// unsigned c2[nx * ny]{
//   encode_fxy(2, 0, 0), encode_fxy(2, 0, 1),
//   encode_fxy(2, 1, 0), encode_fxy(2, 1, 1),
//   encode_fxy(2, 2, 0), encode_fxy(2, 2, 1)};
// unsigned c3[nx * ny]{
//   encode_fxy(3, 0, 0), encode_fxy(3, 0, 1),
//   encode_fxy(3, 1, 0), encode_fxy(3, 1, 1),
//   encode_fxy(3, 2, 0), encode_fxy(3, 2, 1)};

// PhysicalRegion
// attach_col(Context ctx, Runtime* rt, const Column& col, unsigned *base) {

//   const Memory local_sysmem =
//     Machine::MemoryQuery(Machine::get_machine())
//     .has_affinity_to(rt->get_executing_processor(ctx))
//     .only_kind(Memory::SYSTEM_MEM)
//     .first();

//   AttachLauncher task(EXTERNAL_INSTANCE, col.vlr, col.vlr);
//   task.attach_array_soa(base, false, {col.fid}, local_sysmem);
//   PhysicalRegion result = rt->attach_external_resource(ctx, task);
//   AcquireLauncher acq(col.vlr, col.vlr, result);
//   acq.add_field(col.fid);
//   rt->issue_acquire(ctx, acq);
//   return result;
// }

bool
verify_soa_row_major_layout(
  const PhysicalColumnTD<HYPERION_TYPE_UINT, 1, 2, AffineAccessor>& pc) {

  auto a = pc.accessor<READ_ONLY>();
  bool result = true;
  for (PointInRectIterator<2> pir(pc.rect()); result && pir(); pir++)
    result = a.ptr(*pir) == a.ptr({0, 0}) + pir[0] * ny + pir[1];
  return result;
}

bool
verify_aos_column_major_layout(
  const PhysicalColumnTD<HYPERION_TYPE_UINT, 1, 2, AffineAccessor>& pc) {

  auto a = pc.accessor<READ_ONLY>();
  bool result = true;
  for (PointInRectIterator<2> pir(pc.rect()); result && pir(); pir++)
    result = a.ptr(*pir) == a.ptr({0, 0}) + 2 * (pir[1] * nx + pir[0]);
  return result;
}

void
verify_layouts_task(
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

  PhysicalColumnTD<HYPERION_TYPE_UINT, 1, 2, AffineAccessor>
    pc0(*pt.column("c0").value());
  PhysicalColumnTD<HYPERION_TYPE_UINT, 1, 2, AffineAccessor>
    pc1(*pt.column("c1").value());
  PhysicalColumnTD<HYPERION_TYPE_UINT, 1, 2, AffineAccessor>
    pc2(*pt.column("c2").value());
  PhysicalColumnTD<HYPERION_TYPE_UINT, 1, 2, AffineAccessor>
    pc3(*pt.column("c3").value());

  auto a0 = pc0.accessor<WRITE_ONLY>().ptr({0, 0});
  auto a1 = pc1.accessor<WRITE_ONLY>().ptr({0, 0});
  if (a0 < a1)
    recorder.expect_true(
      "In SOA, c1 array begins at c0 array end",
      TE(a1 == a0 + nx * ny));
  else
    recorder.expect_true(
      "In SOA, c0 array begins at c1 array end",
      TE(a1 == a0 + nx * ny));
  recorder.expect_true(
    "c0 array has SOA row-major layout",
    TE(verify_soa_row_major_layout(pc0)));
  recorder.expect_true(
    "c1 array has SOA row-major layout",
    TE(verify_soa_row_major_layout(pc1)));

  auto a2 = pc2.accessor<WRITE_ONLY>().ptr({0, 0});
  auto a3 = pc3.accessor<WRITE_ONLY>().ptr({0, 0});
  if (a2 < a3)
    recorder.expect_true(
      "In AOS, c2 array is interleaved with c3 array",
      TE(a3 == a2 + 1));
  else
    recorder.expect_true(
      "In AOS, c2 array is interleaved with c3 array",
      TE(a2 == a3 + 1));
  recorder.expect_true(
    "c2 array has AOS column-major layout",
    TE(verify_aos_column_major_layout(pc2)));
  recorder.expect_true(
    "c3 array has AOS column-major layout",
    TE(verify_aos_column_major_layout(pc3)));
}

void
mapper_test_suite(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  register_tasks(ctx, rt);

  testing::TestLog<READ_WRITE> log(
    task->regions[0].region,
    regions[0],
    task->regions[1].region,
    regions[1],
    ctx,
    rt);

  IndexSpace is2 =
    rt->create_index_space(ctx, Rect<2>({0, 0}, {nx - 1, ny - 1}));
  ColumnSpace cs2 =
    ColumnSpace::create(
      ctx,
      rt,
      std::vector<Table0Axes>{Table0Axes::A0, Table0Axes::A1},
      is2,
      false);
  std::vector<std::pair<std::string, TableField>> tfs2{
    {"c0",
     TableField(
       HYPERION_TYPE_UINT, COL_C0, Keywords(), MeasRef(), std::nullopt)},
    {"c1",
     TableField(
       HYPERION_TYPE_UINT, COL_C1, Keywords(), MeasRef(), std::nullopt)},
    {"c2",
     TableField(
       HYPERION_TYPE_UINT, COL_C2, Keywords(), MeasRef(), std::nullopt)},
    {"c3",
     TableField(
       HYPERION_TYPE_UINT, COL_C3, Keywords(), MeasRef(), std::nullopt)}};
  // add a dummy column for the A0 axis only (the index axis)
  IndexSpace is1 = rt->create_index_space(ctx, Rect<1>(0, nx - 1));
  ColumnSpace cs1 =
    ColumnSpace::create(
      ctx,
      rt,
      std::vector<Table0Axes>{Table0Axes::A0},
      is1,
      false);
  std::vector<std::pair<std::string, TableField>> tfs1{
    {"foo",
     TableField(
       HYPERION_TYPE_UINT, COL_FOO, Keywords(), MeasRef(), std::nullopt)}};

  Table tb = Table::create(ctx, rt, {{cs1, true, tfs1}, {cs2, false, tfs2}});
  Column::Requirements soa_rm_creqs = Column::default_requirements;
  soa_rm_creqs.values = Column::Req{WRITE_ONLY, EXCLUSIVE, false};
  soa_rm_creqs.tag = DefaultMapper::LayoutTag::SOA_ROW_MAJOR;
  Column::Requirements aos_cm_creqs = Column::default_requirements;
  aos_cm_creqs.values = Column::Req{WRITE_ONLY, EXCLUSIVE, false};
  aos_cm_creqs.tag = DefaultMapper::LayoutTag::AOS_COLUMN_MAJOR;
  auto [reqs, parts] =
    tb.requirements(
      ctx,
      rt,
      ColumnSpacePartition(),
      READ_ONLY,
      {{"c0", soa_rm_creqs},
       {"c1", soa_rm_creqs},
       {"c2", aos_cm_creqs},
       {"c3", aos_cm_creqs},
       {"foo", std::nullopt}});
  TaskLauncher verify(
    VERIFY_LAYOUTS_TASK,
    TaskArgument(NULL, 0),
    Predicate::TRUE_PRED,
    mapper);
  verify.add_region_requirement(task->regions[0]);
  verify.add_region_requirement(task->regions[1]);
  for (auto& r : reqs)
    verify.add_region_requirement(r);
  rt->execute_task(ctx, verify);

  tb.destroy(ctx, rt, true, true);
}

int
main(int argc, char** argv) {

  AxesRegistrar::register_axes<Table0Axes>();

  testing::TestSuiteDriver driver =
    testing::TestSuiteDriver::make<mapper_test_suite>(
      MAPPER_TEST_SUITE,
      "mapper_test_suite",
      200);

  {
    TaskVariantRegistrar registrar(VERIFY_LAYOUTS_TASK, "verify_layouts_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    DefaultMapper::add_layouts(registrar);
    Runtime::preregister_task_variant<verify_layouts_task>(
      registrar,
      "verify_layouts_task");
  }

  return driver.start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
