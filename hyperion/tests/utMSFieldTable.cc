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
#include <hyperion/hyperion.h>
#include <hyperion/utility.h>
#include <hyperion/Table.h>
#include <hyperion/MSFieldTable.h>
#include <hyperion/TableReadTask.h>
#include <hyperion/DefaultMapper.h>

#include <hyperion/testing/TestSuiteDriver.h>
#include <hyperion/testing/TestRecorder.h>
#include <hyperion/testing/TestExpression.h>

#include <casacore/ms/MeasurementSets.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <vector>

using namespace hyperion;
using namespace Legion;

namespace cc = casacore;

enum {
  MS_TEST_TASK,
  VERIFY_FIELD_TABLE_TASK
};

#define TE(f) testing::TestEval([&](){ return f; }, #f)

struct VerifyTableArgs {
  char table_path[1024];
};

void
verify_field_table(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  testing::TestLog<READ_WRITE> log(
    task->regions[0].region,
    regions[0],
    task->regions[1].region,
    regions[1],
    ctx,
    rt);
  testing::TestRecorder<READ_WRITE> recorder(log);

  auto [pt, rit, pit] =
    PhysicalTable::create(
      rt,
      task->regions.begin() + 2,
      task->regions.end(),
      regions.begin() + 2,
      regions.end())
    .value();
  assert(rit == task->regions.end());
  assert(pit == regions.end());

  MSFieldTable table(pt);

  const VerifyTableArgs *args =
    static_cast<const VerifyTableArgs*>(task->args);
  CXX_FILESYSTEM_NAMESPACE::path field_path(args->table_path);
  cc::MeasurementSet ms(
    field_path.parent_path().string(),
    cc::TableLock::PermanentLockingWait);
  cc::ROMSFieldColumns ms_field(ms.field());

  recorder.expect_true(
    "Table has NAME column",
    TE(table.has_name()));
  // TODO: For now, we don't check values, as utMS already has lots of such
  // tests; however, it might be useful to have some tests that confirm the
  // conversion Table->RegionRequiremnts->PhysicalTable->PhysicalColumn is
  // correct.
  recorder.expect_true(
    "Table has CODE column",
    TE(table.has_code()));
  recorder.expect_true(
    "Table has TIME column",
    TE(table.has_time()));
  recorder.expect_true(
    "Table has TIME measures",
    TE(table.has_time_meas()));
  recorder.expect_true(
    "Table TIME measures are correct",
    testing::TestEval(
      [&]() {
        auto col = ms_field.timeMeas();
        auto tm_col = table.time_meas<AffineAccessor>();
        auto tm_meas = tm_col.meas_accessor<READ_ONLY>(rt, "s");
        bool result = true;
        for (PointInRectIterator<1> pir(tm_col.rect(), false);
             result && pir();
             pir++) {
          auto tm = tm_meas.read(*pir);
          decltype(tm) cctm;
          col.get(pir[0], cctm);
          result = (tm.getValue() == cctm.getValue());
        }
        return result;
      }));
  recorder.expect_true(
    "Table has NUM_POLY column",
    TE(table.has_num_poly()));
  recorder.expect_true(
    "Table has DELAY_DIR column",
    TE(table.has_delay_dir()));
  recorder.expect_true(
    "Table has DELAY_DIR measures",
    TE(table.has_delay_dir_meas()));
  recorder.expect_true(
    "Table DELAY_DIR measures are correct",
    testing::TestEval(
      [&]() {
        auto col = ms_field.delayDirMeasCol();
        auto dd_col = table.delay_dir_meas();
        auto dd_meas = dd_col.meas_accessor<READ_ONLY>(rt, "rad");
        bool result = true;
        std::optional<coord_t> prev_row;
        std::optional<Point<2>> prev_midx;
        cc::Vector<cc::MDirection> ccdds;
        for (PointInDomainIterator<3> pid(dd_col.domain(), false);
             result && pid();
             pid++) {
          if (pid[0] != prev_row.value_or(pid[0] + 1)) {
            prev_row = pid[0];
            col.get(prev_row.value(), ccdds, true);
          }
          Point<2> midx(pid[0], pid[1]);
          if (!prev_midx || midx != prev_midx.value()) {
            prev_midx = midx;
            auto dd = dd_meas.read(midx);
            result = (dd.getValue() == ccdds(pid[1]).getValue());
          }
        }
        return result;
      }));
  recorder.expect_true(
    "Table has PHASE_DIR column",
    TE(table.has_phase_dir()));
  recorder.expect_true(
    "Table has PHASE_DIR measures",
    TE(table.has_phase_dir_meas()));
  recorder.expect_true(
    "Table PHASE_DIR measures are correct",
    testing::TestEval(
      [&]() {
        auto col = ms_field.phaseDirMeasCol();
        auto dd_col = table.phase_dir_meas();
        auto dd_meas = dd_col.meas_accessor<READ_ONLY>(rt, "rad");
        bool result = true;
        std::optional<coord_t> prev_row;
        std::optional<Point<2>> prev_midx;
        cc::Vector<cc::MDirection> ccdds;
        for (PointInDomainIterator<3> pid(dd_col.domain(), false);
             result && pid();
             pid++) {
          if (pid[0] != prev_row.value_or(pid[0] + 1)) {
            prev_row = pid[0];
            col.get(prev_row.value(), ccdds, true);
          }
          Point<2> midx(pid[0], pid[1]);
          if (!prev_midx || midx != prev_midx.value()) {
            prev_midx = midx;
            auto dd = dd_meas.read(midx);
            result = (dd.getValue() == ccdds(pid[1]).getValue());
          }
        }
        return result;
      }));
  recorder.expect_true(
    "Table has REFERENCE_DIR column",
    TE(table.has_reference_dir()));
  recorder.expect_true(
    "Table has REFERENCE_DIR measures",
    TE(table.has_reference_dir_meas()));
  recorder.expect_true(
    "Table REFERENCE_DIR measures are correct",
    testing::TestEval(
      [&]() {
        auto col = ms_field.referenceDirMeasCol();
        auto dd_col = table.reference_dir_meas();
        auto dd_meas = dd_col.meas_accessor<READ_ONLY>(rt, "rad");
        bool result = true;
        std::optional<coord_t> prev_row;
        std::optional<Point<2>> prev_midx;
        cc::Vector<cc::MDirection> ccdds;
        for (PointInDomainIterator<3> pid(dd_col.domain(), false);
             result && pid();
             pid++) {
          if (pid[0] != prev_row.value_or(pid[0] + 1)) {
            prev_row = pid[0];
            col.get(prev_row.value(), ccdds, true);
          }
          Point<2> midx(pid[0], pid[1]);
          if (!prev_midx || midx != prev_midx.value()) {
            prev_midx = midx;
            auto dd = dd_meas.read(midx);
            result = (dd.getValue() == ccdds(pid[1]).getValue());
          }
        }
        return result;
      }));
  recorder.expect_true(
    "Table has SOURCE_ID column",
    TE(table.has_source_id()));
  recorder.expect_true(
    "Table has EPHEMERIS_ID column",
    TE(table.has_ephemermis_id()));
  recorder.expect_true(
    "Table has FLAG_ROW column",
    TE(table.has_flag_row()));
}

void
ms_test(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  register_tasks(ctx, rt);

  const CXX_FILESYSTEM_NAMESPACE::path tpath = "data/t0.ms/FIELD";

  // create the table
  auto table =
    Table::create(
      ctx,
      rt,
      std::get<1>(from_ms(ctx, rt, tpath, {"*"})));

  // read values from MS
  {
    auto reqs =
      std::get<0>(
        TableReadTask::requirements(
          ctx,
          rt,
          table,
          ColumnSpacePartition(),
          WRITE_ONLY));
    TableReadTask::Args args;
    fstrcpy(args.table_path, tpath);
    TaskLauncher read(
      TableReadTask::TASK_ID,
      TaskArgument(&args, sizeof(args)),
      Predicate::TRUE_PRED,
      default_mapper);
    for (auto& rq : reqs)
      read.add_region_requirement(rq);
    rt->execute_task(ctx, read);
  }

  // run tests
  {
    VerifyTableArgs args;
    fstrcpy(args.table_path, tpath);
    auto reqs = std::get<0>(table.requirements(ctx, rt));
    TaskLauncher verify(
      VERIFY_FIELD_TABLE_TASK,
      TaskArgument(&args, sizeof(args)),
      Predicate::TRUE_PRED,
      default_mapper);
    verify.add_region_requirement(task->regions[0]);
    verify.add_region_requirement(task->regions[1]);
    for (auto& rq : reqs)
      verify.add_region_requirement(rq);
    rt->execute_task(ctx, verify);
  }

  // clean up
  table.destroy(ctx, rt);
}

int
main(int argc, char** argv) {

  {
    // verify_field_table
    TaskVariantRegistrar
      registrar(VERIFY_FIELD_TABLE_TASK, "verify_field_table");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.add_layout_constraint_set(
      DefaultMapper::cgroup_tag(0),
      default_layout);
    Runtime::preregister_task_variant<verify_field_table>(
      registrar,
      "verify_field_table");

  }
  testing::TestSuiteDriver driver =
    testing::TestSuiteDriver::make<ms_test>(MS_TEST_TASK, "ms_test");

  return driver.start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
