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
#include <hyperion/MSAntennaTable.h>
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

enum {
  MS_TEST_TASK,
  VERIFY_ANTENNA_TABLE_TASK
};

#define TE(f) testing::TestEval([&](){ return f; }, #f)

struct VerifyAntennaTableArgs {
  char table_path[1024];
};

void
verify_antenna_table(
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

  MSAntennaTable table(pt);

  const VerifyAntennaTableArgs *args =
    static_cast<const VerifyAntennaTableArgs*>(task->args);
  CXX_FILESYSTEM_NAMESPACE::path antenna_path(args->table_path);
  casacore::MeasurementSet ms(
    antenna_path.parent_path().string(),
    casacore::TableLock::PermanentLockingWait);
  casacore::ROMSAntennaColumns ms_antenna(ms.antenna());

  recorder.expect_true(
    "Table has NAME column",
    TE(table.has_name()));
  // TODO: For now, we don't check values, as utMS already has lots of such
  // tests; however, it might be useful to have some tests that confirm the
  // conversion Table->RegionRequiremnts->PhysicalTable->PhysicalColumn is
  // correct.
  recorder.expect_true(
    "Table has STATION column",
    TE(table.has_station()));
  recorder.expect_true(
    "Table has TYPE column",
    TE(table.has_type()));
  recorder.expect_true(
    "Table has MOUNT column",
    TE(table.has_mount()));
  recorder.expect_true(
    "Table has POSITION column",
    TE(table.has_position()));
  recorder.expect_true(
    "Table has POSITION measures",
    TE(table.has_position_meas()));
  recorder.expect_true(
    "Table POSITION measures are correct",
    testing::TestEval(
      [&]() {
        auto col = ms_antenna.positionMeas();
        auto position_col = table.position_meas<AffineAccessor>();
        auto position_meas = position_col.meas_accessor<READ_ONLY>(rt, "m");
        bool result = true;
        std::optional<coord_t> prev_row;
        for (PointInRectIterator<2> pir(position_col.rect(), false);
             result && pir();
             pir++) {
          if (pir[0] != prev_row.value_or(pir[0] + 1)) {
            prev_row = pir[0];
            auto pos = position_meas.read(prev_row.value());
            decltype(pos) ccpos;
            col.get(prev_row.value(), ccpos);
            result = (pos.getValue() == ccpos.getValue());
          }
        }
        return result;
      }));
  recorder.expect_true(
    "Table has OFFSET column",
    TE(table.has_offset()));
  recorder.expect_true(
    "Table has OFFSET measures",
    TE(table.has_offset_meas()));
  recorder.expect_true(
    "Table OFFSET measures are correct",
    testing::TestEval(
      [&]() {
        auto col = ms_antenna.offsetMeas();
        auto offset_col = table.offset_meas<AffineAccessor>();
        auto offset_meas = offset_col.meas_accessor<READ_ONLY>(rt, "m");
        bool result = true;
        std::optional<coord_t> prev_row;
          for (PointInRectIterator<2> pir(offset_col.rect(), false);
               result && pir();
               pir++) {
            if (!prev_row || pir[0] != prev_row.value()) {
              prev_row = pir[0];
              auto off = offset_meas.read(prev_row.value());
              decltype(off) ccoff;
              col.get(prev_row.value(), ccoff);
              result = (off.getValue() == ccoff.getValue());
            }
          }
        return result;
      }));
  recorder.expect_true(
    "Table has DISH_DIAMETER column",
    TE(table.has_dish_diameter()));
  recorder.expect_false(
    "Table does not have ORBIT_ID column",
    TE(table.has_orbit_id()));
  recorder.expect_false(
    "Table does not have MEAN_ORBIT column",
    TE(table.has_mean_orbit()));
  recorder.expect_false(
    "Table does not have PHASED_ARRAY_ID column",
    TE(table.has_phased_array_id()));
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

  const CXX_FILESYSTEM_NAMESPACE::path tpath = "data/t0.ms/ANTENNA";

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
    VerifyAntennaTableArgs args;
    fstrcpy(args.table_path, tpath);
    auto reqs = std::get<0>(table.requirements(ctx, rt));
    TaskLauncher verify(
      VERIFY_ANTENNA_TABLE_TASK,
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

  testing::TestSuiteDriver driver =
    testing::TestSuiteDriver::make<ms_test>(MS_TEST_TASK, "ms_test");
  {
    // verify_antenna_table
    TaskVariantRegistrar
      registrar(VERIFY_ANTENNA_TABLE_TASK, "verify_antenna_table");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.add_layout_constraint_set(
      DefaultMapper::cgroup_tag(0),
      default_layout);
    Runtime::preregister_task_variant<verify_antenna_table>(
      registrar,
      "verify_antenna_table");
  }

  return driver.start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
