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
#include <hyperion/MSDataDescriptionTable.h>
#include <hyperion/TableReadTask.h>
#include <hyperion/DefaultMapper.h>

#include <hyperion/testing/TestSuiteDriver.h>
#include <hyperion/testing/TestRecorder.h>
#include <hyperion/testing/TestExpression.h>

#include <casacore/ms/MeasurementSets.h>

using namespace hyperion;
using namespace Legion;

enum {
  MS_TEST_TASK,
  VERIFY_DATA_DESCRIPTION_TABLE_TASK
};

#define TE(f) testing::TestEval([&](){ return f; }, #f)

struct VerifyTableArgs {
  char table_path[1024];
};

void
verify_data_description_table(
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

  MSDataDescriptionTable table(pt);

  const VerifyTableArgs *args =
    static_cast<const VerifyTableArgs*>(task->args);
  CXX_FILESYSTEM_NAMESPACE::path data_desc_path(args->table_path);
  casacore::MeasurementSet ms(
    data_desc_path.parent_path().string(),
    casacore::TableLock::PermanentLockingWait);
  casacore::ROMSDataDescColumns ms_dd(ms.dataDescription());

  recorder.expect_true(
    "Table has SPECTRAL_WINDOW_ID column",
    TE(table.has_spectral_window_id()));
  // TODO: For now, we don't check values, as utMS already has lots of such
  // tests; however, it might be useful to have some tests that confirm the
  // conversion Table->RegionRequiremnts->PhysicalTable->PhysicalColumn is
  // correct.
  recorder.expect_true(
    "Table has POLARIZATION_ID column",
    TE(table.has_polarization_id()));
  recorder.expect_false(
    "Table does not have LAG_ID column",
    TE(table.has_lag_id()));
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

  const CXX_FILESYSTEM_NAMESPACE::path tpath = "data/t0.ms/DATA_DESCRIPTION";

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
      mapper);
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
      VERIFY_DATA_DESCRIPTION_TABLE_TASK,
      TaskArgument(&args, sizeof(args)),
      Predicate::TRUE_PRED,
      mapper);
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
    TaskVariantRegistrar registrar(
      VERIFY_DATA_DESCRIPTION_TABLE_TASK,
      "verify_data_description_table");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    DefaultMapper::add_layouts(registrar);
    Runtime::preregister_task_variant<verify_data_description_table>(
      registrar,
      "verify_data_description_table");
  }

  return driver.start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
