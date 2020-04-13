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
#include <hyperion/MSFeedTable.h>
#include <hyperion/TableReadTask.h>
#include <hyperion/TableMapper.h>

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
  VERIFY_FEED_TABLE_TASK
};

#define TE(f) testing::TestEval([&](){ return f; }, #f)

struct VerifyTableArgs {
  char table_path[1024];
};

void
verify_feed_table(
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

  MSFeedTable<FEED_ROW> table(pt);

  const VerifyTableArgs *args =
    static_cast<const VerifyTableArgs*>(task->args);
  CXX_FILESYSTEM_NAMESPACE::path feed_path(args->table_path);
  casacore::MeasurementSet ms(
    feed_path.parent_path().string(),
    casacore::TableLock::PermanentLockingWait);
  casacore::ROMSFeedColumns ms_feed(ms.feed());

  recorder.expect_true(
    "Table has ANTENNA_ID column",
    TE(table.has_antenna_id()));
  // TODO: For now, we don't check values, as utMS already has lots of such
  // tests; however, it might be useful to have some tests that confirm the
  // conversion Table->RegionRequiremnts->PhysicalTable->PhysicalColumn is
  // correct.
  recorder.expect_true(
    "Table has FEED_ID column",
    TE(table.has_feed_id()));
  recorder.expect_true(
    "Table has SPECTRAL_WINDOW_ID column",
    TE(table.has_spectral_window_id()));
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
        auto col = ms_feed.timeMeas();
        auto time_col = table.time_meas<AffineAccessor>();
        auto time_meas =
          time_col.meas_accessor<READ_ONLY>(
            rt,
            MSTableColumns<MS_FEED>::units.at(
              MSTableColumns<MS_FEED>::col_t::MS_FEED_COL_TIME));
        bool result = true;
        for (PointInRectIterator<1> pir(time_col.rect(), false);
             result && pir();
             pir++) {
          auto time = time_meas.read(*pir);
          decltype(time) cctime;
          col.get(pir[0], cctime);
          result = (time.getValue() == cctime.getValue());
        }
        return result;
      }));
  recorder.expect_true(
    "Table has INTERVAL column",
    TE(table.has_interval()));
  recorder.expect_true(
    "Table has NUM_RECEPTORS column",
    TE(table.has_num_receptors()));
  recorder.expect_true(
    "Table has BEAM_ID column",
    TE(table.has_beam_id()));
  recorder.expect_true(
    "Table has BEAM_OFFSET column",
    TE(table.has_beam_offset()));
  recorder.expect_true(
    "Table has BEAM_OFFSET measures",
    TE(table.has_beam_offset_meas()));
  recorder.expect_true(
    "Table BEAM_OFFSET measures are correct",
    testing::TestEval(
      [&]() {
        auto col = ms_feed.beamOffsetMeas();
        auto bo_col = table.beam_offset_meas();
        auto bo_meas =
          bo_col.meas_accessor<READ_ONLY>(
            rt,
            MSTableColumns<MS_FEED>::units.at(
              MSTableColumns<MS_FEED>::col_t::MS_FEED_COL_BEAM_OFFSET));
        std::optional<coord_t> prev_row;
        std::optional<Point<2>> prev_midx;
        casacore::Vector<casacore::MDirection> ccbo;
        bool result = true;
        for (PointInDomainIterator<3> pid(bo_col.domain(), false);
             result && pid();
             pid++) {
          if (pid[0] != prev_row.value_or(pid[0] + 1)) {
            col.get(pid[0], ccbo, true);
            Point<2> midx(pid[0], pid[1]);
            if (!prev_midx || midx != prev_midx.value()) {
              auto bo = bo_meas.read(midx);
              result = (bo.getValue() == ccbo[pid[1]].getValue());
              prev_midx = midx;
            }
          }
        }
        return result;
      }));
  recorder.expect_true(
    "Table has FOCUS_LENGTH column",
    TE(table.has_focus_length()));
  recorder.expect_true(
    "Table does not have PHASED_FEED_ID column",
    TE(!table.has_phased_feed_id()));
  recorder.expect_true(
    "Table has POLARIZATION_TYPE column",
    TE(table.has_polarization_type()));
  recorder.expect_true(
    "Table has POL_RESPONSE column",
    TE(table.has_pol_response()));
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
        auto col = ms_feed.positionMeas();
        auto position_col = table.position_meas<AffineAccessor>();
        auto position_meas =
          position_col.meas_accessor<READ_ONLY>(
            rt,
            MSTableColumns<MS_FEED>::units.at(
              MSTableColumns<MS_FEED>::col_t::MS_FEED_COL_POSITION));
        bool result = true;
        std::optional<coord_t> prev_row;
          for (PointInRectIterator<2> pir(position_col.rect(), false);
               result && pir();
               pir++) {
            if (pir[0] != prev_row.value_or(pir[0] + 1)) {
              prev_row = pir[0];
              auto pos = position_meas.read(pir[0]);
              decltype(pos) ccpos;
              col.get(prev_row.value(), ccpos);
              result = (pos.getValue() == ccpos.getValue());
            }
          }
        return result;
      }));
  recorder.expect_true(
    "Table has RECEPTOR_ANGLE column",
    TE(table.has_receptor_angle()));
}

void
ms_test(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  register_tasks(ctx, rt);

  const CXX_FILESYSTEM_NAMESPACE::path tpath = "data/t0.ms/FEED";

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
      table_mapper);
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
      VERIFY_FEED_TABLE_TASK,
      TaskArgument(&args, sizeof(args)),
      Predicate::TRUE_PRED,
      table_mapper);
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
    // verify_feed_table
    TaskVariantRegistrar
      registrar(VERIFY_FEED_TABLE_TASK, "verify_feed_table");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.add_layout_constraint_set(
      TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
      soa_row_major_layout);
    TableMapper::add_table_layout_constraint(registrar);
    Runtime::preregister_task_variant<verify_feed_table>(
      registrar,
      "verify_feed_table");
  }

  return driver.start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
