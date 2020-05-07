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
#include <hyperion/MSSpWindowTable.h>
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

namespace cc = casacore;

enum {
  MS_TEST_TASK,
  VERIFY_SPW_TABLE_TASK
};

#define TE(f) testing::TestEval([&](){ return f; }, #f)

struct VerifyTableArgs {
  char table_path[1024];
  Table::Desc desc;
};

void
verify_spw_table(
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

  const VerifyTableArgs *args = static_cast<const VerifyTableArgs*>(task->args);

  auto [pt, rit, pit] =
    PhysicalTable::create(
      rt,
      args->desc,
      task->regions.begin() + 2,
      task->regions.end(),
      regions.begin() + 2,
      regions.end())
    .value();
  assert(rit == task->regions.end());
  assert(pit == regions.end());

  MSSpWindowTable table(pt);

  CXX_FILESYSTEM_NAMESPACE::path spw_path(args->table_path);
  cc::MeasurementSet ms(
    spw_path.parent_path().string(),
    cc::TableLock::PermanentLockingWait);
  cc::ROMSSpWindowColumns ms_spw(ms.spectralWindow());

  recorder.expect_true(
    "Table has NAME column",
    TE(table.has_name()));
  // TODO: For now, we don't check values, as utMS already has lots of such
  // tests; however, it might be useful to have some tests that confirm the
  // conversion Table->RegionRequiremnts->PhysicalTable->PhysicalColumn is
  // correct.
  recorder.expect_true(
    "Table has NUM_CHAN column",
    TE(table.has_num_chan()));
  recorder.expect_true(
    "Table has NAME column",
    TE(table.has_name()));
  recorder.expect_true(
    "Table has REF_FREQUENCY column",
    TE(table.has_ref_frequency()));
  recorder.expect_true(
    "Table has REF_FREQUENCY measures",
    TE(table.has_ref_frequency_meas()));
  recorder.expect_true(
    "Table REF_FREQUENCY measures are correct",
    testing::TestEval(
      [&]() {
        auto col = ms_spw.refFrequencyMeas();
        auto rf_col = table.ref_frequency_meas<AffineAccessor>();
        auto rf_meas =
          rf_col.meas_accessor<READ_ONLY>(
            rt,
            MSSpWindowTable::C::units.at(
              MSSpWindowTable::C::col_t::MS_SPECTRAL_WINDOW_COL_REF_FREQUENCY));
        bool result = true;
        for (PointInRectIterator<1> pir(rf_col.rect(), false);
             result && pir();
             pir++) {
          auto rf = rf_meas.read(*pir);
          decltype(rf) ccrf;
          col.get(pir[0], ccrf);
          result = (rf.getValue() == ccrf.getValue());
        }
        return result;
      }));
  recorder.expect_true(
    "Table has CHAN_FREQ column",
    TE(table.has_chan_freq()));
  recorder.expect_true(
    "Table has CHAN_FREQ measures",
    TE(table.has_chan_freq_meas()));
  recorder.expect_true(
    "Table CHAN_FREQ measures are correct",
    testing::TestEval(
      [&]() {
        auto col = ms_spw.chanFreqMeas();
        auto cf_col = table.chan_freq_meas();
        auto cf_meas =
          cf_col.meas_accessor<READ_ONLY>(
            rt,
            MSSpWindowTable::C::units.at(
              MSSpWindowTable::C::col_t::MS_SPECTRAL_WINDOW_COL_CHAN_FREQ));
        bool result = true;
        std::optional<coord_t> prev_row;
        cc::Vector<cc::MFrequency> cccfs;
        for (PointInDomainIterator<2> pid(cf_col.domain(), false);
             result && pid();
             pid++) {
          if (pid[0] != prev_row.value_or(pid[0] + 1)) {
            prev_row = pid[0];
            col.get(prev_row.value(), cccfs, true);
          }
          auto cf = cf_meas.read(*pid);
          result = (cf.getValue() == cccfs[pid[1]].getValue());
        }
        return result;
      }));
  recorder.expect_true(
    "Table has CHAN_WIDTH column",
    TE(table.has_chan_width()));
  recorder.expect_true(
    "Table has EFFECTIVE_BW column",
    TE(table.has_effective_bw()));
  recorder.expect_true(
    "Table has RESOLUTION column",
    TE(table.has_resolution()));
  recorder.expect_true(
    "Table has TOTAL_BANDWIDTH column",
    TE(table.has_total_bandwidth()));
  recorder.expect_true(
    "Table has NET_SIDEBAND column",
    TE(table.has_net_sideband()));
  recorder.expect_true(
    "Table has BBC_NO column",
    TE(table.has_bbc_no()));
  recorder.expect_false(
    "Table does not have BBC_SIDEBAND column",
    TE(table.has_bbc_sideband()));
  recorder.expect_true(
    "Table has IF_CONV_CHAIN column",
    TE(table.has_if_conv_chain()));
  recorder.expect_false(
    "Table does not have RECEIVER_ID column",
    TE(table.has_receiver_id()));
  recorder.expect_true(
    "Table has FREQ_GROUP column",
    TE(table.has_freq_group()));
  recorder.expect_true(
    "Table has FREQ_GROUP_NAME column",
    TE(table.has_freq_group_name()));
  recorder.expect_false(
    "Table does not have DOPPLER_ID column",
    TE(table.has_doppler_id()));
  recorder.expect_true(
    "Table has ASSOC_SPW_ID column",
    TE(table.has_assoc_spw_id()));
  recorder.expect_true(
    "Table has ASSOC_NATURE column",
    TE(table.has_assoc_nature()));
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

  const CXX_FILESYSTEM_NAMESPACE::path tpath = "data/t0.ms/SPECTRAL_WINDOW";

  // create the table
  auto [nm, index_cs, fields] = from_ms(ctx, rt, tpath, {"*"});
  auto table = Table::create(ctx, rt, index_cs, std::move(fields));

  // read values from MS
  {
    auto [reqs, parts, desc] =
      TableReadTask::requirements(
        ctx,
        rt,
        table,
        ColumnSpacePartition(),
        WRITE_ONLY);
    TableReadTask::Args args;
    fstrcpy(args.table_path, tpath);
    args.table_desc = desc;
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
    auto [reqs, parts, desc] = table.requirements(ctx, rt);
    args.desc = desc;
    TaskLauncher verify(
      VERIFY_SPW_TABLE_TASK,
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
    // verify_spw_table
    TaskVariantRegistrar
      registrar(VERIFY_SPW_TABLE_TASK, "verify_spw_table");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.add_layout_constraint_set(
      TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
      soa_row_major_layout);
    TableMapper::add_table_layout_constraint(registrar);
    Runtime::preregister_task_variant<verify_spw_table>(
      registrar,
      "verify_spw_table");

  }

  return driver.start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
