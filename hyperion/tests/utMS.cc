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
#include <hyperion/IndexTree.h>
#include <hyperion/Column.h>
#include <hyperion/Table.h>
#include <hyperion/TableBuilder.h>
#include <hyperion/TableReadTask.h>
#include <hyperion/Measures.h>
#include <hyperion/TableMapper.h>

#include <hyperion/testing/TestSuiteDriver.h>
#include <hyperion/testing/TestRecorder.h>
#include <hyperion/testing/TestExpression.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <vector>

#ifdef __NVCC__
// to suppress silly nvcc warnings in some template expansions below
#pragma diag_suppress = unsigned_compare_with_zero
#endif // __NVCC__

using namespace hyperion;
using namespace Legion;

enum {
  MS_TEST_SUITE,
  VERIFY_COLUMN_TASK,
};

template <typename T, int DIM>
using RO = FieldAccessor<READ_ONLY, T, DIM, coord_t, AffineAccessor<T, DIM, coord_t>>;

#if HAVE_CXX17
#define TE(f) testing::TestEval([&](){ return f; }, #f)
#else
#define TE(f) testing::TestEval<std::function<bool()>>([&](){ return f; }, #f)
#endif

struct VerifyColumnTaskArgs {
  hyperion::TypeTag dt;
  FieldID fid;
  char table[160];
  char column[32];
  CXX_OPTIONAL_NAMESPACE::optional<hyperion::string> rc;
  bool has_values;
  bool has_keywords;
  bool has_measure;
};

void
verify_scalar_column(
  const casacore::Table& tb,
  const VerifyColumnTaskArgs *targs,
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

  if (targs->has_values) {
    DomainT<1> col_dom(regions[2].get_bounds<1,coord_t>());

    switch (targs->dt) {
#define CMP(DT)                                                         \
      case (DT): {                                                      \
        auto scol =                                                     \
          casacore::ScalarColumn<DataType<DT>::CasacoreType>(           \
            tb,                                                         \
            casacore::String(targs->column));                           \
        recorder.assert_true(                                           \
          std::string("verify bounds, column ") + targs->column,        \
          TE(Domain(col_dom) == Domain(Rect<1>(0, scol.nrow() - 1))));  \
        casacore::Vector<DataType<DT>::CasacoreType> ary =              \
          scol.getColumn();                                             \
        const RO<DataType<DT>::ValueType, 1>                            \
          col(regions[2], targs->fid);                                  \
        PointInDomainIterator<1> pid(col_dom);                          \
        recorder.expect_true(                                           \
          std::string("verify values, column ") + targs->column,        \
          testing::TestEval<std::function<bool()>>(                     \
            [&pid, &col, &ary]() {                                      \
              bool result = true;                                       \
              for (; result && pid(); pid++) {                          \
                DataType<DT>::ValueType a;                              \
                DataType<DT>::from_casacore(a, ary[pid[0]]);            \
                result = DataType<DT>::equiv(a, col[*pid]);             \
              }                                                         \
              return result;                                            \
            }));                                                        \
        break;                                                          \
      }
      HYPERION_FOREACH_CC_DATATYPE(CMP);
      default:
        assert(false);
        break;
#undef CMP
    }
  } else {
    recorder.expect_true(
      std::string("verify empty, column ") + targs->column,
      TE(tb.tableDesc().isColumn(targs->column)));
  }
}

template <unsigned DIM>
void
verify_array_column(
  const casacore::Table& tb,
  const VerifyColumnTaskArgs *targs,
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

  if (targs->has_values) {
    DomainT<DIM> col_dom(regions[2].get_bounds<DIM,coord_t>());
    switch (targs->dt) {
#define CMP(DT)                                                       \
      case (DT): {                                                    \
        auto acol =                                                   \
          casacore::ArrayColumn<DataType<DT>::CasacoreType>(          \
            tb,                                                       \
            casacore::String(targs->column));                         \
        recorder.assert_true(                                         \
          std::string("verify rank, column ") + targs->column,        \
          TE(acol.ndim(0)) == DIM - 1);                               \
        recorder.assert_true(                                         \
          std::string("verify nrows, column ") + targs->column,       \
          TE(Domain(col_dom).hi()[0]) == acol.nrow() - 1);            \
        {                                                             \
          PointInDomainIterator<DIM> pid(col_dom, false);             \
          recorder.assert_true(                                       \
            std::string("verify bounds, column ") + targs->column,    \
            testing::TestEval<std::function<bool()>>(                 \
              [&pid, &acol]() {                                       \
                bool result = true;                                   \
                while (result && pid()) {                             \
                  auto last_p = *pid;                                 \
                  while (result && pid()) {                           \
                    pid++;                                            \
                    if (!pid() || pid[0] != last_p[0]) {              \
                      casacore::IPosition shp(acol.shape(last_p[0])); \
                      Point<DIM> cpt;                                 \
                      cpt[0] = last_p[0];                             \
                      for (size_t i = 0; i < DIM - 1; ++i)            \
                        cpt[i + 1] = shp[DIM - 2 - i] - 1;            \
                      result = cpt == last_p;                         \
                    }                                                 \
                    if (pid())                                        \
                      last_p = *pid;                                  \
                  }                                                   \
                }                                                     \
                return result;                                        \
              }));                                                    \
        }                                                             \
        {                                                             \
          const RO<DataType<DT>::ValueType, DIM>                      \
            col(regions[2], targs->fid);                              \
          PointInDomainIterator<DIM> pid(col_dom, false);             \
          recorder.assert_true(                                       \
            std::string("verify values, column ") + targs->column,    \
            testing::TestEval<std::function<bool()>>(                 \
              [&pid, &acol, &col]() {                                 \
                bool result = true;                                   \
                casacore::Array<DataType<DT>::CasacoreType> ary;      \
                casacore::IPosition ipos(DIM - 1);                    \
                while (result && pid()) {                             \
                  auto row = pid[0];                                  \
                  acol.get(row, ary, true);                           \
                  while (result && pid()) {                           \
                    for (size_t i = 0; i < DIM - 1; ++i)              \
                      ipos[DIM - 2 - i] = pid[i + 1];                 \
                    DataType<DT>::ValueType a;                        \
                    DataType<DT>::from_casacore(a, ary(ipos));        \
                    result = DataType<DT>::equiv(a, col[*pid]);       \
                    pid++;                                            \
                    if (pid() && pid[0] != row) {                     \
                      row = pid[0];                                   \
                      acol.get(row, ary, true);                       \
                    }                                                 \
                  }                                                   \
                }                                                     \
                return result;                                        \
              }));                                                    \
        }                                                             \
        break;                                                        \
      }
      HYPERION_FOREACH_CC_DATATYPE(CMP);
      default:
        assert(false);
        break;
#undef CMP
    }
  } else {
    auto tcol =
      casacore::TableColumn(tb, casacore::String(targs->column));
    recorder.expect_true(
      std::string("verify empty, column ") + targs->column,
      testing::TestEval<std::function<bool()>>(
        [&tcol]() {
          bool result = true;
          for (unsigned i = 0; result && i < tcol.nrow(); ++i)
            result = !tcol.isDefined(i);
          return result;
        }));
  }
}

void
verify_column_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  const VerifyColumnTaskArgs *args =
    static_cast<const VerifyColumnTaskArgs*>(task->args);

  casacore::Table tb(
    casacore::String(args->table),
    casacore::TableLock::PermanentLockingWait);

  auto cdesc = tb.tableDesc()[casacore::String(args->column)];
  if (cdesc.isScalar()) {
    verify_scalar_column(tb, args, task, regions, ctx, rt);
  } else {
    switch (cdesc.ndim() + 1) {
#define VERIFY_ARRAY(N)                                           \
      case (N):                                                   \
        verify_array_column<N>(tb, args, task, regions, ctx, rt); \
        break;
      HYPERION_FOREACH_N(VERIFY_ARRAY);
#undef VERIFY_ARRAY
      default:
        assert(false);
        break;
    }
  }

  testing::TestLog<READ_WRITE> log(
    task->regions[0].region,
    regions[0],
    task->regions[1].region,
    regions[1],
    ctx,
    rt);
  testing::TestRecorder<READ_WRITE> recorder(log);

  auto col = casacore::TableColumn(tb, args->column);
  auto kws = col.keywordSet();
  unsigned region_idx = 2 + (args->has_values ? 1 : 0);
  if (args->has_keywords) {
    Keywords::pair<PhysicalRegion>
      prs{regions[region_idx], regions[region_idx + 1]};
    region_idx += 2;
    Keywords keywords(
      Keywords::pair<LogicalRegion>{
        prs.type_tags.get_logical_region(),
          prs.values.get_logical_region()});
    unsigned nf = kws.nfields();
    bool all_ok = true;
    for (unsigned f = 0; all_ok && f < nf; ++f) {
      std::string nm = kws.name(f);
      if (nm != "MEASINFO" && nm != "QuantumUnits") {
        switch (kws.dataType(f)) {
#define CMP_KW(DT)                                                      \
          case DataType<DT>::CasacoreTypeTag: {                         \
            DataType<DT>::CasacoreType cv;                              \
            kws.get(f, cv);                                             \
            DataType<DT>::ValueType v;                                  \
            DataType<DT>::from_casacore(v, cv);                         \
            auto ofid = keywords.find_keyword(rt, nm);                  \
            all_ok = (bool)ofid;                                        \
            if (all_ok) {                                               \
              all_ok =                                                  \
                (DT == Keywords::value_type(prs.type_tags, ofid.value())); \
              if (all_ok) {                                             \
                auto kv =                                               \
                  Keywords::read<DataType<DT>::ValueType>(prs, ofid.value()) \
                  .value();                                             \
                all_ok = DataType<DT>::equiv(kv, v);                    \
              }                                                         \
            }                                                           \
            break;                                                      \
          }
          HYPERION_FOREACH_CC_RECORD_DATATYPE(CMP_KW)
#undef CMP_KW
        default:
            break;
        }
      }
      recorder.expect_true(
        std::string("verify keywords, column ") + args->column,
        TE(all_ok));
    }
  } else {
    recorder.expect_true(
      std::string("verify no keywords, column ") + args->column,
      testing::TestEval<std::function<bool()>>(
        [&kws]() {
          unsigned num_expected = 0;
          unsigned nf = kws.nfields();
          for (unsigned f = 0; f < nf; ++f) {
            std::string nm = kws.name(f);
            if (nm != "MEASINFO" && nm != "QuantumUnits") {
              switch (kws.dataType(f)) {
#define CMP_KW(DT)                                  \
                case DataType<DT>::CasacoreTypeTag:
                HYPERION_FOREACH_CC_RECORD_DATATYPE(CMP_KW)
                  ++num_expected;
                  break; // break is here intentionally
#undef CMP_KW
              default:
                  break;
              }
            }
          }
          return num_expected == 0;
        }));
  }
  {
    CXX_OPTIONAL_NAMESPACE::optional<unsigned> fmeas;
    unsigned nf = kws.nfields();
    for (unsigned f = 0; !fmeas && f < nf; ++f) {
      std::string nm = kws.name(f);
      if (nm == "MEASINFO" && kws.dataType(f) == casacore::DataType::TpRecord)
        fmeas = f;
    }
    recorder.expect_true(
      std::string("column ") + args->column
      + " has measure only if MS column has a measure",
      TE(args->has_measure == (bool)fmeas));
    if (args->has_measure) {
      MeasRef::DataRegions prs;
      prs.metadata = regions[region_idx++];
      prs.values = regions[region_idx++];
      if (region_idx < regions.size())
        prs.index = regions[region_idx];
      {
        auto mr = MeasRef::make(rt, prs);
#if HAVE_CXX17
        auto& [mrbs, rmap] = mr;
#else
        auto& mrbs = std::get<0>(mr);
        auto& rmap = std::get<1>(mr);
#endif
        auto mrbs_p = &mrbs;
        auto rmap_p = &rmap;
        recorder.assert_true(
          std::string("column ") + args->column + " has a measure",
          TE(mrbs_p->size() == 1));
        recorder.assert_true(
          std::string("column ") + args->column + " measure is simple",
          TE(rmap_p->size() == 1));
      }
      recorder.expect_true(
        std::string("column ") + args->column
        + " has expected measure",
        testing::TestEval<std::function<bool()>>(
          [&kws, &fmeas, &prs, rt]() {
            bool result = false;
            casacore::MeasureHolder mh;
            casacore::String err;
            auto converted = mh.fromType(err, kws.asRecord(fmeas.value()));
            if (converted) {
              // TODO: improve the test for comparing MeasRef values
              if (false) {}
#define MATCH(MC)                                                       \
              else if (MClassT<MC>::holds(mh)) {                        \
                auto mrs =                                              \
                  std::get<0>(                                          \
                    MeasRef::make<MClassT<MC>::type>(rt, prs));         \
                result = mrs.size() == 1;                               \
              }
              HYPERION_FOREACH_MCLASS(MATCH)
#undef MATCH
              else {}
            }
            return result;
          }));
    }
  }
}

void
read_full_ms(
  testing::TestLog<READ_WRITE>& log,
  Context ctx,
  Runtime* rt) {

  testing::TestRecorder<READ_WRITE> recorder(log);

  static const std::string t0_path("data/t0.ms");
  auto nm_ics_flds = from_ms(ctx, rt, t0_path, {"*"});
#if HAVE_CXX17
  auto& [table_name, table_ics, table_fields] = nm_ics_flds;
#else
  auto& table_name = std::get<0>(nm_ics_flds);
  auto& table_ics = std::get<1>(nm_ics_flds);
  auto& table_fields = std::get<2>(nm_ics_flds);
#endif
  auto table_name_p = &table_name;

  recorder.expect_true(
    "main table name is 'MAIN'",
    TE(*table_name_p == "MAIN"));
  {
    auto ixax = table_ics.axes(ctx, rt);
    recorder.assert_true(
      "t0.ms MAIN index is ROW",
      TE(ixax.size() == 1
         && ixax[0] == static_cast<int>(MSTable<MS_MAIN>::ROW_AXIS)));
  }
  auto table = Table::create(ctx, rt, table_ics, std::move(table_fields));
  recorder.assert_true(
    "t0.ms MAIN table successfully read",
    TE(!table.is_empty()));

  std::vector<std::string> expected_columns{
    "UVW",
    "FLAG",
    // "FLAG_CATEGORY",
    "WEIGHT",
    "SIGMA",
    "ANTENNA1",
    "ANTENNA2",
    "ARRAY_ID",
    "DATA_DESC_ID",
    "EXPOSURE",
    "FEED1",
    "FEED2",
    "FIELD_ID",
    "FLAG_ROW",
    "INTERVAL",
    "OBSERVATION_ID",
    "PROCESSOR_ID",
    "SCAN_NUMBER",
    "STATE_ID",
    "TIME",
    "TIME_CENTROID",
    "DATA",
    // "WEIGHT_SPECTRUM"
  };

  auto cols = table.columns();

  recorder.assert_true(
    "table has expected columns",
    testing::TestEval<std::function<bool()>>(
      [&expected_columns, &cols]() {
        std::set<std::string> colnames;
        for (auto& nm_col : cols)
          colnames.insert(std::get<0>(nm_col));
        return
          std::all_of(
            expected_columns.begin(),
            expected_columns.end(),
            [&colnames](auto& nm) { return colnames.count(nm) > 0; });
      }));

  // FIXME: awaiting keyword support in Table
  // recorder.expect_true(
  //   "table has expected MS_VERSION keyword value",
  //   testing::TestEval(
  //     [&table, &ctx, rt]() {
  //       auto fid = table.keywords.find_keyword(rt, "MS_VERSION");
  //       CXX_OPTIONAL_NAMESPACE::optional<float> msv;
  //       if (fid)
  //         msv = table.keywords.read<float>(ctx, rt, fid.value());
  //       return fid && msv && msv.value() == 2.0;
  //     }));
  // recorder.expect_true(
  //   "only table keyword is MS_VERSION",
  //   testing::TestEval(
  //     [&table, &ctx, rt]() {
  //       auto keys = table.keywords.keys(rt);
  //       std::set<std::string> expected{"MS_VERSION"};
  //       std::set<std::string> kw(keys.begin(), keys.end());
  //       return expected == kw;
  //     }));

  //
  // read MS table columns to initialize the Column LogicalRegions
  //
  {
    auto row_part =
      table
      .partition_rows(ctx, rt, {CXX_OPTIONAL_NAMESPACE::optional<size_t>(2000)})
      .get_result<ColumnSpacePartition>();
    auto reqs =
      TableReadTask::requirements(ctx, rt, table, row_part);
#if HAVE_CXX17
    auto& [treqs, tparts, tdesc] = reqs;
#else
    auto& treqs = std::get<0>(reqs);
    auto& tparts = std::get<1>(reqs);
    auto& tdesc = std::get<2>(reqs);
#endif
    TableReadTask::Args args;
    std::strncpy(args.table_path, t0_path.c_str(), sizeof(args.table_path) - 1);
    args.table_desc = tdesc;
    IndexTaskLauncher read(
      TableReadTask::TASK_ID,
      rt->get_index_partition_color_space(row_part.column_ip),
      TaskArgument(&args, sizeof(args)),
      ArgumentMap(),
      Predicate::TRUE_PRED,
      false,
      table_mapper);
    for (auto& rq : treqs)
      read.add_region_requirement(rq);
    rt->execute_index_space(ctx, read);
    for (auto& p : tparts)
      p.destroy(ctx, rt);
  }

  // compare column LogicalRegions to values read using casacore functions
  // directly
  IndexSpace col_is(
    rt->create_index_space(ctx, Rect<1>(0, expected_columns.size() - 1)));
  auto remaining_log =
    log.get_log_references_by_state({testing::TestState::UNKNOWN})[0];
  IndexPartition col_log_ip =
    rt->create_equal_partition(
      ctx,
      remaining_log.log_region().get_index_space(),
      col_is);
  LogicalPartitionT<1> verify_col_logs(
    rt->get_logical_partition(ctx, remaining_log.log_region(), col_log_ip));
  VerifyColumnTaskArgs args;
  fstrcpy(args.table, t0_path.c_str());
  args.table[sizeof(args.table) - 1] = '\0';
  TaskLauncher verify_task(
    VERIFY_COLUMN_TASK,
    TaskArgument(&args, sizeof(args)),
    Predicate::TRUE_PRED,
    table_mapper);
  for (size_t i = 0; i < expected_columns.size(); ++i) {
    auto col = cols.at(expected_columns[i]);
    args.dt = col.dt;
    args.fid = col.fid;
    fstrcpy(args.column, expected_columns[i].c_str());
    args.column[sizeof(args.column) - 1] = '\0';
    args.rc = col.rc;
    verify_task.region_requirements.clear();
    auto log_reqs =
      remaining_log.requirements<READ_WRITE>(
        rt->get_logical_subregion_by_color(verify_col_logs, Point<1>(i)),
        log.log_reference().log_region());
    std::for_each(
      log_reqs.begin(),
      log_reqs.end(),
      [&verify_task](auto& req) {
        verify_task.add_region_requirement(req);
      });
    if (col.is_valid()) {
      RegionRequirement req(col.region, READ_ONLY, EXCLUSIVE, col.region);
      req.add_field(col.fid);
      verify_task.add_region_requirement(req);
      args.has_values = true;
    } else {
      args.has_values = false;
    }
    if (!col.kw.is_empty()) {
      auto n = col.kw.size(rt);
      std::vector<FieldID> fids(n);
      std::iota(fids.begin(), fids.end(), 0);
      auto reqs = col.kw.requirements(rt, fids, READ_ONLY).value();
      verify_task.add_region_requirement(reqs.type_tags);
      verify_task.add_region_requirement(reqs.values);
      args.has_keywords = true;
    } else {
      args.has_keywords = false;
    }
    if (!col.mr.is_empty()) {
      auto reqs = col.mr.requirements(READ_ONLY);
#if HAVE_CXX17
      auto& [mr, vr, oir] = reqs;
#else
      auto& mr = std::get<0>(reqs);
      auto& vr = std::get<1>(reqs);
      auto& oir = std::get<2>(reqs);
#endif
      verify_task.add_region_requirement(mr);
      verify_task.add_region_requirement(vr);
      if (oir)
        verify_task.add_region_requirement(oir.value());
      args.has_measure = true;
    } else {
      args.has_measure = false;
    }
    rt->execute_task(ctx, verify_task);
  }
  rt->destroy_index_partition(ctx, col_log_ip);
  rt->destroy_index_space(ctx, col_is);
  table.destroy(ctx, rt);
}

void
ms_test_suite(
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

  read_full_ms(log, ctx, rt);
}

int
main(int argc, char** argv) {

  testing::TestSuiteDriver driver =
    testing::TestSuiteDriver::make<ms_test_suite>(
      MS_TEST_SUITE,
      "ms_test_suite",
      200);

  TaskVariantRegistrar registrar(VERIFY_COLUMN_TASK, "verify_column_task");
  registrar.add_constraint(ProcessorConstraint(Processor::IO_PROC));
  registrar.add_layout_constraint_set(
    TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
    soa_right_layout);
  Runtime::preregister_task_variant<verify_column_task>(
    registrar,
    "verify_column_task");

  return driver.start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
