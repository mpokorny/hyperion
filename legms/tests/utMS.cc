#include <algorithm>
#include <map>
#include <memory>
#include <vector>

#include <legms/legms.h>
#include <legms/utility.h>
#include <legms/IndexTree.h>
#include <legms/Column.h>
#include <legms/Table.h>
#include <legms/TableBuilder.h>
#include <legms/TableReadTask.h>
#include <legms/Measures.h>

#include <legms/testing/TestSuiteDriver.h>
#include <legms/testing/TestRecorder.h>
#include <legms/testing/TestExpression.h>

using namespace legms;
using namespace Legion;

enum {
  MS_TEST_SUITE,
  VERIFY_COLUMN_TASK,
};

template <typename T, int DIM>
using RO = FieldAccessor<READ_ONLY, T, DIM, coord_t, AffineAccessor<T, DIM, coord_t>>;

#define TE(f) testing::TestEval([&](){ return f; }, #f)

struct VerifyColumnTaskArgs {
  legms::TypeTag tag;
  char table[160];
  char column[32];
  bool has_values;
  bool has_keywords;
  bool has_measures;
};

void
verify_scalar_column(
  const casacore::Table& tb,
  const VerifyColumnTaskArgs *targs,
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context context,
  Runtime* runtime) {

  testing::TestLog<READ_WRITE> log(
    task->regions[0].region,
    regions[0],
    task->regions[1].region,
    regions[1],
    context,
    runtime);
  testing::TestRecorder<READ_WRITE> recorder(log);

  if (targs->has_values) {
    DomainT<1> col_dom(regions[2].get_bounds<1,coord_t>());

    switch (targs->tag) {
#define CMP(TAG)                                                        \
      case (TAG): {                                                     \
        auto scol =                                                     \
          casacore::ScalarColumn<DataType<TAG>::CasacoreType>(          \
            tb,                                                         \
            casacore::String(targs->column));                           \
        recorder.assert_true(                                           \
          std::string("verify bounds, column ") + targs->column,        \
          TE(Domain(col_dom)) == Domain(Rect<1>(0, scol.nrow() - 1)));  \
        casacore::Vector<DataType<TAG>::CasacoreType> ary =             \
          scol.getColumn();                                             \
        const RO<DataType<TAG>::ValueType, 1>                           \
          col(regions[2], Column::VALUE_FID);                           \
        PointInDomainIterator<1> pid(col_dom);                          \
        recorder.expect_true(                                           \
          std::string("verify values, column ") + targs->column,        \
          testing::TestEval(                                            \
            [&pid, &col, &ary, targs]() {                               \
              bool result = true;                                       \
              for (; result && pid(); pid++) {                          \
                DataType<TAG>::ValueType a;                             \
                DataType<TAG>::from_casacore(a, ary[pid[0]]);           \
                result = DataType<TAG>::equiv(a, col[*pid]);            \
              }                                                         \
              return result;                                            \
            }));                                                        \
        break;                                                          \
      }
      LEGMS_FOREACH_DATATYPE(CMP);
#undef CMP
    }
  } else {
    recorder.expect_true(
      std::string("verify empty, column ") + targs->column,
      TE(tb.tableDesc().isColumn(targs->column)));
  }
}

template <int DIM>
void
verify_array_column(
  const casacore::Table& tb,
  const VerifyColumnTaskArgs *targs,
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context context,
  Runtime* runtime) {

  testing::TestLog<READ_WRITE> log(
    task->regions[0].region,
    regions[0],
    task->regions[1].region,
    regions[1],
    context,
    runtime);
  testing::TestRecorder<READ_WRITE> recorder(log);

  if (targs->has_values) {
    DomainT<DIM> col_dom(regions[2].get_bounds<DIM,coord_t>());
    switch (targs->tag) {
#define CMP(TAG)                                                      \
      case (TAG): {                                                   \
        auto acol =                                                   \
          casacore::ArrayColumn<DataType<TAG>::CasacoreType>(         \
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
            testing::TestEval(                                        \
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
          const RO<DataType<TAG>::ValueType, DIM>                     \
            col(regions[2], Column::VALUE_FID);                       \
          PointInDomainIterator<DIM> pid(col_dom, false);             \
          recorder.assert_true(                                       \
            std::string("verify values, column ") + targs->column,    \
            testing::TestEval(                                        \
              [&pid, &acol, &col]() {                                 \
                bool result = true;                                   \
                casacore::Array<DataType<TAG>::CasacoreType> ary;     \
                casacore::IPosition ipos(DIM - 1);                    \
                while (result && pid()) {                             \
                  auto row = pid[0];                                  \
                  acol.get(row, ary, true);                           \
                  while (result && pid()) {                           \
                    for (size_t i = 0; i < DIM - 1; ++i)              \
                      ipos[DIM - 2 - i] = pid[i + 1];                 \
                    DataType<TAG>::ValueType a;                       \
                    DataType<TAG>::from_casacore(a, ary(ipos));       \
                    result = DataType<TAG>::equiv(a, col[*pid]);      \
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
      LEGMS_FOREACH_DATATYPE(CMP);
#undef CMP
    }
  } else {
    auto tcol =
      casacore::TableColumn(tb, casacore::String(targs->column));
    recorder.expect_true(
      std::string("verify empty, column ") + targs->column,
      testing::TestEval(
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
  Context context,
  Runtime* runtime) {

  const VerifyColumnTaskArgs *args =
    static_cast<const VerifyColumnTaskArgs*>(task->args);

  casacore::Table tb(
    casacore::String(args->table),
    casacore::TableLock::PermanentLockingWait);

  auto cdesc = tb.tableDesc()[casacore::String(args->column)];
  if (cdesc.isScalar()) {
    verify_scalar_column(tb, args, task, regions, context, runtime);
  } else {
#define VERIFY_ARRAY(N)                                                 \
    case (N):                                                           \
      verify_array_column<N>(tb, args, task, regions, context, runtime); \
      break;

    switch (cdesc.ndim() + 1) {
      LEGMS_FOREACH_N(VERIFY_ARRAY);
    }
#undef VERIFY_ARRAY
  }

  testing::TestLog<READ_WRITE> log(
    task->regions[0].region,
    regions[0],
    task->regions[1].region,
    regions[1],
    context,
    runtime);
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
            auto ofid = keywords.find_keyword(runtime, nm);             \
            all_ok = ofid.has_value();                                  \
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
          LEGMS_FOREACH_RECORD_DATATYPE(CMP_KW)
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
      testing::TestEval(
        [&kws]() {
          unsigned num_expected = 0;
          unsigned nf = kws.nfields();
          for (unsigned f = 0; f < nf; ++f) {
            std::string nm = kws.name(f);
            if (nm != "MEASINFO" && nm != "QuantumUnits") {
              switch (kws.dataType(f)) {
#define CMP_KW(DT)                                  \
                case DataType<DT>::CasacoreTypeTag:
                LEGMS_FOREACH_RECORD_DATATYPE(CMP_KW)
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
    std::optional<unsigned> fmeas;
    unsigned nf = kws.nfields();
    for (unsigned f = 0; !fmeas && f < nf; ++f) {
      std::string nm = kws.name(f);
      if (nm == "MEASINFO" && kws.dataType(f) == casacore::DataType::TpRecord)
        fmeas = f;
    }
    recorder.expect_true(
      std::string("column ") + args->column
      + " has measure only if MS column has a measure",
      TE(args->has_measures == fmeas.has_value()));
    if (args->has_measures) {
      PhysicalRegion pr = regions[region_idx];
      region_idx += 1;
      std::optional<MeasRefDict::Ref> oref =
        MeasRefContainer::with_measure_references_dictionary(
          context,
          runtime,
          pr,
          false,
          [](Legion::Context c, Legion::Runtime* r, MeasRefDict* dict) {
            auto names = dict->names();
            return
              ((names.size() == 1) ? dict->get(*names.begin()) : std::nullopt);
          });
      recorder.assert_true(
        std::string("column ") + args->column + " has exactly one measure",
        TE(oref.has_value()));
      recorder.expect_true(
        std::string("column ") + args->column + " has expected measure",
        testing::TestEval(
          [&kws, &fmeas, &oref]() {
            bool result = false;
            casacore::MeasureHolder mh;
            casacore::String err;
            auto converted = mh.fromType(err, kws.asRecord(fmeas.value()));
            if (converted) {
              auto ref = oref.value();
              // TODO: improve the test for comparing MeasRef values
              if (false) {}
#define MATCH(MC)                                                       \
              else if (MClassT<MC>::holds(mh)) {                        \
                result = MeasRefDict::holds<MC>(ref);                   \
              }
              LEGMS_FOREACH_MCLASS(MATCH)
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
  Table table = Table::from_ms(ctx, rt, t0_path, {"*"});
  recorder.assert_true(
    "t0.ms MAIN table successfully read",
    !table.is_empty(ctx, rt));
  recorder.expect_true(
    "main table name is 'MAIN'",
    TE(table.name(ctx, rt)) == "MAIN");

  std::vector<std::string> expected_columns{
    "UVW",
    "FLAG",
    "FLAG_CATEGORY",
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
    "WEIGHT_SPECTRUM"
  };
  recorder.assert_true(
    "table has expected columns",
    testing::TestEval(
      [&table, &expected_columns, &ctx, rt]() {
        auto colnames = table.column_names(ctx, rt);
        return
          std::set<std::string>(colnames.begin(), colnames.end()) ==
          std::set<std::string>(expected_columns.begin(), expected_columns.end());
      }));

  recorder.expect_true(
    "table has expected MS_VERSION keyword value",
    testing::TestEval(
      [&table, &ctx, rt]() {
        auto fid = table.keywords.find_keyword(rt, "MS_VERSION");
        std::optional<float> msv;
        if (fid)
          msv = table.keywords.read<float>(ctx, rt, fid.value());
        return fid && msv && msv.value() == 2.0;
      }));
  recorder.expect_true(
    "only table keyword is MS_VERSION",
    testing::TestEval(
      [&table, &ctx, rt]() {
        auto keys = table.keywords.keys(rt);
        std::set<std::string> expected{"MS_VERSION"};
        std::set<std::string> kw(keys.begin(), keys.end());
        return expected == kw;
      }));
  recorder.expect_true(
    "table has no measures",
    TE(table.meas_refs.size(rt) == 0));
  //
  // read MS table columns to initialize the Column LogicalRegions
  //
  {
    TableReadTask table_read_task(
      t0_path,
      table,
      expected_columns.begin(),
      expected_columns.end(),
      2000);
    table_read_task.dispatch(ctx, rt);
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
  // can't use IndexTaskLauncher here since column LogicalRegions are not
  // sub-regions of a common LogicalPartition
  TaskLauncher verify_task(
    VERIFY_COLUMN_TASK,
    TaskArgument(&args, sizeof(args)));
  for (size_t i = 0; i < expected_columns.size(); ++i) {
    auto col = table.column(ctx, rt, expected_columns[i]);
    args.tag = col.datatype(ctx, rt);
    fstrcpy(args.column, col.name(ctx, rt).c_str());
    args.column[sizeof(args.column) - 1] = '\0';
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
    if (!col.is_empty()) {
      RegionRequirement req(col.values_lr, READ_ONLY, EXCLUSIVE, col.values_lr);
      req.add_field(Column::VALUE_FID);
      verify_task.add_region_requirement(req);
      args.has_values = true;
    } else {
      args.has_values = false;
    }
    if (!col.keywords.is_empty()) {
      auto n = col.keywords.size(rt);
      std::vector<FieldID> fids(n);
      std::iota(fids.begin(), fids.end(), 0);
      auto reqs = col.keywords.requirements(rt, fids, READ_ONLY).value();
      verify_task.add_region_requirement(reqs.type_tags);
      verify_task.add_region_requirement(reqs.values);
      args.has_keywords = true;
    } else {
      args.has_keywords = false;
    }
    if (col.meas_refs.size(rt) > 0) {
      args.has_measures = true;
      RegionRequirement
        req(col.meas_refs.lr, READ_ONLY, EXCLUSIVE, col.meas_refs.lr);
      req.add_field(MeasRefContainer::OWNED_FID);
      req.add_field(MeasRefContainer::MEAS_REF_FID);
      verify_task.add_region_requirement(req);
      auto creqs = col.meas_refs.component_requirements(ctx, rt);
      for (auto& r : creqs)
        verify_task.add_region_requirement(r);
    } else {
      args.has_measures = false;
    }
    rt->execute_task(ctx, verify_task);
  }
  rt->destroy_logical_partition(ctx, verify_col_logs);
  rt->destroy_index_partition(ctx, col_log_ip);
  rt->destroy_index_space(ctx, col_is);
  table.destroy(ctx, rt);
}

void
ms_test_suite(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context context,
  Runtime* runtime) {

  register_tasks(context, runtime);

  testing::TestLog<READ_WRITE> log(
    task->regions[0].region,
    regions[0],
    task->regions[1].region,
    regions[1],
    context,
    runtime);

  read_full_ms(log, context, runtime);
}

int
main(int argc, char** argv) {

  TaskVariantRegistrar registrar(VERIFY_COLUMN_TASK, "verify_column_task");
  registrar.add_constraint(ProcessorConstraint(Processor::IO_PROC));
  Runtime::preregister_task_variant<verify_column_task>(
    registrar,
    "verify_column_task");

  testing::TestSuiteDriver driver =
    testing::TestSuiteDriver::make<ms_test_suite>(
      MS_TEST_SUITE,
      "ms_test_suite");

  return driver.start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
