#include <algorithm>
#include <experimental/filesystem>
#include <map>
#include <memory>
#include <vector>

#include "legms.h"
#include "utility.h"
#include "IndexTree.h"
#include "Column.h"
#include "Table.h"
#include "TableBuilder.h"
#include "TableReadTask.h"

#include "TestSuiteDriver.h"
#include "TestRecorder.h"
#include "TestExpression.h"

namespace fs = std::experimental::filesystem;

using namespace legms;
using namespace Legion;

enum {
  MS_TEST_SUITE,
  VERIFY_COLUMN_TASK,
};

#ifdef UNUSED_STUFF
class TopLevelTask {
public:

  static constexpr const char *TASK_NAME = "top_level";
  static const int TASK_ID = TOP_LEVEL_TASK_ID;

  static bool
  pointing_direction_only(
    const std::string& table,
    const std::vector<std::string>& colnames) {
    return table == "POINTING"
      && colnames.size() == 1
      && colnames[0] == "DIRECTION";
  }

  static std::vector<MSTable<MSTables::POINTING>::Axes>
  pointing_direction_axes() {
    std::vector<MSTable<MSTables::POINTING>::Axes> result = {
      MSTable<MSTables::POINTING>::Axes::ROW
    };
    auto dir_axes = MSTable<MSTables::POINTING>::element_axes.at("DIRECTION");
    std::copy(
      dir_axes.begin(),
      dir_axes.end(),
      std::back_inserter(result));
    return result;
  }

  static std::unique_ptr<ColumnPartition>
  read_partition(const Table* table) {

    std::unique_ptr<ColumnPartition> result;

    const unsigned subsample = 10000;
    auto is = table->column(table->max_rank_column_name())->index_space();
    if (table->name() == "POINTING" && subsample > 1 && is.get_dim() == 3) {
      std::cout << "partitioned read" << std::endl;
      auto runtime = table->runtime();
      auto ctx = table->context();
      auto fs = runtime->create_field_space(ctx);
      auto fa = runtime->create_field_allocator(ctx, fs);
      auto fid = fa.allocate_field(sizeof(Point<2>));
      auto lr = runtime->create_logical_region(ctx, is, fs);
      // use InlineLauncher for simplicity
      auto launcher = InlineLauncher(
        RegionRequirement(lr, WRITE_DISCARD, EXCLUSIVE, lr));
      launcher.add_field(fid);
      auto pr = runtime->map_region(ctx, launcher);
      const FieldAccessor<
        WRITE_DISCARD,
        Point<2>,
        3,
        coord_t,
        AffineAccessor<Point<2>, 3, coord_t>,
        false> ps(pr, fid);
      for (PointInDomainIterator<3> pid(runtime->get_index_space_domain(is));
           pid();
           pid++)
        ps[*pid] = Point<2>(((pid[0] % subsample == 0) ? 0 : 1), pid[2]);
      runtime->unmap_region(ctx, pr);
      auto colors =
        runtime->create_index_space(
          ctx,
          Rect<2>(Point<2>(0, 0), Point<2>(1, 1)));
      result =
        std::make_unique<ColumnPartitionT<MSTable<MSTables::POINTING>::Axes>>(
          ctx,
          runtime,
          runtime->create_partition_by_field(ctx, lr, lr, fid, colors),
          pointing_direction_axes());
      runtime->destroy_index_space(ctx, colors);
      runtime->destroy_logical_region(ctx, lr);
      runtime->destroy_field_space(ctx, fs);
    }
    return result;
  }

  static void
  base_impl(
    const Task*,
    const std::vector<PhysicalRegion>&,
    Context ctx,
    Runtime* runtime) {

    // get MS path and table name
    auto input_args = Runtime::get_input_args();
    std::optional<fs::path> table_path;
    std::vector<std::string> colnames;
    for (auto i = 1; i < input_args.argc; ++i) {
      if (*input_args.argv[i] == '-') {
        ++i;// skip option argument
      } else if (!table_path) {
        table_path = fs::path(input_args.argv[i]);
      } else {
        if (strcmp(input_args.argv[i], "*") == 0) {
          colnames.clear();
          colnames.push_back(input_args.argv[i]);
          break;
        }
        colnames.push_back(input_args.argv[i]);
      }
    }
    fs::path ms_path = table_path.value().parent_path();
    std::string table_name = table_path.value().filename();
    if (table_name == ".")
      table_name = "MAIN";

    // register legms library tasks
    TableReadTask::register_task(runtime);
    TreeIndexSpace::register_tasks(runtime);
    Column::register_tasks(runtime);
    ProjectedIndexPartitionTask::register_task(runtime);

    // create the Table instance
    std::unordered_set<std::string>
      colnames_set(colnames.begin(), colnames.end());
    std::unique_ptr<const Table> table;
    if (pointing_direction_only(table_name, colnames)) {
      // special test case: create Table with prior knowledge of its shape
      ColumnT<MSTable<MSTables::POINTING>::Axes>::Generator colgen =
        ColumnT<MSTable<MSTables::POINTING>::Axes>::generator(
          "DIRECTION",
          LEGMS_TYPE_DOUBLE,
          pointing_direction_axes(),
          IndexTreeL(1),
          IndexTreeL({{1, IndexTreeL({{1, IndexTreeL(2)}})}}),
          75107);
      table.reset(
        new TableT<MSTable<MSTables::POINTING>::Axes>(
          ctx,
          runtime,
          table_name,
          {static_cast<int>(MSTable<MSTables::POINTING>::Axes::ROW)},
          {colgen}));
    } else {
      // general test case: create Table by scanning shape of MS table
      table = Table::from_ms(ctx, runtime, table_path.value(), colnames_set);
    }
    std::cout << "table name: "
              << table->name() << std::endl;
    if (table->is_empty()) {
      std::cout << "Empty table" << std::endl;
      return;
    }
    if (colnames_set.count("*") > 0) {
      colnames.clear();
      auto cols = table->column_names();
      std::copy(cols.begin(), cols.end(), std::back_inserter(colnames));
    }

    // check for empty columns, which we will skip hereafter
    auto end_present_colnames =
      std::remove_if(
        colnames.begin(),
        colnames.end(),
        [cols=table->column_names()](auto& nm) {
          return cols.count(nm) == 0;
        });
    if (end_present_colnames != colnames.end()) {
      std::cout << "Empty columns: " << *end_present_colnames;
      std::for_each(
        end_present_colnames + 1,
        colnames.end(),
        [](auto &nm) {
          std::cout << ", " << nm;
        });
      std::cout << std::endl;
    }

    //
    // read MS table columns to initialize the Column LogicalRegions
    //

    {
      TableReadTask table_read_task(
        table_path.value(),
        table.get(),
        colnames.begin(),
        end_present_colnames,
        10000);
      table_read_task.dispatch();
    }

    //
    // compute the LogicalRegions to read back
    //

    // special test case: partitioned read back
    auto read_p = read_partition(table.get());

    std::vector<LogicalRegion> lrs;
    std::transform(
      colnames.begin(),
      end_present_colnames,
      std::back_inserter(lrs),
      [&table](auto& nm) { return table->column(nm)->logical_region(); });
    // read_lr_fids tuple values: colname, region, parent region
    std::vector<
      std::vector<std::tuple<std::string, LogicalRegion, LogicalRegion>>>
      read_lrs;
    if (read_p) {
      // for partitioned read, we select only a couple of the sub-regions per
      // column
      read_lrs.resize(2);
      for (size_t i = 0; i < lrs.size(); ++i) {
        auto lr = lrs[i];
        if (lr != LogicalRegion::NO_REGION) {
          auto col_ip =
            table->column(colnames[i])->projected_column_partition(read_p.get());
          auto lp =
            runtime->get_logical_partition(ctx, lr, col_ip->index_partition());
          read_lrs[0].emplace_back(
            colnames[i],
            runtime->get_logical_subregion_by_color(ctx, lp, Point<2>(0, 0)),
            lr);
          read_lrs[1].emplace_back(
            colnames[i],
            runtime->get_logical_subregion_by_color(ctx, lp, Point<2>(0, 1)),
            lr);
          runtime->destroy_logical_partition(ctx, lp);
        } else {
          std::cout << "skip empty column " << colnames[i] << std::endl;
        }
      }
    } else {
      // general case: read complete columns
      read_lrs.resize(1);
      for (size_t i = 0; i < lrs.size(); ++i) {
        auto lr = lrs[i];
        if (lr != LogicalRegion::NO_REGION)
          read_lrs[0].emplace_back(colnames[i], lr, lr);
        else
          std::cout << "skip empty column " << colnames[i] << std::endl;
      }
    }

    // If all columns are empty, we're done
    if (read_lrs[0].size() == 0)
      return;

    // Find maximum rank of columns to determine index space for read back and
    // output
    unsigned max_col_rank = table->column(std::get<0>(read_lrs[0][0]))->rank();
    size_t max_col_rank_idx = 0;
    for (size_t i = 1; i < read_lrs[0].size(); ++i) {
      auto rank = table->column(std::get<0>(read_lrs[0][i]))->rank();
      if (rank > max_col_rank) {
        max_col_rank = rank;
        max_col_rank_idx = i;
      }
    }
    // Get the IndexSpace for read back and output
    std::vector<IndexSpace> read_is;
    std::transform(
      read_lrs.begin(),
      read_lrs.end(),
      std::back_inserter(read_is),
      [&max_col_rank_idx](auto& rlrs) {
        return std::get<1>(rlrs[max_col_rank_idx]).get_index_space();
      });

    // launch the read tasks inline
    std::vector<std::vector<PhysicalRegion>> prs;
    std::transform(
      read_lrs.begin(),
      read_lrs.end(),
      std::back_inserter(prs),
      [runtime, &ctx](auto& rlrs) {
        std::vector<PhysicalRegion> prs1;
        std::transform(
          rlrs.begin(),
          rlrs.end(),
          std::back_inserter(prs1),
          [runtime, &ctx](auto& rlr) {
            auto launcher = InlineLauncher(
              RegionRequirement(
                std::get<1>(rlr),
                READ_ONLY,
                EXCLUSIVE,
                std::get<2>(rlr)));
            launcher.add_field(Column::value_fid);
            return runtime->map_region(ctx, launcher);
          });
        return prs1;
      });

    // print out the values read, by partition (inc. partition by columns)
    switch (max_col_rank) {
    case 1:
      show_table<1>(ctx, runtime, table.get(), read_lrs, prs, max_col_rank_idx);
      break;
    case 2:
      show_table<2>(ctx, runtime, table.get(), read_lrs, prs, max_col_rank_idx);
      break;
    case 3:
      show_table<3>(ctx, runtime, table.get(), read_lrs, prs, max_col_rank_idx);
      break;
    case 4:
      show_table<4>(ctx, runtime, table.get(), read_lrs, prs, max_col_rank_idx);
      break;
    default:
      assert(false);
      break;
    }
  }

  static void
  register_task() {
    TaskVariantRegistrar registrar(TASK_ID, TASK_NAME);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<base_impl>(registrar, TASK_NAME);
  }

  template <int DIM>
  static void
  show_table(
    Context ctx,
    Runtime* runtime,
    const Table* table,
    const std::vector<
    std::vector<
    std::tuple<std::string, LogicalRegion, LogicalRegion>>>& read_lrs,
    const std::vector<std::vector<PhysicalRegion>>& prs,
    size_t max_rank_idx) {

    std::ostringstream oss;
    oss << "Columns: ";
    for (size_t i = 0; i < read_lrs[0].size(); ++i)
      oss << std::get<0>(read_lrs[0][i]) << " ";

    oss << std::endl;
    auto row_rank = table->index_axes().size();
    auto num_subspaces = read_lrs.size();
    auto num_col = read_lrs[0].size();

    for (unsigned n = 0; n < num_subspaces; ++n) {

      const FieldAccessor<
        READ_ONLY,
        Column::row_number_t,
        DIM,
        coord_t,
        Realm::AffineAccessor<Column::row_number_t, DIM, coord_t>,
        false> row_numbers(prs[n][max_rank_idx], Column::row_number_fid);

      DomainT<DIM> domain =
        runtime->get_index_space_domain(
          ctx,
          std::get<1>(read_lrs[n][max_rank_idx]).get_index_space());
      oss << std::endl;
      std::optional<Column::row_number_t> row;
      for (PointInDomainIterator<DIM> pid(domain, false); pid(); pid++) {
        if (row.value_or(row_numbers[*pid] + 1) != row_numbers[*pid]) {
          if (row)
            oss << ")" << std::endl;
          oss << "([" << pid[0];
          for (size_t i = 1; i < row_rank; ++i)
            oss << "," << pid[i];
          oss << "]:";
          row = row_numbers[*pid];
          const char* sep = "";
          for (size_t i = 0; i < num_col; ++i) {
            oss << sep << "{";
            show_values<DIM>(
              table,
              read_lrs[n][i],
              prs[n][i],
              pid,
              row.value(),
              oss);
            oss << "}";
            sep = ";";
          }
        }
      }
      if (row)
        oss << ")";
      oss << std::endl;
    }
    std::cout << oss.str();
  }

  template <int DIM>
  static void
  show_values(
    const Table* table,
    const std::tuple<std::string, LogicalRegion, LogicalRegion>& rlf,
    const PhysicalRegion& pr,
    const PointInDomainIterator<DIM>& pid0,
    Column::row_number_t row,
    std::ostringstream& oss) {

    auto col = table->column(std::get<0>(rlf));
    switch (col->rank()) {
    case 1:
      show_column_values<DIM, 1>(col, rlf, pr, pid0, row, oss);
      break;
    case 2:
      show_column_values<DIM, 2>(col, rlf, pr, pid0, row, oss);
      break;
    case 3:
      show_column_values<DIM, 3>(col, rlf, pr, pid0, row, oss);
      break;
    default:
      assert(false);
      break;
    }
  }

  template <int TDIM, int CDIM>
  static void
  show_column_values(
    const std::shared_ptr<Column>& col,
    const std::tuple<std::string, LogicalRegion, LogicalRegion>& rlf,
    const PhysicalRegion& pr,
    const PointInDomainIterator<TDIM>& pid0,
    Column::row_number_t row,
    std::ostringstream& oss) {

    switch (col->datatype()) {
    case LEGMS_TYPE_BOOL:
      show_scalar_column_values<TDIM, CDIM, LEGMS_TYPE_BOOL>(
        rlf, pr, pid0, row, oss);
      break;

    case LEGMS_TYPE_CHAR:
      show_scalar_column_values<TDIM, CDIM, LEGMS_TYPE_CHAR>(
        rlf, pr, pid0, row, oss);
      break;

    case LEGMS_TYPE_UCHAR:
      show_scalar_column_values<TDIM, CDIM, LEGMS_TYPE_UCHAR>(
        rlf, pr, pid0, row, oss);
      break;

    case LEGMS_TYPE_SHORT:
      show_scalar_column_values<TDIM, CDIM, LEGMS_TYPE_SHORT>(
        rlf, pr, pid0, row, oss);
      break;

    case LEGMS_TYPE_USHORT:
      show_scalar_column_values<TDIM, CDIM, LEGMS_TYPE_USHORT>(
        rlf, pr, pid0, row, oss);
      break;

    case LEGMS_TYPE_INT:
      show_scalar_column_values<TDIM, CDIM, LEGMS_TYPE_INT>(
        rlf, pr, pid0, row, oss);
      break;

    case LEGMS_TYPE_UINT:
      show_scalar_column_values<TDIM, CDIM, LEGMS_TYPE_UINT>(
        rlf, pr, pid0, row, oss);
      break;

    case LEGMS_TYPE_FLOAT:
      show_scalar_column_values<TDIM, CDIM, LEGMS_TYPE_FLOAT>(
        rlf, pr, pid0, row, oss);
      break;

    case LEGMS_TYPE_DOUBLE:
      show_scalar_column_values<TDIM, CDIM, LEGMS_TYPE_DOUBLE>(
        rlf, pr, pid0, row, oss);
      break;

    case LEGMS_TYPE_COMPLEX:
      show_scalar_column_values<TDIM, CDIM, LEGMS_TYPE_COMPLEX>(
        rlf, pr, pid0, row, oss);
      break;

    case LEGMS_TYPE_DCOMPLEX:
      show_scalar_column_values<TDIM, CDIM, LEGMS_TYPE_DCOMPLEX>(
        rlf, pr, pid0, row, oss);
      break;

    case LEGMS_TYPE_STRING:
      show_scalar_column_values<TDIM, CDIM, LEGMS_TYPE_STRING>(
        rlf, pr, pid0, row, oss);
      break;

    default:
      assert(false);
      break;
    }
  }

  template <int TDIM, int CDIM, legms::TypeTag DT>
  static void
  show_scalar_column_values(
    const std::tuple<std::string, LogicalRegion, LogicalRegion>& rlf,
    const PhysicalRegion& pr,
    const PointInDomainIterator<TDIM>& pid0,
    Column::row_number_t row,
    std::ostringstream& oss) {

    const FieldAccessor<
      READ_ONLY,
      typename DataType<DT>::ValueType,
      CDIM,
      coord_t,
      Realm::AffineAccessor<typename DataType<DT>::ValueType, CDIM, coord_t>,
      false> values(pr, Column::value_fid);

    const FieldAccessor<
      READ_ONLY,
      Column::row_number_t,
      CDIM,
      coord_t,
      Realm::AffineAccessor<Column::row_number_t, CDIM, coord_t>,
      false> row_numbers(pr, Column::row_number_fid);

    auto p0 = pid_prefix<CDIM>(pid0);
    PointInDomainIterator<TDIM> tpid = pid0;
    oss << values[p0];
    tpid++;
    while (tpid()) {
      auto p = pid_prefix<CDIM>(tpid);
      if (row_numbers[p] != row)
        break;
      if (p != p0)
        oss << "," << values[p];
      tpid++;
    }
  }

  template <int DIM>
  static inline Point<DIM>
  to_point(const coord_t vals[DIM]) {
    return Point<DIM>(vals);
  }

  template <int DIM, int TDIM>
  static Point<DIM>
  pid_prefix(const PointInDomainIterator<TDIM>& pid) {
    coord_t pt[DIM];
    for (size_t i = 0; i < DIM; ++i)
      pt[i] = pid[i];
    return to_point<DIM>(pt);
  }
};

template <>
inline Point<1>
TopLevelTask::to_point(const coord_t vals[1]) {
  return Point<1>(vals[0]);
}
#endif // UNUSED_STUFF

template <typename T, int DIM>
using RO = FieldAccessor<READ_ONLY, T, DIM, coord_t, AffineAccessor<T, DIM, coord_t>>;

#define TE(f) testing::TestEval([&](){ return f; }, #f)

struct VerifyColumnTaskArgs {
  legms::TypeTag tag;
  char table[160];
  char column[32];
};

void
verify_scalar_column(
  const casacore::Table& tb,
  const VerifyColumnTaskArgs *targs,
  const std::vector<PhysicalRegion>& regions,
  Context context,
  Runtime* runtime) {

  testing::TestLog<READ_WRITE> log(regions[1], regions[2], context, runtime);
  testing::TestRecorder<READ_WRITE> recorder(log);

  DomainT<1> col_dom(regions[0].get_bounds<1,coord_t>());

#define CMP(TAG)                                                    \
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
      col(regions[0], Column::value_fid);                           \
    PointInDomainIterator<1> pid(col_dom);                          \
    recorder.expect_true(                                           \
      std::string("verify values, column ") + targs->column,        \
      testing::TestEval(                                            \
      [&pid, &col, &ary, targs]() {                                 \
      bool result = true;                                           \
      for (; result && pid(); pid++) {                              \
        DataType<TAG>::ValueType a;                                 \
        DataType<TAG>::from_casacore(a, ary[pid[0]]);               \
        result = DataType<TAG>::equiv(a, col[*pid]);                \
      }                                                             \
      return result;                                                \
    }));                                                            \
    break;                                                          \
  }

  switch (targs->tag) {
    LEGMS_FOREACH_DATATYPE(CMP);
  }
#undef CMP
}

template <int DIM>
void
verify_array_column(
  const casacore::Table& tb,
  const VerifyColumnTaskArgs *targs,
  const std::vector<PhysicalRegion>& regions,
  Context context,
  Runtime* runtime) {

  testing::TestLog<READ_WRITE> log(regions[1], regions[2], context, runtime);
  testing::TestRecorder<READ_WRITE> recorder(log);

  DomainT<DIM> col_dom(regions[0].get_bounds<DIM,coord_t>());
  LogicalRegion ary_lr =
    runtime->create_logical_region(
      context,
      regions[0].get_logical_region().get_index_space(),
      regions[0].get_logical_region().get_field_space());

#define CMP(TAG)                                                        \
  case (TAG): {                                                         \
    auto acol =                                                         \
      casacore::ArrayColumn<DataType<TAG>::CasacoreType>(               \
        tb,                                                             \
        casacore::String(targs->column));                               \
    recorder.assert_true(                                               \
      std::string("verify rank, column ") + targs->column,              \
      TE(acol.ndim(0)) == DIM - 1);                                     \
    recorder.assert_true(                                               \
      std::string("verify nrows, column ") + targs->column,             \
      TE(Domain(col_dom).hi()[0]) == acol.nrow() - 1);                  \
    {                                                                   \
      PointInDomainIterator<DIM> pid(col_dom, false);                   \
      recorder.assert_true(                                             \
        std::string("verify bounds, column ") + targs->column,          \
        testing::TestEval(                                              \
          [&pid, &acol]() {                                             \
            bool result = true;                                         \
            while (result && pid()) {                                   \
              auto last_p = *pid;                                       \
              while (result && pid()) {                                 \
                pid++;                                                  \
                if (!pid() || pid[0] != last_p[0]) {                    \
                  casacore::IPosition shp(acol.shape(last_p[0]));       \
                  Point<DIM> cpt;                                       \
                  cpt[0] = last_p[0];                                   \
                  for (size_t i = 0; i < DIM - 1; ++i)                  \
                    cpt[i + 1] = shp[DIM - 2 - i] - 1;                  \
                  result = cpt == last_p;                               \
                }                                                       \
                if (pid())                                              \
                  last_p = *pid;                                        \
              }                                                         \
            }                                                           \
            return result;                                              \
          }));                                                          \
    }                                                                   \
    {                                                                   \
      const RO<DataType<TAG>::ValueType, DIM>                           \
        col(regions[0], Column::value_fid);                             \
      PointInDomainIterator<DIM> pid(col_dom, false);                   \
      recorder.assert_true(                                             \
        std::string("verify values, column ") + targs->column,          \
        testing::TestEval(                                              \
          [&pid, &acol, &col]() {                                       \
            bool result = true;                                         \
            casacore::Array<DataType<TAG>::CasacoreType> ary;           \
            casacore::IPosition ipos(DIM - 1);                          \
            while (result && pid()) {                                   \
              auto row = pid[0];                                        \
              acol.get(row, ary, true);                                 \
              while (result && pid()) {                                 \
                for (size_t i = 0; i < DIM - 1; ++i)                    \
                  ipos[DIM - 2 - i] = pid[i + 1];                       \
                DataType<TAG>::ValueType a;                             \
                DataType<TAG>::from_casacore(a, ary(ipos));             \
                result = DataType<TAG>::equiv(a, col[*pid]);            \
                pid++;                                                  \
                if (pid() && pid[0] != row) {                           \
                  row = pid[0];                                         \
                  acol.get(row, ary, true);                             \
                }                                                       \
              }                                                         \
            }                                                           \
            return result;                                              \
          }));                                                          \
    }                                                                   \
    break;                                                              \
}

  switch (targs->tag) {
    LEGMS_FOREACH_DATATYPE(CMP);
  }
#undef CMP

  runtime->destroy_logical_region(context, ary_lr);
}

void
verify_column_task(
  const Task* task,
  const std::vector<PhysicalRegion>& region,
  Context context,
  Runtime* runtime) {

  const VerifyColumnTaskArgs *args =
    static_cast<const VerifyColumnTaskArgs*>(task->args);

  casacore::Table tb(
    casacore::String(args->table),
    casacore::TableLock::PermanentLockingWait);

  auto cdesc = tb.tableDesc()[casacore::String(args->column)];
  if (cdesc.isScalar()) {
    verify_scalar_column(tb, args, region, context, runtime);
  } else {
#define VERIFY_ARRAY(N) \
    case (N): verify_array_column<N>(tb, args, region, context, runtime); break;

    switch (cdesc.ndim() + 1) {
      LEGMS_FOREACH_N(VERIFY_ARRAY);
    }
#undef VERIFY_ARRAY
  }
}

void
read_full_ms(
  testing::TestLog<READ_WRITE>& log,
  Context context,
  Runtime* runtime) {

  testing::TestRecorder<READ_WRITE> recorder(log);

  static const std::string t0_path("data/t0.ms");
  std::unique_ptr<const Table> table =
    Table::from_ms(context, runtime, t0_path, {"*"});
  recorder.assert_true(
    "t0.ms MAIN table successfully read",
    bool(table));
  recorder.expect_true(
    "main table name is 'MAIN'",
    TE(table->name()) == "MAIN");
  recorder.expect_true(
    "main table is not empty",
    TE(!table->is_empty()));

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
    TE(std::set<std::string>(
         table->column_names().begin(),
         table->column_names().end()) ==
       std::set<std::string>(
         expected_columns.begin(),
         expected_columns.end())));

  //
  // read MS table columns to initialize the Column LogicalRegions
  //
  {
    TableReadTask table_read_task(
      t0_path,
      table.get(),
      expected_columns.begin(),
      expected_columns.end(),
      2000);
    table_read_task.dispatch();
  }

  // compare column LogicalRegions to values read using casacore functions
  // directly
  IndexSpace col_is(
    runtime->create_index_space(
      context,
      Rect<1>(0, expected_columns.size() - 1)));
  auto remaining_log =
    log.get_log_references_by_state({testing::TestState::UNKNOWN})[0];
  IndexPartition col_log_ip =
    runtime->create_equal_partition(
      context,
      remaining_log.log_region().get_index_space(),
      col_is);
  LogicalPartitionT<1> verify_col_logs(
    runtime->get_logical_partition(
      context,
      remaining_log.log_region(),
      col_log_ip));
  VerifyColumnTaskArgs args;
  std::strncpy(args.table, t0_path.c_str(), sizeof(args.table));
  args.table[sizeof(args.table) - 1] = '\0';
  // can't use IndexTaskLauncher here since column LogicalRegions are not
  // sub-regions of a common LogicalPartition
  TaskLauncher verify_task(
    VERIFY_COLUMN_TASK,
    TaskArgument(&args, sizeof(args)));
  for (size_t i = 0; i < expected_columns.size(); ++i) {
    auto col = table->column(expected_columns[i]);
    if (col->logical_region() != LogicalRegion::NO_REGION) {
      args.tag = col->datatype();
      std::strncpy(args.column, col->name().c_str(), sizeof(args.column));
      args.column[sizeof(args.column) - 1] = '\0';
      verify_task.region_requirements.clear();
      verify_task.add_region_requirement(
        RegionRequirement(
          col->logical_region(),
          READ_ONLY,
          EXCLUSIVE,
          col->logical_region()));
      verify_task.add_field(0, Column::value_fid);
      auto log_reqs =
        remaining_log.requirements<READ_WRITE>(
          runtime->get_logical_subregion_by_color(verify_col_logs, Point<1>(i)),
          log.log_reference().log_region());
      std::for_each(
        log_reqs.begin(),
        log_reqs.end(),
        [&verify_task](auto& req) {
          verify_task.add_region_requirement(req);
        });
      runtime->execute_task(context, verify_task);
    }
  }
  runtime->destroy_logical_partition(context, verify_col_logs);
  runtime->destroy_index_partition(context, col_log_ip);
  runtime->destroy_index_space(context, col_is);
}

void
ms_test_suite(
  const Task*,
  const std::vector<PhysicalRegion>& regions,
  Context context,
  Runtime* runtime) {

  register_tasks(runtime);

  testing::TestLog<READ_WRITE> log(regions[0], regions[1], context, runtime);

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
