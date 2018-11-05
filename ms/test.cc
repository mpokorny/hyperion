#include <algorithm>
#include <experimental/filesystem>
#include <map>
#include <memory>
#include <vector>

#include "legion.h"

#include "IndexTree.h"
#include "Column.h"
#include "Table.h"
#include "TableBuilder.h"
#include "TableReadTask.h"
#include "FillProjectionsTask.h"

namespace fs = std::experimental::filesystem;

using namespace legms;
using namespace legms::ms;
using namespace Legion;

enum {
  TOP_LEVEL_TASK_ID,
};

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

  static std::optional<IndexPartition>
  read_partition(const std::shared_ptr<const Table>& table) {

    std::optional<IndexPartition> result;
    const unsigned subsample = 10000;
    if (table->name() == "POINTING" && subsample > 1) {
      auto runtime = table->runtime();
      auto ctx = table->context();
      auto fs = runtime->create_field_space(ctx);
      auto fa = runtime->create_field_allocator(ctx, fs);
      auto fid = fa.allocate_field(sizeof(Point<2>));
      auto is = table->index_space();
      assert(is.get_dim() == 3);
      auto lr = runtime->create_logical_region(ctx, is, fs);
      // use InlineLauncher for simplicity
      auto launcher = InlineLauncher(
        RegionRequirement(lr, WRITE_DISCARD, EXCLUSIVE, lr));
      launcher.add_field(fid);
      auto pr = runtime->map_region(ctx, launcher);
      const FieldAccessor<WRITE_DISCARD, Point<2>, 3> ps(pr, fid);
      for (PointInDomainIterator<3> pid(runtime->get_index_space_domain(is));
           pid();
           pid++)
        ps[*pid] = Point<2>(((pid[0] % subsample == 0) ? 0 : 1), pid[1]);
      runtime->unmap_region(ctx, pr);
      auto colors =
        runtime->create_index_space(
          ctx,
          Rect<2>(Point<2>(0, 0), Point<2>(1, 1)));
      result = runtime->create_partition_by_field(ctx, lr, lr, fid, colors);
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

    // register legms library tasks
    TableReadTask::register_task(runtime);
    TreeIndexSpace::register_tasks(runtime);
    FillProjectionsTasks::register_tasks(runtime);

    // create the Table instance
    std::unordered_set<std::string>
      colnames_set(colnames.begin(), colnames.end());
    std::shared_ptr<const Table> table;
    if (pointing_direction_only(table_name, colnames)) {
      // special test case: create Table with prior knowledge of its shape
      std::vector<Column::Generator>
        cols {
              Column::generator(
                "DIRECTION",
                casacore::TpDouble,
                IndexTreeL(1),
                IndexTreeL({{1, IndexTreeL({{2, IndexTreeL(1)}})}}),
                75107)
      };
      table.reset(new Table(ctx, runtime, table_path.value(), cols));
    } else {
      // general test case: create Table by scanning shape of MS table
      auto builder =
        TableBuilder::from_casacore_table(
          table_path.value(),
          colnames_set,
          TableBuilder::ms_column_hints(table_name));
      table.reset(
        new Table(
          ctx,
          runtime,
          builder.name(),
          builder.column_generators(),
          builder.keywords()));
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

    TableReadTask table_read_task(
      table_path.value(),
      table,
      colnames.begin(),
      end_present_colnames,
      TableBuilder::ms_column_hints(table_name),
      10000);
    auto lr_fids = table_read_task.dispatch();

    //
    // compute the LogicalRegions to read back
    //

    // special test case: partitioned read back
    std::optional<IndexPartition> read_ip = read_partition(table);
    std::vector<IndexSpace> read_is;

    // read_lr_fids tuple values: colname, region, parent region, field id
    std::vector<std::tuple<std::string, LogicalRegion, LogicalRegion, FieldID>>
      read_lr_fids;
    if (read_ip) {
      // for partitioned read, we select only a couple of the sub-regions per
      // column
      auto col_ip =
        table->index_partitions(
          read_ip.value(),
          colnames.begin(),
          end_present_colnames);
      for (size_t i = 0; i < lr_fids.size(); ++i) {
        auto& [lr, fid] = lr_fids[i];
        auto lp = runtime->get_logical_partition(ctx, lr, col_ip[i]);
        read_lr_fids.emplace_back(
          colnames[i],
          runtime->get_logical_subregion_by_color(ctx, lp, Point<2>(0, 0)),
          lr,
          fid);
        read_lr_fids.emplace_back(
          colnames[i],
          runtime->get_logical_subregion_by_color(ctx, lp, Point<2>(0, 1)),
          lr,
          fid);
        runtime->destroy_logical_partition(ctx, lp);
      }
      std::for_each(
        col_ip.begin(),
        col_ip.end(),
        [runtime, &ctx](auto& ip) {
          runtime->destroy_index_partition(ctx, ip);
        });
      read_is.push_back(
        runtime->get_index_subspace(ctx, read_ip.value(), Point<2>(0, 0)));
      read_is.push_back(
        runtime->get_index_subspace(ctx, read_ip.value(), Point<2>(0, 1)));
    } else {
      // general case: read complete columns
      for (size_t i = 0; i < lr_fids.size(); ++i) {
        auto& [lr, fid] = lr_fids[i];
        read_lr_fids.emplace_back(colnames[i], lr, lr, fid);
      }
      read_is.push_back(table->index_space());
    }

    // launch the read tasks inline
    std::vector<PhysicalRegion> prs;
    std::transform(
      read_lr_fids.begin(),
      read_lr_fids.end(),
      std::back_inserter(prs),
      [runtime, &ctx](auto& rlf) {
        auto launcher = InlineLauncher(
          RegionRequirement(
            std::get<1>(rlf),
            READ_ONLY,
            EXCLUSIVE,
            std::get<2>(rlf)));
        launcher.add_field(std::get<3>(rlf));
        return runtime->map_region(ctx, launcher);
      });

    // print out the values read, by partition (inc. partition by columns)
    switch (table->rank()) {
    case 1:
      show_table<1>(runtime, table, read_lr_fids, prs, read_is);
      break;
    case 2:
      show_table<2>(runtime, table, read_lr_fids, prs, read_is);
      break;
    case 3:
      show_table<3>(runtime, table, read_lr_fids, prs, read_is);
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
  static bool
  same_prefix(
    const PointInDomainIterator<DIM>& pid0,
    const PointInDomainIterator<DIM>& pid1,
    unsigned rank) {

    bool result = true;
    for (unsigned i = 0; result && i < rank; ++i)
      result = pid0[i] == pid1[i];
    return result;
  }

  template <int DIM>
  static void
  show_table(
    Runtime* runtime,
    const std::shared_ptr<const Table>& table,
    const std::vector<
    std::tuple<std::string, LogicalRegion, LogicalRegion, FieldID>>&
    read_lr_fids,
    const std::vector<PhysicalRegion>& prs,
    const std::vector<IndexSpace>& table_subspaces) {

    std::ostringstream oss;
    const char *sep = "";
    oss << "Columns: ";
    for (size_t i = 0; i < read_lr_fids.size(); i += table_subspaces.size()) {
      oss << sep << std::get<0>(read_lr_fids[i]);
      sep = ",";
    }

    oss << std::endl;
    auto row_rank = table->row_rank();
    auto num_col = read_lr_fids.size();

    for (unsigned n = 0; n < table_subspaces.size(); ++n) {
      DomainT<DIM> domain = runtime->get_index_space_domain(table_subspaces[n]);
      oss << std::endl;
      std::optional<PointInDomainIterator<DIM>> row_pid;
      for (PointInDomainIterator<DIM> pid(domain, false); pid(); pid++) {
        if (!row_pid || !same_prefix(row_pid.value(), pid, row_rank)) {
          if (row_pid)
            oss << ")" << std::endl;
          oss << "([" << pid[0];
          for (size_t i = 1; i < row_rank; ++i)
            oss << "," << pid[i];
          oss << "]:";
          row_pid = pid;
          const char* sep = "";
          for (size_t i = n; i < num_col; i += table_subspaces.size()) {
            oss << sep << "{";
            show_values<DIM>(
              table,
              read_lr_fids[i],
              prs[i],
              pid,
              row_rank,
              oss);
            oss << "}";
            sep = ";";
          }
        }
      }
      if (row_pid)
        oss << ")";
      oss << std::endl;
    }
    std::cout << oss.str();
  }

  template <int DIM>
  static void
  show_values(
    const std::shared_ptr<const Table>& table,
    const std::tuple<std::string, LogicalRegion, LogicalRegion, FieldID>& rlf,
    const PhysicalRegion& pr,
    const PointInDomainIterator<DIM>& pid0,
    unsigned row_rank,
    std::ostringstream& oss) {

    auto col = table->column(std::get<0>(rlf));
    switch (col->rank()) {
    case 1:
      show_column_values<DIM, 1>(col, rlf, pr, pid0, row_rank, oss);
      break;
    case 2:
      show_column_values<DIM, 2>(col, rlf, pr, pid0, row_rank, oss);
      break;
    case 3:
      show_column_values<DIM, 3>(col, rlf, pr, pid0, row_rank, oss);
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
    const std::tuple<std::string, LogicalRegion, LogicalRegion, FieldID>& rlf,
    const PhysicalRegion& pr,
    const PointInDomainIterator<TDIM>& pid0,
    unsigned row_rank,
    std::ostringstream& oss) {

    switch (col->datatype()) {
    case casacore::DataType::TpBool:
      show_scalar_column_values<TDIM, CDIM, casacore::DataType::TpBool>(
        rlf, pr, pid0, row_rank, oss);
      break;

    case casacore::DataType::TpArrayBool:
      show_array_column_values<TDIM, CDIM, casacore::DataType::TpArrayBool>(
        rlf, pr, pid0, row_rank, oss);
      break;

    case casacore::DataType::TpChar:
      show_scalar_column_values<TDIM, CDIM, casacore::DataType::TpChar>(
        rlf, pr, pid0, row_rank, oss);
      break;

    case casacore::DataType::TpArrayChar:
      show_array_column_values<TDIM, CDIM, casacore::DataType::TpArrayChar>(
        rlf, pr, pid0, row_rank, oss);
      break;

    case casacore::DataType::TpUChar:
      show_scalar_column_values<TDIM, CDIM, casacore::DataType::TpUChar>(
        rlf, pr, pid0, row_rank, oss);
      break;

    case casacore::DataType::TpArrayUChar:
      show_array_column_values<TDIM, CDIM, casacore::DataType::TpArrayUChar>(
        rlf, pr, pid0, row_rank, oss);
      break;

    case casacore::DataType::TpShort:
      show_scalar_column_values<TDIM, CDIM, casacore::DataType::TpShort>(
        rlf, pr, pid0, row_rank, oss);
      break;

    case casacore::DataType::TpArrayShort:
      show_array_column_values<TDIM, CDIM, casacore::DataType::TpArrayShort>(
        rlf, pr, pid0, row_rank, oss);
      break;

    case casacore::DataType::TpUShort:
      show_scalar_column_values<TDIM, CDIM, casacore::DataType::TpUShort>(
        rlf, pr, pid0, row_rank, oss);
      break;

    case casacore::DataType::TpArrayUShort:
      show_array_column_values<TDIM, CDIM, casacore::DataType::TpArrayUShort>(
        rlf, pr, pid0, row_rank, oss);
      break;

    case casacore::DataType::TpInt:
      show_scalar_column_values<TDIM, CDIM, casacore::DataType::TpInt>(
        rlf, pr, pid0, row_rank, oss);
      break;

    case casacore::DataType::TpArrayInt:
      show_array_column_values<TDIM, CDIM, casacore::DataType::TpArrayInt>(
        rlf, pr, pid0, row_rank, oss);
      break;

    case casacore::DataType::TpUInt:
      show_scalar_column_values<TDIM, CDIM, casacore::DataType::TpUInt>(
        rlf, pr, pid0, row_rank, oss);
      break;

    case casacore::DataType::TpArrayUInt:
      show_array_column_values<TDIM, CDIM, casacore::DataType::TpArrayUInt>(
        rlf, pr, pid0, row_rank, oss);
      break;

    case casacore::DataType::TpFloat:
      show_scalar_column_values<TDIM, CDIM, casacore::DataType::TpFloat>(
        rlf, pr, pid0, row_rank, oss);
      break;

    case casacore::DataType::TpArrayFloat:
      show_array_column_values<TDIM, CDIM, casacore::DataType::TpArrayFloat>(
        rlf, pr, pid0, row_rank, oss);
      break;

    case casacore::DataType::TpDouble:
      show_scalar_column_values<TDIM, CDIM, casacore::DataType::TpDouble>(
        rlf, pr, pid0, row_rank, oss);
      break;

    case casacore::DataType::TpArrayDouble:
      show_array_column_values<TDIM, CDIM, casacore::DataType::TpArrayDouble>(
        rlf, pr, pid0, row_rank, oss);
      break;

    case casacore::DataType::TpComplex:
      show_scalar_column_values<TDIM, CDIM, casacore::DataType::TpComplex>(
        rlf, pr, pid0, row_rank, oss);
      break;

    case casacore::DataType::TpArrayComplex:
      show_array_column_values<TDIM, CDIM, casacore::DataType::TpArrayComplex>(
        rlf, pr, pid0, row_rank, oss);
      break;

    case casacore::DataType::TpDComplex:
      show_scalar_column_values<TDIM, CDIM, casacore::DataType::TpDComplex>(
        rlf, pr, pid0, row_rank, oss);
      break;

    case casacore::DataType::TpArrayDComplex:
      show_array_column_values<TDIM, CDIM, casacore::DataType::TpArrayDComplex>(
        rlf, pr, pid0, row_rank, oss);
      break;

    case casacore::DataType::TpString:
      show_scalar_column_values<TDIM, CDIM, casacore::DataType::TpString>(
        rlf, pr, pid0, row_rank, oss);
      break;

    case casacore::DataType::TpArrayString:
      show_array_column_values<TDIM, CDIM, casacore::DataType::TpArrayString>(
        rlf, pr, pid0, row_rank, oss);
      break;

    default:
      assert(false);
      break;
    }
  }

  template <int TDIM, int CDIM, casacore::DataType DT>
  static void
  show_scalar_column_values(
    const std::tuple<std::string, LogicalRegion, LogicalRegion, FieldID>& rlf,
    const PhysicalRegion& pr,
    const PointInDomainIterator<TDIM>& pid0,
    unsigned row_rank,
    std::ostringstream& oss) {

    FieldID fid = std::get<3>(rlf);
    const FieldAccessor<
      READ_ONLY,
      typename DataType<DT>::ValueType,
      CDIM,
      coord_t,
      Realm::AffineAccessor<typename DataType<DT>::ValueType, CDIM, coord_t>,
      false> values(pr, fid);

    PointInDomainIterator<TDIM> tpid = pid0;
    coord_t pt[CDIM];
    for (size_t i = 0; i < CDIM; ++i)
      pt[i] = tpid[i];
    auto p = to_point<CDIM>(pt);
    oss << values[p];
    tpid++;
    while (tpid() && same_prefix(pid0, tpid, row_rank)) {
      if (!same_prefix(pid0, tpid, CDIM)) {
        for (size_t i = 0; i < CDIM; ++i)
          pt[i] = tpid[i];
        p = to_point<CDIM>(pt);
        oss << "," << values[p];
      }
      tpid++;
    }
  }

  template <int TDIM, int CDIM, casacore::DataType DT>
  static void
  show_array_column_values(
    const std::tuple<std::string, LogicalRegion, LogicalRegion, FieldID>& rlf,
    const PhysicalRegion& pr,
    const PointInDomainIterator<TDIM>& pid0,
    unsigned row_rank,
    std::ostringstream& oss) {

    FieldID fid = std::get<3>(rlf);
    const FieldAccessor<
      READ_ONLY,
      typename DataType<DT>::ValueType,
      CDIM,
      coord_t,
      Realm::AffineAccessor<typename DataType<DT>::ValueType, CDIM, coord_t>,
      false> values(pr, fid);

    PointInDomainIterator<TDIM> tpid = pid0;
    coord_t pt[CDIM];
    for (size_t i = 0; i < CDIM; ++i)
      pt[i] = tpid[i];
    auto p = to_point<CDIM>(pt);
    auto vals = values[p];
    if (vals.size() > 0) {
      oss << "<" << vals[0];
      for (size_t i = 1; i < vals.size(); ++i)
        oss << "," << vals[i];
      oss << ">";
    }
    tpid++;
    while (tpid() && same_prefix(pid0, tpid, row_rank)) {
      if (!same_prefix(pid0, tpid, CDIM)) {
        for (size_t i = 0; i < CDIM; ++i)
          pt[i] = tpid[i];
        p = to_point<CDIM>(pt);
        oss << ",";
        auto vals = values[p];
        if (vals.size() > 0) {
          oss << "<" << vals[0];
          for (size_t i = 1; i < vals.size(); ++i)
            oss << "," << vals[i];
          oss << ">";
        }
      }
      tpid++;
    }
  }

  template <int DIM>
  static Point<DIM>
  to_point(const coord_t vals[DIM]) {
    return Point<DIM>(vals);
  }
};

template <>
Point<1>
TopLevelTask::to_point(const coord_t vals[1]) {
  return Point<1>(vals[0]);
}

void
usage() {
  std::cout << "usage: test [MS](/[TABLE]) [COL]+" << std::endl;
}

int
main(int argc, char** argv) {

  int app_argc = 0;
  std::optional<int> dir_arg;
  for (int i = 1; i < argc; ++i) {
    if (*argv[i] != '-') {
      if (!dir_arg)
        dir_arg = i;
      ++app_argc;
    } else {
      ++i; // skip option argument
    }
  }
  if (app_argc < 2) {
    usage();
    return 1;
  }
  auto arg1_fs_status = fs::status(argv[dir_arg.value()]);
  if (!fs::is_directory(arg1_fs_status)) {
    std::cout << "directory '" << argv[dir_arg.value()]
              << "' does not exist"
              << std::endl;
    usage();
    return 1;
  }

  Runtime::set_top_level_task_id(TopLevelTask::TASK_ID);
  TopLevelTask::register_task();
  SerdezManager::register_ops();
  return Runtime::start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:

