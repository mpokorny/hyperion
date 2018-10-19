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
  read_partition(
    const std::shared_ptr<Table>& table,
    Context ctx,
    Runtime* runtime) {

    std::optional<IndexPartition> result;
    const unsigned subsample = 10000;
    if (table->name() == "POINTING" && subsample > 1) {
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
    std::shared_ptr<Table> table;
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
        TableBuilder::from_casacore_table(table_path.value(), colnames_set);
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
      10000);
    auto lr_fids = table_read_task.dispatch();

    //
    // compute the LogicalRegions to read back
    //

    // special test case: partitioned read back
    std::optional<IndexPartition> read_ip = read_partition(table, ctx, runtime);

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
    } else {
      // general case: read complete columns
      for (size_t i = 0; i < lr_fids.size(); ++i) {
        auto& [lr, fid] = lr_fids[i];
        read_lr_fids.emplace_back(colnames[i], lr, lr, fid);
      }
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
    auto row_index_pattern = table->row_index_pattern();
    for (size_t i = 0; i < read_lr_fids.size(); ++i) {
      auto nm = std::get<0>(read_lr_fids[i]);
      auto lr = std::get<1>(read_lr_fids[i]);
      auto fid = std::get<3>(read_lr_fids[i]);
      auto pr = prs[i];
      std::cout << nm << ":" << std::endl;
      auto col = table->column(nm);
      switch (col->rank()) {
      case 1:
        show<1>(runtime, pr, lr, fid, col, row_index_pattern);
        break;
      case 2:
        show<2>(runtime, pr, lr, fid, col, row_index_pattern);
        break;
      case 3:
        show<3>(runtime, pr, lr, fid, col, row_index_pattern);
        break;
      default:
        assert(false);
        break;
      }
      std::cout << std::endl;
      runtime->unmap_region(ctx, prs[i]);
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
  show(
    Runtime* runtime,
    PhysicalRegion pr,
    LogicalRegion lr,
    FieldID fid,
    std::shared_ptr<Column>& col,
    const IndexTreeL& row_index_pattern) {

    auto row_rank = row_index_pattern.rank().value();
    std::ostringstream oss;
    DomainT<DIM> domain =
      runtime->get_index_space_domain(lr.get_index_space());
    switch (col->datatype()) {
#define SHOW(tp)                                                       \
      casacore::DataType::Tp##tp: {                                     \
      const FieldAccessor<                                              \
        READ_ONLY, \
        DataType<casacore::DataType::Tp##tp>::ValueType, \
        DIM, \
        coord_t, \
        Realm::AffineAccessor< \
          DataType<casacore::DataType::Tp##tp>::ValueType,DIM,coord_t>, \
          false> values(pr, fid); \
      std::array<Legion::coord_t, DIM> pt;                              \
      size_t row_number;                                                \
      {                                                                 \
        Legion::PointInDomainIterator<DIM> pid(domain, false);          \
        for (size_t i = 0; i < DIM; ++i)                                \
          pt[i] = pid[i];                                               \
        row_number = Table::row_number(row_index_pattern, pt.begin(), pt.end()); \
        oss << "([" << pid[0];                                          \
        for (size_t i = 1; i < row_rank; ++i)                           \
          oss << "," << pid[i];                                         \
        oss << "]:";                                                    \
      }                                                                 \
      const char* sep = "";                                             \
      for (PointInDomainIterator<DIM> pid(domain, false); pid(); pid++) { \
        for (size_t i = 0; i < DIM; ++i)                                \
          pt[i] = pid[i];                                               \
        auto rn = Table::row_number(row_index_pattern, pt.begin(), pt.end()); \
        if (rn != row_number) {                                         \
          row_number = rn;                                              \
          oss << ")" << std::endl << "([" << pid[0];                    \
          for (size_t i = 1; i < row_rank; ++i)                         \
            oss << "," << pid[i];                                       \
          oss << "]:";                                                  \
          sep = "";                                                     \
        }                                                               \
        oss << sep << values[*pid];                                     \
        sep = ",";                                                      \
      }                                                                 \
      oss << ")";                                                       \
      }                                                                 \
      break;                                                            \
    case casacore::DataType::TpArray##tp: {                             \
      const FieldAccessor<                                              \
        READ_ONLY, \
        DataType<casacore::DataType::TpArray##tp>::ValueType, \
        DIM, \
        coord_t, \
        Realm::AffineAccessor< \
          DataType<casacore::DataType::TpArray##tp>::ValueType,DIM,coord_t>, \
          false> values(pr, fid); \
      std::array<Legion::coord_t, DIM> pt;                              \
      size_t row_number;                                                \
      {                                                                 \
        Legion::PointInDomainIterator<DIM> pid(domain, false);          \
        for (size_t i = 0; i < DIM; ++i)                                \
          pt[i] = pid[i];                                               \
        row_number = Table::row_number(row_index_pattern, pt.begin(), pt.end()); \
        oss << "([" << pid[0];                                          \
        for (size_t i = 1; i < row_rank; ++i)                           \
          oss << "," << pid[i];                                         \
        oss << "]:";                                                    \
      }                                                                 \
      const char* sep = "";                                             \
      for (PointInDomainIterator<DIM> pid(domain, false); pid(); pid++) { \
        for (size_t i = 0; i < DIM; ++i)                                \
          pt[i] = pid[i];                                               \
        auto rn = Table::row_number(row_index_pattern, pt.begin(), pt.end()); \
        if (rn != row_number) {                                         \
          row_number = rn;                                              \
          oss << ")" << std::endl << "([" << pid[0];                    \
          for (size_t i = 1; i < row_rank; ++i)                         \
            oss << "," << pid[i];                                       \
          oss << "]:";                                                  \
          sep = "";                                                     \
        }                                                               \
        oss << sep;                                                     \
        auto vals = values[*pid];                                       \
        if (vals.size() > 0) {                                          \
          oss << "{" << vals[0];                                        \
          for (size_t i = 1; i < vals.size(); ++i)                      \
            oss << "," << vals[i];                                      \
          oss << "}";                                                   \
        }                                                               \
        sep = ",";                                                      \
      }                                                                 \
      oss << ")";                                                       \
    }

    case SHOW(Bool)
      break;
    case SHOW(Char)
      break;
    case SHOW(UChar)
      break;
    case SHOW(Short)
      break;
    case SHOW(UShort)
      break;
    case SHOW(Int)
      break;
    case SHOW(UInt)
      break;
    case SHOW(Float)
      break;
    case SHOW(Double)
      break;
    case SHOW(Complex)
      break;
    case SHOW(DComplex)
      break;
    case SHOW(String)
      break;
    default:
      assert(false);
      break;
    }
    oss << std::endl;
    std::cout << oss.str();
  };
};

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

