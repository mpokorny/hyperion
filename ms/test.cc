#include <experimental/filesystem>
#include <map>
#include <memory>
#include <vector>

#include "legion.h"

#include "IndexTree.h"
#include "Column.h"
#include "SpectralWindowTable.h"
#include "Table.h"
#include "TableReadTask.h"

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

  static void
  base_impl(
    const Task*,
    const std::vector<PhysicalRegion>&,
    Context ctx,
    Runtime* runtime) {

    auto input_args = Runtime::get_input_args();

    fs::path table_path = fs::path(input_args.argv[1]);
    fs::path ms_path = table_path.parent_path();
    std::string table_name = table_path.filename();

    if (table_name != "SPECTRAL_WINDOW") {
      std::cerr << "Column '" << table_name << "' is unsupported" << std::endl;
      return;
    }

    std::vector<std::string> colnames;
    for (auto i = 2; i < input_args.argc; ++i)
      colnames.push_back(input_args.argv[i]);

    TableReadTask::register_task(runtime);
    TreeIndexSpace::register_tasks(runtime);
    FillProjectionsTasks::register_tasks(runtime);

    SpectralWindowTable spectral_window_table(ms_path);
    std::cout << "name: "
              << spectral_window_table.name() << std::endl;
    std::cout << "columns: ";
    std::for_each (
      colnames.begin(),
      colnames.end(),
      [](auto& nm) { std::cout << nm << " "; });
    std::cout << std::endl;

    TableReadTask spectral_window_read_task(
      spectral_window_table.path(),
      spectral_window_table,
      colnames);
    auto lr_fids = spectral_window_read_task.dispatch(ctx, runtime);
    for (size_t i = 0; i < colnames.size(); ++i) {
      std::cout << colnames[i] << ":" << std::endl;
      auto& [lr, fid] = lr_fids[i];
      auto launcher = InlineLauncher(
        RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
      launcher.add_field(fid);
      PhysicalRegion pr = runtime->map_region(ctx, launcher);
      auto col = spectral_window_table.column(colnames[i]);
      switch (col->rank()) {
      case 1:
        show<1>(
          runtime,
          pr,
          lr,
          fid,
          col,
          spectral_window_table.row_index_shape());
        break;
      case 2:
        show<2>(
          runtime,
          pr,
          lr,
          fid,
          col,
          spectral_window_table.row_index_shape());
        break;
      case 3:
        show<3>(
          runtime,
          pr,
          lr,
          fid,
          col,
          spectral_window_table.row_index_shape());
        break;
      default:
        assert(false);
        break;
      }
      std::cout << std::endl;
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
    const IndexTreeL& row_index_shape) {

    auto row_rank = row_index_shape.rank().value();
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
        row_number = Table::row_number(row_index_shape, pt.begin(), pt.end()); \
        oss << "([" << pid[0];                                          \
        for (size_t i = 1; i < row_rank; ++i)                           \
          oss << "," << pid[i];                                         \
        oss << "]:";                                                    \
      }                                                                 \
      const char* sep = "";                                             \
      for (PointInDomainIterator<DIM> pid(domain, false); pid(); pid++) { \
        for (size_t i = 0; i < DIM; ++i)                                \
          pt[i] = pid[i];                                               \
        auto rn = Table::row_number(row_index_shape, pt.begin(), pt.end()); \
        if (rn != row_number) {                                         \
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
        row_number = Table::row_number(row_index_shape, pt.begin(), pt.end()); \
        oss << "([" << pid[0];                                          \
        for (size_t i = 1; i < row_rank; ++i)                           \
          oss << "," << pid[i];                                         \
        oss << "]:";                                                    \
      }                                                                 \
      const char* sep = "";                                             \
      for (PointInDomainIterator<DIM> pid(domain, false); pid(); pid++) { \
        for (size_t i = 0; i < DIM; ++i)                                \
          pt[i] = pid[i];                                               \
        auto rn = Table::row_number(row_index_shape, pt.begin(), pt.end()); \
        if (rn != row_number) {                                         \
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

  if (argc < 3) {
    usage();
    return 1;
  }
  auto arg1_fs_status = fs::status(argv[1]);
  if (!fs::is_directory(arg1_fs_status)) {
    std::cout << "directory '" << argv[1]
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
// coding: utf-8
// End:
