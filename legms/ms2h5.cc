#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <experimental/filesystem>
#include <unordered_set>

#include "legms.h"
#include "legms_hdf5.h"
#include "Table.h"
#include "TableReadTask.h"

namespace fs = std::experimental::filesystem;

using namespace legms;
using namespace Legion;

enum {
  TOP_LEVEL_TASK_ID,
  TABLE_NAME_COLLECTOR_TASK_ID,  
};

void
get_args(
  const int& argc,
  char** const& argv,
  std::optional<fs::path>& dir,
  std::vector<std::string>& tables) {

  dir.reset();
  tables.clear();
  for (int i = 1; i < argc; ++i) {
    if (*argv[i] != '-') {
      if (!dir) {
        std::string d = argv[i];
        while (d.size() > 0 && d.back() == '/')
          d.pop_back() ;
        dir = d;
      } else {
        tables.push_back(argv[i]);
      }
    } else {
      ++i; // skip option argument
    }
  }
}

fs::path
h5_path(const fs::path& ms) {
  auto result = ms;
  result += ".h5";
  return result;
}

class TableNameCollectorTask {
public:

  // TODO: convert this to an inline task?
  static constexpr const char *TASK_NAME = "TableNameCollector";
  static const int TASK_ID = TABLE_NAME_COLLECTOR_TASK_ID;
  static const int MAX_TABLES = 20;
  static const int NAME_FID = 0;

  TableNameCollectorTask(
    const fs::path& ms,
    LogicalRegion table_names_lr)
    : m_ms(ms)
    , m_table_names_lr(table_names_lr) {}

  void
  dispatch(Context context, Runtime* runtime) {
    TaskArgs args{m_ms};
    auto args_buffer = args.serialize();
    TaskLauncher
      launcher(
        TASK_ID,
        TaskArgument(args_buffer.get(), args.serialized_size()));
    launcher.add_region_requirement(
      RegionRequirement(
        m_table_names_lr,
        WRITE_ONLY,
        EXCLUSIVE,
        m_table_names_lr));
    launcher.add_field(0, NAME_FID);
    runtime->execute_task(context, launcher);
  }

  static LogicalRegion
  table_names_region(Context context, Runtime* runtime) {
    IndexSpace is =
      runtime->create_index_space(context, Rect<1>(0, MAX_TABLES - 1));
    FieldSpace fs = runtime->create_field_space(context);
    FieldAllocator fa = runtime->create_field_allocator(context, fs);
    fa.allocate_field(
      sizeof(DataType<legms::TypeTag::TpString>::ValueType),
      NAME_FID);
    LogicalRegion result = runtime->create_logical_region(context, is, fs);
    runtime->destroy_field_space(context, fs);
    runtime->destroy_index_space(context, is);
    return result;
  }

  static void
  base_impl(
    const Task* task,
    const std::vector<PhysicalRegion>& regions,
    Context,
    Runtime*) {

    TaskArgs args = TaskArgs::deserialize(task->args);
    const FieldAccessor<
      WRITE_ONLY,
      legms::string,
      1,
      coord_t,
      AffineAccessor<legms::string, 1, coord_t>,
      false> names(regions[0], NAME_FID);
    coord_t i = 0;
    if (casacore::Table::isReadable(casacore::String(args.ms))) {
      casacore::Table tb(
        casacore::String(args.ms),
        casacore::TableLock::PermanentLockingWait);
      if (tb.nrow() > 0)
        names[i++] = legms::string("MAIN");
    }
    for (auto& p : fs::directory_iterator(args.ms)) {
      if (i < MAX_TABLES
          && casacore::Table::isReadable(casacore::String(p.path()))) {
        casacore::Table tb(
          casacore::String(p.path()),
          casacore::TableLock::PermanentLockingWait);
        if (tb.nrow() > 0)
          names[i++] = legms::string(p.path().filename().c_str());
      }
    }
    while (i < MAX_TABLES)
      names[i++] = legms::string();
  }

  static void
  register_task() {
    TaskVariantRegistrar registrar(TASK_ID, TASK_NAME);
    registrar.add_constraint(ProcessorConstraint(Processor::IO_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<base_impl>(registrar, TASK_NAME);
  }

private:

  struct TaskArgs {
    fs::path ms;

    size_t
    serialized_size() {
      return string_serdez<std::string>::serialized_size(ms);
    }

    std::unique_ptr<char[]>
    serialize() {
      auto result = std::make_unique<char[]>(serialized_size());
      string_serdez<std::string>::serialize(ms, result.get());
      return result;
    }

    static TaskArgs
    deserialize(void* buffer) {
      std::string s;
      string_serdez<std::string>::deserialize(s, buffer);
      return TaskArgs{s};
    }
  };

  fs::path m_ms;

  LogicalRegion m_table_names_lr;
};

class TopLevelTask {
public:

  static constexpr const char *TASK_NAME = "ms2h5";
  static const int TASK_ID = TOP_LEVEL_TASK_ID;

  static void
  base_impl(
    const Task*,
    const std::vector<PhysicalRegion>&,
    Context context,
    Runtime* runtime) {

    legms::register_tasks(runtime);

    const InputArgs& args = Runtime::get_input_args();
    std::optional<fs::path> ms;
    std::vector<std::string> table_args;
    get_args(args.argc, args.argv, ms, table_args);

    LogicalRegion table_names_lr =
      TableNameCollectorTask::table_names_region(context, runtime);
    TableNameCollectorTask tnames_launcher(ms.value(), table_names_lr);
    tnames_launcher.dispatch(context, runtime);

    auto table_names =
      selected_tables(table_args, table_names_lr, context, runtime);
    runtime->destroy_logical_region(context, table_names_lr);

    std::unordered_set<std::unique_ptr<const Table>> tables;
    std::transform(
      table_names.begin(),
      table_names.end(),
      std::inserter(tables, tables.end()),
      [&ms, &context, runtime](auto& t) {
        fs::path path;
        if (t != "MAIN")
          path = ms.value() / t;
        else
          path = ms.value();
        auto result = Table::from_ms(context, runtime, path, {"*"});
        TableReadTask
          table_read_task(
            path,
            result.get(),
            result->column_names().begin(),
            result->column_names().end(),
            100000);
        table_read_task.dispatch();
        return result;
      });

    // For now, we write the entire MS into a single HDF5 file.
    //
    // TODO: provide options to write HDF5 files with a different structure, for
    // example, one file per table or column, linked in a master file
    auto h5 = h5_path(ms.value());
    hid_t fid = H5DatatypeManager::create(h5.c_str(), H5F_ACC_EXCL);
    assert(fid >= 0);

    std::for_each(
      tables.begin(),
      tables.end(),
      [&fid, &h5](auto& table) {
        hdf5::write_table(h5, fid, table.get());
      });

    herr_t rc = H5Fclose(fid);
    assert(rc >= 0);
  }

  static void
  register_task() {
    TaskVariantRegistrar registrar(TASK_ID, TASK_NAME);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<base_impl>(registrar, TASK_NAME);
  }

  static std::unordered_set<std::string>
  selected_tables(
    std::vector<std::string>& table_args,
    LogicalRegion table_names_lr,
    Context context,
    Runtime* runtime) {

    bool inc =
      std::any_of(
        table_args.begin(),
        table_args.end(),
        [](auto& s){ return s[0] != '~'; });
    bool exc =
      std::any_of(
        table_args.begin(),
        table_args.end(),
        [](auto& s){ return s[0] == '~'; });
    if (inc && exc) {
      std::cerr << "Mixed table inclusion and exclusion: "
                << "exclusions will be ignored"
                << std::endl;
      table_args.erase(
        std::remove_if(
          table_args.begin(),
          table_args.end(),
          [](auto& s){ return s[0] == '~'; }),
        table_args.end());
    }

    PhysicalRegion table_names_pr =
      runtime->map_region(
        context,
        RegionRequirement(
          table_names_lr,
          {TableNameCollectorTask::NAME_FID},
          {TableNameCollectorTask::NAME_FID},
          READ_ONLY,
          EXCLUSIVE,
          table_names_lr));
    const FieldAccessor<
      READ_ONLY,
      legms::string,
      1,
      coord_t,
      AffineAccessor<legms::string, 1, coord_t>,
      false> table_names(table_names_pr, TableNameCollectorTask::NAME_FID);
    std::unordered_set<std::string> result;
    coord_t i = 0;
    if (table_args.size() == 0) {
      while (table_names[i].val[0] != '\0')
        result.insert(table_names[i++].val);
    } else if (table_args.front()[0] != '~') {
      while (table_names[i].val[0] != '\0') {
        if (std::find(
              table_args.begin(),
              table_args.end(),
              std::string(table_names[i].val)) != table_args.end())
          result.insert(table_names[i].val);
        ++i;
      }
    } else {
      while (table_names[i].val[0] != '\0') {
        if (std::find(
              table_args.begin(),
              table_args.end(),
              std::string("~") + table_names[i].val) == table_args.end())
          result.insert(table_names[i].val);
        ++i;
      }
    }
    runtime->unmap_region(context, table_names_pr);
    return result;
  }
};

void
usage() {
  std::cerr << "usage: ms2h5 [OPTION...] MS [TABLE...]" << std::endl;
}

int
main(int argc, char** argv) {

  std::optional<fs::path> dir;
  std::vector<std::string> tables;
  get_args(argc, argv, dir, tables);

  if (!dir) {
    usage();
    return EXIT_FAILURE;
  }

  auto dir_fs_status = fs::status(dir.value());
  if (!fs::is_directory(dir_fs_status)) {
    std::cerr << "MS directory '" << dir.value()
              << "' does not exist"
              << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<std::string> missing_tables;
  std::for_each(
    tables.begin(),
    tables.end(),
    [&missing_tables, d=dir.value()](std::string t) {
      std::string T;
      std::transform(
        t.begin(),
        t.end(),
        std::back_inserter(T),
        [](unsigned char c) { return std::toupper(c); });
      if (T != "MAIN" && T != "." && T[0] != '~') {
        auto stat = fs::status(d / T);
        if (!fs::is_directory(stat))
          missing_tables.push_back(t);
      }
    });

  if (missing_tables.size() > 0) {
    std::ostringstream oss;
    oss << "Tables missing from MS directory";
    const char* sep = ": ";
    std::for_each(
      missing_tables.begin(),
      missing_tables.end(),
      [&oss, &sep](auto& t) {
        oss << sep << t;
        sep = ", ";
      });
    std::cerr << oss.str() << std::endl;
    return EXIT_FAILURE;
  }

  auto h5 = h5_path(dir.value());
  auto h5_status = fs::status(h5);
  if (fs::exists(h5_status)) {
    std::cerr << "HDF5 file path '" << h5
              << "' exists and will not be overwritten"
              << std::endl;
    return EXIT_FAILURE;
  }

  TopLevelTask::register_task();
  TableNameCollectorTask::register_task();
  Runtime::set_top_level_task_id(TopLevelTask::TASK_ID);
  OpsManager::register_ops();
  H5DatatypeManager::register_datatypes();
  return Runtime::start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
