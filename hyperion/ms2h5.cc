/*
 * Copyright 2019 Associated Universities, Inc. Washington DC, USA.
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
#include <hyperion/hdf5.h>
#include <hyperion/Table.h>
#include <hyperion/TableReadTask.h>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include CXX_FILESYSTEM_HEADER
#include <unordered_set>

using namespace hyperion;
using namespace Legion;

enum {
  TOP_LEVEL_TASK_ID,
  TABLE_NAME_COLLECTOR_TASK_ID,
  READ_MS_TABLE_TASK_ID,
};

void
get_args(
  const InputArgs& args,
  CXX_FILESYSTEM_NAMESPACE::path& ms_path,
  std::vector<std::string>& tables,
  CXX_FILESYSTEM_NAMESPACE::path& h5_path) {

  ms_path.clear();
  tables.clear();
  h5_path.clear();
  for (int i = 1; i < args.argc; ++i) {
    if (*args.argv[i] != '-') {
      if (ms_path.empty()) {
        std::string d = args.argv[i];
        while (d.size() > 0 && d.back() == '/')
          d.pop_back() ;
        ms_path = d;
      } else if (i == args.argc - 1) {
        h5_path = args.argv[i];
      } else {
        tables.push_back(args.argv[i]);
      }
    } else if (i < args.argc - 1 && *args.argv[i + 1] != '-') {
      ++i; // skip option argument
    }
  }

  bool inc =
    std::any_of(
      tables.begin(),
      tables.end(),
      [](auto& s){ return s[0] != '~'; });
  bool exc =
    std::any_of(
      tables.begin(),
      tables.end(),
      [](auto& s){ return s[0] == '~'; });
  if (inc && exc) {
    std::cerr << "Mixed table inclusion and exclusion: "
              << "exclusions will be ignored"
              << std::endl;
    tables.erase(
      std::remove_if(
        tables.begin(),
        tables.end(),
        [](auto& s){ return s[0] == '~'; }),
      tables.end());
  }
}

class TableNameCollectorTask {
public:

  // TODO: convert this to an inline task?
  static constexpr const char *TASK_NAME = "TableNameCollector";
  static const int TASK_ID = TABLE_NAME_COLLECTOR_TASK_ID;
  static const int MAX_TABLES = 20;
  static const int NAME_FID = 0;

  TableNameCollectorTask(
    const std::vector<std::string>& table_selection,
    const CXX_FILESYSTEM_NAMESPACE::path& ms) {

    m_args.ms = ms;
    size_t i = 0;
    while (i < table_selection.size()) {
      m_args.selection[i] = table_selection[i];
      ++i;
    }
    while (i < m_args.selection.size())
      m_args.selection[i++] = hyperion::string();
  }

  Future
  dispatch(Context context, Runtime* runtime) {
    TaskLauncher launcher(TASK_ID, TaskArgument(&m_args, sizeof(m_args)));
    return runtime->execute_task(context, launcher);
  }

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using NameAccessor =
    const FieldAccessor<
      MODE,
      hyperion::string,
      1,
      coord_t,
      AffineAccessor<hyperion::string, 1, coord_t>,
      CHECK_BOUNDS>;

  static LogicalRegion
  base_impl(
    const Task* task,
    const std::vector<PhysicalRegion>&,
    Context ctx,
    Runtime* rt) {

    const TaskArgs* args = static_cast<const TaskArgs*>(task->args);
    // collect table names, ordered by number of rows...this, to allow the
    // hdf5::write_table() calls to be initiated in the order in which the
    // read_ms_table tasks are likely to complete, which ought to improve task
    // overlap when writing multiple tables to a single HDF5 file (better still
    // might be to write each table to its own HDF5 file)
    std::set<std::tuple<unsigned, hyperion::string>> names;
    if (casacore::Table::isReadable(casacore::String(args->ms))) {
      casacore::Table tb(
        casacore::String(args->ms),
        casacore::TableLock::PermanentLockingWait);
      hyperion::string nm("MAIN");
      auto nrow = tb.nrow();
      if (nrow > 0 && select(args->selection, nm))
        names.emplace(nrow, nm);
    }
    for (auto& p :
           CXX_FILESYSTEM_NAMESPACE::directory_iterator(
             CXX_FILESYSTEM_NAMESPACE::path(args->ms))) {
      if (casacore::Table::isReadable(casacore::String(p.path()))) {
        casacore::Table tb(
          casacore::String(p.path()),
          casacore::TableLock::PermanentLockingWait);
        hyperion::string nm(p.path().filename().c_str());
        auto nrow = tb.nrow();
        if (nrow > 0 && select(args->selection, nm)) 
          names.emplace(nrow, nm);
      }
    }

    IndexSpace is = rt->create_index_space(ctx, Rect<1>(0, names.size() - 1));
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    fa.allocate_field(sizeof(hyperion::string), NAME_FID);
    LogicalRegion result = rt->create_logical_region(ctx, is, fs);
    RegionRequirement req(result, WRITE_ONLY, EXCLUSIVE, result);
    req.add_field(NAME_FID);
    auto pr = rt->map_region(ctx, req);
    const NameAccessor<WRITE_ONLY> nms(pr, NAME_FID);
    size_t i = 0;
    for (auto& [nr, nm] : names)
      nms[i++] = nm;
    rt->unmap_region(ctx, pr);
    return result;
  }

  static void
  register_task() {
    TaskVariantRegistrar registrar(TASK_ID, TASK_NAME);
    registrar.add_constraint(ProcessorConstraint(Processor::IO_PROC));
    Runtime::preregister_task_variant<LogicalRegion, base_impl>(
      registrar,
      TASK_NAME);
  }

private:

  struct TaskArgs {
    hyperion::string ms;

    std::array<hyperion::string, MAX_TABLES> selection;
  };

  TaskArgs m_args;

  LogicalRegion m_table_names_lr;

  static bool
  select(
    const decltype(TaskArgs::selection)& selection,
    const hyperion::string& nm) {

    assert(nm.size() > 0);
    if (selection[0].size() == 0)
      return true;
    if (selection.front().val[0] != '~')
      return
        std::find(selection.begin(), selection.end(), nm) != selection.end();
    return
      std::find_if(
        selection.begin(),
        selection.end(),
        [&nm](auto& onm) {
          return strcmp(&onm.val[1], nm.val) == 0;
        }) == selection.end();
  }
};

const char* read_ms_table_task_name = "read_ms_table_task";

typedef std::pair<hyperion::string, Table> read_ms_table_result_t;

read_ms_table_result_t
read_ms_table(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  CXX_FILESYSTEM_NAMESPACE::path ms_path =
    std::string(*static_cast<hyperion::string*>(task->args));

  const TableNameCollectorTask::NameAccessor<WRITE_ONLY>
    names(regions[0], TableNameCollectorTask::NAME_FID);

  CXX_FILESYSTEM_NAMESPACE::path tpath;
  if (names[task->index_point] != "MAIN")
    tpath = ms_path / std::string(names[task->index_point]);
  else
    tpath = ms_path;
  return from_ms(ctx, rt, tpath, {"*"});
}

class TopLevelTask {
public:

  static constexpr const char *TASK_NAME = "ms2h5";
  static const int TASK_ID = TOP_LEVEL_TASK_ID;

  static void
  usage(Context context, Runtime* runtime) {
    runtime->print_once(
      context,
      stderr,
      "usage: ms2h5 [OPTION...] MS [TABLE...] OUTPUT\n");
  }

  static bool
  args_ok(const CXX_FILESYSTEM_NAMESPACE::path& ms,
          const std::vector<std::string>& table_args,
          const CXX_FILESYSTEM_NAMESPACE::path& h5,
          Context context,
          Runtime* runtime) {

    if (ms.empty()) {
      runtime->print_once(
        context,
        stderr,
        "MS directory path is missing from arguments\n");
      usage(context, runtime);
      return false;
    }

    if (h5.empty()) {
      runtime->print_once(
        context,
        stderr,
        "Output HDF5 path is missing from arguments\n");
      usage(context, runtime);
      return false;
    }

    auto ms_status = CXX_FILESYSTEM_NAMESPACE::status(ms);
    if (!CXX_FILESYSTEM_NAMESPACE::is_directory(ms)) {
      std::ostringstream oss;
      oss << "MS directory " << ms
          << " does not exist"
          << std::endl;
      runtime->print_once(context, stderr, oss.str().c_str());
      return false;
    }

    if (CXX_FILESYSTEM_NAMESPACE::exists(h5)) {
      std::ostringstream oss;
      oss << "HDF5 file path " << h5
          << " exists and will not be overwritten"
          << std::endl;
      runtime->print_once(context, stderr, oss.str().c_str());
      return false;
    }

    std::vector<std::string> missing_tables;
    std::for_each(
      table_args.begin(),
      table_args.end(),
      [&missing_tables, &ms](std::string t) {
        std::string T;
        std::transform(
          t.begin(),
          t.end(),
          std::back_inserter(T),
          [](unsigned char c) { return std::toupper(c); });
        if (T != "MAIN" && T != "." && T[0] != '~') {
          auto stat = CXX_FILESYSTEM_NAMESPACE::status(ms / T);
          if (!CXX_FILESYSTEM_NAMESPACE::is_directory(stat))
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
      runtime->print_once(context, stderr, oss.str().c_str());
      return false;
    }
    return true;
  }

  static void
  base_impl(
    const Task*,
    const std::vector<PhysicalRegion>&,
    Context ctx,
    Runtime* rt) {

    hyperion::register_tasks(ctx, rt);

    const InputArgs& args = Runtime::get_input_args();
    CXX_FILESYSTEM_NAMESPACE::path ms;
    std::vector<std::string> table_args;
    CXX_FILESYSTEM_NAMESPACE::path h5;
    get_args(args, ms, table_args, h5);

    if (!args_ok(ms, table_args, h5, ctx, rt))
      return;

    TableNameCollectorTask tnames_launcher(table_args, ms);
    LogicalRegion table_names_lr =
      tnames_launcher.dispatch(ctx, rt).get_result<LogicalRegion>();

    FutureMap ftables;
    {
      hyperion::string ms_path(ms);
      IndexTaskLauncher reader(
        READ_MS_TABLE_TASK_ID,
        table_names_lr.get_index_space(),
        TaskArgument(&ms_path, sizeof(ms_path)),
        ArgumentMap());
      RegionRequirement
        req(table_names_lr, READ_ONLY, EXCLUSIVE, table_names_lr);
      req.add_field(TableNameCollectorTask::NAME_FID);
      reader.add_region_requirement(req);
      ftables = rt->execute_index_space(ctx, reader);
    }

    // For now, we write the entire MS into a single HDF5 file.

    hid_t fid = H5DatatypeManager::create(h5.c_str(), H5F_ACC_EXCL);
    assert(fid >= 0);

    hid_t root = H5Gopen(fid, "/", H5P_DEFAULT);
    for (PointInDomainIterator<1> pid(
           rt->get_index_space_domain(table_names_lr.get_index_space()));
         pid();
         pid++) {
      auto [nm, t] = ftables.get_result<read_ms_table_result_t>(*pid);
      hid_t tid =
        H5Gcreate(root, nm.val, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      assert(tid >= 0);
      hdf5::write_table(ctx, rt, tid, t);
      herr_t err = H5Gclose(tid);
      assert(err >= 0);
      t.destroy(ctx, rt);
    }

    herr_t err = H5Gclose(root);
    assert(err >= 0);
    err = H5Fclose(fid);
    assert(err >= 0);

    rt->destroy_field_space(ctx, table_names_lr.get_field_space());
    rt->destroy_index_space(ctx, table_names_lr.get_index_space());
    rt->destroy_logical_region(ctx, table_names_lr);
  }

  static void
  register_task() {
    TaskVariantRegistrar registrar(TASK_ID, TASK_NAME);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<base_impl>(registrar, TASK_NAME);
  }
};

int
main(int argc, char** argv) {

  TopLevelTask::register_task();
  TableNameCollectorTask::register_task();
  {
    // read_ms_table_task
    TaskVariantRegistrar
      registrar(READ_MS_TABLE_TASK_ID, read_ms_table_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::IO_PROC));
    Runtime::preregister_task_variant<read_ms_table_result_t, read_ms_table>(
      registrar,
      read_ms_table_task_name);
  }
  Runtime::set_top_level_task_id(TopLevelTask::TASK_ID);
  hyperion::preregister_all();
  return Runtime::start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
