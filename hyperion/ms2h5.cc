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
  INIT_TABLE_FROM_MS_TASK_ID,
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
    // init_table_from_ms tasks are likely to complete, which ought to improve
    // task overlap when writing multiple tables to a single HDF5 file (better
    // still might be to write each table to its own HDF5 file)
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

const char* init_table_from_ms_task_name = "init_table_from_ms_task";

typedef std::pair<hyperion::string, Table> init_table_from_ms_result_t;

init_table_from_ms_result_t
init_table_from_ms(
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

    // Collect table names, filter them by table_args
    TableNameCollectorTask tnames_launcher(table_args, ms);
    LogicalRegion table_names_lr =
      tnames_launcher.dispatch(ctx, rt).get_result<LogicalRegion>();

    // Initialize Tables from MS (does not initialize the column data) 
    FutureMap f_mstables;
    {
      hyperion::string ms_path(ms);
      IndexTaskLauncher reader(
        INIT_TABLE_FROM_MS_TASK_ID,
        table_names_lr.get_index_space(),
        TaskArgument(&ms_path, sizeof(ms_path)),
        ArgumentMap());
      RegionRequirement
        req(table_names_lr, READ_ONLY, EXCLUSIVE, table_names_lr);
      req.add_field(TableNameCollectorTask::NAME_FID);
      reader.add_region_requirement(req);
      f_mstables = rt->execute_index_space(ctx, reader);
    }

    // Initialize the HDF5 file from Tables
    hid_t fid = CHECK_H5(H5DatatypeManager::create(h5.c_str(), H5F_ACC_EXCL));
    hid_t root = CHECK_H5(H5Gopen(fid, "/", H5P_DEFAULT));
    for (PointInDomainIterator<1> pid(
           rt->get_index_space_domain(table_names_lr.get_index_space()));
         pid();
         pid++) {
      auto [nm, t] = f_mstables.get_result<init_table_from_ms_result_t>(*pid);
      hid_t tid =
        CHECK_H5(
          H5Gcreate(root, nm.val, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
      hdf5::write_table(ctx, rt, tid, t);
      CHECK_H5(H5Gclose(tid));
      t.destroy(ctx, rt); // done with Tables initialized from MS
    }

    // Create new Tables from HDF5. Do this sequentially, since, contrary to
    // Table initialization from MS, no scanning of rows is needed and thus
    // should complete quickly
    std::unordered_map<
      std::string,
      std::tuple<
        Table,
        std::unordered_map<std::string, std::string>>> h5_tables;
    for (PointInDomainIterator<1> pid(
           rt->get_index_space_domain(table_names_lr.get_index_space()));
         pid();
         pid++) {
      auto nm =
        std::get<0>(f_mstables.get_result<init_table_from_ms_result_t>(*pid));
      h5_tables[nm] = hdf5::init_table(ctx, rt, root, nm);
    }
    // Can close the HDF5 file now, the rest is handled by the Legion runtime
    CHECK_H5(H5Gclose(root));
    CHECK_H5(H5Fclose(fid));

    // Copy column values from MS to HDF5 by simply attaching table column
    // regions in h5_tables to the HDF5 datasets, and then initiating
    // TableReadTask tasks to fill the column values
    for (auto& [nm, tb_paths] : h5_tables) {
      auto& [tb, paths] = tb_paths;
      std::vector<PhysicalRegion> h5prs;
      auto tbfields =
        tb.columns(ctx, rt).get_result<Table::columns_result_t>().fields;
      for (auto& [csp, vlr, tflds] : tbfields) {
        std::unordered_set<std::string> cols;
        for (auto& [nm, tfl] : tflds)
          cols.insert(nm);
        h5prs.push_back(
          hdf5::attach_table_columns(
            ctx,
            rt,
            h5,
            "/",
            tb,
            cols,
            paths,
            false,
            false)
          .value());
        auto lr = h5prs.back().get_logical_region();
        AcquireLauncher acquire(lr, lr, h5prs.back());
        rt->get_field_space_fields(lr.get_field_space(), acquire.fields);
        rt->issue_acquire(ctx, acquire);
      }

      TableReadTask copy_task(
        ((nm != "MAIN") ? (ms / nm) : ms),
        tb,
        100000);
      copy_task.dispatch(ctx, rt);

      for (auto& pr : h5prs) {
        auto lr = pr.get_logical_region();
        ReleaseLauncher release(lr, lr, pr);
        rt->get_field_space_fields(lr.get_field_space(), release.fields);
        rt->issue_release(ctx, release);
        rt->detach_external_resource(ctx, pr);
      }
      tb.destroy(ctx, rt); // done with Tables initialized from HDF5
    }

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
    // init_table_from_ms_task
    TaskVariantRegistrar
      registrar(INIT_TABLE_FROM_MS_TASK_ID, init_table_from_ms_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::IO_PROC));
    Runtime::preregister_task_variant<
      init_table_from_ms_result_t,
      init_table_from_ms>(
      registrar,
      init_table_from_ms_task_name);
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
