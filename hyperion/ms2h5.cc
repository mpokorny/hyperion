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
#include <hyperion/PhysicalTable.h>
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
  READ_TABLES_FROM_MS_TASK_ID,
  READ_MS_TABLE_COLUMNS_TASK_ID,
  CREATE_H5_TASK_ID,
};

enum {
  TABLE_NAME_FID,
};

// maximum length of paths (MS and HDF5)
#define MAX_PATHLEN 1024
// minimum number of rows to read per task in parallel
#define MIN_BLOCK_ROWS 100000
// maximum number of tables
#define MAX_TABLES 100

template <PrivilegeMode MODE>
using NameAccessor =
  FieldAccessor<
  MODE,
  hyperion::string,
  1,
  coord_t,
  AffineAccessor<hyperion::string, 1, coord_t>>;

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

struct ReadTableFromMSArgs {
  char ms_path[MAX_PATHLEN];
};

const char* read_tables_from_ms_task_name =
                 "read_tables_from_ms";

typedef std::array<LogicalRegion, MAX_TABLES> read_tables_from_ms_result_t;

read_tables_from_ms_result_t
read_tables_from_ms_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  const ReadTableFromMSArgs* args =
    static_cast<const ReadTableFromMSArgs*>(task->args);

  const NameAccessor<READ_ONLY> names(regions[0], TABLE_NAME_FID);

  CXX_FILESYSTEM_NAMESPACE::path ms_path = std::string(args->ms_path);

  read_tables_from_ms_result_t result;
  size_t i = 0;
  for (PointInDomainIterator<1> pid(
         rt->get_index_space_domain(task->regions[0].region.get_index_space()));
       pid();
       pid++) {
    assert(i < MAX_TABLES);
    CXX_FILESYSTEM_NAMESPACE::path tpath;
    if (names[*pid] != "MAIN")
      tpath = ms_path / std::string(names[*pid]);
    else
      tpath = ms_path;
    auto table =
      Table::create(
        ctx,
        rt,
        std::get<1>(from_ms(ctx, rt, tpath, {"*"})));
    result[i++] = table.fields_lr;
  }
  return result;
}

struct ReadMSTableColumnsTaskArgs {
  char table_path[MAX_PATHLEN];
};

const char* read_ms_table_columns_task_name = "read_ms_table_columns";

void
read_ms_table_columns_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  const ReadMSTableColumnsTaskArgs* args =
    static_cast<const ReadMSTableColumnsTaskArgs*>(task->args);

  auto [table, rit, pit] =
    PhysicalTable::create(
      rt,
      task->regions.begin(),
      task->regions.end(),
      regions.begin(),
      regions.end())
    .value();
  assert(rit == task->regions.end());
  assert(pit == regions.end());

  auto index_column = table.index_column(rt).value();
  assert(index_column->parent().get_index_space().get_dim() == 1);
  auto num_rows =
    rt->get_index_space_domain(index_column->parent().get_index_space())
    .get_volume();
  size_t num_io_processors = task->futures[0].get_result<size_t>();
  size_t block_rows =
    num_rows / min_divisor(num_rows, MIN_BLOCK_ROWS, num_io_processors);
  assert(block_rows >= std::min((size_t)MIN_BLOCK_ROWS, num_rows));

  auto row_part =
    table.partition_rows(ctx, rt, {std::make_optional<size_t>(block_rows)});
  auto [reqs, parts] =
    TableReadTask::requirements(ctx, rt, table, row_part, READ_WRITE);

  TableReadTask::Args tr_args;
  fstrcpy(tr_args.table_path, args->table_path);
  IndexTaskLauncher read(
    TableReadTask::TASK_ID,
    rt->get_index_partition_color_space(parts[0].get_index_partition()),
    TaskArgument(&tr_args, sizeof(tr_args)),
    ArgumentMap(),
    Predicate::TRUE_PRED,
    false,
    table_mapper);
  for (auto& rq : reqs)
    read.add_region_requirement(rq);
  rt->execute_index_space(ctx, read);

  row_part.destroy(ctx, rt);
  for (auto& p : parts)
    rt->destroy_logical_partition(ctx, p);
}

struct CreateH5Args {
  char h5_path[MAX_PATHLEN];
};

const char* create_h5_task_name = "create_h5";

struct create_h5_result_t {
  std::vector<std::unordered_map<std::string, std::string>> maps;

  size_t
  legion_buffer_size() const {
    size_t result = 0;
    for (auto& map : maps) {
      for (auto& [k, v] : map)
        result += k.size() + 1 + v.size() + 1;
      ++result;
    }
    ++result;
    return result;
  }

  size_t
  legion_serialize(void* buffer) const {
    char *buff = static_cast<char*>(buffer);
    for (auto& map : maps) {
      for (auto& [k, v] : map) {
        assert(k.size() > 0 && v.size() > 0);
        std::strcpy(buff, k.c_str());
        buff += k.size() + 1;
        std::strcpy(buff, v.c_str());
        buff += v.size() + 1;
      }
      *buff++ = '\0';
    }
    *buff++ = '\0';
    return buff - static_cast<char*>(buffer);
  }

  size_t
  legion_deserialize(const void* buffer) {
    const char *buff = static_cast<const char*>(buffer);
    while (*buff != '\0') {
      std::unordered_map<std::string, std::string> map;
      while (*buff != '\0') {
        std::string k(buff);
        buff += k.size() + 1;
        std::string v(buff);
        buff += v.size() + 1;
        map[k] = v;
      }
      maps.push_back(std::move(map));
      ++buff;
    }
    ++buff;
    return buff - static_cast<const char*>(buffer);
  }
};

create_h5_result_t
create_h5_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  assert(regions.size() >= 1);

  const CreateH5Args* args = static_cast<const CreateH5Args*>(task->args);

  const NameAccessor<READ_ONLY> names(regions[0], TABLE_NAME_FID);

  std::vector<PhysicalTable> tables;
  auto rq_it = task->regions.begin() + 1;
  auto pr_it = regions.begin() + 1;
  while (rq_it != task->regions.end() && pr_it != regions.end()) {
    auto [pt, rit, pit] =
      PhysicalTable::create(
        rt,
        rq_it,
        task->regions.end(),
        pr_it,
        regions.end())
      .value();
    tables.push_back(std::move(pt));
    rq_it = rit;
    pr_it = pit;
  }
  assert(rq_it == task->regions.end() && pr_it == regions.end());

  // initialize HDF5 file with all tables
  std::vector<std::unordered_map<std::string, std::string>> column_maps;
  hid_t file_id = CHECK_H5(H5DatatypeManager::create(args->h5_path, H5F_ACC_EXCL));
  hid_t root_grp_id = CHECK_H5(H5Gopen(file_id, "/", H5P_DEFAULT));
  for (size_t i = 0; i < tables.size(); ++i) {
    std::unordered_map<std::string, std::string> cmap;
    hid_t table_grp_id =
      CHECK_H5(
        H5Gcreate(
          root_grp_id,
          names[i].val,
          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    hdf5::write_table(rt, table_grp_id, tables[i]);
    CHECK_H5(H5Gclose(table_grp_id));
    auto cols = tables[i].columns();
    std::unordered_set<std::string> cnames;
    for (auto& [cname, pc] : cols)
      cnames.insert(cname);
    std::string tpath = std::string("/") + names[i].val;
    column_maps.push_back(
      hdf5::get_table_column_paths(file_id, tpath, cnames));
  }
  CHECK_H5(H5Gclose(root_grp_id));
  CHECK_H5(H5Fclose(file_id));

  return create_h5_result_t{column_maps};
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

    if (ms.string().size() >= MAX_PATHLEN) {
      runtime->print_once(
        context,
        stderr,
        "MS directory path length exceeds maximum supported value\n");
      usage(context, runtime);
      return false;
    }

    if (h5.string().size() >= MAX_PATHLEN) {
      runtime->print_once(
        context,
        stderr,
        "HDF5 file path length exceeds maximum supported value\n");
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

  static bool
  select(const std::vector<std::string>& selection, const std::string& nm) {

    assert(nm.size() > 0);
    if (selection.size() == 0)
      return true;
    if (selection[0][0] != '~')
      return
        std::find(selection.begin(), selection.end(), nm) != selection.end();
    return
      std::find_if(
        selection.begin(),
        selection.end(),
        [&nm](auto& onm) {
          return onm.substr(1) == nm;
        }) == selection.end();
  }

  static std::vector<std::string>
  collect_table_names(
    const CXX_FILESYSTEM_NAMESPACE::path& ms,
    const std::vector<std::string>& table_args) {

    std::vector<std::string> result;
    if (casacore::Table::isReadable(ms.string())) {
      casacore::Table tb(
        ms.string(),
        casacore::TableLock::PermanentLockingWait);
      std::string nm("MAIN");
      if (tb.nrow() > 0 && select(table_args, nm))
        result.push_back(nm);
    }
    for (auto& p : CXX_FILESYSTEM_NAMESPACE::directory_iterator(ms)) {
      if (casacore::Table::isReadable(p.path().string())) {
        casacore::Table tb(
          p.path().string(),
          casacore::TableLock::PermanentLockingWait);
        hyperion::string nm(p.path().filename().c_str());
        if (tb.nrow() > 0 && select(table_args, nm))
          result.push_back(nm);
      }
    }
    return result;
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

    // Collect table names, filtered by table_args
    auto table_names = collect_table_names(ms, table_args);
    // create LogicalRegion for table names and fields
    LogicalRegion table_info_lr;
    {
      IndexSpace is = rt->create_index_space(ctx, Rect<1>(0, table_names.size() - 1));
      FieldSpace fs = rt->create_field_space(ctx);
      FieldAllocator fa = rt->create_field_allocator(ctx, fs);
      fa.allocate_field(sizeof(hyperion::string), TABLE_NAME_FID);
      table_info_lr = rt->create_logical_region(ctx, is, fs);
    }

    // copy table names to field in table_info_lr
    {
      RegionRequirement req(table_info_lr, WRITE_ONLY, EXCLUSIVE, table_info_lr);
      req.add_field(TABLE_NAME_FID);
      auto pr = rt->map_region(ctx, req);
      const NameAccessor<WRITE_ONLY> names(pr, TABLE_NAME_FID);
      for (size_t i = 0; i < table_names.size(); ++i)
        names[i] = table_names[i];
      rt->unmap_region(ctx, pr);
    }

    // construct Table for each MS table
    std::vector<Table> tables;
    {
      auto ipart =
        partition_over_default_tunable(
          ctx,
          rt,
          table_info_lr.get_index_space(),
          4, // TODO: find a good value
          Mapping::DefaultMapper::DefaultTunables::DEFAULT_TUNABLE_GLOBAL_IOS);
      auto lp = rt->get_logical_partition(ctx, table_info_lr, ipart);
      ReadTableFromMSArgs args;
      fstrcpy(args.ms_path, ms.c_str());
      IndexSpace tables_map_cs = rt->get_index_partition_color_space_name(ctx, ipart);
      IndexTaskLauncher task(
        READ_TABLES_FROM_MS_TASK_ID,
        tables_map_cs,
        TaskArgument(&args, sizeof(args)),
        ArgumentMap());
      {
        RegionRequirement req(lp, 0, READ_ONLY, EXCLUSIVE, table_info_lr);
        req.add_field(TABLE_NAME_FID);
        task.add_region_requirement(req);
      }
      auto tables_map = rt->execute_index_space(ctx, task);

      rt->destroy_logical_partition(ctx, lp);
      rt->destroy_index_partition(ctx, ipart);

      // We're going to need privileges on the Tables in create_h5_task, so wait
      // on tables_map and construct Tables from results
      for (PointInRectIterator<1> pir(
             rt->get_index_space_domain(tables_map_cs));
           pir();
           pir++) {
        auto ary = tables_map.get_result<read_tables_from_ms_result_t>(*pir);
        auto lrp = ary.begin();
        while (lrp != ary.end() && *lrp != LogicalRegion::NO_REGION)
          tables.emplace_back(*lrp++);
      }
      rt->destroy_index_space(ctx, tables_map_cs);
    }

    // Create HDF5 file.
    std::vector<std::unordered_map<std::string, std::string>> column_paths;
    {
      CreateH5Args create_args;
      fstrcpy(create_args.h5_path, h5.c_str());
      TaskLauncher write(
        CREATE_H5_TASK_ID,
        TaskArgument(&create_args, sizeof(create_args)));
      {
        RegionRequirement
          req(table_info_lr, READ_ONLY, EXCLUSIVE, table_info_lr);
        req.add_field(TABLE_NAME_FID);
        write.add_region_requirement(req);
      }
      // use WRITE_ONLY privileges on column values -- even though no values are
      // written, any read privilege triggers a Legion warning or error
      Column::Requirements colreq = Column::default_requirements;
      colreq.values = Column::Req{WRITE_ONLY, EXCLUSIVE, false};
      for (auto& t : tables) {
        auto reqs =
          std::get<0>(
            t.requirements(
              ctx,
              rt,
              ColumnSpacePartition(),
              READ_ONLY,
              {},
              colreq));
        for (auto& rq : reqs)
          write.add_region_requirement(rq);
      }
      column_paths =
        rt->execute_task(ctx, write).get_result<create_h5_result_t>().maps;
    }

    // Attach HDF5 columns
    std::vector<PhysicalTable> ptables;
    for (size_t i = 0; i < tables.size(); ++i) {
      std::unordered_map<std::string, std::tuple<bool, bool, bool>>
        column_modes;
      // modes are: read-write, restricted, unmapped
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
      for (auto& [nm, pth] : column_paths[i])
#pragma GCC diagnostic pop
        column_modes[nm] = {false, true, false};
      ptables.push_back(
        tables[i]
        .attach_columns(ctx, rt, READ_ONLY, h5, column_paths[i], column_modes));
    }

    // use read_ms_table_columns_task to read values from MS into regions of the
    // PhysicalTables
    auto nios =
      rt->select_tunable_value(
        ctx,
        Mapping::DefaultMapper::DefaultTunables::DEFAULT_TUNABLE_GLOBAL_IOS);
    Column::Requirements colreq = Column::default_requirements;
    colreq.values = Column::Req{READ_WRITE, EXCLUSIVE, false};
    for (size_t i = 0; i < ptables.size(); ++i) {
      ReadMSTableColumnsTaskArgs rd_args;
      std::string tname(table_names[i]);
      std::string tpath = ms;
      if (tname != "MAIN")
        tpath += std::string("/") + tname;
      fstrcpy(rd_args.table_path, tpath);
      TaskLauncher task(
        READ_MS_TABLE_COLUMNS_TASK_ID,
        TaskArgument(&rd_args, sizeof(rd_args)));
      auto reqs =
        std::get<0>(
          ptables[i].requirements(
            ctx,
            rt,
            ColumnSpacePartition(),
            READ_ONLY,
            {},
            colreq));
      ptables[i].unmap_regions(ctx, rt);
      for (auto& rq : reqs)
        task.add_region_requirement(rq);
      task.add_future(nios);
      rt->execute_task(ctx, task);
    }

    for (auto& t : tables)
      t.destroy(ctx, rt);

    auto fs = table_info_lr.get_field_space();
    auto is = table_info_lr.get_index_space();
    rt->destroy_logical_region(ctx, table_info_lr);
    rt->destroy_field_space(ctx, fs);
    rt->destroy_index_space(ctx, is);
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

  hyperion::preregister_all();
  TopLevelTask::register_task();
  {
    // read_tables_from_ms_task
    TaskVariantRegistrar registrar(
      READ_TABLES_FROM_MS_TASK_ID,
      read_tables_from_ms_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::IO_PROC));
    Runtime::preregister_task_variant<
      read_tables_from_ms_result_t,
      read_tables_from_ms_task>(
        registrar,
        read_tables_from_ms_task_name);
  }
  {
    // read_ms_table_columns_task
    TaskVariantRegistrar
      registrar(READ_MS_TABLE_COLUMNS_TASK_ID, read_ms_table_columns_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<read_ms_table_columns_task>(
      registrar,
      read_ms_table_columns_task_name);
  }
  {
    // create_h5_task
    TaskVariantRegistrar registrar(CREATE_H5_TASK_ID, create_h5_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<create_h5_result_t, create_h5_task>(
      registrar,
      create_h5_task_name);
  }
  Runtime::set_top_level_task_id(TopLevelTask::TASK_ID);
  return Runtime::start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
