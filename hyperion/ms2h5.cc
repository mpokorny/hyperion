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
  READ_TABLE_FROM_MS_TASK_ID,
  READ_MS_TABLE_COLUMNS_TASK_ID,
  CREATE_H5_TASK_ID,
};

enum {
  TABLE_NAME_FID,
};

// maximum length of paths (MS and HDF5)
#define MAX_PATHLEN 1024
// minimum number of rows to read per task in parallel
#define ROW_BLOCK_SZ 100000
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

const char* read_table_from_ms_task_name = "read_table_from_ms";

Table
read_table_from_ms_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  const ReadTableFromMSArgs* args =
    static_cast<const ReadTableFromMSArgs*>(task->args);

  const NameAccessor<READ_ONLY> names(regions[0], TABLE_NAME_FID);
  std::string tname = names[task->index_point];
  CXX_FILESYSTEM_NAMESPACE::path ms_path = std::string(args->ms_path);
  CXX_FILESYSTEM_NAMESPACE::path tpath;
  if (tname != "MAIN")
    tpath = ms_path / tname;
  else
    tpath = ms_path;
  auto nm_ics_flds = from_ms(ctx, rt, tpath, {"*"});
  return
    Table::create(
      ctx,
      rt,
      std::get<1>(nm_ics_flds),
      std::move(std::get<2>(nm_ics_flds)));
}

struct ReadMSTableColumnsTaskArgs {
  char table_path[MAX_PATHLEN];
  Table::Desc desc;
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

  auto ptcr =
    PhysicalTable::create(
      rt,
      args->desc,
      task->regions.begin(),
      task->regions.end(),
      regions.begin(),
      regions.end())
    .value();
#if HAVE_CXX17
  auto& [table, rit, pit] = ptcr;
#else // !HAVE_CXX17
  auto& table = std::get<0>(ptcr);
  auto& rit = std::get<1>(ptcr);
  auto& pit = std::get<2>(ptcr);
#endif // HAVE_CXX17
  assert(rit == task->regions.end());
  assert(pit == regions.end());

  auto row_part = table.partition_rows(ctx, rt, {ROW_BLOCK_SZ});
  auto reqs =
    TableReadTask::requirements(ctx, rt, table, row_part, READ_WRITE);
#if HAVE_CXX17
  auto& [treqs, tparts, tdesc] = reqs;
#else // !HAVE_CXX17
  auto& treqs = std::get<0>(reqs);
  auto& tparts = std::get<1>(reqs);
  auto& tdesc = std::get<2>(reqs);
#endif // HAVE_CXX17

  TableReadTask::Args tr_args;
  fstrcpy(tr_args.table_path, args->table_path);
  tr_args.table_desc = tdesc;
  IndexTaskLauncher read(
    TableReadTask::TASK_ID,
    rt->get_index_partition_color_space(row_part.column_ip),
    TaskArgument(&tr_args, sizeof(tr_args)),
    ArgumentMap(),
    Predicate::TRUE_PRED,
    false,
    table_mapper);
  for (auto& rq : treqs)
    read.add_region_requirement(rq);
  table.unmap_regions(ctx, rt);
  rt->execute_index_space(ctx, read);

  row_part.destroy(ctx, rt);
  for (auto& p : tparts)
    p.destroy(ctx, rt);
}

struct CreateH5Args {
  char h5_path[MAX_PATHLEN];
  unsigned n_tables;
  Table::DescM<50> desc;
};

const char* create_h5_task_name = "create_h5";

struct create_h5_result_t {
  std::vector<std::unordered_map<std::string, std::string>> maps;

  size_t
  legion_buffer_size() const {
    size_t result = 0;
    for (auto& map : maps) {
      for (auto& k_v : map)
        result += std::get<0>(k_v).size() + 1 + std::get<1>(k_v).size() + 1;
      ++result;
    }
    ++result;
    return result;
  }

  size_t
  legion_serialize(void* buffer) const {
    char *buff = static_cast<char*>(buffer);
    for (auto& map : maps) {
      for (auto& k_v : map) {
#if HAVE_CXX17
        auto& [k, v] = k_v;
#else // !HAVE_CXX17
        auto& k = std::get<0>(k_v);
        auto& v = std::get<0>(k_v);
#endif // HAVE_CXX17
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
  std::vector<Table::Desc> desc;
  desc.reserve(args->n_tables);
  for (size_t i = 0; i < args->n_tables; ++i)
    desc.push_back(args->desc[i]);
  const NameAccessor<READ_ONLY> names(regions[0], TABLE_NAME_FID);

  auto ptcr =
    PhysicalTable::create_many(
      rt,
      desc,
      task->regions.begin() + 1,
      task->regions.end(),
      regions.begin() + 1,
      regions.end())
    .value();
#if HAVE_CXX17
  auto& [tables, rit, pit] = ptcr;
#else // !HAVE_CXX17
  auto& tables = std::get<0>(ptcr);
  auto& rit = std::get<1>(ptcr);
  auto& pit = std::get<2>(ptcr);
#endif // HAVE_CXX17
  assert(rit == task->regions.end() && pit == regions.end());

  // initialize HDF5 file with all tables
  std::vector<std::unordered_map<std::string, std::string>> column_maps;
  hid_t file_id =
    CHECK_H5(H5DatatypeManager::create(args->h5_path, H5F_ACC_EXCL));
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
    for (auto& cname_pc : cols)
      cnames.insert(std::get<0>(cname_pc));
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
      IndexSpace is =
        rt->create_index_space(ctx, Rect<1>(0, table_names.size() - 1));
      FieldSpace fs = rt->create_field_space(ctx);
      FieldAllocator fa = rt->create_field_allocator(ctx, fs);
      fa.allocate_field(sizeof(hyperion::string), TABLE_NAME_FID);
      table_info_lr = rt->create_logical_region(ctx, is, fs);
    }

    // copy table names to field in table_info_lr
    {
      RegionRequirement
        req(table_info_lr, WRITE_ONLY, EXCLUSIVE, table_info_lr);
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
      auto table_info_is = table_info_lr.get_index_space();
      auto ipart =
        rt->create_equal_partition(ctx, table_info_is, table_info_is);
      auto lp = rt->get_logical_partition(ctx, table_info_lr, ipart);
      ReadTableFromMSArgs args;
      fstrcpy(args.ms_path, ms.c_str());
      IndexTaskLauncher task(
        READ_TABLE_FROM_MS_TASK_ID,
        table_info_is,
        TaskArgument(&args, sizeof(args)),
        ArgumentMap());
      {
        RegionRequirement req(lp, 0, READ_ONLY, EXCLUSIVE, table_info_lr);
        req.add_field(TABLE_NAME_FID);
        task.add_region_requirement(req);
      }
      auto tables_map = rt->execute_index_space(ctx, task);

      rt->destroy_index_partition(ctx, ipart);

      // We're going to need privileges on the Tables in create_h5_task, so wait
      // on tables_map and construct Tables from results
      for (PointInRectIterator<1> pir(
             rt->get_index_space_domain(table_info_is));
           pir();
           pir++)
        tables.emplace_back(tables_map.get_result<Table>(*pir));
    }

    // Create HDF5 file.
    std::vector<std::unordered_map<std::string, std::string>> column_paths;
    {
      CreateH5Args create_args;
      fstrcpy(create_args.h5_path, h5.c_str());
      create_args.n_tables = tables.size();
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
      size_t i = 0;
      for (auto& t : tables) {
        auto reqs = t.requirements(ctx, rt, ColumnSpacePartition(), {}, colreq);
#if HAVE_CXX17
        auto& [treqs, tparts, tdesc] = reqs;
#else // !HAVE_CXX17
        auto& treqs = std::get<0>(reqs);
        auto& tdesc = std::get<2>(reqs);
#endif // HAVE_CXX17
        create_args.desc[i++] = tdesc;
        for (auto& rq : treqs)
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
      for (auto& nm_pth : column_paths[i])
        column_modes[std::get<0>(nm_pth)] = {false, true, false};
      ptables.push_back(
        tables[i].attach_columns(ctx, rt, h5, column_paths[i], column_modes));
    }

    // use read_ms_table_columns_task to read values from MS into regions of the
    // PhysicalTables
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
        ptables[i].requirements(ctx, rt, ColumnSpacePartition(), {}, colreq);
#if HAVE_CXX17
      auto& [treqs, tparts, tdesc] = reqs;
#else // !HAVE_CXX17
      auto& treqs = std::get<0>(reqs);
      auto& tdesc = std::get<2>(reqs);
#endif // HAVE_CXX17
      ptables[i].unmap_regions(ctx, rt);
      for (auto& rq : treqs)
        task.add_region_requirement(rq);
      rd_args.desc = tdesc;
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
    // read_table_from_ms_task
    TaskVariantRegistrar registrar(
      READ_TABLE_FROM_MS_TASK_ID,
      read_table_from_ms_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::IO_PROC));
    Runtime::preregister_task_variant<Table, read_table_from_ms_task>(
        registrar,
        read_table_from_ms_task_name);
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
