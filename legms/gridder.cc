#include "legms.h"

// #if LEGION_MAX_DIM < 9
// # error "MAX_DIM too small"
// #endif

#include <algorithm>
#include <array>
#include <cstdio>
#if GCC_VERSION >= 90000
# include <filesystem>
#else
# include <experimental/filesystem>
#endif
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "legms.h"
#include "legms_hdf5.h"
#include "Table.h"
#include "Column.h"

using namespace legms;
using namespace Legion;

template <typename FT, int N, bool CHECK_BOUNDS=false>
using ROAccessor =
  FieldAccessor<
  READ_ONLY,
  FT,
  N,
  coord_t,
  AffineAccessor<FT, N, coord_t>,
  CHECK_BOUNDS>;

template <typename FT, int N, bool CHECK_BOUNDS=false>
using RWAccessor =
  FieldAccessor<
  READ_WRITE,
  FT,
  N,
  coord_t,
  AffineAccessor<FT, N, coord_t>,
  CHECK_BOUNDS>;

template <typename FT, int N, bool CHECK_BOUNDS=false>
using WOAccessor =
  FieldAccessor<
  WRITE_ONLY,
  FT,
  N,
  coord_t,
  AffineAccessor<FT, N, coord_t>,
  CHECK_BOUNDS>;

template <typename FT, int N, bool CHECK_BOUNDS=false>
using WDAccessor =
  FieldAccessor<
  WRITE_DISCARD,
  FT,
  N,
  coord_t,
  AffineAccessor<FT, N, coord_t>,
  CHECK_BOUNDS>;

enum {
  TOP_LEVEL_TASK_ID,
};

template <MSTables T>
struct TableColumns {
  //typedef ColEnums;
  static const std::array<const char*, 0> column_names;
  static const char* table_name;
};

template <MSTables T>
const std::array<const char*, 0> TableColumns<T>::column_names;

template <MSTables T>
const char* TableColumns<T>::table_name = "";

template <>
struct TableColumns<MS_ANTENNA> {
  enum {
    NAME,
    STATION,
    TYPE,
    MOUNT,
    DISH_DIAMETER,
    NUM_COLUMNS
  };
  static constexpr const std::array<const char*,NUM_COLUMNS> column_names = {
    "NAME",
    "STATION",
    "TYPE",
    "MOUNT",
    "DISH_DIAMETER"
  };
  static constexpr const char* table_name = MSTable<MS_ANTENNA>::name;

  static unsigned
  column_offset(const std::string& name) {
    auto p = std::find(column_names.begin(), column_names.end(), name);
    if (p == column_names.end())
      throw std::domain_error(
        "column '" + name + "' not present in table '" + table_name + "'");
    return std::distance(column_names.begin(), p);
  }
};

Table
init_table(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const LEGMS_FS::path& ms,
  const std::string& root,
  MSTables mst) {

  std::unordered_set<std::string> columns;
  std::string table_name;
  switch (mst) {
#define INIT(TBL)                                       \
    case (MS_##TBL): {                                  \
      std::copy(                                        \
        TableColumns<MS_##TBL>::column_names.begin(),   \
        TableColumns<MS_##TBL>::column_names.end(),     \
        std::inserter(columns, columns.end()));         \
      table_name = TableColumns<MS_##TBL>::table_name;  \
      break;                                            \
    }
    LEGMS_FOREACH_MSTABLE(INIT);
  }
  return legms::hdf5::init_table(ctx, rt, ms, root + table_name, columns);
}

static const FieldID antenna_class_fid = 0;

template <typename T>
class ClassifyAntennasTask {
protected:

  ClassifyAntennasTask(const Table& table)
    : m_table(table) {
  }

public:

  // caller must only free returned LogicalRegion and associated FieldSpace, but
  // not the associated IndexSpace
  Legion::LogicalRegion
  dispatch(Legion::Context ctx, Legion::Runtime* rt) const {

    IndexSpace row_is =
      m_table.min_rank_column(ctx, rt).values_lr.get_index_space();
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    fa.allocate_field(sizeof(unsigned), antenna_class_fid);
    Legion::LogicalRegion result = rt->create_logical_region(ctx, row_is, fs);
    assert(result.get_dim() == 1);

    std::vector<RegionRequirement> requirements;
    {
      RegionRequirement
        cols_req(
          m_table.columns_lr,
          READ_ONLY,
          EXCLUSIVE,
          m_table.columns_lr);
      cols_req.add_field(Table::COLUMNS_FID);
      auto cols = rt->map_region(ctx, cols_req);
      for (auto& cn : TableColumns<MS_ANTENNA>::column_names) {
        auto col = m_table.column(ctx, rt, cols, cn);
        RegionRequirement
          vreq(col.values_lr, READ_ONLY, EXCLUSIVE, col.values_lr);
        vreq.add_field(Column::VALUE_FID);
        requirements.push_back(vreq);      
      }
      rt->unmap_region(ctx, cols);
    }
    {
      RegionRequirement req(result, 0, WRITE_ONLY, EXCLUSIVE, result);
      req.add_field(antenna_class_fid);
      requirements.push_back(req);
    }
    IndexTaskLauncher
      launcher(
        T::TASK_ID,
        result.get_index_space(),
        TaskArgument(NULL, 0),
        ArgumentMap());
    for (auto& req : requirements)
      launcher.add_region_requirement(req);
    rt->execute_index_space(ctx, launcher);
    return result;
  }

  static void
  impl(
    const Point<1>& pt,
    const std::vector<Legion::PhysicalRegion>& regions) {

    ROAccessor<legms::string, 1>
      names(regions[TableColumns<MS_ANTENNA>::NAME], Column::VALUE_FID);
    ROAccessor<legms::string, 1>
      stations(regions[TableColumns<MS_ANTENNA>::STATION], Column::VALUE_FID);
    ROAccessor<legms::string, 1>
      types(regions[TableColumns<MS_ANTENNA>::TYPE], Column::VALUE_FID);
    ROAccessor<legms::string, 1>
      mounts(regions[TableColumns<MS_ANTENNA>::MOUNT], Column::VALUE_FID);
    ROAccessor<double, 1>
      diameters(
        regions[TableColumns<MS_ANTENNA>::DISH_DIAMETER],
        Column::VALUE_FID);
    WOAccessor<unsigned, 1> antenna_classes(regions.back(), 0);
    antenna_classes[pt] =
        T::classify(
          names[pt].val,
          stations[pt].val,
          types[pt].val,
          mounts[pt].val,
          diameters[pt]);
  }

  static void
  base_impl(
    const Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context,
    Legion::Runtime*) {

    impl(task->index_point, regions);
  }

protected:

  Table m_table;
};

class UnitaryClassifyAntennasTask
  : public ClassifyAntennasTask<UnitaryClassifyAntennasTask> {
public:

  static FieldID TASK_ID;
  static const constexpr char* TASK_NAME = "UnitaryClassifyAntennasTask";

  UnitaryClassifyAntennasTask(const Table& table)
    : ClassifyAntennasTask(table) {
  }

  static void
  preregister() {
    TASK_ID = Runtime::generate_static_task_id();
    TaskVariantRegistrar registrar(TASK_ID, TASK_NAME, false);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    registrar.set_idempotent();
    Runtime::preregister_task_variant<base_impl>(registrar, TASK_NAME);
  }

  static unsigned
  classify(
    const char* /*name*/,
    const char* /*station*/,
    const char* /*type*/,
    const char* /*mount*/,
    double /*diameter*/) {
    return 0;
  }
};

FieldID UnitaryClassifyAntennasTask::TASK_ID = 0;

class TopLevelTask {
public:

  static const Legion::TaskID TASK_ID = TOP_LEVEL_TASK_ID;
  static constexpr const char* TASK_NAME = "TopLevelTask";

  static void
  get_args(const Legion::InputArgs& args, LEGMS_FS::path& ms) {
    ms.clear();
    for (int i = 1; i < args.argc; ++i) {
      if (std::string(args.argv[i]) == "--ms")
        ms = args.argv[++i];
    }
  }

  static std::optional<std::string>
  args_error(const LEGMS_FS::path& h5) {
    auto h5_abs = LEGMS_FS::canonical(h5);
    if (LEGMS_FS::exists(h5_abs) && LEGMS_FS::is_regular_file(h5_abs)) {
      return std::nullopt;
    } else {
      std::ostringstream oss;
      oss << "Path '" << h5 << "' does not name a regular file";
      return oss.str();
    }
  }

  static void
  base_impl(
    const Legion::Task*,
    const std::vector<Legion::PhysicalRegion>&,
    Legion::Context ctx,
    Legion::Runtime* rt) {

    const Legion::InputArgs& args = Legion::Runtime::get_input_args();
    LEGMS_FS::path h5;
    get_args(args, h5);
    {
      auto errstr = args_error(h5);
      if (errstr) {
        std::cerr << errstr.value() << std::endl;
        return;
      }
    }
    const std::string ms_root = "/";

    legms::register_tasks(ctx, rt);

    MSTables mstables[] = {MS_ANTENNA};
    std::unordered_map<MSTables,Table> tables;
    for (auto& mst : mstables)
      tables[mst] = init_table(ctx, rt, h5, ms_root, mst);

    LogicalRegion antenna_classes;
    {
      Table& antenna_table = tables[MS_ANTENNA];
      antenna_table.with_columns_attached(
        ctx,
        rt,
        h5,
        ms_root,
        [&antenna_classes, &antenna_table](Context ctx, Runtime* rt) {
          UnitaryClassifyAntennasTask task(antenna_table);
          antenna_classes = task.dispatch(ctx, rt);
        });
    }

    rt->destroy_field_space(ctx, antenna_classes.get_field_space());
    rt->destroy_logical_region(ctx, antenna_classes);
  }

  static void
  preregister() {
    Legion::TaskVariantRegistrar registrar(TASK_ID, TASK_NAME);
    registrar.add_constraint(
      Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    Legion::Runtime::preregister_task_variant<base_impl>(registrar, TASK_NAME);
    Legion::Runtime::set_top_level_task_id(TASK_ID);
  }
};

int
main(int argc, char* argv[]) {
  TopLevelTask::preregister();
  legms::preregister_all();
  UnitaryClassifyAntennasTask::preregister();
  return Runtime::start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
