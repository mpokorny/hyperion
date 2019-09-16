#include "legms.h"

#if LEGION_MAX_DIM < 7
# error "MAX_DIM too small"
#endif

#include <algorithm>
#include <array>
#include <cmath>
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

#define DEFAULT_PA_STEP_STR "360.0"
#define DEFAULT_W_PROJ_PLANES_STR "1"

#define AUTO_W_PROJ_PLANES_VALUE -1
#define INVALID_W_PROJ_PLANES_VALUE -2

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

template <typename T>
struct WithColumnLookup {
  static unsigned
  column_offset(const std::string& name) {
    auto p = std::find(T::column_names.begin(), T::column_names.end(), name);
    if (p == T::column_names.end())
      throw std::domain_error(
        "column '" + name + "' not present in table '" + T::table_name + "'");
    return std::distance(T::column_names.begin(), p);
  }
};

template <>
struct TableColumns<MS_ANTENNA>
  : public WithColumnLookup<TableColumns<MS_ANTENNA>> {
  typedef enum {
    NAME,
    STATION,
    TYPE,
    MOUNT,
    DISH_DIAMETER,
    NUM_COLUMNS
  } col;
  static constexpr const std::array<const char*,NUM_COLUMNS> column_names = {
    "NAME",
    "STATION",
    "TYPE",
    "MOUNT",
    "DISH_DIAMETER"
  };
  static constexpr const char* table_name = MSTable<MS_ANTENNA>::name;
};

template <>
struct TableColumns<MS_DATA_DESCRIPTION>
  : public WithColumnLookup<TableColumns<MS_DATA_DESCRIPTION>> {
  typedef enum {
    SPECTRAL_WINDOW_ID,
    POLARIZATION_ID,
    NUM_COLUMNS
  } col;
  static constexpr const std::array<const char*,NUM_COLUMNS> column_names = {
    "SPECTRAL_WINDOW_ID",
    "POLARIZATION_ID"
  };
  static constexpr const char* table_name = MSTable<MS_DATA_DESCRIPTION>::name;
};

template <>
struct TableColumns<MS_POLARIZATION>
  : public WithColumnLookup<TableColumns<MS_POLARIZATION>> {
  typedef enum {
    NUM_CORR,
    NUM_COLUMNS
  } col;
  static constexpr const std::array<const char*,NUM_COLUMNS> column_names = {
    "NUM_CORR",
  };
  static constexpr const char* table_name = MSTable<MS_POLARIZATION>::name;
};

template <>
struct TableColumns<MS_SPECTRAL_WINDOW>
  : public WithColumnLookup<TableColumns<MS_SPECTRAL_WINDOW>> {
  typedef enum {
    NUM_CHAN,
    REF_FREQUENCY,
    CHAN_FREQ,
    CHAN_WIDTH,
    NUM_COLUMNS
  } col;
  static constexpr const std::array<const char*,NUM_COLUMNS> column_names = {
    "NUM_CHAN",
    "REF_FREQUENCY",
    "CHAN_FREQ",
    "CHAN_WIDTH"
  };
  static constexpr const char* table_name = MSTable<MS_SPECTRAL_WINDOW>::name;
};

#define COLUMN_NAME(T, C) TableColumns<T>::column_names[TableColumns<T>::C]

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
  static const unsigned num_classes = 1;

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

struct PAValues {
  static const FieldID PA_ORIGIN_FID = 0;
  static const FieldID PA_STEP_FID = 1;
  static const FieldID PA_NUM_STEP_FID = 2;

  LogicalRegion parameters;

  PAValues(
    Context ctx,
    Runtime* rt,
    const std::unordered_map<MSTables,Table>&,
    float pa_step,
    float pa_origin = 0.0f) {

    IndexSpace is = rt->create_index_space(ctx, Rect<1>(0, 0));
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    fa.allocate_field(sizeof(float), PA_ORIGIN_FID);
    fa.allocate_field(sizeof(float), PA_STEP_FID);
    fa.allocate_field(sizeof(unsigned long), PA_NUM_STEP_FID);
    parameters = rt->create_logical_region(ctx, is, fs);

    RegionRequirement req(parameters, WRITE_ONLY, EXCLUSIVE, parameters);
    req.add_field(PA_ORIGIN_FID);
    req.add_field(PA_STEP_FID);
    req.add_field(PA_NUM_STEP_FID);
    auto pr = rt->map_region(ctx, req);
    const WOAccessor<float, 1> origin(pr, PA_ORIGIN_FID);
    const WOAccessor<float, 1> step(pr, PA_STEP_FID);
    const WOAccessor<unsigned long, 1> num_step(pr, PA_NUM_STEP_FID);
    origin[0] = pa_origin;
    step[0] = pa_step;
    num_step[0] = std::lrint(std::ceil(360.0f / pa_step));
    rt->unmap_region(ctx, pr);
  }

  unsigned long
  num_steps(Context ctx, Runtime* rt) const {
    RegionRequirement req(parameters, READ_ONLY, EXCLUSIVE, parameters);
    req.add_field(PA_NUM_STEP_FID);
    auto pr = rt->map_region(ctx, req);
    auto result = num_steps(pr);
    rt->unmap_region(ctx, pr);
    return result;
  }

  static unsigned long
  num_steps(PhysicalRegion region) {
    const ROAccessor<unsigned long, 1> num_step(region, PA_NUM_STEP_FID);
    return num_step[0];
  }

  std::optional<std::tuple<float, float>>
  pa(Context ctx, Runtime* rt, unsigned long i) const {
    RegionRequirement req(parameters, READ_ONLY, EXCLUSIVE, parameters);
    req.add_field(PA_ORIGIN_FID);
    req.add_field(PA_STEP_FID);
    req.add_field(PA_NUM_STEP_FID);
    auto pr = rt->map_region(ctx, req);
    auto result = pa(pr, i);
    rt->unmap_region(ctx, pr);
    return result;
  }

  static std::optional<std::tuple<float, float>>
  pa(PhysicalRegion region, unsigned long i) {
    const ROAccessor<float, 1> origin(region, PA_ORIGIN_FID);
    const ROAccessor<float, 1> step(region, PA_STEP_FID);
    const ROAccessor<unsigned long, 1> num_step(region, PA_NUM_STEP_FID);
    if (i >= num_step[0])
      return std::nullopt;
    float lo = i * step[0];
    float width = (i == num_step[0] - 1) ? (360.0f - lo) : step[0];
    return std::make_tuple(origin[0] + lo, width);
  }

  unsigned long
  find(Context ctx, Runtime* rt, float pa) const {
    RegionRequirement req(parameters, READ_ONLY, EXCLUSIVE, parameters);
    req.add_field(PA_ORIGIN_FID);
    req.add_field(PA_STEP_FID);
    auto pr = rt->map_region(ctx, req);
    auto result = find(pr, pa);
    rt->unmap_region(ctx, pr);
    return result;
  }

  static unsigned long
  find(PhysicalRegion region, float pa) {
    const ROAccessor<float, 1> origin(region, PA_ORIGIN_FID);
    const ROAccessor<float, 1> step(region, PA_STEP_FID);
    pa -= origin[0];
    pa -= std::floor(pa / 360.0f) * 360.0f;
    return std::lrint(std::floor(pa / step[0]));
  }

  void
  destroy(Context ctx, Runtime* rt) {
    rt->destroy_field_space(ctx, parameters.get_field_space());
    rt->destroy_index_space(ctx, parameters.get_index_space());
    rt->destroy_logical_region(ctx, parameters);
    parameters = LogicalRegion::NO_REGION;
  }
};

class CFMap {
public:

  CFMap(
    Context ctx,
    Runtime* rt,
    const GridderArgs<VALUE_ARGS>& gridder_args,
    const std::string& ms_root,
    const std::unordered_map<MSTables,Table>& tables,
    unsigned num_antenna_classes,
    LogicalRegion antenna_classes,
    const PAValues& pa_values) {

    // compute the map index space bounds
    m_bounds =
      tables[MS_DATA_DESCRIPTION]
      .with_columns_attached(
        ctx,
        rt,
        gridder_args.h5,
        ms_root,
        [&tables, &gridder_args, &pa_values, &ms_root]
        (Context ctx, Runtime* rt, const Table& dd) {
          return 
            tables[MS_SPECTRAL_WINDOW]
            .column(
              ctx,
              rt,
              COLUMN_NAME(MS_SPECTRAL_WINDOW, NUM_CHAN))
            .with_attached(
              ctx,
              rt,
              gridder_args.h5,
              ms_root + TableColumns<MS_SPECTRAL_WINDOW>::table_name,
              [&tables, &gridder_args, &pa_values, &ms_root, &dd]
              (Context ctx, Runtime* rt, const Column& num_chan) {
                return 
                  tables[MS_POLARIZATION]
                  .column(
                    ctx,
                    rt,
                    COLUMN_NAME(MS_POLARIZATION, NUM_CORR))
                  .with_attached(
                    ctx,
                    rt,
                    gridder_args.h5,
                    ms_root + TableColumns<MS_POLARIZATION>::table_name,
                    [&gridder_args, &pa_values, &num_chan, &dd]
                    (Context ctx, Runtime* rt, const Column& num_corr) {
                      return
                        CFMap::bounding_index_space(
                          ctx,
                          rt,
                          dd,
                          num_chan,
                          num_corr,
                          num_antenna_classes,
                          gridder_args.w_proj_planes,
                          pa_values.num_steps(ctx, rt));
                    });
                  });        
        });

    // create and initialize preimage, which records, for every point in
    // m_bounds, a row number that maps to that point
    LogicalRegion preimage;
    {
      FieldSpace fs = rt->create_field_space(ctx);
      FieldAllocator fa = rt->create_field_allocator(ctx, fs);
      fa.allocate_field(sizeof(Point<1>), 0); // row number
      fa.allocate_field(sizeof(Point<1>), 1); // "assigned" flag
      preimage = rt->create_logical_region(ctx, m_bounds, fs);
      rt->fill_field(ctx, preimage, preimage, 0, Point<1>(-1));
      rt->fill_field(ctx, preimage, preimage, 1, Point<1>(0));
      // FIXME: launch index space task over visibilities to write row numbers
      // and flags to preimage via reductions
    }

    // create m_partition, the partition of m_bounds by "assigned" flag of
    // preimage
    {
      IndexSpace flag_values = rt->create_index_space(ctx, Rect<1>(0, 1));
      m_partition =
        rt->create_partition_by_field(
          ctx,
          preimage,
          preimage,
          1,
          flag_values);
      rt->destroy_index_space(ctx, flag_values);
    }

    // create m_cfs, the region of cfs, using as index space the sub-space of
    // m_partition with an "assigned" flag value of 1
    {
      FieldSpace fs = rt->create_field_space(ctx);
      {
        FieldAllocator fa = rt->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(LogicalRegion), 0);
      }
      m_cfs =
        rt->create_logical_region(
          ctx,
          rt->get_index_subspace(ctx, m_partition, Point<1>(1)),
          fs);
    }

    // create common field space of every cf
    m_cf_fs = rt->create_field_space(ctx);
    {
      FieldAllocator fa = rt->create_field_allocator(ctx, m_cf_fs);
      fa.allocate_field(sizeof(std::complex<float>), 0);
    }

    // FIXME: launch index space task on m_cfs to create and initialize its
    // values, using "preimage" to select the row used for initialization

    rt->destroy_field_space(ctx, preimage.get_field_space());
    rt->destroy_logical_region(ctc, preimage);
  }

  struct Key {
    unsigned ant1_class;
    unsigned ant2_class;
    unsigned pa;
    unsigned dd;
    unsigned channel;
    unsigned w_plane;
    unsigned correlation;

    operator Point<7>() const {
      coord_t
        vals[7]{ant1_class, ant2_class, pa, dd, channel, w_plane, correlation};
      return Point<7>(vals);
    }
  };

  LogicalRegion
  lookup(Context ctx, Runtime* rt, const Key& key) {
    RegionRequirement req(m_cfs, READ_ONLY, EXCLUSIVE, m_cfs);
    req.add_field(0);
    auto pr = rt->map_region(ctx, req);
    auto result = lookup(pr, key);
    rt->unmap_region(ctx, pr);
    return result;
  }

  static LogicalRegion
  lookup(PhysicalRegion pr, const Key& key) {
    const ROAccessor<LogicalRegion, 7> lrs(pr, 0);
    return lrs[key];
  }

  void
  destroy(Context ctx, Runtime* rt, bool destroy_cfs = true) {
    if (destroy_cfs) {
      RegionRequirement req(m_cfs, READ_ONLY, EXCLUSIVE, m_cfs);
      req.add_field(0);
      auto pr = rt->map_region(ctx, req);
      const ROAccessor<LogicalRegion, 7> cfs(pr, 0);
      for (PointInDomainIterator<7>
             pid(rt->get_index_space_domain(m_cfs.get_index_space()));
           pid();
           pid++) {
        LogicalRegion cf = cfs[*pid];
        if (cf != LogicalRegion::NO_REGION) {
          rt->destroy_index_space(ctx, cf.get_index_space());
          // DON'T destroy the field space of cf, as cfs share a common field
          // space! TODO: can they share a common index space?
          rt->destroy_logical_region(cf);
        }
      }
      rt->unmap_region(ctx, pr);
      destroy_common_field_space(ctx, rt);
    }
    rt->destroy_field_space(ctx, m_cfs.get_field_space());
    rt->destroy_logical_region(ctx, m_cfs);
    m_cfs = LogicalRegion::NO_REGION;

    rt->destroy_index_space(ctx, m_bounds);
  }

  void
  destroy_common_field_space(Context ctx, Runtime* rt) {
    // call this after destroy() has been called with "destroy_cfs" set to
    // "false"; it is nevertheless safe but unnecessary to call after destroy()
    // was called with a "true" value
    if (m_cf_fs != FieldSpace::NO_SPACE){
      rt->destroy_field_space(ctx, m_cf_fs);
      m_cf_fs = FieldSpace::NO_SPACE;
    }
  }

  static IndexSpace
  bounding_index_space(
    Context ctx,
    Runtime* rt,
    const Table& data_description_tbl,
    const Column& num_chan_col,
    const Column& num_corr_col,
    unsigned num_antenna_classes,
    int w_proj_planes,
    coord_t num_pa_steps) {

    coord_t num_dd;
    PhysicalRegion spw_id_pr;
    {
      const Column& col =
        data_description_tbl.column(
          ctx,
          rt,
          COLUMN_NAME(MS_DATA_DESCRIPTION, SPECTRAL_WINDOW_ID));
      Rect<1> rows(
        rt->get_index_space_domain(col.values_lr.get_index_space()));
      num_dd = rows.hi[0] + 1;
      RegionRequirement req(col.values_lr, READ_ONLY, EXCLUSIVE, col.values_lr);
      req.add_field(Column::VALUE_FID);
      spw_id_pr = rt->map_region(ctx, req);
    }
    const ROAccessor<int, 1> spw_ids(spw_id_pr, Column::VALUE_FID);    

    PhysicalRegion pol_id_pr;
    {
      const Column& col =
        data_description_tbl.column(
          ctx,
          rt,
          COLUMN_NAME(MS_DATA_DESCRIPTION, POLARIZATION_ID));
      RegionRequirement req(col.values_lr, READ_ONLY, EXCLUSIVE, col.values_lr);
      req.add_field(Column::VALUE_FID);
      pol_id_pr = rt->map_region(ctx, req);
    }
    const ROAccessor<int, 1> pol_ids(pol_id_pr, Column::VALUE_FID);

    PhysicalRegion num_chan_pr;
    {
      RegionRequirement
        req(
          num_chan_col.values_lr,
          READ_ONLY,
          EXCLUSIVE,
          num_chan_col.values_lr);
      num_chan_pr = rt->map_region(ctx, req);
    }
    const ROAccessor<int, 1> num_chan(num_chan_pr, Column::VALUE_FID);

    PhysicalRegion num_corr_pr;
    {
      RegionRequirement
        req(
          num_corr_col.values_lr,
          READ_ONLY,
          EXCLUSIVE,
          num_corr_col.values_lr);
      num_corr_pr = rt->map_region(ctx, req);
    }
    const ROAccessor<int, 1> num_corr(num_corr_pr, Column::VALUE_FID);

    std::vector<IndexSpace> is_pieces;
    for (coord_t ant0 = 0; ant0 < num_antenna_classes; ++ant0)
      for (coord_t ant1 = ant0; ant1 < num_antenna_classes; ++ant1)
        for (coord_t pa = 0; pa < num_pa_steps; ++pa)
          for (coord_t dd = 0; dd < num_dd; ++dd) {
            coord_t lo[7]{ant0, ant1, pa, dd, 0, 0, 0};
            coord_t hi[7]{ant0, ant1, pa, dd, num_chan[spw_ids[dd]] - 1,
                w_proj_planes - 1, num_corr[pol_ids[dd]] - 1};
            is_pieces.push_back(
              rt->create_index_space(ctx, Rect<7>(Point<7>(lo), Point<7>(hi))));
          }
    rt->unmap_region(ctx, pol_id_pr);
    rt->unmap_region(ctx, spw_id_pr);
    rt->unmap_region(ctx, num_chan_pr);
    rt->unmap_region(ctx, num_corr_pr);
    return rt->union_index_spaces(ctx, is_pieces);
  }

private:

  IndexSpace m_bounds;

  LogicalRegion m_cfs;

  FieldSpace m_cf_fs;
};

typedef enum {VALUE_ARGS, STRING_ARGS} gridder_args_t;

template <typename T, gridder_args_t G>
struct GridderArgType {
};
template <typename T>
struct GridderArgType<T, VALUE_ARGS> {
  typedef T type;
};
template <typename T>
struct GridderArgType<T, STRING_ARGS> {
  typedef std::string type;
};

template <gridder_args_t G>
struct GridderArgs {
  typename GridderArgType<LEGMS_FS::path, G>::type h5;
  typename GridderArgType<float, G>::type pa_step;
  typename GridderArgType<int, G>::type w_proj_planes;
};

class TopLevelTask {
public:

  static const Legion::TaskID TASK_ID = TOP_LEVEL_TASK_ID;
  static constexpr const char* TASK_NAME = "TopLevelTask";

  static void
  get_args(
    const Legion::InputArgs& args,
    GridderArgs<STRING_ARGS>& gridder_args) {

    for (int i = 1; i < args.argc; ++i) {
      std::string ai = args.argv[i];
      if (ai == "--ms")
        gridder_args.h5 = args.argv[++i];
      else if (ai == "--pastep")
        gridder_args.pa_step = args.argv[++i];
      else if (ai == "--wprojplanes")
        gridder_args.w_proj_planes = args.argv[++i];
    }
  }

  static std::optional<std::string>
  args_error(
    const GridderArgs<STRING_ARGS>& str_args,
    const GridderArgs<VALUE_ARGS>& val_args) {

    std::ostringstream errs;
    std::optional<LEGMS_FS::path> h5_abs;
    if (LEGMS_FS::exists(val_args.h5))
      h5_abs = LEGMS_FS::canonical(val_args.h5);
    if (!h5_abs || !LEGMS_FS::is_regular_file(h5_abs.value()))
      errs << "Path '" << str_args.h5
           << "' does not name a regular file" << std::endl;
    switch (std::fpclassify(val_args.pa_step)) {
    case FP_NORMAL:
      break;
    case FP_ZERO:
    case FP_SUBNORMAL:
      errs << "--pastep value '" << str_args.pa_step
           << "' is too small" << std::endl;
      break;
    default:
      errs << "--pastep value '" << str_args.pa_step
           << "' is invalid" << std::endl;
      break;
    }
    if (val_args.w_proj_planes <= INVALID_W_PROJ_PLANES_VALUE)
      errs << "--wprojplanes value '" << str_args.w_proj_planes
           << "' is less than the minimum valid value of "
           << INVALID_W_PROJ_PLANES_VALUE + 1 << std::endl;
    else if (val_args.w_proj_planes == AUTO_W_PROJ_PLANES_VALUE)
      errs << "Automatic computation of the number of W-projection "
           << "planes is unimplemented" << std::endl;
    if (errs.str().size() > 0)
      return errs.str();
    return std::nullopt;
  }

  static void
  base_impl(
    const Legion::Task*,
    const std::vector<Legion::PhysicalRegion>&,
    Legion::Context ctx,
    Legion::Runtime* rt) {

    legms::register_tasks(ctx, rt);

    // process command line arguments
    GridderArgs<VALUE_ARGS> gridder_args;
    {
      const Legion::InputArgs& input_args = Legion::Runtime::get_input_args();
      GridderArgs<STRING_ARGS> str_args;
      str_args.pa_step = DEFAULT_PA_STEP_STR;
      str_args.w_proj_planes = DEFAULT_W_PROJ_PLANES_STR;
      get_args(input_args, str_args);
      gridder_args.h5 = str_args.h5;
      try {
        std::size_t pos;
        gridder_args.pa_step = std::stof(str_args.pa_step, &pos);
        if (pos != str_args.pa_step.size())
          gridder_args.pa_step = NAN;
      } catch (const std::invalid_argument&) {
        gridder_args.pa_step = NAN;
      } catch (const std::out_of_range&) {
        gridder_args.pa_step = NAN;
      }
      try {
        std::size_t pos;
        gridder_args.w_proj_planes = std::stoi(str_args.w_proj_planes, &pos);
        if (pos != str_args.w_proj_planes.size())
          gridder_args.w_proj_planes = INVALID_W_PROJ_PLANES_VALUE;
        if (gridder_args.w_proj_planes == 0)
          gridder_args.w_proj_planes = 1;
      } catch (const std::invalid_argument&) {
        gridder_args.pa_step = INVALID_W_PROJ_PLANES_VALUE;
      } catch (const std::out_of_range&) {
        gridder_args.pa_step = INVALID_W_PROJ_PLANES_VALUE;
      }
      auto errstr = args_error(str_args, gridder_args);
      if (errstr) {
        std::cerr << errstr.value() << std::endl;
        return;
      }
    }
    gridder_args.pa_step = std::fmod(std::abs(gridder_args.pa_step), 360.f);

    // initialize Tables used by gridder from HDF5 file
    const std::string ms_root = "/";
    MSTables mstables[] = {
      MS_ANTENNA,
      MS_DATA_DESCRIPTION,
      MS_POLARIZATION,
      MS_SPECTRAL_WINDOW
    };
    std::unordered_map<MSTables,Table> tables;
    for (auto& mst : mstables)
      tables[mst] = init_table(ctx, rt, gridder_args.h5, ms_root, mst);

    // create region mapping antenna index to antenna class
    LogicalRegion antenna_classes;
    {
      // note that ClassifyAntennasTask uses an index space from the antenna
      // table for the region it creates
      tables[MS_ANTENNA]
        .with_columns_attached(
          ctx,
          rt,
          gridder_args.h5,
          ms_root,
          [&antenna_classes]
          (Context ctx, Runtime* rt, const Table& tb) {
            UnitaryClassifyAntennasTask task(tb);
            antenna_classes = task.dispatch(ctx, rt);
          });
    }
    rt->attach_name(antenna_classes, "antenna_classes");

    // create vector of parallactic angle values
    PAValues pa_values(ctx, rt, tables, gridder_args.pa_step);
    rt->attach_name(pa_values.parameters, "pa_values");

    // create convolution function map
    CFMap cf_map(
      ctx,
      rt,
      gridder_args,
      ms_root,
      tables,
      UnitaryClassifyAntennasTask::num_classes,
      antenna_classes,
      pa_values);

    // TODO: the rest goes here

    // clean up
    cf_map.destroy(ctx, rt);
    pa_values.destroy(ctx, rt);
    // don't destroy index space of antenna_classes, as it is shared with an
    // index space in the antenna table
    rt->destroy_field_space(ctx, antenna_classes.get_field_space());
    rt->destroy_logical_region(ctx, antenna_classes);
    for (auto& tt : tables)
      tt.second.destroy(ctx, rt);
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
