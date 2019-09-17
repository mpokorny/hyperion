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
#define PARALLACTIC_ANGLE_TYPE float
#define PARALLACTIC_360 ((PARALLACTIC_ANGLE_TYPE)360.0)

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
  UNITARY_CLASSIFY_ANTENNAS_TASK_ID,
  COMPUTE_ROW_AUX_FIELDS_TASK_ID,
};

enum {
  // TODO: fix OpsManager to avoid using an offset here
  LAST_POINT_REDOP=100, // reserve LEGMS_MAX_DIM ids from here
};

template <int DIM>
struct LastPointRedop {
  typedef Point<DIM> LHS;
  typedef Point<DIM> RHS;

  template <bool EXCL>
  static void
  apply(LHS& lhs, RHS rhs);

  template <bool EXCL>
  static void
  fold(RHS& rhs1, RHS rhs2);

  // TODO: I'd prefer not to define "identity", and reduction ops should not be
  // required to define identity or fold according to realm/redop.h, but as of
  // Legion afb79fe, that is not the case
  static const Point<DIM> identity;
};

template <>
const Point<1> LastPointRedop<1>::identity = -1;

template <> template <>
inline void
LastPointRedop<1>::apply<true>(Point<1>& lhs, Point<1> rhs) {
  if (rhs.x != -1)
    lhs = rhs;
}

template <> template <>
inline void
LastPointRedop<1>::apply<false>(Point<1>& lhs, Point<1> rhs) {
  if (rhs != -1)
    __atomic_store_n(&lhs.x, rhs.x, __ATOMIC_RELEASE);
}

template <> template <>
void
LastPointRedop<1>::fold<true>(Point<1>& lhs, Point<1> rhs) {
  if (rhs != -1)
    lhs = rhs;
}

template <> template <>
inline void
LastPointRedop<1>::fold<false>(Point<1>& lhs, Point<1> rhs) {
  if (rhs != -1)
    __atomic_store_n(&lhs.x, rhs.x, __ATOMIC_RELEASE);
}

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
  typename GridderArgType<PARALLACTIC_ANGLE_TYPE, G>::type pa_step;
  typename GridderArgType<int, G>::type w_proj_planes;
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
struct TableColumns<MS_MAIN>
  : public WithColumnLookup<TableColumns<MS_MAIN>> {
  typedef enum {
    TIME,
    ANTENNA1,
    ANTENNA2,
    FEED1,
    FEED2,
    DATA_DESC_ID,
    UVW,
    DATA,
    SIGMA,
    FLAG,
    NUM_COLUMNS
  } col;
  static constexpr const std::array<const char*,NUM_COLUMNS> column_names = {
    "TIME",
    "ANTENNA1",
    "ANTENNA2",
    "FEED1",
    "FEED2",
    "DATA_DESC_ID",
    "UVW",
    "DATA",
    "SIGMA",
    "FLAG"
  };
  static constexpr const char* table_name = MSTable<MS_MAIN>::name;
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
  const char* table_name;
  switch (mst) {
#define INIT(TBL)                                       \
    case (MS_##TBL): {                                  \
      std::copy(                                        \
        TableColumns<MS_##TBL>::column_names.begin(),   \
        TableColumns<MS_##TBL>::column_names.end(),     \
        std::inserter(columns, columns.end()));         \
      table_name = MSTable<MS_##TBL>::name;             \
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
    rt->attach_name(result, "antenna_classes");
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

    const ROAccessor<legms::string, 1>
      names(regions[TableColumns<MS_ANTENNA>::NAME], Column::VALUE_FID);
    const ROAccessor<legms::string, 1>
      stations(regions[TableColumns<MS_ANTENNA>::STATION], Column::VALUE_FID);
    const ROAccessor<legms::string, 1>
      types(regions[TableColumns<MS_ANTENNA>::TYPE], Column::VALUE_FID);
    const ROAccessor<legms::string, 1>
      mounts(regions[TableColumns<MS_ANTENNA>::MOUNT], Column::VALUE_FID);
    const ROAccessor<double, 1>
      diameters(
        regions[TableColumns<MS_ANTENNA>::DISH_DIAMETER],
        Column::VALUE_FID);
    const WOAccessor<unsigned, 1> antenna_classes(regions.back(), 0);
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

  static const constexpr FieldID TASK_ID = UNITARY_CLASSIFY_ANTENNAS_TASK_ID;
  static const constexpr char* TASK_NAME = "UnitaryClassifyAntennasTask";
  static const unsigned num_classes = 1;

  UnitaryClassifyAntennasTask(const Table& table)
    : ClassifyAntennasTask(table) {
  }

  static void
  preregister() {
    TaskVariantRegistrar
      registrar(UNITARY_CLASSIFY_ANTENNAS_TASK_ID, TASK_NAME, false);
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

class PAIntervals {
public:

  static const FieldID PA_ORIGIN_FID = 0;
  static const FieldID PA_STEP_FID = 1;
  static const FieldID PA_NUM_STEP_FID = 2;

  LogicalRegion parameters;

  PAIntervals() {
  };

  PAIntervals(LogicalRegion parameters_)
    :parameters(parameters_) {
  };

  static PAIntervals
  create(
    Context ctx,
    Runtime* rt,
    const std::unordered_map<MSTables,Table>&,
    PARALLACTIC_ANGLE_TYPE pa_step,
    PARALLACTIC_ANGLE_TYPE pa_origin) {

    IndexSpace is = rt->create_index_space(ctx, Rect<1>(0, 0));
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    fa.allocate_field(sizeof(PARALLACTIC_ANGLE_TYPE), PA_ORIGIN_FID);
    fa.allocate_field(sizeof(PARALLACTIC_ANGLE_TYPE), PA_STEP_FID);
    fa.allocate_field(sizeof(unsigned long), PA_NUM_STEP_FID);
    LogicalRegion parameters = rt->create_logical_region(ctx, is, fs);
    rt->attach_name(parameters, "pa_intervals");

    RegionRequirement req(parameters, WRITE_ONLY, EXCLUSIVE, parameters);
    req.add_field(PA_ORIGIN_FID);
    req.add_field(PA_STEP_FID);
    req.add_field(PA_NUM_STEP_FID);
    auto pr = rt->map_region(ctx, req);
    const WOAccessor<PARALLACTIC_ANGLE_TYPE, 1> origin(pr, PA_ORIGIN_FID);
    const WOAccessor<PARALLACTIC_ANGLE_TYPE, 1> step(pr, PA_STEP_FID);
    const WOAccessor<unsigned long, 1> num_step(pr, PA_NUM_STEP_FID);
    origin[0] = pa_origin;
    step[0] = pa_step;
    num_step[0] = std::lrint(std::ceil(360.0f / pa_step));
    rt->unmap_region(ctx, pr);
    return PAIntervals(parameters);
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

  std::optional<std::tuple<PARALLACTIC_ANGLE_TYPE, PARALLACTIC_ANGLE_TYPE>>
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

  static
  std::optional<std::tuple<PARALLACTIC_ANGLE_TYPE, PARALLACTIC_ANGLE_TYPE>>
  pa(PhysicalRegion region, unsigned long i) {
    const ROAccessor<PARALLACTIC_ANGLE_TYPE, 1> origin(region, PA_ORIGIN_FID);
    const ROAccessor<PARALLACTIC_ANGLE_TYPE, 1> step(region, PA_STEP_FID);
    const ROAccessor<unsigned long, 1> num_step(region, PA_NUM_STEP_FID);
    if (i >= num_step[0])
      return std::nullopt;
    PARALLACTIC_ANGLE_TYPE lo = i * step[0];
    PARALLACTIC_ANGLE_TYPE width =
      (i == num_step[0] - 1) ? (PARALLACTIC_360 - lo) : step[0];
    return std::make_tuple(origin[0] + lo, width);
  }

  unsigned long
  find(Context ctx, Runtime* rt, PARALLACTIC_ANGLE_TYPE pa) const {
    RegionRequirement req(parameters, READ_ONLY, EXCLUSIVE, parameters);
    req.add_field(PA_ORIGIN_FID);
    req.add_field(PA_STEP_FID);
    auto pr = rt->map_region(ctx, req);
    auto result = find(pr, pa);
    rt->unmap_region(ctx, pr);
    return result;
  }

  static unsigned long
  find(PhysicalRegion region, PARALLACTIC_ANGLE_TYPE pa) {
    const ROAccessor<PARALLACTIC_ANGLE_TYPE, 1> origin(region, PA_ORIGIN_FID);
    const ROAccessor<PARALLACTIC_ANGLE_TYPE, 1> step(region, PA_STEP_FID);
    pa -= origin[0];
    pa -= std::floor(pa / PARALLACTIC_360) * PARALLACTIC_360;
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

class ComputeRowAuxFieldsTask {
public:

  static const constexpr char* TASK_NAME = "ComputeRowAuxFieldsTask_";
  static const constexpr FieldID PARALLACTIC_ANGLE_FID = 0;

  ComputeRowAuxFieldsTask(
    IndexSpace row_is,
    const Column& antenna1,
    const Column& antenna2,
    const Column& feed1,
    const Column& feed2,
    const Column& data_desc)
    : m_row_is(row_is)
    , m_antenna1(antenna1)
    , m_antenna2(antenna2)
    , m_feed1(feed1)
    , m_feed2(feed2)
    , m_data_desc(data_desc) {
  }

  LogicalRegion
  dispatch(Context ctx, Runtime* rt) {
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    fa.allocate_field(
      sizeof(PARALLACTIC_ANGLE_TYPE),
      PARALLACTIC_ANGLE_FID);
    rt->attach_name(fs, PARALLACTIC_ANGLE_FID, "parallactic_angle");
    LogicalRegion result = rt->create_logical_region(ctx, m_row_is, fs);
    rt->attach_name(result, "row_aux_fields");
    IndexTaskLauncher init_task(
      COMPUTE_ROW_AUX_FIELDS_TASK_ID + m_row_is.get_dim() - 1,
      m_row_is,
      TaskArgument(NULL, 0),
      ArgumentMap());
    for (auto& col : {m_antenna1, m_antenna2, m_feed1, m_feed2, m_data_desc}) {
      RegionRequirement
        req(col.values_lr, 0, READ_ONLY, EXCLUSIVE, col.values_lr);
      req.add_field(Column::VALUE_FID);
      init_task.add_region_requirement(req);
    }
    {
      RegionRequirement req(result, 0, WRITE_ONLY, EXCLUSIVE, result);
      req.add_field(PARALLACTIC_ANGLE_FID);
      init_task.add_region_requirement(req);
    }
    rt->execute_index_space(ctx, init_task);
    return result;
  }

  static PARALLACTIC_ANGLE_TYPE
  parallactic_angle(
    const int& antenna1,
    const int& antenna2,
    const int& feed1,
    const int& feed2,
    const int& data_desc) {

    return 0.0;
  }

  template <int ROW_DIM>
  static void
  base_impl(
    const Task* task,
    const std::vector<PhysicalRegion>& regions,
    Context,
    Runtime *) {

    const ROAccessor<DataType<LEGMS_TYPE_INT>::ValueType, ROW_DIM>
      antenna1(regions[0], Column::VALUE_FID);
    const ROAccessor<DataType<LEGMS_TYPE_INT>::ValueType, ROW_DIM>
      antenna2(regions[1], Column::VALUE_FID);
    const ROAccessor<DataType<LEGMS_TYPE_INT>::ValueType, ROW_DIM>
      feed1(regions[2], Column::VALUE_FID);
    const ROAccessor<DataType<LEGMS_TYPE_INT>::ValueType, ROW_DIM>
      feed2(regions[3], Column::VALUE_FID);
    const ROAccessor<DataType<LEGMS_TYPE_INT>::ValueType, ROW_DIM>
      data_desc(regions[4], Column::VALUE_FID);
    const WOAccessor<PARALLACTIC_ANGLE_TYPE, ROW_DIM>
      pa(regions[5], PARALLACTIC_ANGLE_FID);
    for (PointInDomainIterator<ROW_DIM> pid(task->index_domain);
         pid();
         pid++)
    pa[*pid] =
      parallactic_angle(
        antenna1[*pid],
        antenna2[*pid],
        feed1[*pid],
        feed2[*pid],
        data_desc[*pid]);
  }

  static void
  preregister() {
#define REG_TASK(DIM) {                                   \
      std::string tname = std::string(TASK_NAME) + #DIM;  \
      TaskVariantRegistrar registrar(                     \
        COMPUTE_ROW_AUX_FIELDS_TASK_ID + DIM - 1,         \
        tname.c_str());                                   \
      registrar.add_constraint(                           \
        ProcessorConstraint(Processor::LOC_PROC));        \
      registrar.set_leaf(true);                           \
      registrar.set_idempotent(true);                     \
      Runtime::preregister_task_variant<base_impl<DIM>>(  \
        registrar,                                        \
        tname.c_str());                                   \
    }
    LEGMS_FOREACH_N(REG_TASK);
#undef REG_TASK
  }

private:

  IndexSpace m_row_is;
  Column m_antenna1;
  Column m_antenna2;
  Column m_feed1;
  Column m_feed2;
  Column m_data_desc;
};

class CFMap {
public:

  CFMap() {
  };

  CFMap(IndexSpace bounds, LogicalRegion cfs, FieldSpace cf_fs)
    : m_bounds(bounds)
    , m_cfs(cfs)
    , m_cf_fs(cf_fs) {
  }

  template <int MAIN_ROW_DIM>
  static CFMap
  create(
    Context ctx,
    Runtime* rt,
    const GridderArgs<VALUE_ARGS>& gridder_args,
    const std::string& ms_root,
    std::unordered_map<MSTables,Table>& tables,
    unsigned num_antenna_classes,
    LogicalRegion antenna_classes,
    const PAIntervals& pa_intervals) {

    // compute the map index space bounds
    IndexSpace bounds =
      Table::with_columns_attached(
        ctx,
        rt,
        gridder_args.h5,
        ms_root,
        {{&tables[MS_DATA_DESCRIPTION],
          {COLUMN_NAME(MS_DATA_DESCRIPTION, SPECTRAL_WINDOW_ID),
           COLUMN_NAME(MS_DATA_DESCRIPTION, POLARIZATION_ID)},
          {}},
         {&tables[MS_SPECTRAL_WINDOW],
          {COLUMN_NAME(MS_SPECTRAL_WINDOW, NUM_CHAN)},
          {}},
         {&tables[MS_POLARIZATION],
          {COLUMN_NAME(MS_POLARIZATION, NUM_CORR)},
          {}}},
        [num_antenna_classes, &gridder_args, &pa_intervals]
        (Context c, Runtime* r,
         std::unordered_map<std::string, Table*>& tables) {
          return
            CFMap::bounding_index_space(
              c,
              r,
              *tables[MSTable<MS_DATA_DESCRIPTION>::name],
              tables[MSTable<MS_SPECTRAL_WINDOW>::name]->column(
                c, r, COLUMN_NAME(MS_SPECTRAL_WINDOW, NUM_CHAN)),
              tables[MSTable<MS_POLARIZATION>::name]->column(
                c, r, COLUMN_NAME(MS_POLARIZATION, NUM_CORR)),
              num_antenna_classes,
              gridder_args.w_proj_planes,
              pa_intervals.num_steps(c, r));
        });

    // compute convolution functions
    LogicalRegion cfs;
    FieldSpace cf_fs;
    Table::with_columns_attached(
      ctx,
      rt,
      gridder_args.h5,
      ms_root,
      {{&tables[MS_MAIN],
        {COLUMN_NAME(MS_MAIN, ANTENNA1),
         COLUMN_NAME(MS_MAIN, ANTENNA2),
         COLUMN_NAME(MS_MAIN, FEED1),
         COLUMN_NAME(MS_MAIN, FEED2),
         COLUMN_NAME(MS_MAIN, DATA_DESC_ID),
         COLUMN_NAME(MS_MAIN, UVW)},
        {}}},
      [&pa_intervals, &bounds, &cfs, &cf_fs]
      (Context c, Runtime* r, std::unordered_map<std::string,Table*>& tables) {
        // get some columns that we'll be using
        auto main_table = tables[MSTable<MS_MAIN>::name];
        auto row_is =
          main_table->min_rank_column(c, r).values_lr.get_index_space();
        auto antenna1 =
          main_table->column(c, r, COLUMN_NAME(MS_MAIN, ANTENNA1));
        auto antenna2 =
          main_table->column(c, r, COLUMN_NAME(MS_MAIN, ANTENNA2));
        auto feed1 =
          main_table->column(c, r, COLUMN_NAME(MS_MAIN, FEED1));
        auto feed2 =
          main_table->column(c, r, COLUMN_NAME(MS_MAIN, FEED2));
        auto data_desc =
          main_table->column(c, r, COLUMN_NAME(MS_MAIN, DATA_DESC_ID));
        auto uvw =
          main_table->column(c, r, COLUMN_NAME(MS_MAIN, UVW));

        // get row-wise index partition (only needed for uvw)
        auto row_part =
          uvw.partition_on_axes(c, r, main_table->index_axes(c, r));
        auto uvw_lp =
          r->get_logical_partition(c, uvw.values_lr, row_part.index_partition);

        // compute auxiliary row-wise data
        LogicalRegion row_aux;
        {
          ComputeRowAuxFieldsTask task(
            row_is,
            antenna1,
            antenna2,
            feed1,
            feed2,
            data_desc);
          row_aux = task.dispatch(c, r);
          {
            RegionRequirement req(row_aux, READ_ONLY, EXCLUSIVE, row_aux);
            req.add_field(ComputeRowAuxFieldsTask::PARALLACTIC_ANGLE_FID);
            auto pr = r->map_region(c, req);
            const ROAccessor<PARALLACTIC_ANGLE_TYPE, 1>
              pa(pr, ComputeRowAuxFieldsTask::PARALLACTIC_ANGLE_FID);
            for (PointInDomainIterator<1>
                   pid(r->get_index_space_domain(row_is));
                 pid();
                 pid++)
              std::cout << *pid << ": " << pa[*pid] << std::endl;
          }
        }
        // create and initialize preimage, which records, for every point in
        // bounds, a row number that maps to that point
#define PREIMAGE_ROW_FID 0
#define PREIMAGE_FLAG_FID 1
#define ASSIGNED_FLAG_TYPE int16_t
#define ASSIGNED_FLAG_TRUE (ASSIGNED_FLAG_TYPE)1
#define ASSIGNED_FLAG_FALSE (ASSIGNED_FLAG_TYPE)0
        // LogicalRegion preimage;
        // {
        //   FieldSpace fs = r->create_field_space(c);
        //   FieldAllocator fa = r->create_field_allocator(c, fs);
        //   fa.allocate_field(sizeof(Point<MAIN_ROW_DIM>), PREIMAGE_ROW_FID);
        //   fa.allocate_field(sizeof(ASSIGNED_FLAG_TYPE), PREIMAGE_FLAG_FID);
        //   preimage = r->create_logical_region(c, bounds, fs);
        //   r->fill_field(
        //     c,
        //     preimage,
        //     preimage,
        //     PREIMAGE_ROW_FID,
        //     Point<MAIN_ROW_DIM>(-1));
        //   r->fill_field(
        //     c,
        //     preimage,
        //     preimage,
        //     PREIMAGE_FLAG_FID,
        //     ASSIGNED_FLAG_FALSE);
        //   // launch index space task over visibilities to write row
        //   // numbers and flags to preimage via reductions
        //   IndexTaskLauncher
        //     preimage_task(FIXME, row_is, TaskArgument(NULL, 0), ArgumentMap());
        //   for (auto& col : {antenna1, antenna2, data_desc}) {
        //     RegionRequirement
        //       req(col.values_lr, 0, READ_ONLY, EXCLUSIVE, col.values_lr);
        //     req.add_field(Column::VALUE_FID);
        //     preimage_task.add_region_requirement(req);
        //   }
        //   {
        //     RegionRequirement
        //       req(uvw_lp, 0, READ_ONLY, EXCLUSIVE, uvw.values_lr);
        //     req.add_field(Column::VALUE_FID);
        //     preimage_task.add_region_requirement(req);
        //   }
        //   {
        //     RegionRequirement req(row_aux, 0, READ_ONLY, EXCLUSIVE, row_aux);
        //     req.add_field(ComputeRowAuxFieldsTask::PARALLACTIC_ANGLE_FID);
        //     preimage_task.add_region_requirement(req);
        //   }
        //   {
        //     auto params = pa_intervals.parameters;
        //     RegionRequirement req(params, READ_ONLY, EXCLUSIVE, params);
        //     req.add_field(PAIntervals::PA_ORIGIN_FID);
        //     req.add_field(PAIntervals::PA_STEP_FID);
        //     req.add_field(PAIntervals::PA_NUM_STEP_FID);
        //     preimage_task.add_region_requirement(req);
        //   }
        //   {
        //     RegionRequirement req(preimage, LAST_POINT_REDOP, ATOMIC, preimage);
        //     req.add_field(PREIMAGE_ROW_FID);
        //     preimage_task.add_region_requirement(req);
        //   }
        //   {
        //     RegionRequirement
        //       req(preimage, LEGION_REDOP_MAX_INT16, ATOMIC, preimage);
        //     req.add_field(PREIMAGE_FLAG_FID);
        //     preimage_task.add_region_requirement(req);
        //   }
        //   // TODO: preimage_task tasks will compute index components ant1_class,
        //   // ant2_class, pa, dd, and w_plane, and will write values into all
        //   // elements of preimage with the computed index prefix (i.e, for all
        //   // channel and correlation index components)
        //   preimage_task.dispatch(c, r);
        // }

        // create partition, the partition of bounds by "assigned" flag of
        // preimage
        // IndexPartition partition;
        // {
        //   IndexSpace flag_values = r->create_index_space(c, Rect<1>(0, 1));
        //   partition =
        //     r->create_partition_by_field(
        //       c,
        //       preimage,
        //       preimage,
        //       PREIMAGE_FLAG_FID,
        //       flag_values);
        //   r->destroy_index_space(c, flag_values);
        // }

        // create cfs, the region of cfs, using as index space the sub-space of
        // partition with an "assigned" flag value of 1
#define CFS_REGION_FID 0
        // {
        //   FieldSpace fs = r->create_field_space(c);
        //   {
        //     FieldAllocator fa = r->create_field_allocator(c, fs);
        //     fa.allocate_field(sizeof(LogicalRegion), CFS_REGION_FID);
        //   }
        //   cfs =
        //     r->create_logical_region(
        //       c,
        //       r->get_index_subspace(c, partition, Point<1>(ASSIGNED_FLAG_TRUE)),
        //       fs);
        // }
        // r->destroy_index_partition(c, partition); // TODO: OK?

        // create common field space of every cf
#define CF_VALUE_FID 0
        // cf_fs = r->create_field_space(c);
        // {
        //   FieldAllocator fa = r->create_field_allocator(c, cf_fs);
        //   fa.allocate_field(sizeof(std::complex<float>), CF_VALUE_FID);
        // }

        // launch index space task on cfs to create and initialize its values,
        // using "preimage" to select the row used for initialization
        // {
        //   // we always want to map columns in the main table in pieces (by
        //   // partitions or by index task launches); in this case, since each
        //   // point in the cfs index space only needs to read the columns at the
        //   // row stored in preimage, we first partition cfs completely, and then
        //   // use partition_by_image to derive partitions for the main table
        //   // columns
        //   IndexPartition cfs_ip =
        //     r->create_equal_partition(
        //       c,
        //       cfs.get_index_space(),
        //       cfs.get_index_space());
        //   LogicalPartition preimage_lp =
        //     r->get_logical_partition(c, preimage, cfs_ip);
        //   IndexPartition preimage_ip =
        //     r->create_partition_by_image(
        //       c,
        //       row_is,
        //       preimage_lp,
        //       preimage,
        //       PREIMAGE_ROW_FID,
        //       cfs.get_index_space());

        //   IndexTaskLauncher
        //     cfs_task(
        //       FIXME,
        //       cfs.get_index_space(),
        //       TaskArgument(&cf_fs, sizeof(cf_fs)),
        //       ArgumentMap());
        //   for (auto& col : {antenna1, antenna2, data_desc, uvw}) {
        //     RegionRequirement
        //       req(col.values_lr, READ_ONLY, EXCLUSIVE, col.values_lr);
        //     req.add_field(Column::VALUE_FID);
        //     cfs_task.add_region_requirement(req);
        //   }
        //   {
        //     RegionRequirement req(row_aux, 0, READ_ONLY, EXCLUSIVE, row_aux);
        //     req.add_field(PARALLACTIC_ANGLE_FID);
        //     preimage_task.add_region_requirement(req);
        //   }
        //   {
        //     auto params = pa_intervals.parameters;
        //     RegionRequirement req(params, READ_ONLY, EXCLUSIVE, params);
        //     req.add_field(PAIntervals::PA_ORIGIN_FID);
        //     req.add_field(PAIntervals::PA_STEP_FID);
        //     req.add_field(PAIntervals::PA_NUM_STEP_FID);
        //     preimage_task.add_region_requirement(req);
        //   }
        //   {
        //     RegionRequirement req(preimage, READ_ONLY, EXCLUSIVE, preimage);
        //     req.add_field(PREIMAGE_ROW_FID);
        //     cfs_task.add_region_requirement(req);
        //   }
        //   {
        //     RegionRequirement req(cfs, WRITE_ONLY, EXCLUSIVE, cfs);
        //     req.add_field(CFS_REGION_FID);
        //     cfs_task.add_region_requirement(req);
        //   }
        //   cfs_task.dispatch(c, r);
        // }

        r->destroy_field_space(c, row_aux.get_field_space());
        r->destroy_logical_region(c, row_aux);
        r->destroy_logical_partition(c, uvw_lp);
        row_part.destroy(c, r, true);
        // r->destroy_field_space(c, preimage.get_field_space());
        // r->destroy_logical_region(c, preimage);
      });

    return CFMap(bounds, cfs, cf_fs);
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
    if (m_cfs != LogicalRegion::NO_REGION) {
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
            rt->destroy_logical_region(ctx, cf);
          }
        }
        rt->unmap_region(ctx, pr);
        destroy_common_field_space(ctx, rt);
      }
      rt->destroy_field_space(ctx, m_cfs.get_field_space());
      rt->destroy_logical_region(ctx, m_cfs);
      m_cfs = LogicalRegion::NO_REGION;
    }
    if (m_bounds != IndexSpace::NO_SPACE)
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
      req.add_field(Column::VALUE_FID);
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
      req.add_field(Column::VALUE_FID);
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

template <typename CLASSIFY_ANTENNAS_TASK>
class TopLevelTask {
public:

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
      if (std::abs(val_args.pa_step) > PARALLACTIC_360)
        errs << "--pastep value '" << str_args.pa_step
             << "' is not in valid range [-360, 360]" << std::endl;
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
    gridder_args.pa_step = std::abs(gridder_args.pa_step);

    // initialize Tables used by gridder from HDF5 file
    const std::string ms_root = "/";
    MSTables mstables[] = {
      MS_ANTENNA,
      MS_DATA_DESCRIPTION,
      MS_MAIN,
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
          {COLUMN_NAME(MS_ANTENNA, NAME),
           COLUMN_NAME(MS_ANTENNA, STATION),
           COLUMN_NAME(MS_ANTENNA, TYPE),
           COLUMN_NAME(MS_ANTENNA, MOUNT),
           COLUMN_NAME(MS_ANTENNA, DISH_DIAMETER)},
          {},
          [&antenna_classes]
          (Context ctx, Runtime* rt, const Table* tb) {
            CLASSIFY_ANTENNAS_TASK task(*tb);
            antenna_classes = task.dispatch(ctx, rt);
          });
    }

    // create vector of parallactic angle values
    PAIntervals pa_intervals =
      PAIntervals::create(ctx, rt, tables, gridder_args.pa_step, 0.0f);

    // create convolution function map
    CFMap cf_map;
    switch (tables[MS_MAIN].index_axes(ctx, rt).size()) {
#define MAKE_CFMAP(DIM) \
      case DIM:                                 \
        cf_map = CFMap::create<DIM>(            \
          ctx,                                  \
          rt,                                   \
          gridder_args,                         \
          ms_root,                              \
          tables,                               \
          CLASSIFY_ANTENNAS_TASK::num_classes,  \
          antenna_classes,                      \
          pa_intervals);                        \
        break;
      MAKE_CFMAP(1);
#undef MAKE_CFMAP
    default:
        // for now, we only support 1-d rows (TODO: specialize LastPointRedop
        // for DIM > 1)
        assert(false);
    }

    // TODO: the rest goes here

    // clean up
    cf_map.destroy(ctx, rt);
    pa_intervals.destroy(ctx, rt);
    // don't destroy index space of antenna_classes, as it is shared with an
    // index space in the antenna table
    rt->destroy_field_space(ctx, antenna_classes.get_field_space());
    rt->destroy_logical_region(ctx, antenna_classes);
    for (auto& tt : tables)
      tt.second.destroy(ctx, rt);
  }

  static void
  preregister() {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, TASK_NAME);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<base_impl>(registrar, TASK_NAME);
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  }
};

int
main(int argc, char* argv[]) {
  legms::preregister_all();
  UnitaryClassifyAntennasTask::preregister();
  TopLevelTask<UnitaryClassifyAntennasTask>::preregister();
  ComputeRowAuxFieldsTask::preregister();
  Runtime::register_reduction_op<LastPointRedop<1>>(LAST_POINT_REDOP);
  return Runtime::start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
