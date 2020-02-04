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

#if LEGION_MAX_DIM < 7
# error "MAX_DIM too small"
#endif

#include <hyperion/hdf5.h>
#include <hyperion/Table.h>
#include <hyperion/Column.h>
#include <hyperion/MeasRef.h>
#include <hyperion/MSTableColumns.h>
#include <hyperion/MSAntennaColumns.h>
#include <hyperion/MSFieldColumns.h>
#include <hyperion/gridder/gridder.h>
#include <hyperion/gridder/args.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include CXX_FILESYSTEM_HEADER
#include <forward_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <yaml-cpp/yaml.h>

#include <casacore/measures/Measures/MCDirection.h>
#include <casacore/measures/Measures/MCEpoch.h>
#include <casacore/measures/Measures/MCRadialVelocity.h>

using namespace hyperion;
using namespace Legion;

namespace cc = casacore;

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
  LAST_POINT_REDOP=100, // reserve HYPERION_MAX_DIM ids from here
};

static const char ms_root[] = "/";

static const gridder::Args<gridder::OPT_STRING_ARGS>&
default_config() {
  static bool computed = false;
  static gridder::Args<gridder::OPT_STRING_ARGS> result;
  if (!computed) {
    result.echo = std::string("false");
    result.min_block = std::string("100000");
    result.pa_step = std::string("360.0");
    result.w_planes = std::string("1");
    computed = true;
  }
  return result;
}

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

template <MSTables T>
struct TableColumns {
  //typedef ColEnums;
  static const std::array<const char*, 0> column_names;
};

template <MSTables T>
const std::array<const char*, 0> TableColumns<T>::column_names;

template <>
struct TableColumns<MS_ANTENNA> {
  typedef enum {
    NAME,
    STATION,
    TYPE,
    MOUNT,
    DISH_DIAMETER,
    POSITION,
    NUM_COLUMNS
  } col;
  static const constexpr std::array<const char*,NUM_COLUMNS> column_names = {
    "NAME",
    "STATION",
    "TYPE",
    "MOUNT",
    "DISH_DIAMETER",
    "POSITION"
  };
};

template <>
struct TableColumns<MS_DATA_DESCRIPTION> {
  typedef enum {
    SPECTRAL_WINDOW_ID,
    POLARIZATION_ID,
    NUM_COLUMNS
  } col;
  static const constexpr std::array<const char*,NUM_COLUMNS> column_names = {
    "SPECTRAL_WINDOW_ID",
    "POLARIZATION_ID"
  };
};

template <>
struct TableColumns<MS_FEED> {
  typedef enum {
    ANTENNA_ID,
    FEED_ID,
    SPECTRAL_WINDOW_ID,
    TIME,
    INTERVAL,
    NUM_RECEPTORS,
    BEAM_ID,
    BEAM_OFFSET,
    POL_RESPONSE,
    POSITION,
    RECEPTOR_ANGLE,
    NUM_COLUMNS
  } col;
  static const constexpr std::array<const char*,NUM_COLUMNS> column_names = {
    "ANTENNA_ID",
    "FEED_ID",
    "SPECTRAL_WINDOW_ID",
    "TIME",
    "INTERVAL",
    "NUM_RECEPTORS",
    "BEAM_ID",
    "BEAM_OFFSET",
    "POL_RESPONSE",
    "POSITION",
    "RECEPTOR_ANGLE"
  };
  static const constexpr std::array<unsigned,NUM_COLUMNS> element_rank = {
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    2,
    2,
    1,
    1
  };
  // don't index on TIME and INTERVAL axes, as we don't expect feed
  // columns to vary with time
  static constexpr const std::array<MSTable<MS_FEED>::Axes,3> index_axes = {
    FEED_ANTENNA_ID,
    FEED_FEED_ID,
    FEED_SPECTRAL_WINDOW_ID
  };
};

template <>
struct TableColumns<MS_FIELD> {
  typedef enum {
    TIME,
    NUM_POLY,
    DELAY_DIR,
    PHASE_DIR,
    REFERENCE_DIR,
    SOURCE_ID,
    NUM_COLUMNS
  } col;
  static const constexpr std::array<const char*,NUM_COLUMNS> column_names = {
    "TIME",
    "NUM_POLY",
    "DELAY_DIR",
    "PHASE_DIR",
    "REFERENCE_DIR",
    "SORUCE_ID"
  };
};

template <>
struct TableColumns<MS_MAIN> {
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
  static const constexpr std::array<const char*,NUM_COLUMNS> column_names = {
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
};

template <>
struct TableColumns<MS_POLARIZATION> {
  typedef enum {
    NUM_CORR,
    NUM_COLUMNS
  } col;
  static const constexpr std::array<const char*,NUM_COLUMNS> column_names = {
    "NUM_CORR",
  };
};

template <>
struct TableColumns<MS_SPECTRAL_WINDOW> {
  typedef enum {
    NUM_CHAN,
    REF_FREQUENCY,
    CHAN_FREQ,
    CHAN_WIDTH,
    NUM_COLUMNS
  } col;
  static const constexpr std::array<const char*,NUM_COLUMNS> column_names = {
    "NUM_CHAN",
    "REF_FREQUENCY",
    "CHAN_FREQ",
    "CHAN_WIDTH"
  };
};

#define COLUMN_NAME(T, C) TableColumns<T>::column_names[TableColumns<T>::C]

Table
init_table(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const FS::path& ms,
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
    HYPERION_FOREACH_MS_TABLE(INIT);
  }
  return hyperion::hdf5::init_table(ctx, rt, ms, root + table_name, columns);
}

static const FieldID antenna_class_fid = 0;

template <typename T>
class ClassifyAntennasTask {
protected:

  ClassifyAntennasTask(const Table& table)
    : m_table(table) {
  }

  typedef enum {
    ANTENNA_NAME,
    ANTENNA_STATION,
    ANTENNA_TYPE,
    ANTENNA_MOUNT,
    ANTENNA_DISH_DIAMETER,
    ANTENNA_CLASS,
    NUM_REGIONS
  } region_t;

public:

  static const constexpr std::array<const char*, 5> column_names = {
    MSAntennaColumns::C::column_names[
      MSAntennaColumns::C::col_t::MS_ANTENNA_COL_NAME],
    MSAntennaColumns::C::column_names[
      MSAntennaColumns::C::col_t::MS_ANTENNA_COL_STATION],
    MSAntennaColumns::C::column_names[
      MSAntennaColumns::C::col_t::MS_ANTENNA_COL_TYPE],
    MSAntennaColumns::C::column_names[
      MSAntennaColumns::C::col_t::MS_ANTENNA_COL_MOUNT],
    MSAntennaColumns::C::column_names[
      MSAntennaColumns::C::col_t::MS_ANTENNA_COL_DISH_DIAMETER]
  };

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

    TaskArgs args;
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
#define ADD_COL_REQ(NM) do {                                        \
        auto col =                                                  \
          m_table.column(                                           \
            ctx,                                                    \
            rt,                                                     \
            cols,                                                   \
            MSAntennaColumns::C::column_names[                      \
              MSAntennaColumns::C::col_t::MS_ANTENNA_COL_##NM]);    \
        RegionRequirement                                           \
          vreq(col.values_lr, READ_ONLY, EXCLUSIVE, col.values_lr); \
        vreq.add_field(Column::VALUE_FID);                          \
        args.ridx[ANTENNA_##NM] = requirements.size();              \
        requirements.push_back(vreq);                               \
      } while (0)
      ADD_COL_REQ(NAME);
      ADD_COL_REQ(STATION);
      ADD_COL_REQ(TYPE);
      ADD_COL_REQ(MOUNT);
      ADD_COL_REQ(DISH_DIAMETER);
#undef ADD_COL_REQ
      rt->unmap_region(ctx, cols);
    }
    {
      RegionRequirement req(result, WRITE_ONLY, EXCLUSIVE, result);
      req.add_field(antenna_class_fid);
      args.ridx[ANTENNA_CLASS] = requirements.size();
      requirements.push_back(req);
    }
    TaskLauncher launcher(T::TASK_ID, TaskArgument(&args, sizeof(args)));
    for (auto& req : requirements)
      launcher.add_region_requirement(req);
    rt->execute_task(ctx, launcher);
    return result;
  }

  static void
  base_impl(
    const Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* rt) {

    const TaskArgs* args = static_cast<const TaskArgs*>(task->args);

#define NM_REGION(NM)                                         \
    {std::string(                                             \
        MSAntennaColumns::C::column_names[                    \
          MSAntennaColumns::C::col_t::MS_ANTENNA_COL_##NM]),  \
      {regions[args->ridx[ANTENNA_##NM]]}}

    MSAntennaColumns ac(
      rt,
      task->regions[args->ridx[ANTENNA_NAME]],
      std::unordered_map<std::string, std::vector<PhysicalRegion>>(
        {NM_REGION(NAME),
         NM_REGION(STATION),
         NM_REGION(TYPE),
         NM_REGION(MOUNT),
         NM_REGION(DISH_DIAMETER)}));
    auto names = ac.name<READ_ONLY>();
    auto stations = ac.station<READ_ONLY>();
    auto types = ac.type<READ_ONLY>();
    auto mounts = ac.mount<READ_ONLY>();
    auto diameters = ac.dishDiameter<READ_ONLY>();
    const WOAccessor<unsigned, 1>
      antenna_classes(regions[args->ridx[ANTENNA_CLASS]], 0);

    for (PointInDomainIterator<1> pid(ac.rows(rt)); pid(); pid++)
      antenna_classes[*pid] =
        T::classify(
          names[*pid].val,
          stations[*pid].val,
          types[*pid].val,
          mounts[*pid].val,
          diameters[*pid]);
  }

protected:

  struct TaskArgs {
    std::array<unsigned, NUM_REGIONS> ridx;
  };

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

struct AntennaHelper {
  typedef enum {
    ALT_AZ,
    EQUATORIAL,
    XY,
    ORBITING,
    NASMYTH_R,
    NASMYTH_L,
    OTHER
  } MountCode;

  static MountCode
  mount_code(const std::string& str) {
    if (mount_codes.count(str) > 0)
      return mount_codes[str];
    return OTHER;
  }

private:
  static std::unordered_map<std::string, MountCode> mount_codes;
};

std::unordered_map<std::string, AntennaHelper::MountCode>
AntennaHelper::mount_codes = {
  {"alt-az", ALT_AZ},
  {"", ALT_AZ},
  {"alt-az+rotator", ALT_AZ},
  {"equatorial", EQUATORIAL},
  {"x-y", XY},
  {"orbiting", ORBITING},
  {"alt-az+nasmyth-r", NASMYTH_R},
  {"alt-az+nasmyth-l", NASMYTH_L}};

class ComputeRowAuxFieldsTask {
public:

  static const constexpr char* TASK_NAME = "ComputeRowAuxFieldsTask_";
  static const constexpr FieldID PARALLACTIC_ANGLE_FID = 0;

  ComputeRowAuxFieldsTask(
    const gridder::Args<gridder::VALUE_ARGS>& args,
    Table& tbl_main,
    Table& tbl_data_description,
    Table& tbl_antenna,
    Table& tbl_feed)
    : m_args(args)
    , m_tbl_main(tbl_main)
    , m_tbl_data_description(tbl_data_description)
    , m_tbl_antenna(tbl_antenna)
    , m_tbl_feed(tbl_feed) {
  }

  LogicalRegion
  dispatch(Context ctx, Runtime* rt) {
    return
      Table::with_columns_attached(
        ctx,
        rt,
        m_args.h5_path.value(),
        ms_root,
        {{&m_tbl_main,
         {COLUMN_NAME(MS_MAIN, ANTENNA1),
          COLUMN_NAME(MS_MAIN, DATA_DESC_ID),
          COLUMN_NAME(MS_MAIN, FEED1),
          COLUMN_NAME(MS_MAIN, TIME),
          COLUMN_NAME(MS_MAIN, UVW)},
          {}},
         {&m_tbl_data_description,
          {COLUMN_NAME(MS_DATA_DESCRIPTION, SPECTRAL_WINDOW_ID)},
          {}},
         {&m_tbl_antenna,
          {COLUMN_NAME(MS_ANTENNA, MOUNT),
           COLUMN_NAME(MS_ANTENNA, POSITION)},
          {}}},
        [this](Context c, Runtime *r, std::unordered_map<std::string, Table*>& tbls) {
          Table* tbl_main = tbls[MSTable<MS_MAIN>::name];
          Table* tbl_data_description = tbls[MSTable<MS_DATA_DESCRIPTION>::name];
          Table* tbl_antenna = tbls[MSTable<MS_ANTENNA>::name];

          auto antenna1 =
            tbl_main->column(c, r, COLUMN_NAME(MS_MAIN, ANTENNA1));
          auto data_desc =
            tbl_main->column(c, r, COLUMN_NAME(MS_MAIN, DATA_DESC_ID));
          auto feed1 =
            tbl_main->column(c, r, COLUMN_NAME(MS_MAIN, FEED1));
          auto time =
            tbl_main->column(c, r, COLUMN_NAME(MS_MAIN, TIME));
          auto uvw =
            tbl_main->column(c, r, COLUMN_NAME(MS_MAIN, UVW));

          auto spw =
            tbl_data_description->column(
              c,
              r,
              COLUMN_NAME(MS_DATA_DESCRIPTION, SPECTRAL_WINDOW_ID));

          auto mount =
            tbl_antenna->column(c, r, COLUMN_NAME(MS_ANTENNA, MOUNT));
          auto position =
            tbl_antenna->column(c, r, COLUMN_NAME(MS_ANTENNA, POSITION));

          // using partition_on_axes() to get the index space of the table index
          // axes (a.k.a. row index space), but this also provides the logical
          // regions needed to describe further partitions on those axes
          auto row_part =
            antenna1.partition_on_axes(c, r, tbl_main->index_axes(c, r));
          IndexSpace row_is =
            r->get_index_partition_color_space_name(row_part.index_partition);
          // partition the row index space
          IndexPartition ip =
            partition_over_all_cpus(c, r, row_is, m_args.min_block.value());
          IndexSpace cs = r->get_index_partition_color_space_name(c, ip);

          // create the result logical region
          FieldSpace fs = r->create_field_space(c);
          {
            FieldAllocator fa = r->create_field_allocator(c, fs);
            fa.allocate_field(
              sizeof(PARALLACTIC_ANGLE_TYPE),
              PARALLACTIC_ANGLE_FID);
          }
          r->attach_name(fs, PARALLACTIC_ANGLE_FID, "parallactic_angle");
          LogicalRegion result = r->create_logical_region(c, row_is, fs);
          r->attach_name(result, "row_aux_fields");

          // use an index task launched over the partition of the main table
          // rows
          std::vector<RegionRequirement> reqs;
          // use the logical regions containing the table axes uid and row axes
          // that were provided by the above call to partition_on_axes to create
          // a ColumnPartition based on the table row partition.
          auto col_part = // don't call col_part.destroy()!
            ColumnPartition(row_part.axes_uid_lr, row_part.axes_lr, ip);
          for (auto& col : {antenna1, data_desc, feed1, time, uvw}) {
            // project the row partition onto the entire set of column axes
            auto cp = col.projected_column_partition(c, r, col_part);
            auto lp =
              r->get_logical_partition(c, col.values_lr, cp.index_partition);
            RegionRequirement req(lp, 0, READ_ONLY, EXCLUSIVE, col.values_lr);
            req.add_field(Column::VALUE_FID);
            reqs.push_back(req);
            r->destroy_logical_partition(c, lp);
            cp.destroy(c, r);
          }
          for (auto& col : {spw, mount, position}) {
            RegionRequirement
              req(col.values_lr, READ_ONLY, EXCLUSIVE, col.values_lr);
            req.add_field(Column::VALUE_FID);
            reqs.push_back(req);
          }
          for (auto& nm : TableColumns<MS_FEED>::column_names) {
            auto col = m_tbl_feed.column(c, r, nm);
            RegionRequirement
              req(col.values_lr, READ_ONLY, EXCLUSIVE, col.values_lr);
            req.add_field(Column::VALUE_FID);
            reqs.push_back(req);
          }
          std::vector<unsigned> mr_indexes;
          for (auto& nm :
                 {COLUMN_NAME(MS_FEED, TIME),
                     COLUMN_NAME(MS_FEED, BEAM_OFFSET),
                     COLUMN_NAME(MS_FEED, POSITION)}) {
            mr_indexes.push_back(reqs.size());
            auto col = m_tbl_feed.column(c, r, nm);
            auto [mreq, ovreq] = col.meas_ref.requirements(READ_ONLY);
            reqs.push_back(mreq);
            if (ovreq)
              reqs.push_back(ovreq.value());
          }
          {
            mr_indexes.push_back(reqs.size());
            auto [mreq, ovreq] = position.meas_ref.requirements(READ_ONLY);
            reqs.push_back(mreq);
            if (ovreq)
              reqs.push_back(ovreq.value());
          }
          TaskArgs args;
          args.feed_time_mr_index = mr_indexes[0];
          args.feed_beam_offset_mr_index = mr_indexes[1];
          args.feed_position_mr_index = mr_indexes[2];
          args.antenna_position_mr_index = mr_indexes[3];
          {
            LogicalPartition lp = r->get_logical_partition(c, result, ip);
            RegionRequirement req(lp, 0, WRITE_ONLY, EXCLUSIVE, result);
            req.add_field(PARALLACTIC_ANGLE_FID);
            reqs.push_back(req);
            r->destroy_logical_partition(c, lp);
          }
          IndexTaskLauncher init_task(
            COMPUTE_ROW_AUX_FIELDS_TASK_ID + row_is.get_dim() - 1,
            cs,
            TaskArgument(&args, sizeof(args)),
            ArgumentMap());
          for (auto& req : reqs)
            init_task.add_region_requirement(req);
          r->execute_index_space(c, init_task);

          r->destroy_index_space(c, cs);
          r->destroy_index_partition(c, ip);
          row_part.destroy(c, r, true);
          return result;
        });
  }

  static PARALLACTIC_ANGLE_TYPE
  parallactic_angle(
    const DataType<HYPERION_TYPE_INT>::ValueType& antenna1,
    const DataType<HYPERION_TYPE_INT>::ValueType& data_desc,
    const DataType<HYPERION_TYPE_INT>::ValueType& feed1,
    const DataType<HYPERION_TYPE_DOUBLE>::ValueType& time,
    const AntennaHelper::MountCode& antenna_mount,
    const std::array<cc::MPosition, 3>& antenna_position,
    const DataType<HYPERION_TYPE_INT>::ValueType& feed_beam_num_receptors,
    // feed_beam_offset dimensions: (feed_beam_num_receptors, 2);
    const DataType<HYPERION_TYPE_DOUBLE>::ValueType* feed_beam_offset,
    // feed_receptor_angle dimensions: (feed_beam_num_receptors)
    const DataType<HYPERION_TYPE_DOUBLE>::ValueType* feed_receptor_angle,
    const cc::MeasRef<cc::MEpoch>& feed_time_mr,
    const cc::MeasRef<cc::MDirection>& feed_beam_offset_mr,
    const cc::MeasRef<cc::MPosition>& feed_position_mr) {

    PARALLACTIC_ANGLE_TYPE result;
    switch (antenna_mount) {
    case AntennaHelper::EQUATORIAL:
      result = 0.0;
      break;
    case AntennaHelper::ALT_AZ:
      break;
    case AntennaHelper::NASMYTH_L:
      break;
    case AntennaHelper::NASMYTH_R:
      break;
    default:
      assert(false);
      break;
    }
    return result;
  }

  template <int ROW_DIM>
  static void
  base_impl(
    const Task* task,
    const std::vector<PhysicalRegion>& regions,
    Context ctx,
    Runtime *rt) {

    const TaskArgs* args = static_cast<const TaskArgs*>(task->args);

    const ROAccessor<DataType<HYPERION_TYPE_INT>::ValueType, ROW_DIM>
      antenna1(regions[0], Column::VALUE_FID);
    const ROAccessor<DataType<HYPERION_TYPE_INT>::ValueType, ROW_DIM>
      data_desc(regions[1], Column::VALUE_FID);
    const ROAccessor<DataType<HYPERION_TYPE_INT>::ValueType, ROW_DIM>
      feed1(regions[2], Column::VALUE_FID);
    const ROAccessor<DataType<HYPERION_TYPE_DOUBLE>::ValueType, ROW_DIM>
      time(regions[3], Column::VALUE_FID);
    const ROAccessor<DataType<HYPERION_TYPE_DOUBLE>::ValueType, ROW_DIM + 1>
      uvw(regions[4], Column::VALUE_FID);
    const ROAccessor<DataType<HYPERION_TYPE_INT>::ValueType, 1>
      spw(regions[5], Column::VALUE_FID);
    const ROAccessor<DataType<HYPERION_TYPE_STRING>::ValueType, 1>
      antenna_mount(regions[6], Column::VALUE_FID);
    const ROAccessor<DataType<HYPERION_TYPE_DOUBLE>::ValueType, 2>
      antenna_position(regions[7], Column::VALUE_FID);
#define FEED_COL(var, typ, col)                                       \
    const ROAccessor<DataType<HYPERION_TYPE_##typ>::ValueType,        \
                     TableColumns<MS_FEED>::index_axes.size()         \
                     + TableColumns<MS_FEED>::element_rank[           \
                       TableColumns<MS_FEED>::col]>                   \
      var(regions[8 + TableColumns<MS_FEED>::col], Column::VALUE_FID)

    FEED_COL(feed_num_receptors, INT, NUM_RECEPTORS);
    FEED_COL(feed_beam_offset, DOUBLE, BEAM_OFFSET);
    FEED_COL(feed_pol_response, COMPLEX, POL_RESPONSE);
    FEED_COL(feed_position, DOUBLE, POSITION);
    FEED_COL(feed_receptor_angle, DOUBLE, RECEPTOR_ANGLE);
#undef FEED_COL

    MeasRef::DataRegions drs;
    drs.metadata = regions[args->feed_time_mr_index];
    if (args->feed_beam_offset_mr_index > args->feed_time_mr_index + 1)
      drs.values = regions[args->feed_time_mr_index + 1];
    auto feed_time_mr =
      MeasRef::make<casacore::MEpoch>(rt, drs).value();

    drs.metadata = regions[args->feed_beam_offset_mr_index];
    if (args->feed_position_mr_index > args->feed_beam_offset_mr_index + 1)
      drs.values = regions[args->feed_beam_offset_mr_index + 1];
    auto feed_beam_offset_mr =
      MeasRef::make<casacore::MDirection>(rt, drs).value();

    drs.metadata = regions[args->feed_position_mr_index];
    if (args->antenna_position_mr_index > args->feed_position_mr_index + 1)
      drs.values = regions[args->feed_position_mr_index + 1];
    auto feed_position_mr =
      MeasRef::make<casacore::MPosition>(rt, drs).value();

    drs.metadata = regions[args->antenna_position_mr_index];
    if (regions.size() > args->antenna_position_mr_index + 1)
      drs.values = regions[args->antenna_position_mr_index + 1];
    auto antenna_position_mr =
      MeasRef::make<casacore::MPosition>(rt, drs).value();

    const WOAccessor<PARALLACTIC_ANGLE_TYPE, ROW_DIM>
      pa(regions.back(), PARALLACTIC_ANGLE_FID);

    cc::MeasFrame antFrame; // !
    // Set up the frame for epoch and antenna position. We will
    // adjust this to effect the coordinate transformations
    antFrame.set(cc::MEpoch(), cc::MPosition(), cc::MDirection());
    cc::MDirection::Ref haDecRef(cc::MDirection::HADEC, antFrame);
    // Make the HADec pole as expressed in HADec. The pole is the default.
    cc::MDirection haDecPole;
    haDecPole.set(haDecRef);
    // Set up the machines to convert to AzEl, HADec and LAST
    cc::MDirection::Convert cRADecToAzEl;
    cRADecToAzEl.set(
      cc::MDirection(), 
      cc::MDirection::Ref(cc::MDirection::AZEL, antFrame));
    cc::MDirection::Convert cHADecToAzEl;
    cHADecToAzEl.set(
      haDecPole,
      cc::MDirection::Ref(cc::MDirection::AZEL, antFrame));
    cc::MDirection::Convert cRADecToHADec;
    cRADecToHADec.set(cc::MDirection(), haDecRef);
    cc::MEpoch::Convert cUTCToLAST; // !
    cUTCToLAST.set(cc::MEpoch(), cc::MEpoch::Ref(cc::MEpoch::LAST, antFrame));
    // set up the velocity conversion with zero velocity in the TOPO/antenna 
    // frame. We'll use this to compute the observatory velocity in another
    // frame (often LSR).
    cc::MRadialVelocity::Convert cTOPOToLSR;
    cTOPOToLSR.set(
      cc::MRadialVelocity(
        cc::MVRadialVelocity(0.0),
        cc::MRadialVelocity::Ref(cc::MRadialVelocity::TOPO, antFrame)),
      cc::MRadialVelocity::Ref(cc::MRadialVelocity::LSRK));
    // radialVelocityType = cc::MRadialVelocity::LSRK;
    // frqref = cc::MFrequency::Ref(cc::MFrequency::LSRK);
    // velref = cc::MDoppler::Ref(cc::MDoppler::RADIO);
    // restFreq = cc::Quantity(0.0, "Hz");

    for (PointInDomainIterator<ROW_DIM>
           pid(
             rt->get_index_space_domain(
               task->regions.back().region.get_index_space()));
         pid();
         pid++) {
      coord_t feed_row_index[3] =
        {antenna1[*pid], feed1[*pid], spw[data_desc[*pid]]};
      const coord_t feed_beam_offset_index[5] =
        {feed_row_index[0], feed_row_index[1], feed_row_index[2], 0, 0};
      const coord_t feed_receptor_angle_index[4] =
        {feed_row_index[0], feed_row_index[1], feed_row_index[2], 0};
      coord_t antenna_row_index = antenna1[*pid];
      std::array<cc::MPosition, 3> antenna_mposition{
        cc::MPosition(
          cc::Quantity(
            antenna_position[Point<2>({antenna_row_index, 0})], "m"),
          *antenna_position_mr),
        cc::MPosition(
          cc::Quantity(
            antenna_position[Point<2>({antenna_row_index, 1})], "m"),
          *antenna_position_mr),
        cc::MPosition(
          cc::Quantity(
            antenna_position[Point<2>({antenna_row_index, 2})], "m"),
          *antenna_position_mr)
      };
      pa[*pid] =
        parallactic_angle(
          antenna1[*pid],
          data_desc[*pid],
          feed1[*pid],
          time[*pid],
          AntennaHelper::mount_code(antenna_mount[Point<1>(antenna_row_index)]),
          antenna_mposition,
          feed_num_receptors[Point<3>(feed_row_index)],
          feed_beam_offset.ptr(Point<5>(feed_beam_offset_index)),
          feed_receptor_angle.ptr(Point<4>(feed_receptor_angle_index)),
          *feed_time_mr,
          *feed_beam_offset_mr,
          *feed_position_mr);
    }
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
    HYPERION_FOREACH_N_LESS_MAX(REG_TASK);
#undef REG_TASK
  }

private:

  gridder::Args<gridder::VALUE_ARGS> m_args;

  Table m_tbl_main;

  Table m_tbl_data_description;

  Table m_tbl_antenna;

  Table m_tbl_feed;

  struct TaskArgs {
    unsigned feed_position_mr_index;
    unsigned feed_beam_offset_mr_index;
    unsigned feed_time_mr_index;
    unsigned antenna_position_mr_index;
  };
};

template <typename CLASSIFY_ANTENNAS_TASK>
class TopLevelTask {
public:

  static const constexpr char* TASK_NAME = "TopLevelTask";

  static void
  base_impl(
    const Legion::Task*,
    const std::vector<Legion::PhysicalRegion>&,
    Legion::Context ctx,
    Legion::Runtime* rt) {

    hyperion::register_tasks(ctx, rt);

    // process command line arguments
    std::optional<gridder::Args<gridder::VALUE_ARGS>> gridder_args;
    {
      const Legion::InputArgs& input_args = Legion::Runtime::get_input_args();
      gridder::Args<gridder::OPT_STRING_ARGS> some_str_args = default_config();
      if (gridder::get_args(input_args, some_str_args)) {
        if (!some_str_args.h5_path) {
          std::cerr << "Path to HDF5 data [--"
                    << some_str_args.h5_path.tag
                    << " option] is required, but missing from arguments"
                    << std::endl;
          return;
        }
        std::optional<gridder::Args<gridder::STRING_ARGS>> str_args =
          some_str_args.as_complete();
        assert(str_args);
        try {
          // parse argument values as YAML values, so convert
          // gridder::Args<gridder::STRING_ARGS> to
          // gridder::Args<gridder::VALUE_ARGS> via YAML
          gridder_args = gridder::as_args(str_args.value().as_node());
        } catch (const YAML::Exception& e) {
          std::cerr << "Failed to parse some configuration values: " << std::endl
                    << e.what()
                    << std::endl;
          return;
        }
        auto errstr = gridder::validate_args(gridder_args.value());
        if (errstr) {
          std::cerr << errstr.value() << std::endl;
          return;
        }
      }
    }
    if (!gridder_args)
      return;
    gridder::Args<gridder::VALUE_ARGS>* g_args = &gridder_args.value();
    g_args->pa_step = std::abs(g_args->pa_step.value());

    if (g_args->echo.value())
      std::cout << "*Effective parameters*" << std::endl
                << g_args->as_node() << std::endl;

    // initialize Tables used by gridder from HDF5 file
    MSTables mstables[] = {
      MS_ANTENNA,
      MS_DATA_DESCRIPTION,
      MS_FEED,
      MS_MAIN,
      MS_POLARIZATION,
      MS_SPECTRAL_WINDOW
    };
    std::unordered_map<MSTables, Table> tables;
    for (auto& mst : mstables)
      tables[mst] = init_table(ctx, rt, g_args->h5_path.value(), ms_root, mst);

    // re-index some tables
    std::unordered_map<MSTables, Table> itables;
    {
      std::unordered_map<MSTables, Future> fs;
      fs[MS_FEED] =
        tables[MS_FEED]
        .with_columns_attached(
          ctx,
          rt,
          g_args->h5_path.value(),
          ms_root,
          std::unordered_set<std::string>(
            TableColumns<MS_FEED>::column_names.begin(),
            TableColumns<MS_FEED>::column_names.end()),
          {},
          [](Context c, Runtime* r, const Table* tb) {
            std::vector<MSTable<MS_FEED>::Axes> iaxes(
              TableColumns<MS_FEED>::index_axes.begin(),
              TableColumns<MS_FEED>::index_axes.end());
            return tb->reindexed(c, r, iaxes, false);
          });
      // for convenience, just wait for all reindexing to finish; may want to
      // change this and use the futures directly
      for (auto& [nm, f] : fs)
        itables[nm] = f.template get_result<Table>();
    }

    // create region mapping antenna index to antenna class
    //
    // note that ClassifyAntennasTask uses an index space from the antenna
    // table for the region it creates
    LogicalRegion antenna_classes =
      tables[MS_ANTENNA]
      .with_columns_attached(
        ctx,
        rt,
        g_args->h5_path.value(),
        ms_root,
        std::unordered_set<std::string>(
          CLASSIFY_ANTENNAS_TASK::column_names.begin(),
          CLASSIFY_ANTENNAS_TASK::column_names.end()),
        {},
        [](Context c, Runtime* r, const Table* tb) {
          CLASSIFY_ANTENNAS_TASK task(*tb);
          return task.dispatch(c, r);
        });

    // create vector of parallactic angle values
    PAIntervals pa_intervals =
      PAIntervals::create(ctx, rt, g_args->pa_step.value(), 0.0f);

    // compute auxiliary row-wise data
    LogicalRegion row_aux;
    {
      ComputeRowAuxFieldsTask task(
        *g_args,
        tables[MS_MAIN],
        tables[MS_DATA_DESCRIPTION],
        tables[MS_ANTENNA],
        itables[MS_FEED]);
      row_aux = task.dispatch(ctx, rt);
    }
    {
      RegionRequirement req(row_aux, READ_ONLY, EXCLUSIVE, row_aux);
      req.add_field(ComputeRowAuxFieldsTask::PARALLACTIC_ANGLE_FID);
      auto pr = rt->map_region(ctx, req);
      const ROAccessor<PARALLACTIC_ANGLE_TYPE, 1>
        pa(pr, ComputeRowAuxFieldsTask::PARALLACTIC_ANGLE_FID);
      for (PointInDomainIterator<1>
             pid(rt->get_index_space_domain(row_aux.get_index_space()));
           pid();
           pid++)
        std::cout << *pid << ": " << pa[*pid] << std::endl;
    }

    // TODO: the rest goes here

    // clean up
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
  hyperion::preregister_all();
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
