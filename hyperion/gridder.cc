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
#include <hyperion/MeasRefContainer.h>

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

#define PARALLACTIC_ANGLE_TYPE float
#define PARALLACTIC_360 ((PARALLACTIC_ANGLE_TYPE)360.0)

#define AUTO_W_PROJ_PLANES_VALUE -1
#define INVALID_W_PROJ_PLANES_VALUE -2
#define INVALID_MIN_BLOCK_SIZE_VALUE 0

#define FS CXX_FILESYSTEM_NAMESPACE

using namespace hyperion;
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
  LAST_POINT_REDOP=100, // reserve HYPERION_MAX_DIM ids from here
};

static const char ms_root[] = "/";

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

typedef enum {
  VALUE_ARGS,
  STRING_ARGS,
  OPT_VALUE_ARGS,
  OPT_STRING_ARGS} gridder_args_t;

template <gridder_args_t G>
struct ArgsCompletion {
  static const gridder_args_t val;
};
template <>
struct ArgsCompletion<VALUE_ARGS> {
  static const constexpr gridder_args_t val = VALUE_ARGS;
};
template <>
struct ArgsCompletion<STRING_ARGS> {
  static const constexpr gridder_args_t val = STRING_ARGS;
};
template <>
struct ArgsCompletion<OPT_VALUE_ARGS> {
  static const constexpr gridder_args_t val = VALUE_ARGS;
};
template <>
struct ArgsCompletion<OPT_STRING_ARGS> {
  static const constexpr gridder_args_t val = STRING_ARGS;
};

template <typename T, bool OPT, const char* const* TAG, gridder_args_t G>
struct GridderArgType {
  static const constexpr char* tag = *TAG;
};
template <typename T, const char* const* TAG>
struct GridderArgType<T, false, TAG, VALUE_ARGS> {
  typedef T type;

  T val;

  static const constexpr char* tag = *TAG;

  T
  value() const {
    return val;
  }

  void
  operator=(const T& t) {
    val = t;
  }

  operator bool() const {
    return true;
  }
};
template <typename T, const char* const* TAG>
struct GridderArgType<T, true, TAG, VALUE_ARGS> {
  typedef std::optional<T> type;

  std::optional<T> val;

  static const constexpr char* tag = *TAG;

  T
  value() const {
    return val.value();
  }

  void
  operator=(const T& t) {
    val = t;
  }

  void
  operator=(const std::optional<T>& ot) {
    val = ot;
  }

  operator bool() const {
    return val.has_value();
  }
};
template <typename T, const char* const* TAG>
struct GridderArgType<T, false, TAG, STRING_ARGS> {
  typedef std::string type;

  std::string val;

  static const constexpr char* tag = *TAG;

  std::string
  value() const {
    return val;
  }

  void
  operator=(const std::string& str) {
    val = str;
  }

  operator bool() const {
    return true;
  }
};
template <typename T, const char* const* TAG>
struct GridderArgType<T, true, TAG, STRING_ARGS> {
  typedef std::optional<std::string> type;

  std::optional<std::string> val;

  static const constexpr char* tag = *TAG;

  std::string
  value() const {
    return val.value();
  }

  void
  operator=(const std::string& str) {
    val = str;
  }

  void
  operator=(const std::optional<std::string>& ostr) {
    val = ostr;
  }

  operator bool() const {
    return val.has_value();
  }
};
template <typename T, const char* const* TAG>
struct GridderArgType<T, false, TAG, OPT_VALUE_ARGS> {
  typedef std::optional<T> type;

  std::optional<T> val;

  static const constexpr char* tag = *TAG;

  T
  value() const {
    return val;
  }

  void
  operator=(const T& t) {
    val = t;
  }

  void
  operator=(const std::optional<T>& ot) {
    val = ot;
  }

  operator bool() const {
    return val.has_value();
  }
};
template <typename T, const char* const* TAG>
struct GridderArgType<T, true, TAG, OPT_VALUE_ARGS> {
  typedef std::optional<T> type;

  std::optional<T> val;

  static const constexpr char* tag = *TAG;

  T
  value() const {
    return val.value();
  }

  void
  operator=(const T& t) {
    val = t;
  }

  void
  operator=(const std::optional<T>& ot) {
    val = ot;
  }

  operator bool() const {
    return val.has_value();
  }
};
template <typename T, const char* const* TAG>
struct GridderArgType<T, false, TAG, OPT_STRING_ARGS> {
  typedef std::optional<std::string> type;

  std::optional<std::string> val;

  static const constexpr char* tag = *TAG;

  std::string
  value() const {
    return val.value();
  }

  void
  operator=(const std::string& str) {
    val = str;
  }

  void
  operator=(const std::optional<std::string>& ostr) {
    val = ostr;
  }

  operator bool() const {
    return val.has_value();
  }
};
template <typename T, const char* const* TAG>
struct GridderArgType<T, true, TAG, OPT_STRING_ARGS> {
  typedef std::optional<std::string> type;

  std::optional<std::string> val;

  static const constexpr char* tag = *TAG;

  std::string
  value() const {
    return val.value();
  }

  void
  operator=(const std::string& str) {
    val = str;
  }

  void
  operator=(const std::optional<std::string>& ostr) {
    val = ostr;
  }

  operator bool() const {
    return val.has_value();
  }
};

struct GridderArgsBase {
  static const constexpr char* h5_path_tag = "h5";
  static const constexpr char* config_path_tag = "configuration";
  static const constexpr char* echo_tag = "echo";
  static const constexpr char* min_block_tag = "min_block";
  static const constexpr char* pa_step_tag = "pa_step";
  static const constexpr char* w_planes_tag = "w_proj_planes";

  static const std::vector<std::string>&
  tags() {
    static const std::vector<std::string> result{
      h5_path_tag,
      config_path_tag,
      echo_tag,
      min_block_tag,
      pa_step_tag,
      w_planes_tag
    };
    return result;
  }
};

template <gridder_args_t G>
struct GridderArgs
  : public GridderArgsBase {

  GridderArgType<FS::path, false, &h5_path_tag, G> h5_path;
  GridderArgType<FS::path, true, &config_path_tag, G> config_path;
  GridderArgType<bool, false, &echo_tag, G> echo;
  GridderArgType<size_t, false, &min_block_tag, G> min_block;
  GridderArgType<PARALLACTIC_ANGLE_TYPE, false, &pa_step_tag, G> pa_step;
  GridderArgType<int, false, &w_planes_tag, G> w_planes;

  GridderArgs() {}

  GridderArgs(
    const typename decltype(h5_path)::type& h5_path_,
    const typename decltype(config_path)::type& config_path_,
    const typename decltype(echo)::type& echo_,
    const typename decltype(min_block)::type& min_block_,
    const typename decltype(pa_step)::type& pa_step_,
    const typename decltype(w_planes)::type& w_planes_) {

    h5_path = h5_path_;
    config_path = config_path_;
    echo = echo_;
    min_block = min_block_;
    pa_step = pa_step_;
    w_planes = w_planes_;
  }

  bool
  is_complete() const {
    return
      h5_path
      && echo
      && min_block
      && pa_step
      && w_planes;
  }

  std::optional<GridderArgs<ArgsCompletion<G>::val>>
  as_complete() const {
    std::optional<GridderArgs<ArgsCompletion<G>::val>> result;
    if (is_complete())
      result =
        std::make_optional<GridderArgs<ArgsCompletion<G>::val>>(
          h5_path.value(),
          (config_path
           ? config_path.value()
           : std::optional<std::string>()),
          echo.value(),
          min_block.value(),
          pa_step.value(),
          w_planes.value());
    return result;
  }

  YAML::Node
  as_node() const {
    YAML::Node result;
    if (h5_path)
      result[h5_path.tag] = h5_path.value().c_str();
    if (config_path)
      result[config_path.tag] = config_path.value().c_str();
    if (echo)
      result[echo.tag] = echo.value();
    if (min_block)
      result[min_block.tag] = min_block.value();
    if (pa_step)
      result[pa_step.tag] = pa_step.value();
    if (w_planes)
      result[w_planes.tag] = w_planes.value();
    return result;
  }
};

GridderArgs<VALUE_ARGS>
as_args(YAML::Node&& node) {
  FS::path h5_path = node[GridderArgsBase::h5_path_tag].as<std::string>();
  std::optional<FS::path> config_path;
  if (node[GridderArgsBase::config_path_tag])
    config_path = node[GridderArgsBase::config_path_tag].as<std::string>();
  bool echo = node[GridderArgsBase::echo_tag].as<bool>();
  size_t min_block = node[GridderArgsBase::min_block_tag].as<size_t>();
  PARALLACTIC_ANGLE_TYPE pa_step =
    node[GridderArgsBase::pa_step_tag].as<PARALLACTIC_ANGLE_TYPE>();
  int w_planes = node[GridderArgsBase::w_planes_tag].as<int>();
  return
    GridderArgs<VALUE_ARGS>(
      h5_path,
      config_path,
      echo,
      min_block,
      pa_step,
      w_planes);
}

static const GridderArgs<OPT_STRING_ARGS>&
default_config() {
  static bool computed = false;
  static GridderArgs<OPT_STRING_ARGS> result;
  if (!computed) {
    result.echo = std::string("false");
    result.min_block = std::string("100000");
    result.pa_step = std::string("360.0");
    result.w_planes = std::string("1");
    computed = true;
  }
  return result;
}

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
  static const constexpr std::array<const char*,NUM_COLUMNS> column_names = {
    "NAME",
    "STATION",
    "TYPE",
    "MOUNT",
    "DISH_DIAMETER"
  };
  static const constexpr char* table_name = MSTable<MS_ANTENNA>::name;
};

template <>
struct TableColumns<MS_DATA_DESCRIPTION>
  : public WithColumnLookup<TableColumns<MS_DATA_DESCRIPTION>> {
  typedef enum {
    SPECTRAL_WINDOW_ID,
    POLARIZATION_ID,
    NUM_COLUMNS
  } col;
  static const constexpr std::array<const char*,NUM_COLUMNS> column_names = {
    "SPECTRAL_WINDOW_ID",
    "POLARIZATION_ID"
  };
  static const constexpr char* table_name = MSTable<MS_DATA_DESCRIPTION>::name;
};

template <>
struct TableColumns<MS_FEED>
  : public WithColumnLookup<TableColumns<MS_FEED>> {
  typedef enum {
    ANTENNA_ID,
    FEED_ID,
    SPECTRAL_WINDOW_ID,
    TIME,
    INTERVAL,
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
  static const constexpr char* table_name = MSTable<MS_FEED>::name;
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
  static const constexpr char* table_name = MSTable<MS_MAIN>::name;
};

template <>
struct TableColumns<MS_POLARIZATION>
  : public WithColumnLookup<TableColumns<MS_POLARIZATION>> {
  typedef enum {
    NUM_CORR,
    NUM_COLUMNS
  } col;
  static const constexpr std::array<const char*,NUM_COLUMNS> column_names = {
    "NUM_CORR",
  };
  static const constexpr char* table_name = MSTable<MS_POLARIZATION>::name;
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
  static const constexpr std::array<const char*,NUM_COLUMNS> column_names = {
    "NUM_CHAN",
    "REF_FREQUENCY",
    "CHAN_FREQ",
    "CHAN_WIDTH"
  };
  static const constexpr char* table_name = MSTable<MS_SPECTRAL_WINDOW>::name;
};

#define COLUMN_NAME(T, C) TableColumns<T>::column_names[TableColumns<T>::C]

Table
init_table(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const FS::path& ms,
  const std::string& root,
  const MeasRefContainer& ms_meas_ref,
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
    HYPERION_FOREACH_MSTABLE(INIT);
  }
  return
    hyperion::hdf5::init_table(
      ctx,
      rt,
      ms,
      root + table_name,
      columns,
      ms_meas_ref);
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
      RegionRequirement req(result, WRITE_ONLY, EXCLUSIVE, result);
      req.add_field(antenna_class_fid);
      requirements.push_back(req);
    }
    TaskLauncher launcher(T::TASK_ID, TaskArgument(NULL, 0));
    for (auto& req : requirements)
      launcher.add_region_requirement(req);
    rt->execute_task(ctx, launcher);
    return result;
  }

  static void
  base_impl(
    const Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context,
    Legion::Runtime* rt) {

    const ROAccessor<hyperion::string, 1>
      names(regions[TableColumns<MS_ANTENNA>::NAME], Column::VALUE_FID);
    const ROAccessor<hyperion::string, 1>
      stations(regions[TableColumns<MS_ANTENNA>::STATION], Column::VALUE_FID);
    const ROAccessor<hyperion::string, 1>
      types(regions[TableColumns<MS_ANTENNA>::TYPE], Column::VALUE_FID);
    const ROAccessor<hyperion::string, 1>
      mounts(regions[TableColumns<MS_ANTENNA>::MOUNT], Column::VALUE_FID);
    const ROAccessor<double, 1>
      diameters(
        regions[TableColumns<MS_ANTENNA>::DISH_DIAMETER],
        Column::VALUE_FID);
    const WOAccessor<unsigned, 1> antenna_classes(regions.back(), 0);

    for (PointInRectIterator<1>
           pir(
             rt->get_index_space_domain(
               task->regions.back().region.get_index_space()));
         pir();
         pir++)
      antenna_classes[*pir] =
        T::classify(
          names[*pir].val,
          stations[*pir].val,
          types[*pir].val,
          mounts[*pir].val,
          diameters[*pir]);
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
    const GridderArgs<VALUE_ARGS>& args,
    Table& tbl_main,
    Table& tbl_feed)
    : m_args(args)
    , m_tbl_main(tbl_main)
    , m_tbl_feed(tbl_feed) {
  }

  LogicalRegion
  dispatch(Context ctx, Runtime* rt) {
    return
      m_tbl_main
      .with_columns_attached(
        ctx,
        rt,
        m_args.h5_path.value(),
        ms_root,
        {COLUMN_NAME(MS_MAIN, ANTENNA1),
         COLUMN_NAME(MS_MAIN, ANTENNA2),
         COLUMN_NAME(MS_MAIN, DATA_DESC_ID),
         COLUMN_NAME(MS_MAIN, FEED1),
         COLUMN_NAME(MS_MAIN, FEED2),
         COLUMN_NAME(MS_MAIN, TIME),
         COLUMN_NAME(MS_MAIN, UVW)},
        {},
        [this](Context c, Runtime *r, Table* tbl_main) {

          auto antenna1 =
            tbl_main->column(c, r, COLUMN_NAME(MS_MAIN, ANTENNA1));
          auto antenna2 =
            tbl_main->column(c, r, COLUMN_NAME(MS_MAIN, ANTENNA2));
          auto data_desc =
            tbl_main->column(c, r, COLUMN_NAME(MS_MAIN, DATA_DESC_ID));
          auto feed1 =
            tbl_main->column(c, r, COLUMN_NAME(MS_MAIN, FEED1));
          auto feed2 =
            tbl_main->column(c, r, COLUMN_NAME(MS_MAIN, FEED2));
          auto time =
            tbl_main->column(c, r, COLUMN_NAME(MS_MAIN, TIME));
          auto uvw =
            tbl_main->column(c, r, COLUMN_NAME(MS_MAIN, UVW));

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
          for (auto& col :
                 {antenna1, antenna2, data_desc, feed1, feed2, time, uvw}) {
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
          for (auto& nm : TableColumns<MS_FEED>::column_names) {
            auto col = m_tbl_feed.column(c, r, nm);
            RegionRequirement
              req(col.values_lr, READ_ONLY, EXCLUSIVE, col.values_lr);
            req.add_field(Column::VALUE_FID);
            reqs.push_back(req);
          }
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
            TaskArgument(NULL, 0),
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
    const DataType<HYPERION_TYPE_INT>::ValueType& antenna2,
    const DataType<HYPERION_TYPE_INT>::ValueType& data_desc,
    const DataType<HYPERION_TYPE_INT>::ValueType& feed1,
    const DataType<HYPERION_TYPE_INT>::ValueType& feed2,
    const DataType<HYPERION_TYPE_DOUBLE>::ValueType& time) {

    return time * (antenna1 + antenna2 + data_desc + feed1 + feed2);
  }

  template <int ROW_DIM>
  static void
  base_impl(
    const Task* task,
    const std::vector<PhysicalRegion>& regions,
    Context,
    Runtime *rt) {


    const ROAccessor<DataType<HYPERION_TYPE_INT>::ValueType, ROW_DIM>
      antenna1(regions[0], Column::VALUE_FID);
    const ROAccessor<DataType<HYPERION_TYPE_INT>::ValueType, ROW_DIM>
      antenna2(regions[1], Column::VALUE_FID);
    const ROAccessor<DataType<HYPERION_TYPE_INT>::ValueType, ROW_DIM>
      data_desc(regions[2], Column::VALUE_FID);
    const ROAccessor<DataType<HYPERION_TYPE_INT>::ValueType, ROW_DIM>
      feed1(regions[3], Column::VALUE_FID);
    const ROAccessor<DataType<HYPERION_TYPE_INT>::ValueType, ROW_DIM>
      feed2(regions[4], Column::VALUE_FID);
    const ROAccessor<DataType<HYPERION_TYPE_DOUBLE>::ValueType, ROW_DIM>
      time(regions[5], Column::VALUE_FID);
    const ROAccessor<DataType<HYPERION_TYPE_DOUBLE>::ValueType, ROW_DIM + 1>
      uvw(regions[6], Column::VALUE_FID);
#define FEED_COL(var, typ, col)                                       \
    const ROAccessor<DataType<HYPERION_TYPE_##typ>::ValueType,        \
                     TableColumns<MS_FEED>::index_axes.size()         \
                     + TableColumns<MS_FEED>::element_rank[           \
                       TableColumns<MS_FEED>::col]>                   \
      var(regions[7 + TableColumns<MS_FEED>::col], Column::VALUE_FID)

    FEED_COL(feed_beam_offset, DOUBLE, BEAM_OFFSET);
    FEED_COL(feed_pol_response, COMPLEX, POL_RESPONSE);
    FEED_COL(feed_position, DOUBLE, POSITION);
    FEED_COL(feed_receptor_angle, DOUBLE, RECEPTOR_ANGLE);
#undef FEED_COL
    const WOAccessor<PARALLACTIC_ANGLE_TYPE, ROW_DIM>
      pa(regions.back(), PARALLACTIC_ANGLE_FID);
    for (PointInDomainIterator<ROW_DIM>
           pid(
             rt->get_index_space_domain(
               task->regions.back().region.get_index_space()));
         pid();
         pid++)
      pa[*pid] =
        parallactic_angle(
          antenna1[*pid],
          antenna2[*pid],
          data_desc[*pid],
          feed1[*pid],
          feed2[*pid],
          time[*pid]);
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

  GridderArgs<VALUE_ARGS> m_args;

  Table m_tbl_main;

  Table m_tbl_feed;
};

std::variant<std::forward_list<std::string>, GridderArgs<OPT_STRING_ARGS>>
read_config(const YAML::Node& config) {
  std::forward_list<std::string> invalid_tags;
  GridderArgs<OPT_STRING_ARGS> args;
  for (auto it = config.begin(); it != config.end(); ++it) {
    auto key = it->first.as<std::string>();
    auto val = it->second.as<std::string>();
    if (key == args.h5_path.tag)
      args.h5_path = val;
    else if (key == args.echo.tag)
      args.echo = val;
    else if (key == args.min_block.tag)
      args.min_block = val;
    else if (key == args.pa_step.tag)
      args.pa_step = val;
    else if (key == args.w_planes.tag)
      args.w_planes = val;
    else
      invalid_tags.push_front(key);  
  }
  if (invalid_tags.empty())
    return args;
  else
    return invalid_tags;
}

std::optional<std::string>
load_config(
  const FS::path& path,
  const std::string& config_provenance,
  GridderArgs<OPT_STRING_ARGS>& gridder_args) {

  std::optional<std::string> result;
  try {
    auto node = YAML::LoadFile(path);
    auto read_result = read_config(node);
    std::visit(
      overloaded {
        [&path, &config_provenance, &result]
        (std::forward_list<std::string>& invalid_tags) {
          std::ostringstream oss;
          oss << "Invalid configuration variable tags in file named by "
              << config_provenance << std::endl
              << " at '" << path << "': " << std::endl
              << invalid_tags.front();
          invalid_tags.pop_front();
          std::for_each(
            invalid_tags.begin(),
            invalid_tags.end(),
            [&oss](auto tg) { oss << ", " << tg; });
          oss << std::endl;
          result = oss.str();
        },
        [&gridder_args](GridderArgs<OPT_STRING_ARGS>& args) {
          if (args.h5_path)
            gridder_args.h5_path = args.h5_path.value();
          if (args.echo)
            gridder_args.echo = args.echo.value();
          if (args.min_block)
            gridder_args.min_block = args.min_block.value();
          if (args.pa_step)
            gridder_args.pa_step = args.pa_step.value();
          if (args.w_planes)
            gridder_args.w_planes = args.w_planes.value();
        }
      },
      read_result);
  } catch (const YAML::Exception& e) {
    std::ostringstream oss;
    oss << "Failed to parse configuration file named by "
        << config_provenance << std::endl
        << " at '" << path << "':" << std::endl
        << e.what()
        << std::endl;
    result = oss.str();
  }
  return result;
}

template <typename CLASSIFY_ANTENNAS_TASK>
class TopLevelTask {
public:

  static const constexpr char* TASK_NAME = "TopLevelTask";

  static bool
  get_args(
    const Legion::InputArgs& args,
    GridderArgs<OPT_STRING_ARGS>& gridder_args) {

    // first look for configuration file given by environment variable
    const char* env_config_pathname =
      std::getenv("HYPERION_GRIDDER_CONFIG");
    if (env_config_pathname != nullptr) {
      auto errs =
        load_config(
          env_config_pathname,
          "HYPERION_GRIDDER_CONFIG environment variable",
          gridder_args);
      if (errs) {
        std::cerr << errs.value();
        return false;
      }
    }

    // tokenize command line arguments
    std::vector<std::pair<std::string, std::string>> arg_pairs;
    for (int i = 1; i < args.argc;) {
      std::string tag = args.argv[i++];
      if (tag.substr(0, 2) == "--") {
        std::optional<std::string> val;
        tag = tag.substr(2);
        auto eq = tag.find('=');
        if (eq == std::string::npos) {
          if (i < args.argc) {
            val = args.argv[i++];
            if (val.value() == "=") {
              if (i < args.argc)
                val = args.argv[i++];
              else
                val.reset();
            }
          }
        } else {
          val = tag.substr(eq + 1);
          tag = tag.substr(0, eq);
          if (val.value().size() == 0) {
            if (i < args.argc)
              val = args.argv[i++];
            else
              val.reset();
          }
        }
        if (val) {
          arg_pairs.emplace_back(tag, val.value());
        } else {
          std::cerr << "No value provided for argument '"
                    << tag << "'" << std::endl;
          return false;
        }
      }
    }

    // apply variables from config file before the rest of the command line
    {
      auto config =
        std::find_if(
          arg_pairs.begin(),
          arg_pairs.end(),
          [&gridder_args](auto& kv) {
            return kv.first == gridder_args.config_path.tag;
          });
      if (config != arg_pairs.end()) {
        auto errs =
          load_config(
            config->second,
            (std::string("command line argument '")
             + gridder_args.config_path.tag + "' value"),
            gridder_args);
        if (errs) {
          std::cerr << errs.value();
          return false;
        }
      }
    }
    // apply other variables from the command line
    for (auto& [tag, val] : arg_pairs) {
      std::vector<std::string> matches;
      for (auto& tg : gridder_args.tags())
        if (tg.substr(0, tag.size()) == tag)
          matches.push_back(tg);
      if (matches.size() == 1) {
        auto match = matches.front();
        if (match == gridder_args.h5_path.tag)
          gridder_args.h5_path = val;
        else if (match == gridder_args.pa_step.tag)
          gridder_args.pa_step = val;
        else if (match == gridder_args.w_planes.tag)
          gridder_args.w_planes = val;
        else if (match == gridder_args.min_block.tag)
          gridder_args.min_block = val;
        else if (match == gridder_args.echo.tag)
          gridder_args.echo = val;
        else if (match == gridder_args.config_path.tag)
          gridder_args.config_path = val;
        else
          assert(false);
      }
      else {
        if (matches.size() == 0)
          std::cerr << "Unrecognized command line argument '"
                    << tag << "'" << std::endl;
        else
          std::cerr << "Command line argument '"
                    << tag << "' is a prefix of a more than one option"
                    << std::endl;
        return false;
      }
    }

    return true;
  }

  template <typename T>
  static std::ostringstream&
  arg_error(std::ostringstream& oss, const T& arg, const std::string& reason) {
    oss << "'" << arg.tag << "' value '" << arg.value()
        << "': " << reason << std::endl;
    return oss;
  }

  static std::optional<std::string>
  validate_args(const GridderArgs<VALUE_ARGS>& args) {

    std::ostringstream errs;
    if (!FS::is_regular_file(args.h5_path.value()))
      arg_error(errs, args.h5_path, "not a regular file");

    switch (std::fpclassify(args.pa_step.value())) {
    case FP_NORMAL:
      if (std::abs(args.pa_step.value()) > PARALLACTIC_360)
        arg_error(errs, args.pa_step, "not in valid range [-360, 360]");
      break;
    case FP_ZERO:
    case FP_SUBNORMAL:
      arg_error(errs, args.pa_step, "too small");
      break;
    default:
      arg_error(errs, args.pa_step, "invalid");
      break;
    }

    if (args.w_planes.value() <= INVALID_W_PROJ_PLANES_VALUE)
      arg_error(
        errs,
        args.w_planes,
        std::string("less than the minimum valid value of ")
        + std::to_string(INVALID_W_PROJ_PLANES_VALUE + 1));
    else if (args.w_planes.value() == AUTO_W_PROJ_PLANES_VALUE)
      arg_error(
        errs,
        args.w_planes,
        "automatic computation of the number of W-projection "
        "planes is unimplemented");

    if (args.min_block.value() == INVALID_MIN_BLOCK_SIZE_VALUE)
      arg_error(
        errs,
        args.min_block,
        "invalid, value must be at least one");

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

    hyperion::register_tasks(ctx, rt);

    // process command line arguments
    std::optional<GridderArgs<VALUE_ARGS>> gridder_args;
    {
      const Legion::InputArgs& input_args = Legion::Runtime::get_input_args();
      GridderArgs<OPT_STRING_ARGS> some_str_args = default_config();
      if (get_args(input_args, some_str_args)) {
        if (!some_str_args.h5_path) {
          std::cerr << "Path to HDF5 data [--"
                    << some_str_args.h5_path.tag
                    << " option] is required, but missing from arguments"
                    << std::endl;
          return;
        }
        std::optional<GridderArgs<STRING_ARGS>> str_args =
          some_str_args.as_complete();
        assert(str_args);
        try {
          // parse argument values as YAML values, so convert
          // GridderArgs<STRING_ARGS> to GridderArgs<VALUE_ARGS> via YAML
          gridder_args = as_args(str_args.value().as_node());
        } catch (const YAML::Exception& e) {
          std::cerr << "Failed to parse some configuration values: " << std::endl
                    << e.what()
                    << std::endl;
          return;
        }
        auto errstr = validate_args(gridder_args.value());
        if (errstr) {
          std::cerr << errstr.value() << std::endl;
          return;
        }
      }
    }
    if (!gridder_args)
      return;
    GridderArgs<VALUE_ARGS>* g_args = &gridder_args.value();
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
    MeasRefContainer ms_meas_ref; // TODO: account for measures at top level
    for (auto& mst : mstables)
      tables[mst] =
        init_table(ctx, rt, g_args->h5_path.value(), ms_root, ms_meas_ref, mst);

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
        {COLUMN_NAME(MS_ANTENNA, NAME),
         COLUMN_NAME(MS_ANTENNA, STATION),
         COLUMN_NAME(MS_ANTENNA, TYPE),
         COLUMN_NAME(MS_ANTENNA, MOUNT),
         COLUMN_NAME(MS_ANTENNA, DISH_DIAMETER)},
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
