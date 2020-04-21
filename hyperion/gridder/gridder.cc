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

#if LEGION_MAX_DIM < 7
# error "MAX_DIM too small"
#endif

#include <hyperion/hdf5.h>
#include <hyperion/utility.h>
#include <hyperion/Table.h>
#include <hyperion/MSTable.h>
#include <hyperion/PhysicalTable.h>
#include <hyperion/PhysicalColumn.h>
#include <hyperion/TableMapper.h>
#include <hyperion/MSMainTable.h>
#include <hyperion/MSFeedTable.h>
#include <hyperion/MSAntennaTable.h>
#include <hyperion/MSDataDescriptionTable.h>

#include <hyperion/gridder/gridder.h>
#include <hyperion/gridder/args.h>

#include <algorithm>
#include <array>
#include <experimental/array>
#include <cmath>
#include CXX_FILESYSTEM_HEADER
#include <iomanip>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <yaml-cpp/yaml.h>

using namespace hyperion;
using namespace Legion;

namespace cc = casacore;

#ifdef HYPERION_DEBUG
# define CHECK_BOUNDS true
#else
# define CHECK_BOUNDS false
#endif

enum {
  GRIDDER_TASK_ID,
  CLASSIFY_ANTENNAS_TASK_ID,
  COMPUTE_PARALLACTIC_ANGLES_TASK_ID,
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
    result.pa_block = std::string("1000");
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

#define FEED_AXES FEED_ANTENNA_ID, FEED_FEED_ID, FEED_SPECTRAL_WINDOW_ID

std::tuple<
  Table,
  std::unordered_map<std::string, std::string>>
init_table(Context ctx, Runtime* rt, hid_t loc, MSTables mst) {

  std::string table_name;
  switch (mst) {
#define INIT(TBL)                               \
    case (MS_##TBL): {                          \
      table_name = MSTable<MS_##TBL>::name;     \
      break;                                    \
    }
    HYPERION_FOREACH_MS_TABLE(INIT);
  }
  return hdf5::init_table(ctx, rt, loc, table_name);
}

typedef unsigned antenna_class_t;
const constexpr hyperion::TypeTag antenna_class_dt =
  ValueType<antenna_class_t>::DataType;
const char* antenna_class_column_name = "CLASS";
const Legion::FieldID antenna_class_fid =
  MSTableColumns<MS_ANTENNA>::user_fid_base;

antenna_class_t
trivially_classify_antenna(
  const char* /*name*/,
  const char* /*station*/,
  const char* /*type*/,
  const char* /*mount*/,
  double /*diameter*/) {
  return 0;
}

void
classify_antennas_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  auto [pt, rit, pit] =
    PhysicalTable::create(
      rt,
      task->regions.begin(),
      task->regions.end(),
      regions.begin(),
      regions.end())
    .value();
  assert(rit == task->regions.end());
  assert(pit == regions.end());
  MSAntennaTable antenna_table(pt);
  auto name =
    antenna_table.name<AffineAccessor>()
    .accessor<READ_ONLY, CHECK_BOUNDS>();;
  auto station =
    antenna_table.station<AffineAccessor>()
    .accessor<READ_ONLY, CHECK_BOUNDS>();
  auto type =
    antenna_table.type<AffineAccessor>()
    .accessor<READ_ONLY, CHECK_BOUNDS>();
  auto mount =
    antenna_table.mount<AffineAccessor>()
    .accessor<READ_ONLY, CHECK_BOUNDS>();
  auto dish_diameter =
    antenna_table.dish_diameter<AffineAccessor>()
    .accessor<READ_ONLY, CHECK_BOUNDS>();
  PhysicalColumnTD<antenna_class_dt, 1, 1, AffineAccessor>
    antenna_class_column(*pt.column(antenna_class_column_name).value());
  auto aclass = antenna_class_column.accessor<WRITE_ONLY, CHECK_BOUNDS>();
  for (PointInRectIterator<1> pir(antenna_class_column.rect());
       pir();
       pir++)
    aclass[*pir] =
      trivially_classify_antenna(
        name[*pir].val,
        station[*pir].val,
        type[*pir].val,
        mount[*pir].val,
        dish_diameter[*pir]);
}

constexpr hyperion::TypeTag parallactic_angle_dt =
  ValueType<PARALLACTIC_ANGLE_TYPE>::DataType;
const char* parallactic_angle_column_name = "PARANG";
const constexpr Legion::FieldID parallactic_angle_fid =
  MSTableColumns<MS_MAIN>::user_fid_base;

struct PAIntervals {

  PARALLACTIC_ANGLE_TYPE origin;
  PARALLACTIC_ANGLE_TYPE step;
  unsigned long num_steps;

  PAIntervals(PARALLACTIC_ANGLE_TYPE origin_, PARALLACTIC_ANGLE_TYPE step_)
    : origin(origin_)
    , step(step_)
    , num_steps(std::lrint(std::ceil(PARALLACTIC_360 / step))) {
  }

  std::optional<std::tuple<PARALLACTIC_ANGLE_TYPE, PARALLACTIC_ANGLE_TYPE>>
  pa(unsigned long i) const {
    std::optional<std::tuple<PARALLACTIC_ANGLE_TYPE, PARALLACTIC_ANGLE_TYPE>>
      result;
    if (i < num_steps) {
      auto lo = i * step;
      auto width = ((i == num_steps - 1) ? (PARALLACTIC_360 - lo) : step);
      result = {origin + lo, width};
    }
    return result;
  }

  unsigned long
  find(PARALLACTIC_ANGLE_TYPE pa) const {
    pa -= origin;
    pa -= std::floor(pa / PARALLACTIC_360) * PARALLACTIC_360;
    return std::lrint(std::floor(pa / step));
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
  {"", ALT_AZ},
  {"alt-az", ALT_AZ},
  {"ALT-AZ", ALT_AZ},
  {"alt-az+rotator", ALT_AZ},
  {"ALT-AZ+ROTATOR", ALT_AZ},
  {"equatorial", EQUATORIAL},
  {"EQUATORIAL", EQUATORIAL},
  {"x-y", XY},
  {"X-Y", XY},
  {"orbiting", ORBITING},
  {"ORBITING", ORBITING},
  {"alt-az+nasmyth-r", NASMYTH_R},
  {"ALT-AZ+NASMYTH-R", NASMYTH_R},
  {"alt-az+nasmyth-l", NASMYTH_L},
  {"ALT-AZ+NASMYTH-L", NASMYTH_L}};

void
compute_parallactic_angles_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  // main table columns
  auto [pt0, reqs0, prs0] =
    PhysicalTable::create(
      rt,
      task->regions.begin(),
      task->regions.end(),
      regions.begin(),
      regions.end())
    .value();
  MSMainTable<MAIN_ROW> main(pt0);
  typedef decltype(main)::C MainCols;
  auto main_antenna1_col = main.antenna1<AffineAccessor>();
  auto main_antenna1 =
    main_antenna1_col.accessor<READ_ONLY, CHECK_BOUNDS>();
  auto main_data_desc_id =
    main.data_desc_id<AffineAccessor>().accessor<READ_ONLY, CHECK_BOUNDS>();
  auto main_feed1 =
    main.feed1<AffineAccessor>().accessor<READ_ONLY, CHECK_BOUNDS>();
  auto main_time =
    main.time<AffineAccessor>().accessor<READ_ONLY, CHECK_BOUNDS>();
  auto main_time_meas =
    main.time_meas<AffineAccessor>().meas_accessor<READ_ONLY, CHECK_BOUNDS>(
      rt,
      MainCols::units.at(MainCols::col_t::MS_MAIN_COL_TIME));
  PhysicalColumnTD<
    parallactic_angle_dt,
    main.row_rank,
    main.row_rank,
    AffineAccessor> main_parallactic_angle_col(
      *pt0.column(parallactic_angle_column_name).value());
  auto main_parallactic_angle =
    main_parallactic_angle_col.accessor<WRITE_ONLY, CHECK_BOUNDS>();

  // data description table columns
  auto [pt1, reqs1, prs1] =
    PhysicalTable::create(rt, reqs0, task->regions.end(), prs0, regions.end())
    .value();
  MSDataDescriptionTable data_desc(pt1);
  auto dd_spectral_window_id =
    data_desc.spectral_window_id<AffineAccessor>()
    .accessor<READ_ONLY, CHECK_BOUNDS>();

  // antenna table columns
  auto [pt2, reqs2, prs2] =
    PhysicalTable::create(rt, reqs1, task->regions.end(), prs1, regions.end())
    .value();
  MSAntennaTable antenna(pt2);
  typedef decltype(antenna)::C AntennaCols;
  auto antenna_mount =
    antenna.mount<AffineAccessor>().accessor<READ_ONLY, CHECK_BOUNDS>();
  auto antenna_position =
    antenna.position_meas<AffineAccessor>()
    .meas_accessor<READ_ONLY, CHECK_BOUNDS>(
      rt,
      AntennaCols::units.at(AntennaCols::col_t::MS_ANTENNA_COL_POSITION));

  // feed table columns
  auto [pt3, reqs3, prs3] =
    PhysicalTable::create(rt, reqs2, task->regions.end(), prs2, regions.end())
    .value();
  MSFeedTable<FEED_AXES> feed(pt3);
  typedef decltype(feed)::C FeedCols;
  auto feed_time =
    feed.time_meas<AffineAccessor>().meas_accessor<READ_ONLY, CHECK_BOUNDS>(
      rt,
      FeedCols::units.at(FeedCols::col_t::MS_FEED_COL_TIME));
  auto feed_receptor_angle =
    feed.receptor_angle<AffineAccessor>().accessor<READ_ONLY, CHECK_BOUNDS>();

  cc::MeasFrame ant_frame;
  // Set up the frame for epoch and antenna position. We will
  // adjust this to effect the coordinate transformations
  ant_frame.set(cc::MEpoch(), cc::MPosition(), cc::MDirection());
  cc::MDirection::Ref ha_dec_ref(cc::MDirection::HADEC, ant_frame);
  // Make the HADec pole as expressed in HADec. The pole is the default.
  cc::MDirection ha_dec_pole;
  ha_dec_pole.set(ha_dec_ref);
  // Set up the machines to convert to AzEl, HADec and LAST
  cc::MDirection::Convert cvt_ra_dec_to_az_el;
  cvt_ra_dec_to_az_el.set(
    cc::MDirection(),
    cc::MDirection::Ref(cc::MDirection::AZEL, ant_frame));
  cc::MDirection::Convert cvt_ha_dec_to_az_el;
  cvt_ha_dec_to_az_el.set(
    ha_dec_pole,
    cc::MDirection::Ref(cc::MDirection::AZEL, ant_frame));
  cc::MEpoch::Convert cvt_utc_to_last;
  cvt_utc_to_last.set(cc::MEpoch(), cc::MEpoch::Ref(cc::MEpoch::LAST, ant_frame));

  std::optional<double> last_time;
  std::optional<int> last_antenna;
  std::optional<AntennaHelper::MountCode> last_mount;
  double par_angle = 0.0;
  for (PointInRectIterator<main.row_rank> row(main_antenna1_col.rect());
       row();
       row++) {
    bool ant_frame_changed = false;
    if (main_time[*row] != last_time.value_or(main_time[*row] + 1)) {
      last_time = main_time[*row];
      cc::MEpoch epoch = main_time_meas.read(*row);
      cvt_utc_to_last.setModel(epoch);
      ant_frame.resetEpoch(epoch);
      ant_frame_changed = true;
    }
    if (main_antenna1[*row] != last_antenna.value_or(main_antenna1[*row] + 1)) {
      last_antenna = main_antenna1[*row];
      ant_frame.resetPosition(antenna_position.read(last_antenna.value()));
      ant_frame_changed = true;
    }
    auto mount = AntennaHelper::mount_code(antenna_mount[last_antenna.value()]);
    if (!last_mount || mount != last_mount.value()) {
      last_mount = mount;
      ant_frame_changed = true;
    }
    if (mount != AntennaHelper::MountCode::EQUATORIAL) {
      if (ant_frame_changed) {
        // Now we can do the conversions using the machines
        auto ra_dec_in_az_el = cvt_ra_dec_to_az_el();
        auto ha_dec_pole_in_az_el = cvt_ha_dec_to_az_el();
        // Get the parallactic angle
        par_angle =
          ra_dec_in_az_el
          .getValue()
          .positionAngle(ha_dec_pole_in_az_el.getValue());
        switch (mount) {
        case AntennaHelper::MountCode::ALT_AZ:
          break;
        case AntennaHelper::MountCode::NASMYTH_R:
          // Get the parallactic angle
          par_angle += ra_dec_in_az_el.getAngle().getValue()[1];
          break;
        case AntennaHelper::MountCode::NASMYTH_L:
          // Get the parallactic angle
          par_angle -= ra_dec_in_az_el.getAngle().getValue()[1];
          break;
        default:
          assert(false);
          break;
        }
      }
    } else {
      par_angle = 0.0;
    }
    Point<4> ra_idx(
      main_antenna1[*row],
      main_feed1[*row],
      dd_spectral_window_id[main_data_desc_id[*row]],
      0);
    main_parallactic_angle[*row] = par_angle + feed_receptor_angle[ra_idx];
  }
}

// write values to antenna_class_column_name
void
init_antenna_classes(
  Context ctx,
  Runtime* rt,
  const PhysicalTable& antenna_table) {

  Column::Requirements default_colreqs = Column::default_requirements;
  default_colreqs.values.mapped = true;
  Column::Requirements class_colreq = Column::default_requirements;
  class_colreq.values = Column::Req{WRITE_ONLY, EXCLUSIVE, true};
  auto reqs =
    std::get<0>(
      antenna_table
      .requirements(
        ctx,
        rt,
        ColumnSpacePartition(),
        READ_ONLY,
        {{antenna_class_column_name, class_colreq}},
        default_colreqs));
  TaskLauncher task(
    CLASSIFY_ANTENNAS_TASK_ID,
    TaskArgument(NULL, 0),
    Predicate::TRUE_PRED,
    table_mapper);
  task.enable_inlining = true;
  for (auto& rq : reqs)
    task.add_region_requirement(rq);
  antenna_table.unmap_regions(ctx, rt);
  rt->execute_task(ctx, task);
  antenna_table.remap_regions(ctx, rt);
}


// write values to parallactic angle column
void
init_parallactic_angles(
  Context ctx,
  Runtime* rt,
  size_t block_size,
  const PhysicalTable& main_table,
  const PhysicalTable& data_description_table,
  const PhysicalTable& antenna_table,
  const Table& feed_table) {

  ColumnSpacePartition partition =
    main_table.partition_rows(ctx, rt, {block_size});
  IndexTaskLauncher task(
    COMPUTE_PARALLACTIC_ANGLES_TASK_ID,
    rt->get_index_partition_color_space_name(partition.column_ip),
    TaskArgument(NULL, 0),
    ArgumentMap(),
    Predicate::TRUE_PRED,
    false,
    table_mapper);

  Column::Requirements pareqs = Column::default_requirements;
  pareqs.values.privilege = WRITE_ONLY;
  auto [main_reqs, main_parts] =
    main_table
    .requirements(
      ctx,
      rt,
      partition,
      READ_ONLY,
      {{parallactic_angle_column_name, pareqs},
       {HYPERION_COLUMN_NAME(MAIN, ANTENNA1),
        Column::default_requirements},
       {HYPERION_COLUMN_NAME(MAIN, DATA_DESC_ID),
        Column::default_requirements},
       {HYPERION_COLUMN_NAME(MAIN, FEED1),
        Column::default_requirements},
       {HYPERION_COLUMN_NAME(MAIN, TIME),
        Column::default_requirements},
       {HYPERION_COLUMN_NAME(MAIN, UVW),
        Column::default_requirements}},
      std::nullopt);
  for (auto& rq : main_reqs)
    task.add_region_requirement(rq);
  main_table.unmap_regions(ctx, rt);

  auto dd_reqs =
    std::get<0>(
      data_description_table
      .requirements(
        ctx,
        rt,
        ColumnSpacePartition(),
        READ_ONLY,
        {{HYPERION_COLUMN_NAME(DATA_DESCRIPTION, SPECTRAL_WINDOW_ID),
          Column::default_requirements}},
        std::nullopt));
  for (auto& rq : dd_reqs)
    task.add_region_requirement(rq);
  data_description_table.unmap_regions(ctx, rt);

  auto ant_reqs =
    std::get<0>(
      antenna_table
      .requirements(
        ctx,
        rt,
        ColumnSpacePartition(),
        READ_ONLY,
        {{HYPERION_COLUMN_NAME(ANTENNA, MOUNT),
          Column::default_requirements},
         {HYPERION_COLUMN_NAME(ANTENNA, POSITION),
          Column::default_requirements}},
        std::nullopt));
  for (auto& rq : ant_reqs)
    task.add_region_requirement(rq);
  antenna_table.unmap_regions(ctx, rt);

  auto feed_reqs = std::get<0>(feed_table.requirements(ctx, rt));
  for (auto& rq : feed_reqs)
    task.add_region_requirement(rq);

  rt->execute_index_space(ctx, task);
  main_table.remap_regions(ctx, rt);
  data_description_table.remap_regions(ctx, rt);
  antenna_table.remap_regions(ctx, rt);

  for (auto& lp : main_parts)
    rt->destroy_logical_partition(ctx, lp);
}

template <typename gridder::args_t G>
void
show_help(const gridder::Args<G>& args) {
  std::ostringstream oss;
  oss << "Usage: gridder [LEGION OPTIONS] [GRIDDER OPTIONS]"
      << std::endl
      << "  valid GRIDDER OPTIONS (all options have values):"
      << std::endl;
  size_t max_tag_len = 0;
  std::map<std::string, std::string> h = args.help();
  for (auto& [tag, desc] : h)
    max_tag_len = std::max(tag.size(), max_tag_len);
  for (auto& [tag, desc] : h) {
    oss << std::setw(max_tag_len + 4) << std::right << tag
        << std::string(": ") << desc
        << std::endl;
  }
  std::cout << oss.str();
}

void
gridder_task(
  const Task*,
  const std::vector<PhysicalRegion>&,
  Context ctx,
  Runtime* rt) {

  hyperion::register_tasks(ctx, rt);

  // process command line arguments
  //
  std::optional<gridder::Args<gridder::VALUE_ARGS>> gridder_args;
  {
    const InputArgs& input_args = Runtime::get_input_args();
    gridder::Args<gridder::OPT_STRING_ARGS> some_str_args = default_config();
    if (gridder::has_help_flag(input_args)) {
      show_help(some_str_args);
      return;
    }
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
  //
  std::unordered_map<
    MSTables,
    std::tuple<Table, std::unordered_map<std::string, std::string>>> tables =
    using_resource(
      [&g_args]() {
        return
          CHECK_H5(
            H5Fopen(
              g_args->h5_path.value().c_str(),
              H5F_ACC_RDONLY,
              H5P_DEFAULT));
      },
      [&](hid_t h5f) {
        return
          using_resource(
            [h5f]() {
              return CHECK_H5(H5Gopen(h5f, "/", H5P_DEFAULT));
            },
            [&](hid_t h5root) {
              decltype(tables) result;
              auto mstables =
                std::experimental::make_array<MSTables>(
                  MS_ANTENNA,
                  MS_DATA_DESCRIPTION,
                  MS_FEED,
                  MS_MAIN,
                  MS_POLARIZATION,
                  MS_SPECTRAL_WINDOW);
              for (auto& mst : mstables)
                result[mst] = init_table(ctx, rt, h5root, mst);
              return result;
            },
            [](hid_t h5root) {
              CHECK_H5(H5Gclose(h5root));
            });
      },
      [](hid_t h5f) {
        CHECK_H5(H5Fclose(h5f));
      });

  // attach table columns to HDF5
  //
  std::unordered_map<MSTables, PhysicalTable> ptables;
  for (auto& [mst, tb_pths] : tables) {
    auto& [tb, pths] = tb_pths;
    std::unordered_map<std::string, std::tuple<bool, bool, bool>> modes;
    for (auto& pth : pths) {
      auto& nm = std::get<0>(pth);
      modes[nm] = {true/*read-only*/, true/*restricted*/, false/*mapped*/};
    }
    ptables.emplace(
      mst,
      tb.attach_columns(
        ctx,
        rt,
        READ_WRITE,
        g_args->h5_path.value(),
        pths,
        modes));
  }

  // re-index some tables
  //
  std::unordered_map<MSTables, Table> itables;
  {
    std::vector<MSTable<MS_FEED>::Axes> iaxes{FEED_AXES};
    itables.emplace(
      MS_FEED,
      ptables.at(MS_FEED).reindexed(ctx, rt, iaxes, false));
  }

  // create column in ANTENNA table for mapping antenna to its class
  //
  ptables
    .at(MS_ANTENNA)
    .add_columns(
      ctx,
      rt,
      {{ptables.at(MS_ANTENNA).index_column(rt).value()->column_space(),
        true,
        {{antenna_class_column_name,
          TableField(antenna_class_dt, antenna_class_fid)}}}});
  init_antenna_classes(ctx, rt, ptables.at(MS_ANTENNA));

  // create parallactic angle column in MAIN table
  //
  ptables
    .at(MS_MAIN)
    .add_columns(
      ctx,
      rt,
      {{ptables.at(MS_MAIN).index_column(rt).value()->column_space(),
        true,
        {{parallactic_angle_column_name,
          TableField(parallactic_angle_dt, parallactic_angle_fid)}}}});
  init_parallactic_angles(
    ctx,
    rt,
    g_args->pa_block.value(),
    ptables.at(MS_MAIN),
    ptables.at(MS_DATA_DESCRIPTION),
    ptables.at(MS_ANTENNA),
    itables.at(MS_FEED));

  // TODO: the rest goes here

  // clean up
  //
  ptables.at(MS_MAIN).remove_columns(ctx, rt, {parallactic_angle_column_name});
  ptables.at(MS_ANTENNA).remove_columns(ctx, rt, {antenna_class_column_name});

  for (auto& [mst, tb_pths] : tables) {
    auto& [tb, pths] = tb_pths;
    std::unordered_set<std::string> cols;
    for (auto& pth : pths)
      cols.insert(std::get<0>(pth));
    auto& pt = ptables.at(mst);
    pt.detach_columns(ctx, rt, cols);
    pt.unmap_regions(ctx, rt);
    tb.destroy(ctx, rt);
  }

  for (auto& [mst, tb] : itables)
    tb.destroy(ctx, rt);
}

int
main(int argc, char* argv[]) {
  hyperion::preregister_all();
  {
    TaskVariantRegistrar registrar(GRIDDER_TASK_ID, "gridder_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<gridder_task>(registrar, "gridder_task");
    Runtime::set_top_level_task_id(GRIDDER_TASK_ID);
  }
  {
    TaskVariantRegistrar registrar(
      CLASSIFY_ANTENNAS_TASK_ID,
      "classify_antennas_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    registrar.set_idempotent();
    TableMapper::add_table_layout_constraint(registrar);
    registrar.add_layout_constraint_set(
      TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
      aos_row_major_layout);
    Runtime::preregister_task_variant<classify_antennas_task>(
      registrar,
      "classify_antennas_task");
  }
  {
    TaskVariantRegistrar registrar(
      COMPUTE_PARALLACTIC_ANGLES_TASK_ID,
      "compute_parallactic_angles_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    registrar.set_idempotent();
    TableMapper::add_table_layout_constraint(registrar);
    registrar.add_layout_constraint_set(
      TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
      aos_row_major_layout);
    Runtime::preregister_task_variant<compute_parallactic_angles_task>(
      registrar,
      "compute_parallactic_angles_task");
  }
  //Runtime::register_reduction_op<LastPointRedop<1>>(LAST_POINT_REDOP);
  return Runtime::start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
