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
#include <hyperion/synthesis/DirectionCoordinateTable.h>

#include <casacore/casa/Arrays.h>

#include <memory>

using namespace hyperion::synthesis;
using namespace hyperion;
using namespace Legion;

namespace cc = casacore;

#if !HAVE_CXX17
const constexpr unsigned DirectionCoordinateTable::d_pa;
const constexpr unsigned DirectionCoordinateTable::worldc_rank;
const constexpr Legion::FieldID DirectionCoordinateTable::WORLD_X_FID;
const constexpr Legion::FieldID DirectionCoordinateTable::WORLD_Y_FID;
const constexpr char* DirectionCoordinateTable::WORLD_X_NAME;
const constexpr char* DirectionCoordinateTable::WORLD_Y_NAME;
#endif

DirectionCoordinateTable::DirectionCoordinateTable(
  Context ctx,
  Runtime* rt,
  const std::array<size_t, 2>& cf_size,
  const std::vector<typename cf_table_axis<CF_PARALLACTIC_ANGLE>::type>&
    parallactic_angles)
  : CFTable(
    ctx,
    rt,
    cf_size,
    Axis<CF_PARALLACTIC_ANGLE>(parallactic_angles)) {

  auto cs = columns().at(CF_VALUE_COLUMN_NAME).cs;
  auto dt = ValueType<worldc_t>::DataType;
  Table::fields_t tflds =
    {{cs,
      {{WORLD_X_NAME, TableField(dt, WORLD_X_FID)},
       {WORLD_Y_NAME, TableField(dt, WORLD_Y_FID)}}}};
  add_columns(ctx, rt, std::move(tflds));
}

void
DirectionCoordinateTable::compute_world_coordinates(
  Context ctx,
  Runtime* rt,
  const cc::DirectionCoordinate& direction,
  const ColumnSpacePartition& partition) const {

  auto wd_colreqs = Column::default_requirements;
  wd_colreqs.values.privilege = WRITE_DISCARD;
  wd_colreqs.values.mapped = true;
  auto reqs =
    requirements(
      ctx,
      rt,
      partition,
      {{WORLD_X_NAME, wd_colreqs}, {WORLD_Y_NAME, wd_colreqs}},
      CXX_OPTIONAL_NAMESPACE::nullopt);
#if HAVE_CXX17
  auto& [treqs, tparts, tdesc] = reqs;
#else
  auto& treqs = std::get<0>(reqs);
  auto& tparts = std::get<1>(reqs);
  auto& tdesc = std::get<2>(reqs);
#endif

  // Set the reference pixel of the coordinate system at the center of the
  // grid. The grid defined by the constructor is always centered at the origin,
  // and we must account for the fact that casacore::Coordinates pixel
  // coordinates are always zero-based and non-negative, so casacore::pixel_i =
  // hyperion::pixel_i + floor(hyperion::size_i / 2), where pixel coordinates
  // are floating point values, and size_t is a positive integer.
  cc::DirectionCoordinate dc(direction);
  Rect<cf_rank> value_rect(
    rt->get_index_space_domain(
      columns().at(CF_VALUE_COLUMN_NAME).cs.column_is));
  std::array<Legion::coord_t, 2> cf_size{
    value_rect.hi[0] - value_rect.lo[0] + 1,
    value_rect.hi[1] - value_rect.lo[1] + 1};
  cc::Vector<double> ref_pixel;
  // hyperion origin[.] is at (cf_size[.] % 2) / 2.0
  ref_pixel[0] = (cf_size[0] % 2) / 2.0 + cf_size[0] / 2;
  ref_pixel[1] = (cf_size[1] % 2) / 2.0 + cf_size[1] / 2;
  dc.setReferencePixel(ref_pixel);

  ComputeWorldCoordinatesTaskArgs args;
  args.desc = tdesc;
  args.pixel_offset[0] = cf_size[0] / 2 + 0.5; // center of grid cell
  args.pixel_offset[1] = cf_size[1] / 2 + 0.5; // center of grid cell
  assert(direction_coordinate_serdez::serialized_size(direction)
         <= direction_coordinate_serdez::MAX_SERIALIZED_SIZE);
  direction_coordinate_serdez::serialize(direction, args.dc.data());
  TaskArgument ta(&args, sizeof(args));

  if (tparts.size() == 0) {
    TaskLauncher task(
      compute_world_coordinates_task_id,
      ta,
      Predicate::TRUE_PRED,
      table_mapper);
    for (auto& r : treqs)
      task.add_region_requirement(r);
    rt->execute_task(ctx, task);
  } else {
    IndexTaskLauncher task(
      compute_world_coordinates_task_id,
      rt->get_index_partition_color_space(ctx, partition.column_ip),
      ta,
      ArgumentMap(),
      Predicate::TRUE_PRED,
      table_mapper);
    for (auto& r : treqs)
      task.add_region_requirement(r);
    rt->execute_index_space(ctx, task);
  }
  for (auto& p : tparts)
    p.destroy(ctx, rt);
}

#if !HAVE_CXX17
const constexpr char*
DirectionCoordinateTable::compute_world_coordinates_task_name;
#endif

Legion::TaskID DirectionCoordinateTable::compute_world_coordinates_task_id;

void
DirectionCoordinateTable::compute_world_coordinates(
  const CFPhysicalTable<CF_PARALLACTIC_ANGLE>& dc_tbl,
  const cc::DirectionCoordinate& dc0,
  const std::array<double, 2>& pixel_offset) {

  // world coordinates columns
  auto wx_col =
    WorldCColumn<AffineAccessor>(*dc_tbl.column(WORLD_X_NAME).value());
  auto wx_rect = wx_col.rect();
  auto wxs = wx_col.accessor<WRITE_DISCARD>();
  [[maybe_unused]] auto wys =
    WorldCColumn<AffineAccessor>(*dc_tbl.column(WORLD_Y_NAME).value())
    .accessor<WRITE_DISCARD>();

  // parallactic angles
  auto pas = dc_tbl.parallactic_angle<AffineAccessor>().accessor<READ_ONLY>();

  // cc::Matrix for pixel coordinates
  size_t nx = wx_rect.hi[d_x] - wx_rect.lo[d_x] + 1;
  size_t ny = wx_rect.hi[d_y] - wx_rect.lo[d_y] + 1;
  cc::Matrix<double> pixel(nx * ny, 2);
  for (size_t i = 0; i < nx * ny; ++i) {
    pixel(i, 0) = i / ny + wx_rect.lo[d_x] + pixel_offset[0];
    pixel(i, 1) = i % ny + wx_rect.lo[d_y] + pixel_offset[1];
  }
  cc::Vector<bool> failures(nx * ny);
  // I'd like to write the world coordinates directly into the physical region
  // using cc::DirectionCoordinate::toWorldMany(), but that isn't possible
  // because I can't assign a region pointer with a given layout to a cc::Matrix
  // (in particular, the lack of offsets or strides in the cc::Matrix
  // constructor is a problem), so we use an auxiliary buffer
  cc::Matrix<double> world(nx * ny, 2);

  Point<worldc_rank> pt;
  pt[d_x] = wx_rect.lo[d_x];
  pt[d_y] = wx_rect.lo[d_y];
  for (coord_t pa = wx_rect.lo[d_pa]; pa <= wx_rect.hi[d_pa]; ++pa) {
    // rotate dc0
    auto dc = std::unique_ptr<cc::DirectionCoordinate>(
      dynamic_cast<cc::DirectionCoordinate*>(dc0.rotate(pas[pa])));
    // do the conversions
    [[maybe_unused]] auto rc = dc->toWorldMany(world, pixel, failures);
    assert(rc);
    pt[d_pa] = pa;
    // assume an AOS layout of wxs/wys
    assert(wxs.ptr(pt) + 1 == wys.ptr(pt));
    bool delstorage;
    const worldc_t* wcs = world.getStorage(delstorage);
    std::memcpy(wxs.ptr(pt), wcs, nx * ny * sizeof(worldc_t));
    world.freeStorage(wcs, delstorage);
  }
}

void
DirectionCoordinateTable::compute_world_coordinates_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  const ComputeWorldCoordinatesTaskArgs& args =
    *static_cast<const ComputeWorldCoordinatesTaskArgs*>(task->args);

  cc::DirectionCoordinate dc0;
  direction_coordinate_serdez::deserialize(dc0, args.dc.data());

  auto ptcr =
    PhysicalTable::create(
      rt,
      args.desc,
      task->regions.begin(),
      task->regions.end(),
      regions.begin(),
      regions.end())
    .value();
#if HAVE_CXX17
  auto& [pt, rit, pit] = ptcr;
#else
  auto& pt = std::get<0>(ptcr);
  auto& rit = std::get<1>(ptcr);
  auto& pit = std::get<2>(ptcr);
#endif
  assert(rit == task->regions.end());
  assert(pit == regions.end());

  compute_world_coordinates(
    CFPhysicalTable<CF_PARALLACTIC_ANGLE>(pt),
    dc0,
    args.pixel_offset);
}

void
DirectionCoordinateTable::preregister_tasks() {
  //
  // compute_world_coordinates_task
  {
    TaskVariantRegistrar registrar(
      compute_world_coordinates_task_id,
      compute_world_coordinates_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    registrar.set_idempotent();

    // standard column layout
    LayoutConstraintRegistrar
      constraints(
        FieldSpace::NO_SPACE,
        "DirectionCoordinateTable::compute_world_coordinates_constraints");
    add_aos_right_ordering_constraint(constraints);
    constraints.add_constraint(SpecializedConstraint(LEGION_AFFINE_SPECIALIZE));
    registrar.add_layout_constraint_set(
      TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
      Runtime::preregister_layout(constraints));

    Runtime::preregister_task_variant<compute_world_coordinates_task>(
      registrar,
      compute_world_coordinates_task_name);
  }
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
