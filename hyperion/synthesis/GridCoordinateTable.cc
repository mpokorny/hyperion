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
#include <hyperion/synthesis/GridCoordinateTable.h>

#include <casacore/casa/Arrays.h>

#include <memory>

using namespace hyperion::synthesis;
using namespace hyperion;
using namespace Legion;

namespace cc = casacore;

#if !HAVE_CXX17
const constexpr unsigned GridCoordinateTable::d_pa;
const constexpr unsigned GridCoordinateTable::coord_rank;
const constexpr Legion::FieldID GridCoordinateTable::COORD_X_FID;
const constexpr Legion::FieldID GridCoordinateTable::COORD_Y_FID;
const constexpr char* GridCoordinateTable::COORD_X_NAME;
const constexpr char* GridCoordinateTable::COORD_Y_NAME;
#endif

GridCoordinateTable::GridCoordinateTable(
  Context ctx,
  Runtime* rt,
  const size_t& grid_size,
  const std::vector<typename cf_table_axis<CF_PARALLACTIC_ANGLE>::type>&
    parallactic_angles)
  : CFTable(
    ctx,
    rt,
    grid_size,
    Axis<CF_PARALLACTIC_ANGLE>(parallactic_angles))
  , m_grid_size(grid_size) {

  auto cs = columns().at(CF_VALUE_COLUMN_NAME).cs;
  auto dt = ValueType<GridCoordinateTable::coord_t>::DataType;
  Table::fields_t tflds =
    {{cs,
      {{COORD_X_NAME, TableField(dt, COORD_X_FID)},
       {COORD_Y_NAME, TableField(dt, COORD_Y_FID)}}}};
  add_columns(ctx, rt, std::move(tflds));
}

void
GridCoordinateTable::compute_coordinates(
  Context ctx,
  Runtime* rt,
  const casacore::Coordinate& cf_coordinates,
  const double& cf_radius,
  const ColumnSpacePartition& partition) const {

  auto wd_colreqs = Column::default_requirements;
  wd_colreqs.values.privilege = LEGION_WRITE_DISCARD;
  wd_colreqs.values.mapped = true;
  auto ro_colreqs = Column::default_requirements;
  ro_colreqs.values.privilege = LEGION_READ_ONLY;
  ro_colreqs.values.mapped = true;
  auto reqs =
    requirements(
      ctx,
      rt,
      partition,
      {{COORD_X_NAME, wd_colreqs},
       {COORD_Y_NAME, wd_colreqs},
       {cf_table_axis<CF_PARALLACTIC_ANGLE>::name, ro_colreqs}},
      CXX_OPTIONAL_NAMESPACE::nullopt);
#if HAVE_CXX17
  auto& [treqs, tparts, tdesc] = reqs;
#else
  auto& treqs = std::get<0>(reqs);
  auto& tparts = std::get<1>(reqs);
  auto& tdesc = std::get<2>(reqs);
#endif

  ComputeCoordinatesTaskArgs args;
  args.desc = tdesc;
  auto coord = std::unique_ptr<cc::Coordinate>(cf_coordinates.clone());
  // Set the reference pixel of the coordinate system before serializing it
  auto origin = domain_origin();
  auto origin_p = origin.data();
  coord->setReferencePixel(cc::Vector(cc::Block<double>(2, origin_p, false)));
  auto increment = (2 * cf_radius) / m_grid_size;
  coord->setIncrement(std::vector<double>{increment, increment});

  switch (coord->type()) {
  case cc::Coordinate::LINEAR: {
    const cc::LinearCoordinate* lc =
      dynamic_cast<cc::LinearCoordinate*>(coord.get());
    assert(linear_coordinate_serdez::serialized_size(*lc)
           <= linear_coordinate_serdez::MAX_SERIALIZED_SIZE);
    linear_coordinate_serdez::serialize(*lc, args.lc.data());
    args.is_linear_coordinate = true;
    break;
  }
  case cc::Coordinate::DIRECTION: {
    const cc::DirectionCoordinate* dc =
      dynamic_cast<cc::DirectionCoordinate*>(coord.get());
    assert(direction_coordinate_serdez::serialized_size(*dc)
           <= direction_coordinate_serdez::MAX_SERIALIZED_SIZE);
    direction_coordinate_serdez::serialize(*dc, args.dc.data());
    args.is_linear_coordinate = false;
    break;
  }
  default:
    assert(false); // user error; TODO: throw exception?
    break;
  }
  TaskArgument ta(&args, sizeof(args));

  if (tparts.size() == 0) {
    TaskLauncher task(
      compute_coordinates_task_id,
      ta,
      Predicate::TRUE_PRED,
      table_mapper);
    for (auto& r : treqs)
      task.add_region_requirement(r);
    rt->execute_task(ctx, task);
  } else {
    IndexTaskLauncher task(
      compute_coordinates_task_id,
      rt->get_index_partition_color_space(ctx, partition.column_ip),
      ta,
      ArgumentMap(),
      Predicate::TRUE_PRED,
      false,
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
GridCoordinateTable::compute_coordinates_task_name;
#endif

Legion::TaskID GridCoordinateTable::compute_coordinates_task_id;

void
GridCoordinateTable::compute_coordinates_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  const ComputeCoordinatesTaskArgs& args =
    *static_cast<const ComputeCoordinatesTaskArgs*>(task->args);

  cc::LinearCoordinate lc0;
  cc::DirectionCoordinate dc0;
  cc::Coordinate* coord0;
  if (args.is_linear_coordinate) {
    linear_coordinate_serdez::deserialize(lc0, args.lc.data());
    coord0 = &lc0;
  } else {
    direction_coordinate_serdez::deserialize(dc0, args.dc.data());
    coord0 = &dc0;
  }

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

  auto gc_tbl = CFPhysicalTable<CF_PARALLACTIC_ANGLE>(pt);
  // coordinates columns
  auto cx_col =
    CoordColumn<AffineAccessor>(*gc_tbl.column(COORD_X_NAME).value());
  auto cx_rect = cx_col.rect();
  auto cxs = cx_col.accessor<WRITE_DISCARD>();
  [[maybe_unused]] auto cys =
    CoordColumn<AffineAccessor>(*gc_tbl.column(COORD_Y_NAME).value())
    .accessor<WRITE_DISCARD>();

  // parallactic angles
  auto pas = gc_tbl.parallactic_angle<AffineAccessor>().accessor<READ_ONLY>();

  // I'd prefer to write the coordinates directly into the physical region
  // using cc::LinearCoordinate::toWorldMany(), but that isn't possible
  // because I can't assign a region pointer with a given layout to a cc::Matrix
  // (in particular, the lack of offsets or strides in the cc::Matrix
  // constructor is a problem), so we use an auxiliary buffer
  const size_t nx = cx_rect.hi[d_x] - cx_rect.lo[d_x] + 1;
  const size_t ny = cx_rect.hi[d_y] - cx_rect.lo[d_y] + 1;
  cc::Matrix<double> pixel(2, nx * ny);
  for (size_t i = 0; i < nx * ny; ++i) {
    pixel(0, i) = i / ny + cx_rect.lo[d_x] + 0.5;
    pixel(1, i) = i % ny + cx_rect.lo[d_y] + 0.5;
  }

  cc::Matrix<double> world(2, nx * ny);
  bool delstorage;
  const GridCoordinateTable::coord_t* wcs = world.getStorage(delstorage);
  cc::Vector<bool> failures(nx * ny);

  Point<coord_rank> cpt;
  cpt[d_x] = cx_rect.lo[d_x];
  cpt[d_y] = cx_rect.lo[d_y];
  for (Legion::coord_t pa = cx_rect.lo[d_pa]; pa <= cx_rect.hi[d_pa]; ++pa) {
    // rotate coord0
    auto coord =
      std::unique_ptr<cc::Coordinate>(
        coord0->rotate(cc::Quantity(-pas[pa], "rad")));
    // do the conversions
    [[maybe_unused]] auto ok = coord->toWorldMany(world, pixel, failures);
    assert(ok);
    coord->makeWorldRelativeMany(world);
    cpt[d_pa] = pa;
    // assume an AOS layout of cxs/cys
    assert(cxs.ptr(cpt) + 1 == cys.ptr(cpt));
    std::memcpy(
      cxs.ptr(cpt),
      wcs,
      2 * nx * ny * sizeof(GridCoordinateTable::coord_t));
  }
  pixel.freeStorage(wcs, delstorage);
}

void
GridCoordinateTable::preregister_tasks() {
  //
  // compute_coordinates_task
  {
    compute_coordinates_task_id = Runtime::generate_static_task_id();

    TaskVariantRegistrar registrar(
      compute_coordinates_task_id,
      compute_coordinates_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    registrar.set_idempotent();

    // standard column layout
    LayoutConstraintRegistrar
      constraints(
        FieldSpace::NO_SPACE,
        "GridCoordinateTable::compute_coordinates_constraints");
    add_aos_right_ordering_constraint(constraints);
    constraints.add_constraint(SpecializedConstraint(LEGION_AFFINE_SPECIALIZE));
    registrar.add_layout_constraint_set(
      TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
      Runtime::preregister_layout(constraints));

    Runtime::preregister_task_variant<compute_coordinates_task>(
      registrar,
      compute_coordinates_task_name);
  }
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
