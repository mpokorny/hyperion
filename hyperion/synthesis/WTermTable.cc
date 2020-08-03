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
#include <hyperion/synthesis/WTermTable.h>

#include <cmath>
#include <complex>

using namespace hyperion::synthesis;
using namespace hyperion;
using namespace Legion;

#define USE_KOKKOS_SERIAL_COMPUTE_CFS_TASK // undef to disable
#define USE_KOKKOS_OPENMP_COMPUTE_CFS_TASK //undef to disable
#define USE_KOKKOS_CUDA_COMPUTE_CFS_TASK //undef to disable

Legion::TaskID WTermTable::compute_cfs_task_id;

WTermTable::WTermTable(
  Context ctx,
  Runtime* rt,
  const size_t& grid_size,
  const std::vector<typename cf_table_axis<CF_W>::type>& w_values)
  : CFTable(ctx, rt, grid_size, Axis<CF_W>(w_values)) {}

#ifndef HYPERION_USE_KOKKOS
void
WTermTable::compute_cfs_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  const ComputeCFsTaskArgs& args =
    *static_cast<const ComputeCFsTaskArgs*>(task->args);
  std::vector<Table::Desc> tdesc{args.w, args.gc};

  auto ptcrs =
    PhysicalTable::create_many(
      rt,
      tdesc,
      task->regions.begin(),
      task->regions.end(),
      regions.begin(),
      regions.end())
    .value();
#if HAVE_CXX17
  auto& [pts, rit, pit] = ptcrs;
#else // !HAVE_CXX17
  auto& pts = std::get<0>(ptcrs);
  auto& rit = std::get<1>(ptcrs);
  auto& pit = std::get<2>(ptcrs);
#endif // HAVE_CXX17
  assert(rit == task->regions.end());
  assert(pit == regions.end());

  auto w_tbl = CFPhysicalTable<CF_W>(pts[0]);
  auto gc_tbl = CFPhysicalTable<CF_PARALLACTIC_ANGLE>(pts[1]);

  auto w_values =
    w_tbl.w<Legion::AffineAccessor>().accessor<LEGION_READ_ONLY>();

  auto value_col = w_tbl.value<Legion::AffineAccessor>();
  auto value_rect = value_col.rect();
  auto values = values_col.accessor<LEGION_WRITE_ONLY>();
  auto weights = weights_col.accessor<LEGION_WRITE_ONLY>();

  auto cs_x_col =
    GridCoordinateTable::CoordColumn<Legion::AffineAccessor>(
      *gc_tbl.column(GridCoordinateTable::COORD_X_NAME).value());
  auto cs_x = cs_x_col.accessor<LEGION_READ_ONLY>();
  auto cs_y =
    GridCoordinateTable::CoordColumn<Legion::AffineAccessor>(
      *gc_tbl.column(GridCoordinateTable::COORD_Y_NAME).value())
    .accessor<LEGION_READ_ONLY>();
  auto i_pa = cs_x_col.rect().lo[GridCoordinateTable::d_pa];

  auto& r = value_rect;
  for (coord_t i_w = r.lo[d_w]; i_w <= r.hi[d_w]; ++i_w) {
    const cf_fp_t twoPiW = (cf_fp_t)twopi * w_values[i_w];
    for (coord_t i_x = r.lo[d_x]; i_x <= r.hi[d_x]; ++i_x) {
      for (coord_t i_y = r.lo[d_y]; i_y <= r.hi[d_y]; ++i_y) {
        const cf_fp_t l = cs_x[{i_pa, i_x, i_y}];
        const cf_fp_t m = cs_y[{i_pa, i_x, i_y}];
        const cf_fp_t r2 = l * l + m * m;
        if (r2 <= (cf_fp_t)1.0) {
          const cf_fp_t phase =
            towPiW * (std::sqrt((cf_fp_t)1.0 - r2) - (cf_fp_t)1.0);
          values[{i_w, i_x, i_y}] = std::polar((cf_fp_t)1.0, phase);
          weights[{i_w, i_x, i_y}] = (cf_fp_t)1.0;
        } else {
          values[{i_w, i_x, i_y}] = (cf_fp_t)0.0;
          weights[{i_w, i_x, i_y}] = std::numeric_limits<cf_fp_t>::quiet_NaN();
        }
      }
    }
  }
}
#endif

void
WTermTable::compute_cfs(
  Context ctx,
  Runtime* rt,
  const GridCoordinateTable& gc,
  const ColumnSpacePartition& partition) const {

  auto ro_colreqs = Column::default_requirements;
  ro_colreqs.values.mapped = true;

  ComputeCFsTaskArgs args;
  std::vector<RegionRequirement> all_reqs;
  std::vector<ColumnSpacePartition> all_parts;
  {
    auto wo_colreqs = Column::default_requirements;
    wo_colreqs.values.privilege = LEGION_WRITE_ONLY;
    wo_colreqs.values.mapped = true;

    auto reqs =
      requirements(
        ctx,
        rt,
        partition,
        {{CF_VALUE_COLUMN_NAME, wo_colreqs},
         {CF_WEIGHT_COLUMN_NAME, wo_colreqs}},
        ro_colreqs);
#if HAVE_CXX17
    auto& [treqs, tparts, tdesc] = reqs;
#else // !HAVE_CXX17
    auto& treqs = std::get<0>(reqs);
    auto& tparts = std::get<1>(reqs);
    auto& tdesc = std::get<2>(reqs);
#endif // HAVE_CXX17
    std::copy(treqs.begin(), treqs.end(), std::back_inserter(all_reqs));
    std::copy(tparts.begin(), tparts.end(), std::back_inserter(all_parts));
    args.w = tdesc;
  }
  {
    auto reqs =
      gc.requirements(
        ctx,
        rt,
        partition,
        {{GridCoordinateTable::COORD_X_NAME, ro_colreqs},
         {GridCoordinateTable::COORD_Y_NAME, ro_colreqs}},
        CXX_OPTIONAL_NAMESPACE::nullopt);
#if HAVE_CXX17
    auto& [treqs, tparts, tdesc] = reqs;
#else // !HAVE_CXX17
    auto& treqs = std::get<0>(reqs);
    auto& tparts = std::get<1>(reqs);
    auto& tdesc = std::get<2>(reqs);
#endif // HAVE_CXX17
    std::copy(treqs.begin(), treqs.end(), std::back_inserter(all_reqs));
    std::copy(tparts.begin(), tparts.end(), std::back_inserter(all_parts));
    args.gc = tdesc;
  }
  if (!partition.is_valid()) {
    TaskLauncher task(
      compute_cfs_task_id,
      TaskArgument(&args, sizeof(args)),
      Predicate::TRUE_PRED,
      table_mapper);
    for (auto& r : all_reqs)
      task.add_region_requirement(r);
    rt->execute_task(ctx, task);
  } else {
    IndexTaskLauncher task(
      compute_cfs_task_id,
      rt->get_index_partition_color_space(ctx, partition.column_ip),
      TaskArgument(&args, sizeof(args)),
      ArgumentMap(),
      Predicate::TRUE_PRED,
      table_mapper);
    for (auto& r : all_reqs)
      task.add_region_requirement(r);
    rt->execute_index_space(ctx, task);
  }
  for (auto& p : all_parts)
    p.destroy(ctx, rt);
}

#define USE_KOKKOS_VARIANT(V, T)                \
  (defined(USE_KOKKOS_##V##_COMPUTE_##T) &&     \
   defined(HYPERION_USE_KOKKOS) &&              \
   defined(KOKKOS_ENABLE_##V))

#define USE_PLAIN_SERIAL_VARIANT(T)             \
  (!USE_KOKKOS_VARIANT(SERIAL, T) &&            \
   !USE_KOKKOS_VARIANT(OPENMP, T) &&            \
   !USE_KOKKOS_VARIANT(CUDA, T))

void
WTermTable::preregister_tasks() {
  //
  // compute_cfs_task
  //
  {
#if USE_KOKKOS_VARIANT(SERIAL, CFS_TASK) || \
  USE_KOKKOS_VARIANT(OPENMP, CFS_TASK) || \
  USE_PLAIN_SERIAL_VARIANT(CFS_TASK)
    LayoutConstraintRegistrar
      cpu_constraints(FieldSpace::NO_SPACE, "WTermTable::compute_cfs");
    add_aos_right_ordering_constraint(cpu_constraints);
    cpu_constraints.add_constraint(
      SpecializedConstraint(LEGION_AFFINE_SPECIALIZE));
    auto cpu_layout_id = Runtime::preregister_layout(cpu_constraints);
#endif

#if USE_KOKKOS_VARIANT(CUDA, CFS_TASK)
    LayoutConstraintRegistrar
      gpu_constraints(FieldSpace::NO_SPACE, "WTermTable::compute_cfs");
    add_soa_left_ordering_constraint(gpu_constraints);
    gpu_constraints.add_constraint(
      SpecializedConstraint(LEGION_AFFINE_SPECIALIZE));
    auto gpu_layout_id = Runtime::preregister_layout(gpu_constraints);
#endif

    compute_cfs_task_id = Runtime::generate_static_task_id();

#if USE_KOKKOS_VARIANT(SERIAL, CFS_TASK)
    // register a serial version on the CPU
    {
      TaskVariantRegistrar
        registrar(compute_cfs_task_id, compute_cfs_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();

      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        cpu_layout_id);

      Runtime::preregister_task_variant<compute_cfs_task<Kokkos::Serial>>(
        registrar,
        compute_cfs_task_name);
    }
#endif

#if USE_KOKKOS_VARIANT(OPENMP, CFS_TASK)
    // register an openmp version
    {
      TaskVariantRegistrar
        registrar(compute_cfs_task_id, compute_cfs_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();

      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        cpu_layout_id);

      Runtime::preregister_task_variant<compute_cfs_task<Kokkos::OpenMP>>(
        registrar,
        compute_cfs_task_name);
    }
#endif

#if USE_KOKKOS_VARIANT(CUDA, CFS_TASK)
    // register a cuda version
    {
      TaskVariantRegistrar
        registrar(compute_cfs_task_id, compute_cfs_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();

      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        gpu_layout_id);

      Runtime::preregister_task_variant<compute_cfs_task<Kokkos::Cuda>>(
        registrar,
        compute_cfs_task_name);
    }
#endif

#if USE_PLAIN_SERIAL_VARIANT(CFS_TASK)
    {
      TaskVariantRegistrar
        registrar(compute_cfs_task_id, compute_cfs_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();

      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        cpu_layout_id);

      Runtime::preregister_task_variant<compute_cfs_task>(
        registrar,
        compute_cfs_task_name);
    }
#endif
  }
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
