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
#if defined(USE_KOKKOS_SERIAL_COMPUTE_CFS_TASK) &&  \
  defined(HYPERION_USE_KOKKOS) &&                   \
  defined(KOKKOS_ENABLE_SERIAL)
# define ENABLE_KOKKOS_SERIAL_COMPUTE_CFS_TASK
#else
# undef ENABLE_KOKKOS_SERIAL_COMPUTE_CFS_TASK
#endif

#define USE_KOKKOS_OPENMP_COMPUTE_CFS_TASK // undef to disable
#if defined(USE_KOKKOS_OPENMP_COMPUTE_CFS_TASK) &&  \
  defined(HYPERION_USE_KOKKOS) &&                   \
  defined(KOKKOS_ENABLE_OPENMP)
# define ENABLE_KOKKOS_OPENMP_COMPUTE_CFS_TASK
#else
# undef ENABLE_KOKKOS_OPENMP_COMPUTE_CFS_TASK
#endif

#define USE_KOKKOS_CUDA_COMPUTE_CFS_TASK // undef to disable
#if defined(USE_KOKKOS_CUDA_COMPUTE_CFS_TASK) &&  \
  defined(HYPERION_USE_KOKKOS) &&                 \
  defined(KOKKOS_ENABLE_CUDA)
# define ENABLE_KOKKOS_CUDA_COMPUTE_CFS_TASK
#else
# undef ENABLE_KOKKOS_CUDA_COMPUTE_CFS_TASK
#endif

Legion::TaskID WTermTable::compute_cfs_task_id;

WTermTable::WTermTable(
  Context ctx,
  Runtime* rt,
  const std::array<Legion::coord_t, 2>& cf_size,
  const std::vector<typename cf_table_axis<CF_W>::type>& w_values)
  : CFTable(
    ctx,
    rt,
    Legion::Rect<2>(
      {-(std::abs(cf_size[0]) / 2), std::abs(cf_size[0]) / 2},
      {-(std::abs(cf_size[1]) / 2), std::abs(cf_size[1]) / 2}),
    Axis<CF_W>(w_values)) {}

#ifndef HYPERION_USE_KOKKOS
void
WTermTable::compute_cfs_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  const ComputeCFSTaskArgs& args =
    *static_cast<const ComputeCFSTaskArgs*>(task->args);

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
#else // !HAVE_CXX17
  auto& pt = std::get<0>(ptcr);
  auto& rit = std::get<1>(ptcr);
  auto& pit = std::get<2>(ptcr);
#endif // HAVE_CXX17
  assert(rit == task->regions.end());
  assert(pit == regions.end());

  auto tbl = CFPhysicalTable<CF_W>(pt);

  auto w_values = tbl.w<Legion::AffineAccessor>().accessor<READ_ONLY>();
  auto values_col = tbl.value<Legion::AffineAccessor>();
  auto values = values_col.accessor<WRITE_ONLY>();
  auto weights = weights_col.accessor<WRITE_ONLY>();

  auto rect = values_col.rect();
  const coord_t& w_lo = rect.lo[0];
  const coord_t& w_hi = rect.hi[0];
  const coord_t& x_lo = rect.lo[1];
  const coord_t& x_hi = rect.hi[1];
  const coord_t& y_lo = rect.lo[2];
  const coord_t& y_hi = rect.hi[2];
  typedef typename cf_table_axis<CF_W>::type fp_t;
  for (coord_t w_idx = w_lo; w_idx <= w_hi; ++w_idx) {
    const fp_t twoPiW = (fp_t)twopi * w_values[w_idx];
    for (coord_t x_idx = x_lo; x_idx <= x_hi; ++x_idx) {
      const fp_t l = args.cell_size[0] * x_idx;
      const fp_t l2 = l * l;
      for (coord_t y_idx = y_lo; y_idx <= y_hi; ++y_idx) {
        const fp_t m = args.cell_size[1] * y_idx;
        const fp_t r2 = l2 + m * m;
        const fp_t phase =
          ((r2 <= (fp_t)1.0)
           ? (twoPiW * (std::sqrt((fp_t)1.0 - r2) - (fp_t)1.0))
           : (fp_t)0.0);
        values[{w_idx, x_idx, y_idx}] = std::polar((fp_t)1.0, phase);
        weights[{w_idx, x_idx, y_idx}] = (fp_t)1.0;
      }
    }
  }
}
#endif

void
WTermTable::compute_cfs(
  Context ctx,
  Runtime* rt,
  const std::array<double, 2>& cell_size,
  const ColumnSpacePartition& partition) const {

    auto cf_colreqs = Column::default_requirements;
    cf_colreqs.values = Column::Req{
      WRITE_ONLY /* privilege */,
      EXCLUSIVE /* coherence */,
      true /* mapped */
    };

    auto default_colreqs = Column::default_requirements;
    default_colreqs.values.mapped = true;

    auto reqs =
      requirements(
        ctx,
        rt,
        partition,
        {{CF_VALUE_COLUMN_NAME, cf_colreqs},
         {CF_WEIGHT_COLUMN_NAME, CXX_OPTIONAL_NAMESPACE::nullopt}},
        default_colreqs);
#if HAVE_CXX17
    auto& [treqs, tparts, tdesc] = reqs;
#else // !HAVE_CXX17
    auto& treqs = std::get<0>(reqs);
    auto& tparts = std::get<1>(reqs);
    auto& tdesc = std::get<2>(reqs);
#endif // HAVE_CXX17
    ComputeCFSTaskArgs args;
    args.desc = tdesc;
    args.cell_size[0] = cell_size[0];
    args.cell_size[1] = cell_size[1];
    TaskLauncher task(
      compute_cfs_task_id,
      TaskArgument(&args, sizeof(args)),
      Predicate::TRUE_PRED,
      table_mapper);
    for (auto& r : treqs)
      task.add_region_requirement(r);
    rt->execute_task(ctx, task);
    for (auto& p : tparts)
      p.destroy(ctx, rt);
}

void
WTermTable::preregister_tasks() {
  //
  // compute_cfs_task
  //
  {
#if defined(ENABLE_KOKKOS_SERIAL_COMPUTE_CFS_TASK) || \
  defined(ENABLE_KOKKOS_OPENMP_COMPUTE_CFS_TASK)
    LayoutConstraintRegistrar
      cpu_constraints(FieldSpace::NO_SPACE, "WTermTable::compute_cfs");
    add_aos_right_ordering_constraint(cpu_constraints);
    cpu_constraints.add_constraint(
      SpecializedConstraint(LEGION_AFFINE_SPECIALIZE));
    auto cpu_layout_id = Runtime::preregister_layout(cpu_constraints);
#endif

#if defined(ENABLE_KOKKOS_CUDA_COMPUTE_CFS_TASK)
    LayoutConstraintRegistrar
      gpu_constraints(FieldSpace::NO_SPACE, "WTermTable::compute_cfs");
    add_soa_left_ordering_constraint(gpu_constraints);
    gpu_constraints.add_constraint(
      SpecializedConstraint(LEGION_AFFINE_SPECIALIZE));
    auto gpu_layout_id = Runtime::preregister_layout(gpu_constraints);
#endif

    compute_cfs_task_id = Runtime::generate_static_task_id();

#ifdef ENABLE_KOKKOS_SERIAL_COMPUTE_CFS_TASK
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

#ifdef ENABLE_KOKKOS_OPENMP_COMPUTE_CFS_TASK
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

#ifdef ENABLE_KOKKOS_CUDA_COMPUTE_CFS_TASK
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

#ifndef HYPERION_USE_KOKKOS
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
  }
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
