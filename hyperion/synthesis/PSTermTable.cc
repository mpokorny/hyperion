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
#include <hyperion/synthesis/PSTermTable.h>

#include <cmath>
#ifdef HYPERION_USE_KOKKOS
# include <Kokkos_Core.hpp>
#endif

using namespace hyperion::synthesis;
using namespace hyperion;
using namespace Legion;

#if !HAVE_CXX17
constexpr const char* compute_cfs_task_name;
#endif
TaskID PSTermTable::compute_cfs_task_id;

hyperion::synthesis::PSTermTable::PSTermTable(
  Context ctx,
  Runtime* rt,
  const std::array<coord_t, 2>& cf_bounds_lo,
  const std::array<coord_t, 2>& cf_bounds_hi,
  const std::vector<typename cf_table_axis<CF_PB_SCALE>::type>& pb_scales)
  : CFTable<CF_PB_SCALE>(
    ctx,
    rt,
    Rect<2>(
      {cf_bounds_lo[0], cf_bounds_lo[1]},
      {cf_bounds_hi[0], cf_bounds_hi[1]}),
    Axis<CF_PB_SCALE>(pb_scales)) {}

hyperion::synthesis::PSTermTable::PSTermTable(
  Context ctx,
  Runtime* rt,
  const coord_t& cf_x_radius,
  const coord_t& cf_y_radius,
  const std::vector<typename cf_table_axis<CF_PB_SCALE>::type>& pb_scales)
  : CFTable<CF_PB_SCALE>(
    ctx,
    rt,
    Rect<2>({-cf_x_radius, -cf_y_radius}, {cf_x_radius, cf_y_radius}),
    Axis<CF_PB_SCALE>(pb_scales)) {}

#ifndef HYPERION_USE_KOKKOS
void
hyperion::synthesis::PSTermTable::compute_cfs_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  const Table::Desc& desc = *static_cast<Table::Desc*>(task->args);

  auto ptcr =
    PhysicalTable::create(
      rt,
      desc,
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

  auto tbl = CFPhysicalTable<CF_PB_SCALE>(pt);

  auto pb_scales = tbl.pb_scale<AffineAccessor>().accessor<READ_ONLY>();
  auto values_col = tbl.value<AffineAccessor>();
  auto values = values_col.accessor<WRITE_ONLY>();

  auto rect = values_col.rect();
  const coord_t& pb_lo = rect.lo[0];
  const coord_t& pb_hi = rect.hi[0];
  const coord_t& x_lo = rect.lo[1];
  const coord_t& x_hi = rect.hi[1];
  const coord_t& y_lo = rect.lo[2];
  const coord_t& y_hi = rect.hi[2];
  typedef typename cf_table_axis<CF_PB_SCALE>::type fp_t;
  for (coord_t pb_idx = pb_lo; pb_idx <= pb_hi; ++pb_idx) {
    const fp_t pb_scale = pb_scales[pb_idx];
    for (coord_t x_idx = x_lo; x_idx <= x_hi; ++x_idx) {
      const fp_t xp = x_idx * x_idx;
      for (coord_t y_idx = y_lo; y_idx <= y_hi; ++y_idx) {
        const fp_t yp = std::sqrt(xp + y_idx * y_idx) * pb_scale;
        const fp_t v =
          static_cast<fp_t>(spheroidal(yp)) * ((fp_t)1.0 - yp * yp);
        // we're creating PS terms that are intended to be multiplied in some
        // domain, thus, if v == 0.0, we'll just multiply by 1.0
        values[{pb_idx, x_idx, y_idx}] = ((v > (fp_t)0.0) ? v : (fp_t)1.0);
      }
    }
  }
}
#endif

void
hyperion::synthesis::PSTermTable::compute_cfs(
  Context ctx,
  Runtime* rt,
  const ColumnSpacePartition& partition) const {

  Column::Requirements cf_colreqs = Column::default_requirements;
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
  TaskLauncher task(
    compute_cfs_task_id,
    TaskArgument(&tdesc, sizeof(tdesc)),
    Predicate::TRUE_PRED,
    table_mapper);
  for (auto& r : treqs)
    task.add_region_requirement(r);
  rt->execute_task(ctx, task);
  for (auto& p : tparts)
    rt->destroy_logical_partition(ctx, p);
}

void
hyperion::synthesis::PSTermTable::preregister_tasks() {
  {
    // compute_cfs_task
    compute_cfs_task_id = Runtime::generate_static_task_id();
#ifdef HYPERION_USE_KOKKOS
# ifdef KOKKOS_ENABLE_SERIAL
    // register a serial version on the CPU
    {
      TaskVariantRegistrar
        registrar(compute_cfs_task_id, compute_cfs_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();
      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        soa_right_layout);
      Runtime::preregister_task_variant<compute_cfs_task<Kokkos::Serial>>(
        registrar,
        compute_cfs_task_name);
    }
# endif

# ifdef KOKKOS_ENABLE_OPENMP
    // register an openmp version, if available
    {
      TaskVariantRegistrar
        registrar(compute_cfs_task_id, compute_cfs_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();
      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        soa_right_layout);
      Runtime::preregister_task_variant<compute_cfs_task<Kokkos::OpenMP>>(
        registrar,
        compute_cfs_task_name);
    }
# endif

# ifdef KOKKOS_ENABLE_CUDA
    // register a serial version on the GPU
    {
      TaskVariantRegistrar
        registrar(compute_cfs_task_id, compute_cfs_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();
      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        soa_left_layout);
      Runtime::preregister_task_variant<compute_cfs_task<Kokkos::Cuda>>(
        registrar,
        compute_cfs_task_name);
    }
# endif
#else // !HYPERION_USE_KOKKOS  
    {
      TaskVariantRegistrar
        registrar(compute_cfs_task_id, compute_cfs_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();
      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        soa_right_layout);
      Runtime::preregister_task_variant<compute_cfs_task>(
        registrar,
        compute_cfs_task_name);
    }  
#endif // HYPERION_USE_KOKKOS
  }
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End: