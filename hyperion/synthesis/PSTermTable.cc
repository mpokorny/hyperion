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
const constexpr unsigned PSTermTable::d_ps;
const constexpr char* PSTermTable::compute_cfs_task_name;
#endif
TaskID PSTermTable::compute_cfs_task_id;

PSTermTable::PSTermTable(
  Context ctx,
  Runtime* rt,
  const std::array<coord_t, 2>& cf_size,
  const std::vector<typename cf_table_axis<CF_PS_SCALE>::type>& ps_scales)
  : CFTable(
    ctx,
    rt,
    Rect<2>(
      {-(std::abs(cf_size[0]) / 2), std::abs(cf_size[0]) / 2},
      {-(std::abs(cf_size[1]) / 2), std::abs(cf_size[1]) / 2}),
    Axis<CF_PS_SCALE>(ps_scales)) {}

#ifndef HYPERION_USE_KOKKOS
void
PSTermTable::compute_cfs_task(
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

  auto tbl = CFPhysicalTable<CF_PS_SCALE>(pt);

  auto ps_scales = tbl.ps_scale<AffineAccessor>().accessor<READ_ONLY>();
  auto value_col = tbl.value<AffineAccessor>();
  auto values = value_col.accessor<WRITE_ONLY>();
  auto weight_col = tbl.value<AffineAccessor>();
  auto weights = weight_col.accessor<WRITE_ONLY>();
  typedef decltype(value_col)::value_t::value_type fp_t;

  for (PointInRectIterator<3> pir(value_col.rect()); pir(); pir++) {
    const fp_t x = pir[d_x];
    const fp_t y = pir[d_y];
    const fp_t yp =
      std::sqrt((static_cast<fp_t>(x) * x) + (static_cast<fp_t>(y) * y))
      * ps_scales[pir[d_ps]];
    const fp_t v =
      std::max(
        static_cast<fp_t>(spheroidal(yp)) * ((fp_t)1.0 - yp * yp),
        (fp_t)0.0);
    values[*pir] = v;
    weights[*pir] = v * v;
  }
}
#endif

void
PSTermTable::compute_cfs(
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
       {CF_WEIGHT_COLUMN_NAME, cf_colreqs}},
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
    p.destroy(ctx, rt);
}

void
PSTermTable::preregister_tasks() {
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

      // standard column layout
      LayoutConstraintRegistrar
        constraints(
          FieldSpace::NO_SPACE,
          "PSTermTable::compute_cfs_constraints");
      add_soa_right_ordering_constraint(constraints);
      constraints.add_constraint(
        SpecializedConstraint(LEGION_AFFINE_SPECIALIZE));
      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        Runtime::preregister_layout(constraints));

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

      // standard column layout
      LayoutConstraintRegistrar
        constraints(
          FieldSpace::NO_SPACE,
          "PSTermTable::compute_cfs_constraints");
      add_soa_right_ordering_constraint(constraints);
      constraints.add_constraint(
        SpecializedConstraint(LEGION_AFFINE_SPECIALIZE));
      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        Runtime::preregister_layout(constraints));

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

      // standard column layout
      LayoutConstraintRegistrar
        constraints(
          FieldSpace::NO_SPACE,
          "PSTermTable::compute_cfs_constraints");
      add_soa_left_ordering_constraint(constraints);
      constraints.add_constraint(
        SpecializedConstraint(LEGION_AFFINE_SPECIALIZE));
      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        Runtime::preregister_layout(constraints));

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

      // standard column layout
      LayoutConstraintRegistrar
        constraints(
          FieldSpace::NO_SPACE,
          "PSTermTable::compute_cfs_constraints");
      add_soa_right_ordering_constraint(constraints);
      constraints.add_constraint(
        SpecializedConstraint(LEGION_AFFINE_SPECIALIZE));
      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        Runtime::preregister_layout(constraints));

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
