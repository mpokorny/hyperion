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
#include <hyperion/synthesis/ATermAux1.h>
#include <hyperion/synthesis/ATermAux0.h>
#include <hyperion/synthesis/ATermTable.h>

using namespace hyperion;
using namespace hyperion::synthesis;
using namespace Legion;

TaskID ATermAux1::compute_epts_task_id;

#if !HAVE_CXX17
const constexpr unsigned ATermAux1::d_blc;
const constexpr unsigned ATermAux1::d_pa;
const constexpr unsigned ATermAux1::d_frq;
const constexpr unsigned ATermAux1::d_sto;
const constexpr unsigned ATermAux1::d_x;
const constexpr unsigned ATermAux1::d_y;
const constexpr unsigned ATermAux1::d_power;
const constexpr unsigned ATermAux1::ept_rank;
const constexpr Legion::FieldID ATermAux1::EPT_X_FID;
const constexpr Legion::FieldID ATermAux1::EPT_Y_FID;
const constexpr char* ATermAux1::EPT_X_NAME;
const constexpr char* ATermAux1::EPT_Y_NAME;
#endif // !HAVE_CXX17

ATermAux1::ATermAux1(
  Context ctx,
  Runtime* rt,
  const Rect<2>& cf_bounds,
  unsigned zernike_order,
  const std::vector<typename cf_table_axis<CF_BASELINE_CLASS>::type>&
    baseline_classes,
  const std::vector<typename cf_table_axis<CF_PARALLACTIC_ANGLE>::type>&
    parallactic_angles,
  const std::vector<typename cf_table_axis<CF_FREQUENCY>::type>&
    frequencies,
  const std::vector<typename cf_table_axis<CF_STOKES>::type>&
    stokes_values)
  : CFTable(
    ctx,
    rt,
    cf_bounds,
    Axis<CF_BASELINE_CLASS>(baseline_classes),
    Axis<CF_PARALLACTIC_ANGLE>(parallactic_angles),
    Axis<CF_FREQUENCY>(frequencies),
    Axis<CF_STOKES>(stokes_values)) {

  Table::fields_t tflds;
  {
    // EPT column
    Legion::coord_t
      c_hi[ept_rank]{
          static_cast<Legion::coord_t>(baseline_classes.size()) - 1,
          static_cast<Legion::coord_t>(parallactic_angles.size()) - 1,
          static_cast<Legion::coord_t>(frequencies.size()) - 1,
          static_cast<Legion::coord_t>(stokes_values.size()) - 1,
          cf_bounds.hi[0],
          cf_bounds.hi[1],
          static_cast<Legion::coord_t>(zernike_order)};
    Legion::Point<ept_rank> hi(c_hi);
    Legion::coord_t
      c_lo[ept_rank]{0, 0, 0, 0, cf_bounds.lo[0], cf_bounds.lo[1], 0};
    Legion::Point<ept_rank> lo(c_lo);
    Rect<ept_rank> rect(lo, hi);
    auto is = rt->create_index_space(ctx, rect);
    auto cs =
      ColumnSpace::create<cf_table_axes_t>(
        ctx,
        rt,
        {CF_BASELINE_CLASS,
         CF_PARALLACTIC_ANGLE,
         CF_FREQUENCY,
         CF_STOKES,
         CF_X,
         CF_Y,
         CF_ORDER0},
        is,
        false);
    tflds.push_back(
      {cs,
       {{EPT_X_NAME, TableField(ValueType<ept_t>::DataType, EPT_X_FID)},
        {EPT_Y_NAME, TableField(ValueType<ept_t>::DataType, EPT_Y_FID)}}});
  }
  add_columns(ctx, rt, std::move(tflds));
}

void
ATermAux1::compute_epts(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const ATermTable& aterm_table,
  const ATermAux0& aux0_table,
  const ColumnSpacePartition& partition) const {

  std::vector<RegionRequirement> all_reqs;
  std::vector<ColumnSpacePartition> all_parts;
  ComputeEPtsTaskArgs args;
  {
    // ATermTable, parallactic angle column
    auto default_colreqs = Column::default_requirements;
    default_colreqs.values.mapped = true;
    auto reqs =
      aterm_table.requirements(
        ctx,
        rt,
        partition,
        {{cf_table_axis<CF_PARALLACTIC_ANGLE>::name, default_colreqs}},
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
    args.aterm = tdesc;
  }
  {
    // ATermAux0 table, polynomial coefficients column
    auto default_colreqs = Column::default_requirements;
    default_colreqs.values.mapped = true;
    auto reqs =
      aux0_table.requirements(
        ctx,
        rt,
        partition,
        {{ATermAux0::PC_NAME, default_colreqs}},
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
    args.aux0 = tdesc;
  }
  {
    // ATermAux1 table, epts columns
    auto default_colreqs = Column::default_requirements;
    default_colreqs.values.privilege = WRITE_DISCARD;
    default_colreqs.values.mapped = true;
    auto reqs =
      requirements(
        ctx,
        rt,
        partition,
        {{ATermAux1::EPT_X_NAME, default_colreqs},
         {ATermAux1::EPT_Y_NAME, default_colreqs}},
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
    args.aux1 = tdesc;
  }
  TaskArgument ta(&args, sizeof(args));
  if (!partition.is_valid()) {
    TaskLauncher task(
      compute_epts_task_id,
      ta,
      Predicate::TRUE_PRED,
      table_mapper);
    for (auto& r : all_reqs)
      task.add_region_requirement(r);
    rt->execute_task(ctx, task);
  } else {
    IndexTaskLauncher task(
      compute_epts_task_id,
      rt->get_index_partition_color_space(ctx, partition.column_ip),
      ta,
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

void
ATermAux1::preregister_tasks() {
  //
  // compute_epts_task
  //
  {
    compute_epts_task_id = Runtime::generate_static_task_id();

    // the only table with two columns sharing an index space is ATermAux1,
    // using EPT_X and EPT_Y; use an AOS layout for CPUs as default for that
    // reason
#ifdef HYPERION_USE_KOKKOS
# ifdef KOKKOS_ENABLE_SERIAL
    {
      TaskVariantRegistrar
        registrar(compute_epts_task_id, compute_epts_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();
      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        aos_right_layout);
      Runtime::preregister_task_variant<compute_epts_task<Kokkos::Serial>>(
        registrar,
        compute_epts_task_name);
    }
# endif // KOKKOS_ENABLE_SERIAL
# ifdef KOKKOS_ENABLE_OPENMP
    {
      TaskVariantRegistrar
        registrar(compute_epts_task_id, compute_epts_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();
      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        aos_right_layout);
      Runtime::preregister_task_variant<compute_epts_task<Kokkos::OpenMP>>(
        registrar,
        compute_epts_task_name);
    }
# endif // KOKKOS_ENABLE_OPENMP
# ifdef KOKKOS_ENABLE_CUDA
    {
      TaskVariantRegistrar
        registrar(compute_epts_task_id, compute_epts_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();
      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        soa_left_layout);
      Runtime::preregister_task_variant<compute_epts_task<Kokkos::Cuda>>(
        registrar,
        compute_epts_task_name);
    }
# endif // KOKKOS_ENABLE_CUDA
#else // !HYPERION_USE_KOKKOS
    {
      TaskVariantRegistrar
        registrar(compute_epts_task_id, compute_epts_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();
      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        aos_right_layout);
      Runtime::preregister_task_variant<compute_epts_task>(
        registrar,
        compute_epts_task_name);
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
