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
#include <hyperion/synthesis/ATermIlluminationFunction.h>
#include <hyperion/synthesis/FFT.h>

using namespace hyperion;
using namespace hyperion::synthesis;
using namespace Legion;

TaskID ATermIlluminationFunction::compute_epts_task_id;
TaskID ATermIlluminationFunction::evaluate_polynomials_task_id;

#if !HAVE_CXX17
const constexpr unsigned ATermIlluminationFunction::d_blc;
const constexpr unsigned ATermIlluminationFunction::d_pa;
const constexpr unsigned ATermIlluminationFunction::d_frq;
const constexpr unsigned ATermIlluminationFunction::d_sto;
const constexpr unsigned ATermIlluminationFunction::d_x;
const constexpr unsigned ATermIlluminationFunction::d_y;
const constexpr unsigned ATermIlluminationFunction::d_power;
const constexpr unsigned ATermIlluminationFunction::ept_rank;
const constexpr Legion::FieldID ATermIlluminationFunction::EPT_X_FID;
const constexpr Legion::FieldID ATermIlluminationFunction::EPT_Y_FID;
const constexpr char* ATermIlluminationFunction::EPT_X_NAME;
const constexpr char* ATermIlluminationFunction::EPT_Y_NAME;
#endif // !HAVE_CXX17

#define ENABLE_KOKKOS_SERIAL_EPTS_TASK
#define ENABLE_KOKKOS_OPENMP_EPTS_TASK
#define ENABLE_KOKKOS_CUDA_EPTS_TASK

#define ENABLE_KOKKOS_SERIAL_EVAL_POLY_TASK
#define ENABLE_KOKKOS_OPENMP_EVAL_POLY_TASK
#define ENABLE_KOKKOS_CUDA_EVAL_POLY_TASK

ATermIlluminationFunction::ATermIlluminationFunction(
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
          (Legion::coord_t)2};
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
ATermIlluminationFunction::compute_epts(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const ColumnSpacePartition& partition) const {

  // ATermIlluminationFunction table, epts columns
  auto wd_colreqs = Column::default_requirements;
  wd_colreqs.values.privilege = WRITE_DISCARD;
  wd_colreqs.values.mapped = true;
  auto ro_colreqs = Column::default_requirements;
  ro_colreqs.values.mapped = true;
  auto reqs =
    requirements(
      ctx,
      rt,
      partition,
      {{ATermIlluminationFunction::EPT_X_NAME, wd_colreqs},
       {ATermIlluminationFunction::EPT_Y_NAME, wd_colreqs},
       {cf_table_axis<CF_PARALLACTIC_ANGLE>::name, ro_colreqs}},
      CXX_OPTIONAL_NAMESPACE::nullopt);
#if HAVE_CXX17
  auto& [treqs, tparts, tdesc] = reqs;
#else // !HAVE_CXX17
  auto& treqs = std::get<0>(reqs);
  auto& tparts = std::get<1>(reqs);
  auto& tdesc = std::get<2>(reqs);
#endif // HAVE_CXX17

  TaskArgument ta(&tdesc, sizeof(tdesc));
  if (!partition.is_valid()) {
    TaskLauncher task(
      compute_epts_task_id,
      ta,
      Predicate::TRUE_PRED,
      table_mapper);
    for (auto& r : treqs)
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
    for (auto& r : treqs)
      task.add_region_requirement(r);
    rt->execute_index_space(ctx, task);
  }
  for (auto& p : tparts)
    p.destroy(ctx, rt);
}

void
ATermIlluminationFunction::compute_aifs(
  Context ctx,
  Runtime* rt,
  const ATermZernikeModel& zmodel,
  const ColumnSpacePartition& partition,
  unsigned fftw_flags,
  double fftw_timelimit) const {

  std::vector<RegionRequirement> all_reqs;
  std::vector<ColumnSpacePartition> all_parts;

  // execute evaluate_polynomials_task
  {
    EvaluatePolynomialsTaskArgs args;

    auto ro_colreqs = Column::default_requirements;
    ro_colreqs.values.privilege = READ_ONLY;
    ro_colreqs.values.mapped = true;

    // ATermZernikeModel table, polynomial coefficients column
    {
      auto reqs =
        zmodel.requirements(
          ctx,
          rt,
          partition,
          {{ATermZernikeModel::PC_NAME, ro_colreqs}},
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
      args.zmodel = tdesc;
    }
    // this table, READ_ONLY privileges on epts columns, WRITE_DISCARD on values
    {
      auto wd_colreqs = Column::default_requirements;
      wd_colreqs.values.privilege = WRITE_DISCARD;
      wd_colreqs.values.mapped = true;
      auto reqs =
        requirements(
          ctx,
          rt,
          partition,
          {{EPT_X_NAME, ro_colreqs},
           {EPT_Y_NAME, ro_colreqs},
           {CF_VALUE_COLUMN_NAME, wd_colreqs}},
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
      args.aif = tdesc;
    }
    TaskArgument ta(&args, sizeof(args));
    if (!partition.is_valid()) {
      TaskLauncher task(
        evaluate_polynomials_task_id,
        ta,
        Predicate::TRUE_PRED,
        table_mapper);
      for (auto& r : all_reqs)
        task.add_region_requirement(r);
      rt->execute_task(ctx, task);
    } else {
      IndexTaskLauncher task(
        evaluate_polynomials_task_id,
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
  // FFT on the values region
  {
    // this table, READ_WRITE privileges on values
    auto rw_colreqs = Column::default_requirements;
    rw_colreqs.values.privilege = LEGION_READ_WRITE;
    rw_colreqs.values.mapped = true;
    auto reqs =
      requirements(
        ctx,
        rt,
        partition,
        {{CF_VALUE_COLUMN_NAME, rw_colreqs}},
        CXX_OPTIONAL_NAMESPACE::nullopt);
#if HAVE_CXX17
    auto& [treqs, tparts, tdesc] = reqs;
#else // !HAVE_CXX17
    auto& treqs = std::get<0>(reqs);
    auto& tparts = std::get<1>(reqs);
    auto& tdesc = std::get<2>(reqs);
#endif // HAVE_CXX17
    // FFT::in_place needs a simple RegionRequirement: find the requirement for
    // values column
    RegionRequirement req;
    for (auto& r : treqs)
      if (r.privilege_fields.count(CF_VALUE_FID) > 0)
        req = r;
    assert(req.privilege_fields.size() == 1);
    FFT::Args args;
    args.desc.rank = 2;
    args.desc.precision =
      ((typeid(cf_fp_t) == typeid(float))
       ? FFT::Precision::SINGLE
       : FFT::Precision::DOUBLE);
    args.desc.transform = FFT::Type::C2C;
    args.desc.sign = -1;
    args.fid = CF_VALUE_FID;
    args.seconds = fftw_timelimit;
    args.flags = fftw_flags;
    if (!partition.is_valid()) {
      TaskLauncher
        task(FFT::in_place_task_id, TaskArgument(&args, sizeof(args)));
      task.add_region_requirement(req);
      rt->execute_task(ctx, task);
    } else {
      IndexTaskLauncher task(
        FFT::in_place_task_id,
        rt->get_index_partition_color_space(ctx, partition.column_ip),
        TaskArgument(&args, sizeof(args)),
        ArgumentMap());
      task.add_region_requirement(req);
      rt->execute_index_space(ctx, task);
    }
    for (auto& p : all_parts)
      p.destroy(ctx, rt);
  }
}

void
ATermIlluminationFunction::preregister_tasks() {
  //
  // compute_epts_task
  //
  {
    compute_epts_task_id = Runtime::generate_static_task_id();

    // the only table with two columns sharing an index space is
    // ATermIlluminationFunction, using EPT_X and EPT_Y; use an AOS layout for
    // CPUs as default for that reason
#ifdef HYPERION_USE_KOKKOS
# if defined(KOKKOS_ENABLE_SERIAL) && defined(ENABLE_KOKKOS_SERIAL_EPTS_TASK)
    {
      TaskVariantRegistrar
        registrar(compute_epts_task_id, compute_epts_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();

      // standard column layout
      LayoutConstraintRegistrar
        constraints(
          FieldSpace::NO_SPACE,
          "ATermIlluminationFunction::compute_epts_constraints");
      add_aos_right_ordering_constraint(constraints);
      constraints.add_constraint(
        SpecializedConstraint(LEGION_AFFINE_SPECIALIZE));
      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        Runtime::preregister_layout(constraints));

      Runtime::preregister_task_variant<compute_epts_task<Kokkos::Serial>>(
        registrar,
        compute_epts_task_name);
    }
# endif // KOKKOS_ENABLE_SERIAL
# if defined(KOKKOS_ENABLE_OPENMP) && defined(ENABLE_KOKKOS_OPENMP_EPTS_TASK)
    {
      TaskVariantRegistrar
        registrar(compute_epts_task_id, compute_epts_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();

      // standard column layout
      LayoutConstraintRegistrar
        constraints(
          FieldSpace::NO_SPACE,
          "ATermIlluminationFunction::compute_epts_constraints");
      add_aos_right_ordering_constraint(constraints);
      constraints.add_constraint(
        SpecializedConstraint(LEGION_AFFINE_SPECIALIZE));
      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        Runtime::preregister_layout(constraints));

      Runtime::preregister_task_variant<compute_epts_task<Kokkos::OpenMP>>(
        registrar,
        compute_epts_task_name);
    }
# endif // KOKKOS_ENABLE_OPENMP
# if defined(KOKKOS_ENABLE_CUDA) && defined(ENABLE_KOKKOS_CUDA_EPTS_TASK)
    {
      TaskVariantRegistrar
        registrar(compute_epts_task_id, compute_epts_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();

      // standard column layout
      LayoutConstraintRegistrar
        constraints(
          FieldSpace::NO_SPACE,
          "ATermIlluminationFunction::compute_epts_constraints");
      add_soa_left_ordering_constraint(constraints);
      constraints.add_constraint(
        SpecializedConstraint(LEGION_AFFINE_SPECIALIZE));
      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        Runtime::preregister_layout(constraints));

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

      // standard column layout
      LayoutConstraintRegistrar
        constraints(
          FieldSpace::NO_SPACE,
          "ATermIlluminationFunction::compute_epts_constraints");
      add_aos_right_ordering_constraint(constraints);
      constraints.add_constraint(
        SpecializedConstraint(LEGION_AFFINE_SPECIALIZE));
      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        Runtime::preregister_layout(constraints));

      Runtime::preregister_task_variant<compute_epts_task>(
        registrar,
        compute_epts_task_name);
    }
#endif // HYPERION_USE_KOKKOS
  }

  //
  // evaluate_polynomials_task
  //
  {
    evaluate_polynomials_task_id = Runtime::generate_static_task_id();

    // the only table with two columns sharing an index space is
    // ATermIlluminationFunction, using EPT_X and EPT_Y; use an AOS layout for
    // CPUs as default for that reason
#ifdef HYPERION_USE_KOKKOS
# if defined(KOKKOS_ENABLE_SERIAL) && defined(ENABLE_KOKKOS_SERIAL_EVAL_POLY_TASK)
    {
      TaskVariantRegistrar
        registrar(evaluate_polynomials_task_id, evaluate_polynomials_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();

      // standard column layout
      LayoutConstraintRegistrar
        constraints(
          FieldSpace::NO_SPACE,
          "ATermIlluminationFunction::evaluate_polynomials_constraints");
      add_aos_right_ordering_constraint(constraints);
      constraints.add_constraint(
        SpecializedConstraint(LEGION_AFFINE_SPECIALIZE));
      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        Runtime::preregister_layout(constraints));

      Runtime::preregister_task_variant<evaluate_polynomials_task<Kokkos::Serial>>(
        registrar,
        evaluate_polynomials_task_name);
    }
# endif // KOKKOS_ENABLE_SERIAL
# if defined(KOKKOS_ENABLE_OPENMP) && defined(ENABLE_KOKKOS_OPENMP_EVAL_POLY_TASK)
    {
      TaskVariantRegistrar
        registrar(evaluate_polynomials_task_id, evaluate_polynomials_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();

      // standard column layout
      LayoutConstraintRegistrar
        constraints(
          FieldSpace::NO_SPACE,
          "ATermIlluminationFunction::evaluate_polynomials_constraints");
      add_aos_right_ordering_constraint(constraints);
      constraints.add_constraint(
        SpecializedConstraint(LEGION_AFFINE_SPECIALIZE));
      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        Runtime::preregister_layout(constraints));

      Runtime::preregister_task_variant<evaluate_polynomials_task<Kokkos::OpenMP>>(
        registrar,
        evaluate_polynomials_task_name);
    }
# endif // KOKKOS_ENABLE_OPENMP
# if defined(KOKKOS_ENABLE_CUDA) && defined(ENABLE_KOKKOS_CUDA_EVAL_POLY_TASK)
    {
      TaskVariantRegistrar
        registrar(evaluate_polynomials_task_id, evaluate_polynomials_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();

      // standard column layout
      LayoutConstraintRegistrar
        constraints(
          FieldSpace::NO_SPACE,
          "ATermIlluminationFunction::evaluate_polynomials_constraints");
      add_soa_left_ordering_constraint(constraints);
      constraints.add_constraint(
        SpecializedConstraint(LEGION_AFFINE_SPECIALIZE));
      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        Runtime::preregister_layout(constraints));

      Runtime::preregister_task_variant<evaluate_polynomials_task<Kokkos::Cuda>>(
        registrar,
        evaluate_polynomials_task_name);
    }
# endif // KOKKOS_ENABLE_CUDA
#else // !HYPERION_USE_KOKKOS
    {
      TaskVariantRegistrar
        registrar(evaluate_polynomials_task_id, evaluate_polynomials_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();

      // standard column layout
      LayoutConstraintRegistrar
        constraints(
          FieldSpace::NO_SPACE,
          "ATermIlluminationFunction::evaluate_polynomials_constraints");
      add_aos_right_ordering_constraint(constraints);
      constraints.add_constraint(
        SpecializedConstraint(LEGION_AFFINE_SPECIALIZE));
      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        Runtime::preregister_layout(constraints));

      Runtime::preregister_task_variant<evaluate_polynomials_task>(
        registrar,
        evaluate_polynomials_task_name);
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
