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
#include <hyperion/synthesis/ATermZernikeModel.h>

#include <cmath>
#include <cstring>
#include <limits>

using namespace hyperion;
using namespace hyperion::synthesis;
using namespace Legion;

namespace stdex = std::experimental;

TaskID ATermZernikeModel::compute_pcs_task_id;
#if !HAVE_CXX17
const constexpr float ATermZernikeModel::zc_exact_frequency_tolerance;
const constexpr unsigned ATermZernikeModel::d_blc;
const constexpr unsigned ATermZernikeModel::d_frq;
const constexpr unsigned ATermZernikeModel::d_sto;
const constexpr Legion::FieldID ATermZernikeModel::ZC_FID;
const constexpr char* ATermZernikeModel::ZC_NAME;
const constexpr Legion::FieldID ATermZernikeModel::PC_FID;
const constexpr char* ATermZernikeModel::PC_NAME;
#endif

#define USE_KOKKOS_SERIAL_COMPUTE_PCS_TASK // undef to disable
#if defined(USE_KOKKOS_SERIAL_COMPUTE_PCS_TASK) &&  \
  defined(HYPERION_USE_KOKKOS) &&                   \
  defined(KOKKOS_ENABLE_SERIAL)
# define ENABLE_KOKKOS_SERIAL_COMPUTE_PCS_TASK
#else
# undef ENABLE_KOKKOS_SERIAL_COMPUTE_PCS_TASK
#endif

#define USE_KOKKOS_OPENMP_COMPUTE_PCS_TASK // undef to disable
#if defined(USE_KOKKOS_OPENMP_COMPUTE_PCS_TASK) &&  \
  defined(HYPERION_USE_KOKKOS) &&                   \
  defined(KOKKOS_ENABLE_OPENMP)
# define ENABLE_KOKKOS_OPENMP_COMPUTE_PCS_TASK
#else
# undef ENABLE_KOKKOS_OPENMP_COMPUTE_PCS_TASK
#endif

#undef USE_KOKKOS_CUDA_COMPUTE_PCS_TASK // define to enable
#if defined(USE_KOKKOS_CUDA_COMPUTE_PCS_TASK) &&  \
  defined(HYPERION_USE_KOKKOS) &&                 \
  defined(KOKKOS_ENABLE_CUDA)
# define ENABLE_KOKKOS_CUDA_COMPUTE_PCS_TASK
#else
# undef ENABLE_KOKKOS_CUDA_COMPUTE_PCS_TASK
#endif


ATermZernikeModel::ATermZernikeModel(
  Context ctx,
  Runtime* rt,
  const std::vector<ZCoeff>& zernike_coefficients,
  const std::vector<typename cf_table_axis<CF_BASELINE_CLASS>::type>&
    baseline_classes,
  const std::vector<typename cf_table_axis<CF_FREQUENCY>::type>&
    frequencies,
  const std::vector<typename cf_table_axis<CF_STOKES>::type>&
    stokes_values)
  : CFTable(
    ctx,
    rt,
    Rect<2>(Point<2>::ZEROES(), Point<2>::ZEROES()),
    Axis<CF_BASELINE_CLASS>(baseline_classes),
    Axis<CF_FREQUENCY>(frequencies),
    Axis<CF_STOKES>(stokes_values)) {

  unsigned zernike_order = 0;
  for (auto& zc : zernike_coefficients)
    zernike_order = std::max(zernike_order, zc.n);

  // Add ZC and PC columns to table
  Table::fields_t tflds;
  {
    // ZC column
    size_t
      c_n[4]{
         baseline_classes.size(),
         frequencies.size(),
         stokes_values.size(),
         zernike_num_terms(zernike_order)};
    Point<4> n(c_n);
    Rect<4> rect(Point<4>::ZEROES(), n - Point<4>::ONES());
    auto is = rt->create_index_space(ctx, rect);
    auto cs =
      ColumnSpace::create<cf_table_axes_t>(
        ctx,
        rt,
        {CF_BASELINE_CLASS, CF_FREQUENCY, CF_STOKES, CF_ORDER0},
        is,
        false);
    tflds.push_back(
      {cs,
       {{ZC_NAME, TableField(ValueType<zc_t>::DataType, ZC_FID)}}});
  }
  {
    // PC column
    size_t
      c_n[5]{
         baseline_classes.size(),
         frequencies.size(),
         stokes_values.size(),
         zernike_order + 1,
         zernike_order + 1};
    Point<5> n(c_n);
    Rect<5> rect(Point<5>::ZEROES(), n - Point<5>::ONES());
    auto is = rt->create_index_space(ctx, rect);
    auto cs =
      ColumnSpace::create<cf_table_axes_t>(
        ctx,
        rt,
        {CF_BASELINE_CLASS, CF_FREQUENCY, CF_STOKES, CF_ORDER0, CF_ORDER1},
        is,
        false);
    tflds.push_back(
      {cs,
       {{PC_NAME, TableField(ValueType<pc_t>::DataType, PC_FID)}}});
  }
  add_columns(ctx, rt, std::move(tflds));

  init_zc_region(
    ctx,
    rt,
    zernike_coefficients,
    baseline_classes,
    frequencies,
    stokes_values);
}

void
ATermZernikeModel::init_zc_region(
  Context ctx,
  Runtime* rt,
  const std::vector<ZCoeff>& zernike_coefficients,
  const std::vector<typename cf_table_axis<CF_BASELINE_CLASS>::type>&
    baseline_classes,
  const std::vector<typename cf_table_axis<CF_FREQUENCY>::type>&
    frequencies,
  const std::vector<typename cf_table_axis<CF_STOKES>::type>&
    stokes_values) const {

  std::vector<ZCoeff> zcs = zernike_coefficients;
  // copy (needed) coefficients from zernike_coefficients to ZC_FID field
  using_resource(
    [&]() {
      auto reqs = Column::default_requirements;
      reqs.values.mapped = true;
      reqs.values.privilege = WRITE_ONLY;
      return
        map_inline(ctx, rt, {{ZC_NAME, reqs}}, CXX_OPTIONAL_NAMESPACE::nullopt);
    },
    [&](PhysicalTable& tbl) {
      auto col = ZCColumn<AffineAccessor>(*tbl.column(ZC_NAME).value());
      auto zc_rect = col.rect();
      auto values = col.accessor<WRITE_ONLY>();
      for (size_t blc = 0; blc < baseline_classes.size(); ++blc) {
        auto& baseline_class = baseline_classes[blc];
        auto zcs_blc_end =
          std::partition(
            zcs.begin(),
            zcs.end(),
            [&baseline_class](auto& zc) {
              return zc.baseline_class == baseline_class;
            });
        for (size_t sto = 0; sto <= stokes_values.size(); ++sto) {
          auto& stokes_value = stokes_values[sto];
          auto zcs_blc_sto_end =
            std::partition(
              zcs.begin(),
              zcs_blc_end,
              [&stokes_value](auto& zc) {
                return zc.stokes == stokes_value;
              });
          for (size_t frq = 0; frq <= frequencies.size(); ++frq) {
            auto frequency = frequencies[frq];
            // first, we determine the nearest frequency in zernike_coefficients
            auto nearest_frequency =
              std::numeric_limits<typename cf_table_axis<CF_FREQUENCY>::type>
              ::quiet_NaN();
            {
              auto frequency_diff = std::abs(frequency - nearest_frequency);
              for (auto zc = zcs.begin(); zc != zcs_blc_sto_end; ++zc) {
                auto d = std::abs(frequency - zc->frequency);
                if (std::isnan(frequency_diff) || d < frequency_diff) {
                  frequency_diff = d;
                  nearest_frequency = zc->frequency;
                  if (d <= zc_exact_frequency_tolerance * frequency)
                    break;
                }
              }
            }
            Point<zc_rank> p;
            p[d_blc] = blc;
            p[d_frq] = frq;
            p[d_sto] = sto;
            // set all coefficients for this stokes value and frequency to zero
            for (coord_t i = zc_rect.lo[d_zc]; i <= zc_rect.hi[d_zc]; ++i) {
              p[d_zc] = i;
              values[p] = static_cast<zc_t>(0);
            }
            // copy all coefficients that match nearest_frequency, stokes and
            // baseline_class values from zernike_coefficients
            if (!std::isnan(nearest_frequency)) {
              for (auto zc = zcs.begin(); zc != zcs_blc_sto_end; ++zc) {
                if (zc->frequency == nearest_frequency) {
                  p[d_zc] = zernike_index(zc->m, zc->n);
                  values[p] = zc->coefficient;
                }
              }
            }
          }
        }
      }
    },
    [&](PhysicalTable& tbl) {
      tbl.unmap_regions(ctx, rt);
    });
}

void
ATermZernikeModel::compute_pcs(
  Context ctx,
  Runtime* rt,
  const ColumnSpacePartition& partition) const {

  auto zc_reqs = Column::default_requirements;
  zc_reqs.values.mapped = true;
  auto pc_reqs = Column::default_requirements;
  pc_reqs.values.privilege = WRITE_ONLY;
  pc_reqs.values.mapped = true;

  auto reqs =
    requirements(
      ctx,
      rt,
      partition,
      {{ZC_NAME, zc_reqs}, {PC_NAME, pc_reqs}},
      CXX_OPTIONAL_NAMESPACE::nullopt);
#if HAVE_CXX17
  auto& [treqs, tparts, tdesc] = reqs;
#else // !HAVE_CXX17
  auto& treqs = std::get<0>(reqs);
  auto& tparts = std::get<1>(reqs);
  auto& tdesc = std::get<2>(reqs);
#endif // HAVE_CXX17
  if (!partition.is_valid()) {
    TaskLauncher task(
      compute_pcs_task_id,
      TaskArgument(&tdesc, sizeof(tdesc)),
      Predicate::TRUE_PRED,
      table_mapper);
    for (auto& r : treqs)
      task.add_region_requirement(r);
    rt->execute_task(ctx, task);
  } else {
    IndexTaskLauncher task(
      compute_pcs_task_id,
      rt->get_index_partition_color_space(ctx, partition.column_ip),
      TaskArgument(&tdesc, sizeof(tdesc)),
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

#ifndef HYPERION_USE_KOKKOS
void
ATermZernikeModel::compute_pcs_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  const Table::Desc& tdesc = *static_cast<Table::Desc*>(task->args);

  // ATermZernikeModel physical instance
  auto ptcr =
    PhysicalTable::create(
      rt,
      tdesc,
      task->regions.begin(),
      task->regions.end(),
      regions.begin(),
      regions.end())
    .value();
#if HAVE_CXX17
  auto& [tbl, rit, pit] = ptcr;
#else // !HAVE_CXX17
  auto& tbl = std::get<0>(ptcr);
  auto& rit = std::get<1>(ptcr);
  auto& pit = std::get<2>(ptcr);
#endif // HAVE_CXX17
  assert(rit == task->regions.end());
  assert(pit == regions.end());

  // Zernike coefficients
  auto zc_col =
    ZCColumn<AffineAccessor>(*tbl.column(ZC_NAME).value());
  auto zc_rect = zc_col.rect();
  auto zcs = zc_col.span<READ_ONLY>();

  // polynomial function coefficients
  auto pc_col = PCColumn<AffineAccessor>(*tbl.column(PC_NAME).value());
  auto pcs = pc_col.span<WRITE_ONLY>();

  for (coord_t blc = zc_rect.lo[d_blc]; blc <= zc_rect.hi[d_blc]; ++blc)
    for (coord_t frq = zc_rect.lo[d_frq]; frq <= zc_rect.hi[d_frq]; ++frq)
      for (coord_t sto = zc_rect.lo[d_sto]; sto <= zc_rect.hi[d_sto]; ++sto) {

        auto zcs0 = stdex::subspan(zcs, blc, frq, sto, stdex::all);
        auto pcs0 = stdex::subspan(pcs, blc, frq, sto, stdex::all, stdex::all);
        switch (zcs.extent(3) - 1) {
#define ZEXP(N)                                         \
          case N:                                       \
            zernike_basis<zc_t, N>::expand(pcs0, zcs0); \
            break
          ZEXP(0);
          ZEXP(1);
          ZEXP(2);
          ZEXP(3);
          ZEXP(4);
          ZEXP(5);
          ZEXP(6);
          ZEXP(7);
          ZEXP(8);
          ZEXP(9);
          ZEXP(10);
#undef ZEXP
        default:
          assert(false);
          break;
        }
      }
}
#endif // !HYPERION_USE_KOKKOS

void
ATermZernikeModel::preregister_tasks() {
  //
  // compute_pcs_task
  //
  {
#if defined(ENABLE_KOKKOS_SERIAL_COMPUTE_PCS_TASK) || \
  defined(ENABLE_KOKKOS_OPENMP_COMPUTE_PCS_TASK) ||   \
  !defined(HYPERION_USE_KOKKOS)
    LayoutConstraintRegistrar
      cpu_constraints(
        FieldSpace::NO_SPACE,
        "ATermZernikeModel::compute_pcs");
    add_soa_right_ordering_constraint(cpu_constraints);
    cpu_constraints.add_constraint(
      SpecializedConstraint(LEGION_AFFINE_SPECIALIZE));
    auto cpu_layout_id = Runtime::preregister_layout(cpu_constraints);
#endif

#if defined(ENABLE_KOKKOS_CUDA_COMPUTE_PCS_TASK)
    LayoutConstraintRegistrar
      gpu_constraints(FieldSpace::NO_SPACE, "ATermZernikeModel::compute_cfs");
    add_soa_left_ordering_constraint(gpu_constraints);
    gpu_constraints.add_constraint(
      SpecializedConstraint(LEGION_AFFINE_SPECIALIZE));
    auto gpu_layout_id = Runtime::preregister_layout(gpu_constraints);
#endif

    compute_pcs_task_id = Runtime::generate_static_task_id();

#ifdef ENABLE_KOKKOS_SERIAL_COMPUTE_PCS_TASK
    {
      TaskVariantRegistrar
        registrar(compute_pcs_task_id, compute_pcs_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();

      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        cpu_layout_id);

      Runtime::preregister_task_variant<compute_pcs_task<Kokkos::Serial>>(
        registrar,
        compute_pcs_task_name);
    }
#endif

#ifdef ENABLE_KOKKOS_OPENMP_COMPUTE_PCS_TASK
    {
      TaskVariantRegistrar
        registrar(compute_pcs_task_id, compute_pcs_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();

      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        cpu_layout_id);

      Runtime::preregister_task_variant<compute_pcs_task<Kokkos::OpenMP>>(
        registrar,
        compute_pcs_task_name);
    }
#endif

#ifdef ENABLE_KOKKOS_CUDA_COMPUTE_PCS_TASK
    {
      TaskVariantRegistrar
        registrar(compute_pcs_task_id, compute_pcs_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();
      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        gpu_layout_id);
      Runtime::preregister_task_variant<compute_pcs_task<Kokkos::Cuda>>(
        registrar,
        compute_pcs_task_name);
    }
#endif

#ifndef HYPERION_USE_KOKKOS
    {
      TaskVariantRegistrar
        registrar(compute_pcs_task_id, compute_pcs_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();

      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        cpu_layout_id);

      Runtime::preregister_task_variant<compute_pcs_task>(
        registrar,
        compute_pcs_task_name);
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
