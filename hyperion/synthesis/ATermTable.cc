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
#include <hyperion/synthesis/ATermTable.h>

#include <cmath>
#include <cstring>
#include <limits>

using namespace hyperion;
using namespace hyperion::synthesis;
using namespace Legion;

TaskID ATermTable::compute_cfs_task_id;
TaskID ATermTable::compute_pcs_task_id;
#if !HAVE_CXX17
constexpr const float ATermTable::zc_exact_frequency_tolerance;
#endif

ATermTable::ATermTable(
  Context ctx,
  Runtime* rt,
  const std::array<coord_t, 2>& cf_bounds_lo,
  const std::array<coord_t, 2>& cf_bounds_hi,
  unsigned zernike_order,
  const std::vector<ZCoeff>& zernike_coefficients,
  const std::vector<typename cf_table_axis<CF_BASELINE_CLASS>::type>&
    baseline_classes,
  const std::vector<typename cf_table_axis<CF_PARALLACTIC_ANGLE>::type>&
    parallactic_angles,
  const std::vector<typename cf_table_axis<CF_STOKES>::type>&
    stokes_values,
  const std::vector<typename cf_table_axis<CF_FREQUENCY>::type>&
    frequencies)
  : CFTable(
    ctx,
    rt,
    Rect<2>(
      {cf_bounds_lo[0], cf_bounds_lo[1]},
      {cf_bounds_hi[0], cf_bounds_hi[1]}),
    Axis<CF_BASELINE_CLASS>(baseline_classes),
    Axis<CF_PARALLACTIC_ANGLE>(parallactic_angles),
    Axis<CF_STOKES>(stokes_values),
    Axis<CF_FREQUENCY>(frequencies))
  , m_zernike_order(zernike_order) {

  create_zc_region(
    ctx,
    rt,
    zernike_order,
    zernike_coefficients,
    baseline_classes,
    stokes_values,
    frequencies);
  create_pc_region(
    ctx,
    rt,
    zernike_order,
    baseline_classes.size(),
    stokes_values.size(),
    frequencies.size());
}

ATermTable::ATermTable(
  Context ctx,
  Runtime* rt,
  const coord_t& cf_x_radius,
  const coord_t& cf_y_radius,
  unsigned zernike_order,
  const std::vector<ZCoeff>& zernike_coefficients,
  const std::vector<typename cf_table_axis<CF_BASELINE_CLASS>::type>&
    baseline_classes,
  const std::vector<typename cf_table_axis<CF_PARALLACTIC_ANGLE>::type>&
    parallactic_angles,
  const std::vector<typename cf_table_axis<CF_STOKES>::type>&
    stokes_values,
  const std::vector<typename cf_table_axis<CF_FREQUENCY>::type>&
    frequencies)
  : CFTable(
    ctx,
    rt,
    Rect<2>({-cf_x_radius, -cf_y_radius}, {cf_x_radius, cf_y_radius}),
    Axis<CF_BASELINE_CLASS>(baseline_classes),
    Axis<CF_PARALLACTIC_ANGLE>(parallactic_angles),
    Axis<CF_STOKES>(stokes_values),
    Axis<CF_FREQUENCY>(frequencies))
  , m_zernike_order(zernike_order) {

  create_zc_region(
    ctx,
    rt,
    zernike_order,
    zernike_coefficients,
    baseline_classes,
    stokes_values,
    frequencies);
  create_pc_region(
    ctx,
    rt,
    zernike_order,
    baseline_classes.size(),
    stokes_values.size(),
    frequencies.size());
}

void
ATermTable::create_zc_region(
  Context ctx,
  Runtime* rt,
  unsigned zernike_order,
  const std::vector<ZCoeff>& zernike_coefficients,
  const std::vector<typename cf_table_axis<CF_BASELINE_CLASS>::type>&
    baseline_classes,
  const std::vector<typename cf_table_axis<CF_STOKES>::type>&
    stokes_values,
  const std::vector<typename cf_table_axis<CF_FREQUENCY>::type>&
    frequencies) {

  // create region for Zernike coefficients
  Rect<4> zc_rect(
    {0, 0, 0, 0},
    {static_cast<coord_t>(baseline_classes.size()) - 1,
     static_cast<coord_t>(stokes_values.size()) - 1,
     static_cast<coord_t>(frequencies.size()) - 1,
     static_cast<coord_t>(zernike_num_terms(zernike_order)) - 1});
  {
    auto is = rt->create_index_space(ctx, zc_rect);
    auto fs = rt->create_field_space(ctx);
    auto fa = rt->create_field_allocator(ctx, fs);
    fa.allocate_field(sizeof(zc_t), ZC_FID);
    fa.allocate_field(sizeof(pc_t), PC_FID);
    m_zc_region = rt->create_logical_region(ctx, is, fs);
  }
  // copy (needed) coefficients from zernike_coefficients to m_zc_region ZC_FID
  // field
  {
    RegionRequirement req(m_zc_region, WRITE_ONLY, EXCLUSIVE, m_zc_region);
    req.add_field(ZC_FID);
    auto pr = rt->map_region(ctx, req);
    const FieldAccessor<
      WRITE_ONLY,
      zc_t,
      4,
      coord_t,
      AffineAccessor<zc_t, 4, coord_t>> acc(pr, ZC_FID);
    for (coord_t blc = 0; blc <= zc_rect.hi[0]; ++blc) {
      auto baseline_class = baseline_classes[blc];
      for (coord_t frq = 0; frq <= zc_rect.hi[2]; ++frq) {
        auto frequency = frequencies[frq];
        // first, we determine the nearest frequency in zernike_coefficients
        // (the following assumes bit-wise identical frequency values across all
        // stokes values associated with a nominal frequency (and
        // baseline_class) in zernike_coefficients)
        auto nearest_frequency =
          std::numeric_limits<typename cf_table_axis<CF_FREQUENCY>::type>
          ::quiet_NaN();
        {
          auto frequency_diff = std::abs(frequency - nearest_frequency);
          for (auto& zcoeff : zernike_coefficients) {
            auto d = std::abs(frequency - zcoeff.frequency);
            if (baseline_class == zcoeff.baseline_class
                && (std::isnan(frequency_diff) || d < frequency_diff)) {
              frequency_diff = d;
              nearest_frequency = zcoeff.frequency;
              if (d <= zc_exact_frequency_tolerance * frequency)
                break;
            }
          }
        }
        for (coord_t sto = 0; sto <= zc_rect.hi[1]; ++sto) {
          auto stokes = stokes_values[sto];
          Point<4> p{blc, sto, frq, 0};
          // set all coefficients for this stokes and frequency to zero
          for (coord_t i = 0; i <= zc_rect.hi[3]; ++i) {
            p[3] = i;
            acc[p] = static_cast<zc_t>(0);
          }

          // copy all coefficients that match nearest_frequency, stokes and
          // baseline_class values from zernike_coefficients
          if (!std::isnan(nearest_frequency)) {
            for (auto& zcoeff : zernike_coefficients) {
              if (zcoeff.stokes == stokes
                  && zcoeff.frequency == nearest_frequency) {
                p[2] = zernike_index(zcoeff.m, zcoeff.n);
                acc[p] = zcoeff.coefficient;
              }
            }
          }
        }
      }
    }
    rt->unmap_region(ctx, pr);
  }
}

void
ATermTable::create_pc_region(
  Context ctx,
  Runtime* rt,
  unsigned zernike_order,
  unsigned num_baseline_classes,
  unsigned num_stokes_values,
  unsigned num_frequencies) {

  coord_t nzt = zernike_num_terms(zernike_order);
  coord_t
    hi[5]{num_baseline_classes - 1,
      num_stokes_values - 1,
      num_frequencies - 1,
      nzt - 1,
      nzt - 1};
  Rect<5> pc_rect(Point<5>::ZEROES(), Point<5>(hi));
  auto is = rt->create_index_space(ctx, pc_rect);
  auto fs = rt->create_field_space(ctx);
  auto fa = rt->create_field_allocator(ctx, fs);
  fa.allocate_field(sizeof(pc_t), PC_FID);
  m_pc_region = rt->create_logical_region(ctx, is, fs);

  TaskLauncher task(compute_pcs_task_id, TaskArgument());
  {
    RegionRequirement req(m_zc_region, READ_ONLY, EXCLUSIVE, m_zc_region);
    req.add_field(ZC_FID);
    task.add_region_requirement(req);
  }
  {
    RegionRequirement req(m_pc_region, WRITE_ONLY, EXCLUSIVE, m_pc_region);
    req.add_field(PC_FID);
    task.add_region_requirement(req);
  }
  rt->execute_task(ctx, task);
}

void
ATermTable::compute_cfs(
  Context ctx,
  Runtime* rt,
  const ColumnSpacePartition& partition) const {

    auto cf_colreqs = Column::default_requirements;
    cf_colreqs.values = Column::Req{
      WRITE_DISCARD /* privilege */,
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
    RegionRequirement pc_req(m_pc_region, READ_ONLY, EXCLUSIVE, m_pc_region);
    pc_req.add_field(PC_FID);
    task.add_region_requirement(pc_req);
    for (auto& r : treqs)
      task.add_region_requirement(r);
    rt->execute_task(ctx, task);
    for (auto& p : tparts)
      p.destroy(ctx, rt);
}

#ifndef HYPERION_USE_KOKKOS
void
ATermTable::compute_cfs_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  const Table::Desc& tdesc = *static_cast<const Table::Desc*>(task->args);

  // polynomial coefficients region (the array Z, as described elsewhere),
  // computed for each value of (baseline_class, stokes_value, frequency)
  const FieldAccessor<
    READ_ONLY,
    pc_t,
    5,
    coord_t,
    AffineAccessor<pc_t, 5, coord_t>> pcoeffs(regions[0], PC_FID);
  Rect<5> pcoeffs_rect =
    rt->get_index_space_domain(task->regions[0].region.get_index_space());
  auto zernike_order = pcoeffs_rect.hi[4];

  // ATermTable physical instance
  auto ptcr =
    PhysicalTable::create(
      rt,
      tdesc,
      task->regions.begin() + 1,
      task->regions.end(),
      regions.begin() + 1,
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

  auto tbl = CFPhysicalTable<HYPERION_A_TERM_TABLE_AXES>(pt);

  // baseline class column
  auto baseline_class_col = tbl.baseline_class<AffineAccessor>();
  auto baseline_class_rect = baseline_class_col.rect();
  auto baseline_class_extent =
    baseline_class_rect.hi[0] - baseline_class_rect.lo[0] + 1;
  auto baseline_classes = baseline_class_col.accessor<READ_ONLY>();
  typedef typename cf_table_axis<CF_BASELINE_CLASS>::type bl_t;

  // frequency column
  auto frequency_col = tbl.frequency<AffineAccessor>();
  auto frequency_rect = frequency_col.rect();
  auto frequency_extent =
    frequency_rect.hi[0] - frequency_rect.lo[0] + 1;
  auto frequencies = frequency_col.accessor<READ_ONLY>();
  typedef typename cf_table_axis<CF_FREQUENCY>::type frq_t;

  // parallactic angle column
  auto parallactic_angle_col = tbl.parallactic_angle<AffineAccessor>();
  auto parallactic_angle_rect = parallactic_angle_col.rect();
  auto parallactic_angle_extent =
    parallactic_angle_rect.hi[0] - parallactic_angle_rect.lo[0] + 1;
  auto parallactic_angles = parallactic_angle_col.accessor<READ_ONLY>();
  typedef typename cf_table_axis<CF_PARALLACTIC_ANGLE>::type pa_t;

  // stokes column
  auto stokes_value_col = tbl.stokes<AffineAccessor>();
  auto stokes_value_rect = stokes_value_col.rect();
  auto stokes_value_extent =
    stokes_value_rect.hi[0] - stokes_value_rect.lo[0] + 1;
  auto stokes_values = stokes_value_col.accessor<READ_ONLY>();
  typedef typename cf_table_axis<CF_STOKES>::type me_t;

  // A-term cf values column
  auto cf_value_col = tbl.value<AffineAccessor>();
  auto cf_value_rect = cf_value_col.rect();
  auto cf_values = cf_value_col.accessor<WRITE_DISCARD>();
#ifdef A_TERM_TABLE_USE_BLAS
  // initialize cf_values values to 0; not sure that it's necessary, it depends
  // on whether GEMM accesses the array even when beta is 0
  for (PointInRectIterator pir(cf_value_rect); pir(); pir++)
    cf_values[*pir] = 0;
#endif

  // the matrices X and Y (for each parallactic angle); these need to have
  // complex values in order to use (optional) GEMM for the matrix
  // multiplication implementation
  std::vector<pc_t> xp_buff(
    parallactic_angle_extent * (zernike_order + 1)
    * (cf_value_rect.hi[4] - cf_value_rect.lo[4] + 1));
  std::experimental::mdspan<
    pc_t***,
    std::experimental::dynamic_extent,
    std::experimental::dynamic_extent,
    std::experimental::dynamic_extent> xp(
      xp_buff.data(),
      parallactic_angle_extent,
      zernike_order + 1,
      cf_value_rect.hi[4] - cf_value_rect.lo[4] + 1);
  std::vector<pc_t> yp_buff(
    parallactic_angle_extent * (zernike_order + 1)
    * (cf_value_rect.hi[5] - cf_value_rect.lo[5] + 1));
  std::experimental::mdspan<
    pc_t***,
    std::experimental::dynamic_extent,
    std::experimental::dynamic_extent,
    std::experimental::dynamic_extent> yp(
      yp_buff.data(),
      parallactic_angle_extent,
      zernike_order + 1,
      cf_value_rect.hi[5] - cf_value_rect.lo[5] + 1);
  {
    assert(xp.extent(2) == yp.extent(2)); // squares only
    std::vector<double> g(xp.extent(2));
    const double step = 2.0 / xp.extent(2);
    const double offset = -1.0 + step / 2.0;
    for (int i = 0; i < g.size(); ++i)
      g[i] = i * step + offset;

    for (coord_t pa0 = 0; pa0 < parallactic_angle_extent; ++pa0)
      for (coord_t x0 = 0; x < xp.extent(0); ++x)
        for (coord_t y0 = 0; y < yp.extent(0); ++y) {
          auto powx =
            std::experimental::subspan(xp, pa0, std::experimental::all, x0);
          auto powy =
            std::experimental::subspan(yp, pa0, std::experimental::all, y0);

          // apply parallactic angle rotation
          auto neg_parallactic_angle =
            -parallactic_angles(pa0 + parallactic_angle_rect.lo[0]);
          double cs = std::cos(neg_parallactic_angle);
          double sn = std::sin(neg_parallactic_angle);
          double rx = cs * g(x0) - sn * g(y0);
          double ry = sn * g(x0) + cs * g(y0);
          // Outside of the unit disk, the function should evaluate to zero, which
          // can be achieved by setting the X and Y vectors to zero. Recall that
          // xp and yp were created without value initialization.
          powx(0) = powy(0) = ((rx * rx + ry * ry <= 1.0) ? 1.0 : 0.0);
          for (unsigned d = 1; d <= zernike_order; ++d) {
            powx(d) = rx * powx(d - 1).real();
            powy(d) = ry * powy(d - 1).real();
          }
        }
  }

  // do X^T Z Y products
  {
    // do multiplication as Q = Z Y, result = X^T Q; need a scratch array for
    // Q
    std::vector<pc_t> zy_buff(
      baseline_class_extent * parallactic_angle_extent * stokes_value_extent
      * frequency_extent * yp.extent(1) * yp.extent(2));
    std::experimental::mdspan<
      pc_t******,
      std::experimental::dynamic_extent,
      std::experimental::dynamic_extent,
      std::experimental::dynamic_extent,
      std::experimental::dynamic_extent,
      std::experimental::dynamic_extent,
      std::experimental::dynamic_extent> zy(
        zy_buff,
        baseline_class_extent,
        parallactic_angle_extent,
        stokes_value_extent,
        frequency_extent,
        yp.extent(1),
        yp.extent(2));
    for (coord_t blc0 = 0; blc0 < baseline_class_extent; ++blc0)
      for (coord_t pa0 = 0; pa0 < parallactic_angle_extent; ++pa0)
        for (coord_t sto0 = 0; sto0 < stokes_value_extent; ++sto0)
          for (coord_t frq0 = 0; frq0 < frequency_extent; ++frq0) {
#ifndef A_TERM_TABLE_USE_BLAS
            coord_t p5[]{
              blc0 + pcoeffs_rect.lo[0],
              sto0 + pcoeffs_rect.lo[1],
              frq0 + pcoeffs_rect.lo[2],
              0,
              0};
            Point<5> pz(p5);
            auto X =
              std::experimental::subspan(
                xp,
                pa0,
                std::experimental::all,
                std::experimental::all);
            auto Y =
              std::experimental::subspan(
                yp,
                pa0,
                std::experimental::all,
                std::experimental::all);
            auto Q =
              std::experimental::subspan(
                zy,
                blc0,
                pa0,
                sto0,
                frq0,
                std::experimental::all,
                std::experimental::all);
            coord_t p6[]{
              blc0 + cf_value_rect.lo[0],
              pa0 + cf_value_rect.lo[1],
              sto0 + cf_value_rect.lo[2],
              frq0 + cf_value_rect.lo[3],
              0,
              0};
            Point<5> pcf(p6);
            for (coord_t i = 0; i < Z.extent(0); ++i) {
              pz[3] = i;
              for (coord_t j = 0; j < Y.extent(1); ++j) {
                Q(i, j) = 0.0;
                for (coord_t k = 0; k < Y.extent(0); ++k) {
                  pz[4] = k;
                  Q(i, j) += pcoeffs[pz] * Y(k, j);
                }
              }
            }
            for (coord_t i = 0; i < X.extent(1); ++i)
              pcf[4] = i;
              for (coord_t j = 0; j < Q.extent(1); ++j) {
                pcf[5] = j;
                cf_values[pcf] = 0.0;
                for (coord_t k = 0; k < X.extent(0); ++k)
                  cf_values[pcf] += X(k, i) * Q(k, j);
              }
          };
#else
# error "ATermTable::compute_cfs_task BLAS implementation is missing"
#endif
  }
}

void
ATermTable::compute_pcs_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  Rect<4> pr_rect = regions[0];

  Rect<4> rect =
    rt->get_index_space_domain(task->regions[0].region.get_index_space());
  auto n_blc = rect.hi[0] - rect.lo[0] + 1;
  auto n_pa = rect.hi[1] - rect.lo[1] + 1;
  auto n_sto = rect.hi[2] - rect.lo[2] + 1;
  auto n_z = rect.hi[3] - rect.lo[3] + 1;

  typedef std::experimental::mdspan<
    zc_t,
    std::experimental::dynamic_extent,
    std::experimental::dynamic_extent,
    std::experimental::dynamic_extent,
    std::experimental::dynamic_extent> z4d_t;
  typedef std::experimental::mdspan<
    pc_t,
    std::experimental::dynamic_extent,
    std::experimental::dynamic_extent,
    std::experimental::dynamic_extent,
    std::experimental::dynamic_extent,
    std::experimental::dynamic_extent> p5d_t;

  // Zernike coefficients
  const FieldAccessor<
    READ_ONLY,
    zc_t,
    4,
    coord_t,
    AffineAccessor<zc_t, 4, coord_t>> zc_acc(regions[0], ZC_FID);
  z4d_t zcs_array(
    zc_acc.ptr(Point<4>::ZEROES()),
    pr_rect.hi[0] + 1,
    pr_rect.hi[1] + 1,
    pr_rect.hi[2] + 1,
    pr_rect.hi[3] + 1);

  // polynomial expansion coefficients
  const FieldAccessor<
    WRITE_ONLY,
    pc_t,
    5,
    coord_t,
    AffineAccessor<pc_t, 5, coord_t>> pc_acc(regions[1], PC_FID);
  p5d_t pcs_array(
    pc_acc.ptr(Point<5>::ZEROES()),
    pr_rect.hi[0] + 1,
    pr_rect.hi[1] + 1,
    pr_rect.hi[2] + 1,
    pr_rect.hi[3] + 1,
    pr_rect.hi[3] + 1);

  for (coord_t blc = rect.lo[0]; blc <= rect.hi[0]; ++blc)
    for (coord_t pa = rect.lo[1]; pa <= rect.hi[1]; ++pa)
      for (coord_t sto = rect.lo[2]; sto <= rect.hi[2]; ++sto) {
        auto zcs =
          std::experimental::subspan(
            zcs_array,
            blc,
            pa,
            sto,
            std::experimental::all);
        auto pcs =
          std::experimental::subspan(
            pcs_array,
            blc,
            pa,
            sto,
            std::experimental::all,
            std::experimental::all);
        switch (zcs.extent(3) - 1) {
#define ZEXP(N)                                       \
          case N:                                     \
            zernike_basis<N, zc_t>::expand(zcs, pcs); \
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
#endif

void
ATermTable::preregister_tasks() {
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
    // register an openmp version
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
    // register a cuda version
    {
      TaskVariantRegistrar
        registrar(compute_cfs_task_id, compute_cfs_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
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
  {
    // compute_pcs_task
    compute_pcs_task_id = Runtime::generate_static_task_id();
#ifdef HYPERION_USE_KOKKOS
# ifdef KOKKOS_ENABLE_SERIAL
    // register a serial version on the CPU
    {
      TaskVariantRegistrar
        registrar(compute_cfs_task_id, compute_cfs_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();
      registrar.add_layout_constraint_set(0, soa_right_layout);
      registrar.add_layout_constraint_set(1, soa_right_layout);
      Runtime::preregister_task_variant<compute_cfs_task<Kokkos::Serial>>(
        registrar,
        compute_cfs_task_name);
    }
# endif

# ifdef KOKKOS_ENABLE_OPENMP
    // register an openmp version
    {
      TaskVariantRegistrar
        registrar(compute_cfs_task_id, compute_cfs_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();
      registrar.add_layout_constraint_set(0, soa_right_layout);
      registrar.add_layout_constraint_set(1, soa_right_layout);
      Runtime::preregister_task_variant<compute_cfs_task<Kokkos::OpenMP>>(
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
      registrar.add_layout_constraint_set(0, soa_right_layout);
      registrar.add_layout_constraint_set(1, soa_right_layout);
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
