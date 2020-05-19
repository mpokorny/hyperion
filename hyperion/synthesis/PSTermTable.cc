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

using namespace hyperion::synthesis;
using namespace hyperion;
using namespace Legion;

template <size_t N>
static double
sph(const std::array<double, N>& ary, double nu_lo, double nu_hi) {
  const double dn2 = std::pow(nu_lo, 2) - std::pow(nu_hi, 2);
  double result = 0.0;
  for (unsigned k = 0; k < N; ++k) 
    result += ary[k] * std::pow(dn2, k);
  return result;
}

static double
spheroidal(double nu) {
  double result;
  if (nu <= 0) {
    result = 1.0;
  } else if (nu < 0.75) {
    const std::array<double, 5>
      p{8.203343e-2,-3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1};
    const std::array<double, 3>
      q{1.0000000e0, 8.212018e-1, 2.078043e-1};
    result = sph(p, nu, 0.75) / sph(q, nu, 0.75);
  } else if (nu < 1.0) {
    const std::array<double, 5>
      p{4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2};
    const std::array<double, 3>
      q{1.0000000e0, 9.599102e-1, 2.918724e-1};
    result = sph(p, nu, 1.0) / sph(q, nu, 1.0);
  } else {
    result = 0.0;
  }
  return result;
}

Legion::TaskID PSTermTable::compute_cfs_task_id;

hyperion::synthesis::PSTermTable::PSTermTable(
  Context ctx,
  Runtime* rt,
  const std::array<Legion::coord_t, 2>& cf_bounds_lo,
  const std::array<Legion::coord_t, 2>& cf_bounds_hi,
  const std::vector<typename cf_table_axis<CF_PB_SCALE>::type>& pb_scales)
  : CFTable<CF_PB_SCALE>(
    ctx,
    rt,
    Legion::Rect<2>(
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
    Legion::Rect<2>({-cf_x_radius, -cf_y_radius}, {cf_x_radius, cf_y_radius}),
    Axis<CF_PB_SCALE>(pb_scales)) {}

void
hyperion::synthesis::PSTermTable::compute_cfs_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  const Table::Desc& desc = *static_cast<Table::Desc*>(task->args);

  auto [pt, rit, pit] =
    PhysicalTable::create(
      rt,
      desc,
      task->regions.begin(),
      task->regions.end(),
      regions.begin(),
      regions.end())
    .value();
  assert(rit == task->regions.end());
  assert(pit == regions.end());

  auto tbl = CFPhysicalTable<CF_PB_SCALE>(pt);

  auto pb_scales = tbl.pb_scale<Legion::AffineAccessor>().accessor<READ_ONLY>();
  auto values_col = tbl.value<Legion::AffineAccessor>();
  auto values = values_col.accessor<WRITE_ONLY>();

  auto rect = values_col.rect();
  const coord_t& pb_lo = rect.lo[0];
  const coord_t& pb_hi = rect.hi[0];
  const coord_t& x_lo = rect.lo[1];
  const coord_t& x_hi = rect.hi[1];
  const coord_t& y_lo = rect.lo[2];
  const coord_t& y_hi = rect.hi[2];
  for (coord_t pb_idx = pb_lo; pb_idx <= pb_hi; ++pb_idx) {
    const auto pb_scale = pb_scales[pb_idx];
    for (coord_t x_idx = x_lo; x_idx <= x_hi; ++x_idx) {
      const float xp = x_idx * x_idx;
      for (coord_t y_idx = y_lo; y_idx <= y_hi; ++y_idx) {
        const float yp = std::sqrt(xp + y_idx * y_idx) * pb_scale;
        const float v = (float)spheroidal(yp) * (1.0f - yp * yp);
        // we're creating PS terms that are intended to be multiplied in some
        // domain, thus, if v == 0.0, we'll just multiply by 1.0
        values[{pb_idx, x_idx, y_idx}] = ((v > 0.0) ? v : 1.0);
      }
    }
  }
}

void
hyperion::synthesis::PSTermTable::compute_cfs(
  Context ctx,
  Runtime* rt,
  const ColumnSpacePartition& partition) const {

    auto cf_colreqs = Column::default_requirements;
    cf_colreqs.values = Column::Req{
      WRITE_ONLY /* privilege */,
      EXCLUSIVE /* coherence */,
      true /* mapped */
    };

    auto default_colreqs = Column::default_requirements;
    default_colreqs.values.mapped = true;

    auto [reqs, parts, desc] =
      requirements(
        ctx,
        rt,
        partition,
        {{CF_VALUE_COLUMN_NAME, cf_colreqs},
         {CF_WEIGHT_COLUMN_NAME, std::nullopt}},
        default_colreqs);
    TaskLauncher task(compute_cfs_task_id, TaskArgument(&desc, sizeof(desc)));
    for (auto& r : reqs)
      task.add_region_requirement(r);
    rt->execute_task(ctx, task);
}

void
hyperion::synthesis::PSTermTable::preregister_tasks() {
  {
    // compute_cfs_task
    compute_cfs_task_id = Runtime::generate_static_task_id();
    TaskVariantRegistrar registrar(compute_cfs_task_id, compute_cfs_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    Runtime::preregister_task_variant<compute_cfs_task>(
      registrar,
      compute_cfs_task_name);
  }
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
