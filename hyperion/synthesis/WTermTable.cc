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

static const constexpr double pi = 3.141592653589793;

Legion::TaskID WTermTable::compute_cfs_task_id;

hyperion::synthesis::WTermTable::WTermTable(
  Context ctx,
  Runtime* rt,
  const std::array<Legion::coord_t, 2>& cf_bounds_lo,
  const std::array<Legion::coord_t, 2>& cf_bounds_hi,
  const std::array<double, 2>& cell_size,
  const std::vector<typename cf_table_axis<CF_W>::type>& w_values)
  : CFTable<CF_W>(
    ctx,
    rt,
    Legion::Rect<2>(
      {cf_bounds_lo[0], cf_bounds_lo[1]},
      {cf_bounds_hi[0], cf_bounds_hi[1]}),
    Axis<CF_W>(w_values))
  , m_cell_size(cell_size) {}

hyperion::synthesis::WTermTable::WTermTable(
  Context ctx,
  Runtime* rt,
  const coord_t& cf_x_radius,
  const coord_t& cf_y_radius,
  const std::array<double, 2>& cell_size,
  const std::vector<typename cf_table_axis<CF_W>::type>& w_values)
  : CFTable<CF_W>(
    ctx,
    rt,
    Legion::Rect<2>({-cf_x_radius, -cf_y_radius}, {cf_x_radius, cf_y_radius}),
    Axis<CF_W>(w_values))
  , m_cell_size(cell_size) {}

void
hyperion::synthesis::WTermTable::compute_cfs_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  const ComputeCFSTaskArgs& args =
    *static_cast<ComputeCFSTaskArgs*>(task->args);

  auto [pt, rit, pit] =
    PhysicalTable::create(
      rt,
      args.desc,
      task->regions.begin(),
      task->regions.end(),
      regions.begin(),
      regions.end())
    .value();
  assert(rit == task->regions.end());
  assert(pit == regions.end());

  auto tbl = CFPhysicalTable<CF_W>(pt);

  auto ws = tbl.w<Legion::AffineAccessor>().accessor<READ_ONLY>();
  auto values_col = tbl.value<Legion::AffineAccessor>();
  auto values = values_col.accessor<WRITE_ONLY>();

  auto rect = values_col.rect();
  const coord_t& w_lo = rect.lo[0];
  const coord_t& w_hi = rect.hi[0];
  const coord_t& x_lo = rect.lo[1];
  const coord_t& x_hi = rect.hi[1];
  const coord_t& y_lo = rect.lo[2];
  const coord_t& y_hi = rect.hi[2];
  for (coord_t w_idx = w_lo; w_idx <= w_hi; ++w_idx) {
    const double twoPiW = 2.0 * pi * ws[w_idx];
    for (coord_t x_idx = x_lo; x_idx <= x_hi; ++x_idx) {
      const double l = args.cell_size[0] * x_idx;
      const double l2 = l * l;
      for (coord_t y_idx = y_lo; y_idx <= y_hi; ++y_idx) {
        const double m = args.cell_size[1] * y_idx;
        const double r2 = l2 + m * m;
        const double phase =
          ((r2 <= 1.0) ? (twoPiW * (std::sqrt(1.0 - r2) - 1.0)) : 0.0);
        values[{w_idx, x_idx, y_idx}] = std::polar(1.0, phase);
      }
    }
  }
}

void
hyperion::synthesis::WTermTable::compute_cfs(
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
    ComputeCFSTaskArgs args;
    args.desc = desc;
    args.cell_size = m_cell_size;
    TaskLauncher task(compute_cfs_task_id, TaskArgument(&args, sizeof(args)));
    for (auto& r : reqs)
      task.add_region_requirement(r);
    rt->execute_task(ctx, task);
}

void
hyperion::synthesis::WTermTable::preregister_tasks() {
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
