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
#include <set>

using namespace hyperion;
using namespace hyperion::synthesis;
using namespace Legion;

TaskID ATermTable::compute_cfs_task_id;

#define ENABLE_KOKKOS_SERIAL_CFS_TASK
#define ENABLE_KOKKOS_OPENMP_CFS_TASK
#define ENABLE_KOKKOS_CUDA_CFS_TASK

ATermTable::ATermTable(
  Context ctx,
  Runtime* rt,
  const std::array<coord_t, 2>& cf_bounds_lo,
  const std::array<coord_t, 2>& cf_bounds_hi,
  const std::vector<typename cf_table_axis<CF_BASELINE_CLASS>::type>&
    baseline_classes,
  const std::vector<typename cf_table_axis<CF_PARALLACTIC_ANGLE>::type>&
    parallactic_angles,
  const std::vector<typename cf_table_axis<CF_FREQUENCY>::type>&
    frequencies,
  const std::vector<typename cf_table_axis<CF_STOKES_OUT>::type>&
    stokes_out_values,
  const std::vector<typename cf_table_axis<CF_STOKES_IN>::type>&
    stokes_in_values)
  : CFTable(
    ctx,
    rt,
    Rect<2>(
      {cf_bounds_lo[0], cf_bounds_lo[1]},
      {cf_bounds_hi[0], cf_bounds_hi[1]}),
    Axis<CF_BASELINE_CLASS>(baseline_classes),
    Axis<CF_PARALLACTIC_ANGLE>(parallactic_angles),
    Axis<CF_FREQUENCY>(frequencies),
    Axis<CF_STOKES_OUT>(stokes_out_values),
    Axis<CF_STOKES_IN>(stokes_in_values)) {
}

ATermTable::ATermTable(
  Context ctx,
  Runtime* rt,
  const coord_t& cf_x_radius,
  const coord_t& cf_y_radius,
  const std::vector<typename cf_table_axis<CF_BASELINE_CLASS>::type>&
    baseline_classes,
  const std::vector<typename cf_table_axis<CF_PARALLACTIC_ANGLE>::type>&
    parallactic_angles,
  const std::vector<typename cf_table_axis<CF_FREQUENCY>::type>&
    frequencies,
  const std::vector<typename cf_table_axis<CF_STOKES_OUT>::type>&
    stokes_out_values,
  const std::vector<typename cf_table_axis<CF_STOKES_IN>::type>&
    stokes_in_values)
  : CFTable(
    ctx,
    rt,
    Rect<2>({-cf_x_radius, -cf_y_radius}, {cf_x_radius - 1, cf_y_radius - 1}),
    Axis<CF_BASELINE_CLASS>(baseline_classes),
    Axis<CF_PARALLACTIC_ANGLE>(parallactic_angles),
    Axis<CF_FREQUENCY>(frequencies),
    Axis<CF_STOKES_OUT>(stokes_out_values),
    Axis<CF_STOKES_IN>(stokes_in_values)) {
}

template <unsigned D>
static IndexPartition
map_mueller_to_stokes(
  Context ctx,
  Runtime* rt,
  const ColumnSpacePartition& aterm_part,
  const std::vector<stokes_t>& stokes_out_values,
  const std::vector<stokes_t>& stokes_in_values,
  const IndexSpace& aif_cf_is,
  const std::vector<stokes_t>& stokes_values) {

  auto aterm_part_color_space =
    rt->get_index_partition_color_space_name(ctx, aterm_part.column_ip);

  auto result =
    rt->create_pending_partition(ctx, aif_cf_is, aterm_part_color_space);

  std::map<stokes_t, unsigned> stokes_value_indexes;
  for (size_t i = 0; i < stokes_values.size(); ++i)
    stokes_value_indexes[stokes_values[i]] = i;

  // find location (dimension index) of CF_STOKES_OUT and CF_STOKES_IN in
  // partition index
  unsigned stokes_out_dim = std::numeric_limits<unsigned>::max();
  unsigned stokes_in_dim = std::numeric_limits<unsigned>::max();
  static_assert(ColumnSpace::MAX_DIM <= std::numeric_limits<unsigned>::max());
  for (unsigned i = 0;
       aterm_part.partition[i].dim >= 0 && i < ColumnSpace::MAX_DIM;
       ++i) {
    if (aterm_part.partition[i].dim == CF_STOKES_OUT)
      stokes_out_dim = i;
    else if (aterm_part.partition[i].dim == CF_STOKES_IN)
      stokes_in_dim = i;
  }
  assert(stokes_out_dim < std::numeric_limits<unsigned>::max());
  assert(stokes_in_dim < std::numeric_limits<unsigned>::max());
  const unsigned stokes_dim = stokes_out_dim;

  // create a partition subspace for every color in aterm_part_color_space
  for (PointInDomainIterator<D> pid(
         rt->get_index_space_domain(ctx, aterm_part_color_space));
       pid();
       pid++) {
    // create rectangles in aif_cf_is of Stokes indexes for this set of Mueller
    // elements
    Rect<ATermTable::index_rank + 2> aterm_bounds =
      rt->get_index_space_domain(
        ctx,
        rt->get_index_subspace(ctx, aterm_part.column_ip, DomainPoint(*pid)));
    // rectangles in aif are the same as aterm_bounds except that they have a
    // single Stokes axis with a single index in the stokes_values domain (the
    // dimension index of this single Stokes axis is assumed to be the same as
    // stokes_out_dim)
    Rect<ATermIlluminationFunction::index_rank + 2> aif_bounds;
    for (size_t i = 0, j = 0; i < ATermTable::index_rank + 2; ++i) {
      if (i != stokes_in_dim) {
        aif_bounds.lo[j] = aterm_bounds.lo[i];
        aif_bounds.hi[j] = aterm_bounds.hi[i];
        ++j;
      }
    }
    // get the set of all indexes in stokes_values referenced by aterm_bounds
    // (CF_STOKES_OUT and CF_STOKES_IN)
    std::set<unsigned> sto_idxs;
    for (Legion::coord_t s = aterm_bounds.lo[stokes_out_dim];
         s <= aterm_bounds.hi[stokes_out_dim];
         ++s)
      sto_idxs.insert(stokes_value_indexes[stokes_out_values[s]]);
    for (Legion::coord_t s = aterm_bounds.lo[stokes_in_dim];
         s <= aterm_bounds.hi[stokes_in_dim];
         ++s)
      sto_idxs.insert(stokes_value_indexes[stokes_in_values[s]]);
    // create the subspace using create_partition_by_domain()
    //
    // the size of the color space could be reduced by merging contiguous values
    // in sto_idxs, but that seems a minor point since there are at most four
    // values
    auto cs = rt->create_index_space(ctx, Rect<1>(0, sto_idxs.size() - 1));
    std::map<DomainPoint, Domain> domains;
    unsigned i = 0;
    for (auto& s : sto_idxs) {
      Rect<ATermIlluminationFunction::index_rank + 2> r = aif_bounds;
      r.lo[stokes_dim] = r.hi[stokes_dim] = s;
      domains[i++] = r;
    }
    auto ip = rt->create_partition_by_domain(ctx, aif_cf_is, domains, cs);
    std::vector<IndexSpace> iss;
    for (unsigned i = 0; i < sto_idxs.size(); ++i)
      iss.push_back(rt->get_index_subspace(ctx, ip, i));
    // create the partition subspace for the color *pid as the union of all
    // subspaces in ip
    rt->create_index_space_union(ctx, result, *pid, iss);
    rt->destroy_index_partition(ctx, ip); // TODO: OK?
    rt->destroy_index_space(ctx, cs);
  }
  return result;
}

void
ATermTable::compute_cfs(
  Context ctx,
  Runtime* rt,
  const std::vector<ZCoeff>& zernike_coefficients,
  const ColumnSpacePartition& partition) const {

  // Get vectors of values for all index columns, and bounding box of CFs
  std::vector<typename cf_table_axis<CF_BASELINE_CLASS>::type>
    baseline_classes;
  std::vector<typename cf_table_axis<CF_PARALLACTIC_ANGLE>::type>
    parallactic_angles;
  std::vector<typename cf_table_axis<CF_FREQUENCY>::type>
    frequencies;
  std::vector<typename cf_table_axis<CF_STOKES_OUT>::type>
    stokes_out_values;
  std::vector<typename cf_table_axis<CF_STOKES_IN>::type>
    stokes_in_values;
  Rect<2> cf_bounds;
  using_resource(
    [&]() {
      auto colreqs = Column::default_requirements;
      colreqs.values.mapped = true;
      return
        CFPhysicalTable<HYPERION_A_TERM_TABLE_AXES>(
          map_inline(
            ctx,
            rt,
            {{cf_table_axis<CF_BASELINE_CLASS>::name, colreqs},
             {cf_table_axis<CF_PARALLACTIC_ANGLE>::name, colreqs},
             {cf_table_axis<CF_FREQUENCY>::name, colreqs},
             {cf_table_axis<CF_STOKES_OUT>::name, colreqs},
             {cf_table_axis<CF_STOKES_IN>::name, colreqs},
             {CF_VALUE_COLUMN_NAME, Column::default_requirements}},
            CXX_OPTIONAL_NAMESPACE::nullopt));
    },
    [&](CFPhysicalTable<HYPERION_A_TERM_TABLE_AXES>& tbl) {
      {
        auto blc_col = tbl.baseline_class<AffineAccessor>();
        auto blcs = blc_col.accessor<READ_ONLY>();
        for (PointInRectIterator<1> pir(blc_col.rect()); pir(); pir++)
          baseline_classes.push_back(blcs[*pir]);
      }
      {
        auto pa_col = tbl.parallactic_angle<AffineAccessor>();
        auto pas = pa_col.accessor<READ_ONLY>();
        for (PointInRectIterator<1> pir(pa_col.rect()); pir(); pir++)
          parallactic_angles.push_back(pas[*pir]);
      }
      {
        auto frq_col = tbl.frequency<AffineAccessor>();
        auto frqs = frq_col.accessor<READ_ONLY>();
        for (PointInRectIterator<1> pir(frq_col.rect()); pir(); pir++)
          frequencies.push_back(frqs[*pir]);
      }
      {
        auto sto_col = tbl.stokes_out<AffineAccessor>();
        auto stos = sto_col.accessor<READ_ONLY>();
        for (PointInRectIterator<1> pir(sto_col.rect()); pir(); pir++)
          stokes_out_values.push_back(stos[*pir]);
      }
      {
        auto sto_col = tbl.stokes_in<AffineAccessor>();
        auto stos = sto_col.accessor<READ_ONLY>();
        for (PointInRectIterator<1> pir(sto_col.rect()); pir(); pir++)
          stokes_in_values.push_back(stos[*pir]);
      }
      {
        auto value_col = tbl.value<AffineAccessor>();
        auto rect = value_col.rect();
        cf_bounds.lo[0] = rect.lo[index_rank];
        cf_bounds.hi[0] = rect.hi[index_rank];
        cf_bounds.lo[1] = rect.lo[index_rank + 1];
        cf_bounds.hi[1] = rect.hi[index_rank + 1];
      }
    },
    [&](CFPhysicalTable<HYPERION_A_TERM_TABLE_AXES>& tbl) {
      tbl.unmap_regions(ctx, rt);
    });
  // created vector of all referenced Stokes values
  std::vector<stokes_t> stokes_values;
  {
    std::set<stokes_t> sto;
    std::copy(
      stokes_out_values.begin(),
      stokes_out_values.end(),
      std::inserter(sto, sto.end()));
    std::copy(
      stokes_in_values.begin(),
      stokes_in_values.end(),
      std::inserter(sto, sto.end()));
    stokes_values.reserve(sto.size());
    std::copy(sto.begin(), sto.end(), std::back_inserter(stokes_values));
  }
  // create table for Zernike expansion and polynomial expansion coefficients
  ATermZernikeModel zmodel(
    ctx,
    rt,
    zernike_coefficients,
    baseline_classes,
    frequencies,
    stokes_values);
  // compute the polynomial function coefficients column in zmodel
  {
    auto p =
      zmodel.columns().at(CF_VALUE_COLUMN_NAME)
      .narrow_partition(ctx, rt, partition)
      .value_or(partition);
    zmodel.compute_pcs(ctx, rt, p);
    if (p != partition)
      p.destroy(ctx, rt);
  }
  // create table for aperture illumination functions
  unsigned zernike_order = 0;
  for (auto& zc : zernike_coefficients)
    zernike_order = std::max(zernike_order, zc.n);
  ATermIlluminationFunction aif(
    ctx,
    rt,
    cf_bounds,
    zernike_order,
    baseline_classes,
    parallactic_angles,
    frequencies,
    stokes_values);
  // compute polynomial function evaluation points for aperture illumination
  // functions
  {
    auto p =
      aif.columns().at(CF_VALUE_COLUMN_NAME)
      .narrow_partition(ctx, rt, partition)
      .value_or(partition);
    aif.compute_epts(ctx, rt, p);
    if (p != partition)
      p.destroy(ctx, rt);
  }

  // evaluate aperture illumination polynomial function values for each
  // Stokes value
  {
    // no partition on X/Y, as ATermIlluminationFunction::compute_aifs() doesn't
    // know how to do a distributed FFT
    auto p =
      aif.columns().at(CF_VALUE_COLUMN_NAME)
      .narrow_partition(ctx, rt, partition, {CF_X, CF_Y})
      .value_or(partition);
    aif.compute_jones(ctx, rt, zmodel, p);
    if (p != partition)
      p.destroy(ctx, rt);
  }
  zmodel.destroy(ctx, rt); // don't need zmodel again

  auto aterm_part = // partition of illumination function value/weight columns
    columns().at(CF_VALUE_COLUMN_NAME)
    .narrow_partition(ctx, rt, partition)
    .value_or(partition);
  auto aterm_part_color_space =
    rt->get_index_partition_color_space_name(ctx, aterm_part.column_ip);

  // check for CF_STOKES_OUT or CF_STOKES_IN axis in partition (these are the
  // Mueller matrix axes)
  auto m_p =
    columns().at(CF_VALUE_COLUMN_NAME)
    .narrow_partition(
      ctx,
      rt,
      partition,
      {CF_BASELINE_CLASS, CF_PARALLACTIC_ANGLE, CF_FREQUENCY, CF_STOKES});
  IndexPartition aif_read_ip;
  LogicalPartition aif_read_lp;

  if (m_p) {
    // Because every Mueller element depends on one or two Stokes components
    // only, every subspace of a partition of the CF_STOKES_OUT and CF_STOKES_IN
    // axes (the combination of which we refer to as the Mueller axes) may
    // depend on a strict subset of the Stokes components. We therefore create a
    // map from Mueller axes partition sub-space indexes to Stokes value subsets
    // as a first step in creating the minimal dependency relations of Mueller
    // CF regions on Stokes aperture illumination function regions.

    // create an aliased partition of aif based on the Stokes value subsets
    // needed by the Mueller axis partition
    {
      auto aif_cf_col = aif.columns().at(CF_VALUE_COLUMN_NAME);
      auto aif_cf_is = aif_cf_col.cs.column_is;
      static_assert(index_rank == ATermIlluminationFunction::index_rank + 1);

      // To create the partition using Runtime::create_partition_by_domain(), we
      // need the mapping from subspaces in aterm_part to Domains in
      // aif_cf_is
      switch (aterm_part.color_dim(rt)) {
#define MAP_MUELLER_TO_STOKES(N)                \
        case N:                                 \
          aif_read_ip =                         \
            map_mueller_to_stokes<N>(           \
              ctx,                              \
              rt,                               \
              aterm_part,                       \
              stokes_out_values,                \
              stokes_in_values,                 \
              aif_cf_is,                        \
              stokes_values);                   \
          break;
      HYPERION_FOREACH_N(MAP_MUELLER_TO_STOKES);
#undef MAP_ME_TO_STO
      default:
        assert(false);
        break;
      }
      aif_read_lp =
        rt->get_logical_partition(ctx, aif_cf_col.region, aif_read_ip);
    }
  }
  // compute the elements of the Mueller matrix
  std::vector<RegionRequirement> all_reqs;
  std::vector<ColumnSpacePartition> all_parts;
  ComputeCFsTaskArgs args;
  {
    // ATermIlluminationFunction value and weight columns
    auto sto_part_colreqs = Column::default_requirements;
    sto_part_colreqs.values.mapped = true;
    sto_part_colreqs.partition = aif_read_lp; // OK to be NO_PART

    // we'll need to map Stokes values to region indexes in compute_cfs_task,
    // which may be derived from the Stokes index column (we don't bother to
    // make a special partition for this purpose -- aterm_part will leave the
    // CF_STOKES axis unpartitioned)
    auto colreqs = Column::default_requirements;
    colreqs.values.mapped = true;
    auto reqs =
      aif.requirements(
        ctx,
        rt,
        aterm_part,
        {{CF_VALUE_COLUMN_NAME, sto_part_colreqs},
         {CF_WEIGHT_COLUMN_NAME, sto_part_colreqs},
         {cf_table_axis<CF_STOKES>::name, colreqs}},
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
  {
    // ATermTable, Mueller element CF value and weight columns
    auto colreqs = Column::default_requirements;
    colreqs.values.mapped = true;
    colreqs.values.privilege = WRITE_DISCARD;
    auto ro_colreqs = Column::default_requirements;
    ro_colreqs.values.mapped = true;
    auto reqs =
      requirements(
        ctx,
        rt,
        aterm_part,
        {{CF_VALUE_COLUMN_NAME, colreqs},
         {CF_WEIGHT_COLUMN_NAME, colreqs},
         {cf_table_axis<CF_STOKES_OUT>::name, ro_colreqs},
         {cf_table_axis<CF_STOKES_IN>::name, ro_colreqs}},
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
  // launch the compute_cfs task
  if (!aterm_part.is_valid()) {
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
      aterm_part_color_space,
      TaskArgument(&args, sizeof(args)),
      ArgumentMap(),
      Predicate::TRUE_PRED,
      table_mapper);
    for (auto& r : all_reqs)
      task.add_region_requirement(r);
    rt->execute_index_space(ctx, task);
  }
  // clean up
  for (auto& p : all_parts)
    p.destroy(ctx, rt);
  if (aif_read_ip != IndexPartition::NO_PART)
    rt->destroy_index_partition(ctx, aif_read_ip);
  if (aterm_part != partition)
    aterm_part.destroy(ctx, rt);
  aif.destroy(ctx, rt);
}

#ifndef HYPERION_USE_KOKKOS
void
ATermTable::compute_cfs_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {
}
#endif

void
ATermTable::preregister_tasks() {
  //
  // compute_cfs_task
  //
  {
    compute_cfs_task_id = Runtime::generate_static_task_id();

#ifdef HYPERION_USE_KOKKOS
# if defined(KOKKOS_ENABLE_SERIAL) && defined(ENABLE_KOKKOS_SERIAL_CFS_TASK)
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
          "ATermTable::compute_cfs_constraints");
      add_aos_right_ordering_constraint(constraints);
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
# if defined(KOKKOS_ENABLE_OPENMP) && defined(ENABLE_KOKKOS_OPENMP_CFS_TASK)
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
          "ATermTable::compute_cfs_constraints");
      add_aos_right_ordering_constraint(constraints);
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
# if defined(KOKKOS_ENABLE_CUDA) && defined(ENABLE_KOKKOS_CUDA_CFS_TASK)
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
          "ATermTable::compute_cfs_constraints");
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
          "ATermTable::compute_cfs_constraints");
      add_aos_right_ordering_constraint(constraints);
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
