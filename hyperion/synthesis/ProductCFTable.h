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
#ifndef HYPERION_SYNTHESIS_PRODUCT_CF_TABLE_H_
#define HYPERION_SYNTHESIS_PRODUCT_CF_TABLE_H_

#include <hyperion/synthesis/CFTable.h>
#include <hyperion/synthesis/PSTermTable.h>
#include <hyperion/synthesis/WTermTable.h>
#include <hyperion/synthesis/ATermTable.h>
#include <hyperion/PhysicalTableGuard.h>

namespace hyperion {
namespace synthesis {

template <cf_table_axes_t...Axes>
class HYPERION_EXPORT ProductCFTable
  : public CFTable<Axes...> {
public:

  template <typename...Ts>
  ProductCFTable(Legion::Context ctx, Legion::Runtime* rt, const Ts&...ts)
    : CFTable<Axes...>(
      product(
        ctx,
        rt,
        PhysicalTableGuard<typename Ts::physical_table_t>(
          ctx,
          rt,
          typename Ts::physical_table_t(
            ts.map_inline(
              ctx,
              rt,
              {},
              Column::default_requirements_mapped)))...)) {}

  ProductCFTable() {}

  static Legion::TaskID multiply_ps_task_id;
  static const constexpr char* multiply_ps_task_name =
    "ProductCFTable::multiply_ps_task";

  static Legion::TaskID multiply_w_task_id;
  static const constexpr char* multiply_w_task_name =
    "ProductCFTable::multiply_w_task";

  static Legion::TaskID multiply_a_task_id;
  static const constexpr char* multiply_a_task_name =
    "ProductCFTable::multiply_a_task";

  struct MultiplyCFTermArgs {
    Table::Desc left;
    Table::Desc right;
    bool do_multiply;
  };

protected:

  static void
  multiply_by(
    Legion::Context ctx,
    Legion::Runtime* rt,
    Legion::TaskID task_id,
    const Table& left,
    const Table& right,
    const ColumnSpacePartition& partition,
    bool do_multiply) {

    MultiplyCFTermArgs args;
    args.do_multiply = do_multiply;
    std::vector<Legion::RegionRequirement> treqs;
    std::vector<ColumnSpacePartition> tparts;
    {
      auto colreqs = Column::default_requirements_mapped;
      colreqs.values.privilege =
        do_multiply ? LEGION_READ_WRITE : LEGION_WRITE_ONLY;
      auto reqs =
        left.requirements(
          ctx,
          rt,
          partition,
          {{CFTableBase::CF_VALUE_COLUMN_NAME, colreqs},
           {CFTableBase::CF_WEIGHT_COLUMN_NAME, colreqs}},
          CXX_OPTIONAL_NAMESPACE::nullopt);
      args.left = std::get<2>(reqs);
      for (auto& r : std::get<0>(reqs))
        treqs.push_back(r);
      for (auto& p : std::get<1>(reqs))
        tparts.push_back(p);
    }
    {
      auto colreqs = Column::default_requirements_mapped;
      auto reqs =
        right.requirements(
          ctx,
          rt,
          partition,
          {{CFTableBase::CF_VALUE_COLUMN_NAME, colreqs},
           {CFTableBase::CF_WEIGHT_COLUMN_NAME, colreqs}},
          CXX_OPTIONAL_NAMESPACE::nullopt);
      args.right = std::get<2>(reqs);
      for (auto& r : std::get<0>(reqs))
        treqs.push_back(r);
      for (auto& p : std::get<1>(reqs))
        tparts.push_back(p);
    }
    if (!partition.is_valid()) {
      Legion::TaskLauncher task(
        task_id,
        Legion::TaskArgument(&args, sizeof(args)),
        Legion::Predicate::TRUE_PRED,
        table_mapper);
      for (auto& r : treqs)
        task.add_region_requirement(r);
      rt->execute_task(ctx, task);
    } else {
      Legion::IndexTaskLauncher task(
        task_id,
        rt->get_index_partition_color_space(ctx, partition.column_ip),
        Legion::TaskArgument(&args, sizeof(args)),
        Legion::ArgumentMap(),
        Legion::Predicate::TRUE_PRED,
        false,
        table_mapper);
      for (auto& r : treqs)
        task.add_region_requirement(r);
      rt->execute_index_space(ctx, task);
    }
    for (auto& p : tparts)
      p.destroy(ctx, rt);
  }

public:

  void
  multiply_by(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const PSTermTable& ps_term,
    const ColumnSpacePartition& partition = ColumnSpacePartition(),
    bool do_multiply = true) const {

    multiply_by(
      ctx,
      rt,
      multiply_ps_task_id,
      *this,
      ps_term,
      partition,
      do_multiply);
  }

  void
  multiply_by(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const WTermTable& w_term,
    const ColumnSpacePartition& partition = ColumnSpacePartition(),
    bool do_multiply = true) const {

    multiply_by(
      ctx,
      rt,
      multiply_w_task_id,
      *this,
      w_term,
      partition,
      do_multiply);
  }

  void
  multiply_by(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const ATermTable& a_term,
    const ColumnSpacePartition& partition = ColumnSpacePartition(),
    bool do_multiply = true) const {

    multiply_by(
      ctx,
      rt,
      multiply_a_task_id,
      *this,
      a_term,
      partition,
      do_multiply);
  }

  /**
   * create a ProductCFTable and fill it with the products of an ordered list of
   * CFTables
   *
   * N.B: the first element of the CFTable arguments must have the largest value
   * of grid_size() of all CFTable arguments
   */
  template <typename T0, typename...Ts>
  static ProductCFTable
  create_and_fill(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const ColumnSpacePartition& partition,
    const T0& t0,
    const Ts&...ts) {

    ProductCFTable<Axes...> result(ctx, rt, t0, ts...);
    auto grid_size =
      [&](auto& tbl) {
        typedef
          typename std::remove_reference_t<std::remove_const_t<decltype(tbl)>>
          table_t;
        return
          PhysicalTableGuard<typename table_t::physical_table_t>(
            ctx,
            rt,
            typename table_t::physical_table_t(
              tbl.map_inline(
                ctx,
                rt,
                {{CFTableBase::CF_VALUE_COLUMN_NAME,
                  Column::default_requirements}},
                CXX_OPTIONAL_NAMESPACE::nullopt)))->grid_size();
      };
    std::array<size_t, sizeof...(Ts) + 1>
      grid_sizes{grid_size(t0), grid_size(ts)...};
    bool first_is_largest = true;
    for (size_t i = 1; first_is_largest && i < grid_sizes.size(); ++i)
      first_is_largest = grid_sizes[i] <= grid_sizes[0];
    assert(first_is_largest);

    result.multiply_by(ctx, rt, t0, partition, false);
    [[maybe_unused]] std::array<int, sizeof...(Ts)>
      rc{(result.multiply_by(ctx, rt, ts, partition, true), 1)...};
    return result;
  }

  template <cf_table_axes_t A, typename Left, typename Right>
  static HYPERION_INLINE_FUNCTION void
  set_index(
    array<coord_t, Right::row_rank>& to,
    const array<coord_t, Left::row_rank>& from) {

    if (Right::template has_index_axis<A>())
      to[Right::template index_axis_index<A>()] =
        Left::template has_index_axis<A>()
        ? from[Left::template index_axis_index<A>()]
        : 0;
  }

  template <unsigned INDEX_RANK, typename T>
  using cf_col_t =
    PhysicalColumnTD<
      ValueType<T>::DataType,
      INDEX_RANK,
      INDEX_RANK + 2,
      Legion::AffineAccessor>;

#ifdef HYPERION_USE_KOKKOS
  template <
    typename execution_space,
    typename T,
    unsigned N,
    typename P>
  static void
  cf_multiply(
    Legion::Context ctx,
    Legion::Runtime* rt,
    bool do_multiply,
    const cf_col_t<ProductCFTable::index_rank, T>& left,
    size_t left_grid_size,
    const cf_col_t<N, T>& right,
    size_t right_grid_size,
    P prj) {

    assert(left_grid_size >= right_grid_size);
    assert(left_grid_size % 2 == right_grid_size % 2);
    auto left_slice =
      Kokkos::make_pair(
        (left_grid_size - right_grid_size) / 2,
        (left_grid_size - right_grid_size) / 2 + right_grid_size);
    auto right_slice = Kokkos::make_pair((size_t)0, right_grid_size);
    Legion::Rect<ProductCFTable::index_rank> left_cf_pts;
    auto left_rect = left.rect();
    for (size_t i = 0; i < ProductCFTable::index_rank; ++i) {
      left_cf_pts.lo[i] = left_rect.lo[i];
      left_cf_pts.hi[i] = left_rect.hi[i];
    }
    auto left_cf = left.template view<execution_space, LEGION_READ_WRITE>();
    auto right_cf = right.template view<execution_space, LEGION_READ_ONLY>();

    auto kokkos_work_space =
      rt->get_executing_processor(ctx).kokkos_work_space();
    typedef typename Kokkos::TeamPolicy<execution_space>::member_type
      member_type;
    if (do_multiply)
      Kokkos::parallel_for(
        Kokkos::TeamPolicy<execution_space>(
          kokkos_work_space,
          CFTableBase::linearized_index_range(left_cf_pts),
          Kokkos::AUTO()),
        KOKKOS_LAMBDA(const member_type& team_member) {
          auto left_cf_pt =
            CFTableBase::multidimensional_index(
              static_cast<Legion::coord_t>(team_member.league_rank()),
              left_cf_pts);
          auto right_cf_pt = prj(left_cf_pt);
          auto left_cf_subview =
            cf_subview(left_cf, left_cf_pt, left_slice, left_slice);
          auto right_cf_subview =
            cf_subview(right_cf, right_cf_pt, right_slice, right_slice);
          Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, right_cf_subview.extent(0)),
            [=](const int& i) {
              Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(
                  team_member,
                  right_cf_subview.extent(1)),
                [=](const int& j) {
                  left_cf_subview(i, j) *= right_cf_subview(i, j);
                });
            });
        });
    else
      Kokkos::parallel_for(
        Kokkos::TeamPolicy<execution_space>(
          kokkos_work_space,
          CFTableBase::linearized_index_range(left_cf_pts),
          Kokkos::AUTO()),
        KOKKOS_LAMBDA(const member_type& team_member) {
          auto left_cf_pt =
            CFTableBase::multidimensional_index(
              static_cast<Legion::coord_t>(team_member.league_rank()),
              left_cf_pts);
          auto right_cf_pt = prj(left_cf_pt);
          auto left_cf_subview =
            cf_subview(left_cf, left_cf_pt, left_slice, left_slice);
          auto right_cf_subview =
            cf_subview(right_cf, right_cf_pt, right_slice, right_slice);
          Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, right_cf_subview.extent(0)),
            [=](const int& i) {
              Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(
                  team_member,
                  right_cf_subview.extent(1)),
                [=](const int& j) {
                  left_cf_subview(i, j) = right_cf_subview(i, j);
                });
            });
        });
  }

  template <typename execution_space, typename Right>
  static void
  multiply_impl(
    Legion::Context ctx,
    Legion::Runtime* rt,
    bool do_multiply,
    const typename CFTable<Axes...>::physical_table_t& left,
    const Right& right) {

    using physical_table_t = typename CFTable<Axes...>::physical_table_t;

    auto prj =
      KOKKOS_LAMBDA(const array<coord_t, physical_table_t::row_rank>& pt) {
      array<coord_t, Right::row_rank> result;
      set_index<CF_PS_SCALE, physical_table_t, Right>(result, pt);
      set_index<CF_BASELINE_CLASS, physical_table_t, Right>(result, pt);
      set_index<CF_FREQUENCY, physical_table_t, Right>(result, pt);
      set_index<CF_W, physical_table_t, Right>(result, pt);
      set_index<CF_PARALLACTIC_ANGLE, physical_table_t, Right>(result, pt);
      set_index<CF_STOKES_OUT, physical_table_t, Right>(result, pt);
      set_index<CF_STOKES_IN, physical_table_t, Right>(result, pt);
      set_index<CF_STOKES, physical_table_t, Right>(result, pt);
      return result;
    };

    cf_multiply<execution_space, CFTableBase::cf_value_t>(
      ctx,
      rt,
      do_multiply,
      left.template value<Legion::AffineAccessor>(),
      left.grid_size(),
      right.template value<Legion::AffineAccessor>(),
      right.grid_size(),
      prj);
    cf_multiply<execution_space, CFTableBase::cf_weight_t>(
      ctx,
      rt,
      do_multiply,
      left.template weight<Legion::AffineAccessor>(),
      left.grid_size(),
      right.template weight<Legion::AffineAccessor>(),
      right.grid_size(),
      prj);
  }

  template <typename execution_space, cf_table_axes_t...RightAxes>
  struct MultiplyTermTask{

    static void
    task_body(
      const Legion::Task* task,
      const std::vector<Legion::PhysicalRegion>& regions,
      Legion::Context ctx,
      Legion::Runtime* rt) {

      const MultiplyCFTermArgs& args =
        *static_cast<const MultiplyCFTermArgs*>(task->args);
      std::vector<Table::Desc> tdesc{args.left, args.right};
      auto pts =
        PhysicalTable::create_all_unsafe(rt, tdesc, task->regions, regions);

      CFPhysicalTable<Axes...> left(pts[0]);
      CFPhysicalTable<RightAxes...> right(pts[1]);
      multiply_impl<execution_space>(ctx, rt, args.do_multiply, left, right);
    }
    ;
  };

  template <template<typename> typename T>
  static void
  preregister_task_variants(
    const char* task_name,
    Legion::TaskID& task_id,
    Legion::LayoutConstraintID cpu_layout_id,
    Legion::LayoutConstraintID gpu_layout_id) {

    task_id = Legion::Runtime::generate_static_task_id();

#ifdef KOKKOS_ENABLE_SERIAL
    {
      Legion::TaskVariantRegistrar registrar(task_id, task_name);
      registrar.add_constraint(
        Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();
      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        cpu_layout_id);
      Legion::Runtime::preregister_task_variant<T<Kokkos::Serial>::task_body>(
        registrar,
        task_name);
    }
#endif

#ifdef KOKKOS_ENABLE_OPENMP
    {
      Legion::TaskVariantRegistrar registrar(task_id, task_name);
      registrar.add_constraint(
        Legion::ProcessorConstraint(Legion::Processor::OMP_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();
      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        cpu_layout_id);
      Legion::Runtime::preregister_task_variant<T<Kokkos::OpenMP>::task_body>(
        registrar,
        task_name);
    }
#endif

#ifdef KOKKOS_ENABLE_CUDA
    {
      Legion::TaskVariantRegistrar registrar(task_id, task_name);
      registrar.add_constraint(
        Legion::ProcessorConstraint(Legion::Processor::TOC_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();
      registrar.add_layout_constraint_set(
        TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag),
        gpu_layout_id);
      Legion::Runtime::preregister_task_variant<T<Kokkos::Cuda>::task_body>(
        registrar,
        task_name);
    }
#endif
  }
#endif // HYPERION_USE_KOKKOS

protected:

  template <typename E>
  using MultiplyPSTermTask = MultiplyTermTask<E, CF_PS_SCALE>;

  template <typename E>
  using MultiplyWTermTask = MultiplyTermTask<E, CF_W>;

  template <typename E>
  using MultiplyATermTask = MultiplyTermTask<E, HYPERION_A_TERM_TABLE_AXES>;

public:

  static void
  preregister_tasks() {

    Legion::LayoutConstraintRegistrar
      cpu_constraints(Legion::FieldSpace::NO_SPACE);
    add_aos_right_ordering_constraint(cpu_constraints);
    cpu_constraints.add_constraint(
      Legion::SpecializedConstraint(LEGION_AFFINE_SPECIALIZE));
    auto cpu_layout_id = Legion::Runtime::preregister_layout(cpu_constraints);

    Legion::LayoutConstraintRegistrar
      gpu_constraints(Legion::FieldSpace::NO_SPACE);
    add_soa_left_ordering_constraint(gpu_constraints);
    gpu_constraints.add_constraint(
      Legion::SpecializedConstraint(LEGION_AFFINE_SPECIALIZE));
    auto gpu_layout_id = Legion::Runtime::preregister_layout(gpu_constraints);

    preregister_task_variants<MultiplyPSTermTask>(
      multiply_ps_task_name,
      multiply_ps_task_id,
      cpu_layout_id,
      gpu_layout_id);
    preregister_task_variants<MultiplyWTermTask>(
      multiply_w_task_name,
      multiply_w_task_id,
      cpu_layout_id,
      gpu_layout_id);
    preregister_task_variants<MultiplyATermTask>(
      multiply_a_task_name,
      multiply_a_task_id,
      cpu_layout_id,
      gpu_layout_id);
  }

protected:

  template <typename PT0, typename...PTs>
  static CFTable<Axes...>
  product(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const PT0& pt0,
    const PTs&...pts) {

    auto grid_sizes =
      std::array<size_t, sizeof...(PTs) + 1>{
      pt0->grid_size(),
      pts->grid_size()...};
    size_t grid_size = 0;
    for (auto& s : grid_sizes)
      grid_size = std::max(grid_size, s);

    return
      CFTable<Axes...>(
        ctx,
        rt,
        grid_size,
        index_axis_h<Axes>(*pt0, hcons<typename PTs::table_t...>(*pts...))...);
  }

  template <typename PT0>
    static CFTable<Axes...>
    product(
      Legion::Context ctx,
      Legion::Runtime* rt,
      const PT0& pt0) {

    return
      CFTable<Axes...>(
        ctx,
        rt,
        pt0->grid_size(),
        index_axis_h<Axes>(*pt0, hnil())...);
  }
};

template <cf_table_axes_t...Axes>
Legion::TaskID ProductCFTable<Axes...>::multiply_ps_task_id;
template <cf_table_axes_t...Axes>
Legion::TaskID ProductCFTable<Axes...>::multiply_w_task_id;
template <cf_table_axes_t...Axes>
Legion::TaskID ProductCFTable<Axes...>::multiply_a_task_id;
template <cf_table_axes_t...Axes>
const constexpr char* ProductCFTable<Axes...>::multiply_ps_task_name;
template <cf_table_axes_t...Axes>
const constexpr char* ProductCFTable<Axes...>::multiply_w_task_name;
template <cf_table_axes_t...Axes>
const constexpr char* ProductCFTable<Axes...>::multiply_a_task_name;

}  // synthesis

}  // hyperion

#endif /* HYPERION_SYNTHESIS_PRODUCT_CF_TABLE_H_ */
// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
