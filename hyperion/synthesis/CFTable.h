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
#ifndef HYPERION_SYNTHESIS_CF_TABLE_H_
#define HYPERION_SYNTHESIS_CF_TABLE_H_

#include <hyperion/synthesis/CFTableBase.h>
#include <hyperion/synthesis/CFPhysicalTable.h>

#include <memory>

namespace hyperion {
namespace synthesis {

template <cf_table_axes_t ...AXES>
class CFTable
  : public CFTableBase {
public:

  static const constexpr unsigned index_rank = sizeof...(AXES);

  static const constexpr unsigned d_x = index_rank;
  static const constexpr unsigned d_y = d_x + 1;
  static const constexpr unsigned cf_rank = d_y + 1;

  typedef CFPhysicalTable<AXES...> physical_table_t;

  template <Legion::PrivilegeMode MODE, bool CHECK_BOUNDS=HYPERION_CHECK_BOUNDS>
  using value_accessor_t =
    Legion::FieldAccessor<
      MODE,
      cf_value_t,
      cf_rank,
      Legion::coord_t,
      Legion::AffineAccessor<cf_value_t, cf_rank, Legion::coord_t>,
      CHECK_BOUNDS>;
  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  using ValueColumn =
    PhysicalColumnTD<
      ValueType<cf_value_t>::DataType,
      index_rank,
      cf_rank,
      A,
      COORD_T>;

  template <Legion::PrivilegeMode MODE, bool CHECK_BOUNDS=HYPERION_CHECK_BOUNDS>
  using weight_accessor_t =
    Legion::FieldAccessor<
      MODE,
      cf_weight_t,
      cf_rank,
      Legion::coord_t,
      Legion::AffineAccessor<cf_weight_t, cf_rank, Legion::coord_t>,
      CHECK_BOUNDS>;
  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  using WeightColumn =
    PhysicalColumnTD<
      ValueType<cf_weight_t>::DataType,
      index_rank,
      cf_rank,
      A,
      COORD_T>;

  CFTable() {}

  CFTable(const CFTable& t)
    : CFTableBase(t) {}

  CFTable(hyperion::Table&& t)
    : CFTableBase(std::move(t)) {}

  CFTable(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const size_t& grid_size,
    const Axis<AXES>&...axes)
    : CFTableBase() {

    // table index ColumnSpace
    ColumnSpace index_cs;
    {
      Legion::coord_t c_lo[]{axes.bounds().lo[0]...};
      Legion::Point<index_rank> lo(c_lo);
      Legion::coord_t c_hi[]{axes.bounds().hi[0]...};
      Legion::Point<index_rank> hi(c_hi);
      Legion::Rect<index_rank> bounds(lo, hi);
      Legion::IndexSpace is = rt->create_index_space(ctx, bounds);
      index_cs =
        ColumnSpace::create<cf_table_axes_t>(ctx, rt, {AXES...}, is, false);
    }

    // index columns
    Table::fields_t fields;
    fields.reserve(index_rank + 1);
    {
      std::vector<ColumnSpace>
        index_axes_cs{
        ColumnSpace::create<cf_table_axes_t>(
          ctx,
          rt,
          {AXES},
          rt->create_index_space(ctx, axes.bounds()),
          true)...};
      std::vector<std::pair<std::string, TableField>>
        index_axes_fields{
        {cf_table_axis<AXES>::name,
         TableField(
           ValueType<typename cf_table_axis<AXES>::type>::DataType,
           INDEX_VALUE_FID)}...};
      for (size_t i = 0; i < index_rank; ++i)
        fields.emplace_back(
          index_axes_cs[i],
          std::vector<std::pair<std::string, TableField>>{index_axes_fields[i]});
    }

    // CF columns
    {
      assert(grid_size > 0);
      Legion::coord_t
        c_hi[]{axes.bounds().hi[0]...,
          static_cast<Legion::coord_t>(grid_size - 1),
          static_cast<Legion::coord_t>(grid_size - 1)};
      Legion::Point<cf_rank> hi(c_hi);
      Legion::Rect<cf_rank> bounds(Legion::Point<cf_rank>::ZEROES(), hi);
      Legion::IndexSpace is = rt->create_index_space(ctx, bounds);
      ColumnSpace cs =
        ColumnSpace::create<cf_table_axes_t>(
          ctx,
          rt,
          {AXES..., CF_X, CF_Y},
          is,
          false);
      fields.emplace_back(
        cs,
        std::vector<std::pair<std::string, TableField>>{
          {CF_VALUE_COLUMN_NAME,
           TableField(ValueType<cf_value_t>::DataType, CF_VALUE_FID)},
          {CF_WEIGHT_COLUMN_NAME,
           TableField(ValueType<cf_weight_t>::DataType, CF_WEIGHT_FID)}});
    }

    auto tbl =
      hyperion::Table::create(ctx, rt, std::move(index_cs), std::move(fields));

    // initialize columns -- but not the CF values and weights
    {
      auto colreqs = Column::default_requirements;
      colreqs.values = Column::Req{
        WRITE_ONLY /* privilege */,
        EXCLUSIVE /* coherence */,
        true /* mapped */
      };
      auto reqs =
        tbl.requirements(
          ctx,
          rt,
          ColumnSpacePartition(),
          {{CF_VALUE_COLUMN_NAME, CXX_OPTIONAL_NAMESPACE::nullopt},
           {CF_WEIGHT_COLUMN_NAME, CXX_OPTIONAL_NAMESPACE::nullopt}},
          colreqs);
#if HAVE_CXX17
      auto& [treqs, tparts, tdesc] = reqs;
#else // !HAVE_CXX17
      auto& treqs = std::get<0>(reqs);
      auto& tdesc = std::get<2>(reqs);
#endif // HAVE_CXX17
      InitIndexColumnTaskArgs args;
      args.desc = tdesc;
      InitIndexColumnTaskArgs::initializer<AXES...>::init(args, axes...);
      //((args.values<AXES>() = axes.values), ...);
      std::unique_ptr<char[]> buf =
        std::make_unique<char[]>(args.serialized_size());
      args.serialize(buf.get());
      Legion::TaskLauncher task(
        init_index_column_task_id,
        Legion::TaskArgument(buf.get(), args.serialized_size()));
      for (auto& r : treqs)
        task.add_region_requirement(r);
      rt->execute_task(ctx, task);
    }

    *this = std::move(tbl);
  }
};


} // end namespace synthesis
} // end namespace hyperion

#endif // HYPERION_SYNTHESIS_CF_TABLE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
