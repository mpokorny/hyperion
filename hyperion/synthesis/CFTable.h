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


  CFTable(hyperion::Table&& t)
    : CFTableBase(std::move(t)) {}

  CFTable(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const Legion::Rect<2>& cf_bounds,
    const Axis<AXES>&...axes)
    : CFTableBase() {

    // table index ColumnSpace
    ColumnSpace index_cs;
    constexpr size_t N = sizeof...(AXES);
    {
      Legion::Point<N> lo{axes.bounds().lo[0]...};
      Legion::Point<N> hi{axes.bounds().hi[0]...};
      Legion::Rect<N> bounds(lo, hi);
      Legion::IndexSpace is = rt->create_index_space(ctx, bounds);
      index_cs =
        ColumnSpace::create<cf_table_axes_t>(ctx, rt, {AXES...}, is, false);
    }

    // index columns
    Table::fields_t fields;
    fields.reserve(N + 1);
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
      for (size_t i = 0; i < N; ++i)
        fields.emplace_back(
          index_axes_cs[i],
          std::vector<std::pair<std::string, TableField>>{index_axes_fields[i]});
    }

    // CF columns
    {
      Legion::Point<N + 2>
        lo{axes.bounds().lo[0]..., cf_bounds.lo[0], cf_bounds.lo[1]};
      Legion::Point<N + 2>
        hi{axes.bounds().hi[0]..., cf_bounds.hi[0], cf_bounds.hi[1]};
      Legion::Rect<N + 2> bounds(lo, hi);
      Legion::IndexSpace is = rt->create_index_space(ctx, bounds);
      ColumnSpace cs =
        ColumnSpace::create<cf_table_axes_t>(
          ctx,
          rt, {AXES..., CF_X, CF_Y},
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
      hyperion::Table::create(ctx, rt, index_cs, std::move(fields));

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
