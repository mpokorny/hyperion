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
#ifndef HYPERION_PHYSICAL_TABLE_H_
#define HYPERION_PHYSICAL_TABLE_H_

#include <hyperion/hyperion.h>
#include <hyperion/PhysicalColumn.h>
#include <hyperion/Table.h>

#pragma GCC visibility push(default)
# include <memory>
# include <optional>
# include <string>
# include <unordered_map>
# include <vector>
#pragma GCC visibility pop

namespace hyperion {

class HYPERION_API PhysicalTable {
public:

  PhysicalTable(
    Legion::LogicalRegion table_parent,
    Legion::PhysicalRegion table_pr,
    const std::unordered_map<std::string, std::shared_ptr<PhysicalColumn>>&
    columns);

  static std::optional<
    std::tuple<
      PhysicalTable,
      std::vector<Legion::RegionRequirement>::const_iterator,
      std::vector<Legion::PhysicalRegion>::const_iterator>>
  create(
    Legion::Runtime *rt,
    const std::vector<Legion::RegionRequirement>::const_iterator& reqs_begin,
    const std::vector<Legion::RegionRequirement>::const_iterator& reqs_end,
    const std::vector<Legion::PhysicalRegion>::const_iterator& prs_begin,
    const std::vector<Legion::PhysicalRegion>::const_iterator& prs_end);

  Table
  table() const;

  std::optional<ColumnSpace::AXIS_SET_UID_TYPE>
  axes_uid() const;

  std::optional<std::shared_ptr<PhysicalColumn>>
  column(const std::string& name) const;

  std::optional<Legion::Point<1>>
  index_column_space(Legion::Runtime* rt) const;

  static std::optional<Legion::Point<1>>
  index_column_space(
    Legion::Runtime* rt,
    const Legion::LogicalRegion& parent,
    const Legion::PhysicalRegion& pr);

  std::optional<std::shared_ptr<PhysicalColumn>>
  index_column(Legion::Runtime* rt) const;

  unsigned
  index_rank(Legion::Runtime* rt) const;

  static unsigned
  index_rank(
    Legion::Runtime* rt,
    const Legion::LogicalRegion& parent,
    const Legion::PhysicalRegion& pr);

  bool
  is_conformant(
    Legion::Runtime* rt,
    const Legion::IndexSpace& cs_is,
    const Legion::PhysicalRegion& cs_md_pr) const;

  std::tuple<
    std::vector<Legion::RegionRequirement>,
    std::vector<Legion::LogicalPartition>>
  requirements(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const ColumnSpacePartition& table_partition = ColumnSpacePartition(),
    Legion::PrivilegeMode table_privilege = READ_ONLY,
    const std::map<std::string, std::optional<Column::Requirements>>&
      column_requirements = {},
    const Column::Requirements& default_column_requirements =
      Column::default_requirements) const;

  const std::unordered_map<std::string, std::shared_ptr<PhysicalColumn>>&
  columns() const {
    return m_columns;
  }

  decltype(Table::columns_result_t::fields)
  column_fields(Legion::Runtime *rt) const;

  bool
  add_columns(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<
      std::tuple<
        ColumnSpace,
        bool,
        std::vector<std::pair<std::string, TableField>>>>& columns);

  bool
  remove_columns(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::unordered_set<std::string>& columns,
    bool destroy_orphan_column_spaces = true,
    bool destroy_field_data = true);

  void
  unmap_regions(Legion::Context ctx, Legion::Runtime* rt) const;

  void
  remap_regions(Legion::Context ctx, Legion::Runtime* rt) const;

  ColumnSpacePartition
  partition_rows(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<std::optional<size_t>>& block_sizes) const;

  Legion::LogicalRegion
  reindexed(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<std::pair<int, std::string>>& index_axes,
    bool allow_rows) const;

  template <typename D>
  Legion::LogicalRegion
  reindexed(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<D>& index_axes,
    bool allow_rows) const {

    std::vector<std::pair<int, std::string>> iax;
    iax.reserve(index_axes.size());
    for (auto& d : index_axes) {
      int i = static_cast<int>(d);
      iax.emplace_back(i, Axes<D>::names[i]);
    }
    return reindexed(ctx, rt, iax, allow_rows);
  }

  // boolean values in 'column_modes': (read-only, restricted, mapped)
  bool
  attach_columns(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const CXX_FILESYSTEM_NAMESPACE::path& file_path,
    const std::unordered_map<std::string, std::string>& column_paths,
    const std::unordered_map<std::string, std::tuple<bool, bool, bool>>&
    column_modes);

  void
  detach_columns(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::unordered_set<std::string>& columns);

  void
  acquire_columns(Legion::Context ctx, Legion::Runtime* rt);

  void
  release_columns(Legion::Context ctx, Legion::Runtime* rt);

protected:

  Legion::LogicalRegion m_table_parent;

  Legion::PhysicalRegion m_table_pr;

  std::unordered_map<std::string, std::shared_ptr<PhysicalColumn>> m_columns;

  std::unordered_map<std::string, Legion::PhysicalRegion> m_attached;
};

} // end namespace hyperion

#endif // HYPERION_PHYSICAL_TABLE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
