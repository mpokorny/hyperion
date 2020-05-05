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
    const Legion::PhysicalRegion& index_col_md,
    const Legion::LogicalRegion& index_col_parent,
    const std::tuple<Legion::LogicalRegion, Legion::PhysicalRegion>&
      index_col,
    const Legion::LogicalRegion& fields_parent,
    const std::tuple<Legion::LogicalRegion, Legion::PhysicalRegion>&
      fixed_fields,
    const std::optional<
      std::tuple<Legion::LogicalRegion, Legion::PhysicalRegion>>& free_fields,
    const std::unordered_map<std::string, std::shared_ptr<PhysicalColumn>>&
      columns);

  PhysicalTable(const PhysicalTable& other);

  PhysicalTable(PhysicalTable&& other);

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

  static std::optional<
    std::tuple<
      std::vector<PhysicalTable>,
      std::vector<Legion::RegionRequirement>::const_iterator,
      std::vector<Legion::PhysicalRegion>::const_iterator>>
  create_many(
    Legion::Runtime *rt,
    const std::vector<Legion::RegionRequirement>::const_iterator& reqs_begin,
    const std::vector<Legion::RegionRequirement>::const_iterator& reqs_end,
    const std::vector<Legion::PhysicalRegion>::const_iterator& prs_begin,
    const std::vector<Legion::PhysicalRegion>::const_iterator& prs_end,
    std::optional<unsigned> max_number = std::nullopt);

  Table
  table(Legion::Context ctx, Legion::Runtime* rt) const;

  std::optional<std::string>
  axes_uid() const;

  std::vector<int>
  index_axes() const;

  unsigned
  index_rank() const;

  ColumnSpace
  index_column_space(Legion::Context ctx, Legion::Runtime* rt) const;

  Legion::IndexSpace
  index_column_space_index_space() const;

  const Legion::PhysicalRegion&
  index_column_space_metadata() const;

  std::optional<std::shared_ptr<PhysicalColumn>>
  column(const std::string& name) const;

  const std::unordered_map<std::string, std::shared_ptr<PhysicalColumn>>&
  columns() const;

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
    const std::optional<Column::Requirements>& default_column_requirements =
      Column::default_requirements) const;

  bool
  add_columns(
    Legion::Context ctx,
    Legion::Runtime* rt,
    std::vector<
      std::tuple<
        ColumnSpace,
        std::vector<std::pair<std::string, TableField>>>>&& columns);

  bool
  remove_columns(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::unordered_set<std::string>& columns);

  void
  unmap_regions(Legion::Context ctx, Legion::Runtime* rt) const;

  void
  remap_regions(Legion::Context ctx, Legion::Runtime* rt) const;

  ColumnSpacePartition
  partition_rows(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<std::optional<size_t>>& block_sizes) const;

  Table
  reindexed(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<std::pair<int, std::string>>& index_axes,
    bool allow_rows) const;

  template <typename D>
  Table
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

  static void
  preregister_tasks();

  static void
  reindex_copy_values_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *rt);

  static Legion::TaskID reindex_copy_values_task_id;

  static const char* reindex_copy_values_task_name;

protected:

  Legion::PhysicalRegion m_index_col_md;

  Legion::LogicalRegion m_index_col_parent;

  std::tuple<Legion::LogicalRegion, Legion::PhysicalRegion> m_index_col;

  Legion::LogicalRegion m_fields_parent;

  std::tuple<Legion::LogicalRegion, Legion::PhysicalRegion> m_fixed_fields;

  std::optional<std::tuple<Legion::LogicalRegion, Legion::PhysicalRegion>>
    m_free_fields;

  std::unordered_map<std::string, std::shared_ptr<PhysicalColumn>> m_columns;

  std::unordered_map<std::string, Legion::PhysicalRegion> m_attached;

  std::string m_axes_uid;

  std::vector<int> m_index_axes;
};

} // end namespace hyperion

#endif // HYPERION_PHYSICAL_TABLE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
