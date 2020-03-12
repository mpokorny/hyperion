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
# include <optional>
# include <string>
# include <unordered_map>
# include <vector>
#pragma GCC visibility pop

namespace hyperion {

class HYPERION_API PhysicalTable {

  PhysicalTable(
    Legion::LogicalRegion table_parent,
    Legion::PhysicalRegion table_pr,
    std::unordered_map<std::string, PhysicalColumn> columns);

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

  std::optional<PhysicalColumn>
  column(const std::string& name) const;

  std::optional<Legion::Point<1>>
  index_column_space(Legion::Runtime* rt) const;

  static std::optional<Legion::Point<1>>
  index_column_space(
    Legion::Runtime* rt,
    const Legion::LogicalRegion& parent,
    const Legion::PhysicalRegion& pr);

  std::optional<PhysicalColumn>
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
    const std::map<
      std::string,
      std::optional<
        std::tuple<bool, Legion::PrivilegeMode, Legion::CoherenceProperty>>>&
    column_modes = {},
    bool columns_mapped = true,
    Legion::PrivilegeMode columns_privilege = READ_ONLY,
    Legion::CoherenceProperty columns_coherence = EXCLUSIVE) const;

  Table::columns_result_t
  columns(Legion::Runtime *rt) const;

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
    bool destroy_orphan_column_spaces = true);

protected:

  Legion::LogicalRegion m_table_parent;

  Legion::PhysicalRegion m_table_pr;

  std::unordered_map<std::string, PhysicalColumn> m_columns;
};

} // end namespace hyperion

#endif // HYPERION_PHYSICAL_TABLE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
