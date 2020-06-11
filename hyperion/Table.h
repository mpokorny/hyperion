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
#ifndef HYPERION_TABLE_H_
#define HYPERION_TABLE_H_

#include <hyperion/hyperion.h>
#include <hyperion/utility.h>
#include <hyperion/Column.h>
#include <hyperion/ColumnSpace.h>
#include <hyperion/ColumnSpacePartition.h>
#include <hyperion/TableField.h>

#include <array>
#include <limits>
#include <map>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hyperion {

class PhysicalTable;

// helper template classes for handling different table index spaces
template <typename T>
struct table_indexing {
  template <T...AXES>
  struct axes {};

  template <size_t N, T A0, typename...AXES>
  struct index_of_ {
    //typedef ... type;
  };

  template <size_t N, T A0, T ...AXES>
  struct index_of_<N, A0, axes<A0, AXES...>> {
    typedef std::integral_constant<size_t, N> type;
  };

  template <size_t N, T A0, T A1, T ...AXES>
  struct index_of_<N, A0, axes<A1, AXES...>> {
    typedef typename index_of_<N + 1, A0, axes<AXES...>>::type type;
  };

  template <size_t N, T A0>
  struct index_of_<N, A0, axes<>> {
    typedef void type;
  };

  template <T A0, T ...AXES>
  struct index_of
    : public index_of_<0, A0, axes<AXES...>> {};

  template <T A0, T ...AXES>
  static constexpr bool includes =
    !std::is_void<typename index_of<A0, AXES...>::type>::value;

  template <
    int M,
    int N,
    T A0,
    typename AXES,
    typename V = typename index_of_<0, A0, AXES>::type,
    std::enable_if_t<(M <= N), int> = 0>
  struct Pt_ {
    Pt_(const Legion::Point<N>& _pt)
      : pt(_pt[V::value]) {};
    Legion::Point<1> pt;
  };

  template <int M, int N, T A0, typename AXES>
  struct Pt_<M, N, A0, AXES, void> {
    Pt_(const Legion::Point<N>& _pt) {
      for (size_t i = 0; i < M; ++i)
        pt[i] = _pt[i];
    }
    Legion::Point<M> pt;
  };

  template <int M, int N, T A0, T...AXES>
  using Pt = Pt_<M, N, A0, axes<AXES...>>;
};

class HYPERION_EXPORT Table {

  // FIXME: add support for table keywords

public:

  static const constexpr size_t MAX_COLUMNS = HYPERION_MAX_NUM_TABLE_COLUMNS;

  struct add_columns_result_t {
    typedef std::tuple<std::string, Column> col_t;

    std::vector<col_t> cols;

    size_t
    legion_buffer_size(void) const;

    size_t
    legion_serialize(void* buffer) const;

    size_t
    legion_deserialize(const void* buffer);
  };

  struct Desc {
    unsigned num_columns;
    std::array<Column::Desc, HYPERION_MAX_NUM_TABLE_COLUMNS> columns;
  };

  Desc
  desc() const {
    Desc result;
    result.num_columns = m_columns.size();
    assert(result.num_columns <= result.columns.size());
    std::transform(
      m_columns.begin(),
      m_columns.end(),
      &result.columns[0],
      [](const auto& nm_col) {
#if HAVE_CXX17
        auto& [nm, col] = nm_col;
#else // !HAVE_CXX17
        auto& nm = std::get<0>(nm_col);
        auto& col = std::get<1>(nm_col);
#endif // HAVE_CXX17
        return col.desc(nm);
      });
    return result;
  }

  template <size_t N>
  using DescM = std::array<Desc, N>;

public:

  Table() {}

  Table(
    Legion::Runtime* rt,
    ColumnSpace&& index_col_cs,
    const Legion::LogicalRegion& index_col_region,
    const std::unordered_map<std::string, Column>& columns);

  Table(
    ColumnSpace&& index_col_cs,
    const Legion::LogicalRegion& index_col_region,
    const Legion::LogicalRegion& index_col_parent,
    const std::unordered_map<std::string, Column>& columns)
    : m_index_col_cs(index_col_cs)
    , m_index_col_region(index_col_region)
    , m_index_col_parent(index_col_parent)
    , m_columns(columns) {}

  Table(const Table& t);

  Table(Table&& t);

  Table&
  operator=(const Table& rhs);

  Table&
  operator=(Table&& rhs);

  typedef std::vector<
    std::tuple<
      ColumnSpace,
      std::vector<std::pair<std::string, TableField>>>> fields_t;

  static Table
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    ColumnSpace&& index_cs,
    fields_t&& fields);

  static Table
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const ColumnSpace& index_cs,
    fields_t&& fields) {
    return create(ctx, rt, index_cs.clone(ctx, rt), std::move(fields));
  }

  ColumnSpace
  index_column_space(Legion::Context ctx, Legion::Runtime* rt) const;

  bool
  is_empty() const;

  // Any ColumnSpacePartitions returned from requirements() should eventually be
  // destroyed by the caller. Such ColumnSpacePartitions should only appear when
  // "table_partition" is not empty or column_modes deselects some columns.
  std::tuple<
    std::vector<Legion::RegionRequirement>,
    std::vector<ColumnSpacePartition>,
    Desc>
  requirements(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const ColumnSpacePartition& table_partition = ColumnSpacePartition(),
    const std::map<
      std::string,
      CXX_OPTIONAL_NAMESPACE::optional<Column::Requirements>>&
      column_requirements = {},
    const CXX_OPTIONAL_NAMESPACE::optional<Column::Requirements>&
      default_column_requirements = Column::default_requirements) const;

private:

  std::tuple<std::vector<Legion::RegionRequirement>, Desc>
  requirements() const;

public:

  static std::tuple<
    std::vector<Legion::RegionRequirement>,
    std::vector<ColumnSpacePartition>,
    Desc>
  requirements(
    CXX_OPTIONAL_NAMESPACE::optional<Legion::Context> ctx,
    CXX_OPTIONAL_NAMESPACE::optional<Legion::Runtime*> rt,
    const ColumnSpace& index_col_cs,
    const Legion::LogicalRegion& index_col_region,
    const Legion::LogicalRegion& index_col_parent,
    const std::unordered_map<std::string, Column>& columns,
    const ColumnSpacePartition& table_partition = ColumnSpacePartition(),
    const std::map<
      std::string,
      CXX_OPTIONAL_NAMESPACE::optional<Column::Requirements>>&
      column_requirements = {},
    const CXX_OPTIONAL_NAMESPACE::optional<Column::Requirements>&
      default_column_requirements = Column::default_requirements);

  Legion::Future /* bool */
  is_conformant(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const ColumnSpace& cs) const;

  static bool
  is_conformant(
    Legion::Runtime* rt,
    const std::unordered_map<std::string, Column>& columns,
    const std::tuple<Legion::IndexSpace, Legion::PhysicalRegion>& index_cs,
    const Legion::IndexSpace& cs_is,
    const Legion::PhysicalRegion& cs_md_pr);

  bool
  add_columns(Legion::Context ctx, Legion::Runtime* rt, fields_t&& fields);

  static std::unordered_map<std::string, Column>
  add_columns(
    Legion::Context ctx,
    Legion::Runtime* rt,
    std::vector<
      std::tuple<
        ColumnSpace,
        size_t,
        std::vector<std::pair<hyperion::string, TableField>>>>&& new_columns,
    const std::unordered_map<std::string, Column>& columns,
    const std::vector<Legion::PhysicalRegion>& cs_md_prs,
    const std::tuple<Legion::IndexSpace, Legion::PhysicalRegion>& index_cs);

  bool
  remove_columns(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::unordered_set<std::string>& columns);

  static bool
  remove_columns(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::unordered_set<std::string>& rm_columns,
    const std::unordered_map<std::string, Column>& columns,
    const std::vector<ColumnSpace>& css,
    const std::vector<Legion::PhysicalRegion>& cs_md_prs);

  // each element of the block_sizes vector is the block size on the
  // corresponding axis of the index axes vector, with an empty value indicating
  // that there is no partitioning on that axis; if the length of block_sizes is
  // less than the length of the index axes vector the "missing" axes will not
  // be partitioned
  Legion::Future /* ColumnSpacePartition */
  partition_rows(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<CXX_OPTIONAL_NAMESPACE::optional<size_t>>& block_sizes)
    const;

  static ColumnSpacePartition
  partition_rows(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<CXX_OPTIONAL_NAMESPACE::optional<size_t>>& block_sizes,
    const Legion::IndexSpace& ics_is,
    const Legion::PhysicalRegion& ics_md_pr);

  bool
  is_valid() const {
    return m_index_col_region != Legion::LogicalRegion::NO_REGION;
  }

  // 'allow_rows' is intended to support the case where reindexing may not
  // result in a single value in a column per aggregate index, necessitating the
  // maintenance of a row index. A value of 'true' for this argument is always
  // safe, but may result in a degenerate axis when an aggregate index always
  // identifies a single value in a column. If the value is 'false' and a
  // non-degenerate axis is required by the reindexing, this method will return
  // an empty value. TODO: remove degenerate axes after the fact, and do that
  // automatically in this method, which would allow us to remove the
  // 'allow_rows' argument.
  Legion::Future /* Table */
  reindexed(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<std::pair<int, std::string>>& index_axes,
    bool allow_rows) const;

  template <typename D>
  Legion::Future /* Table */
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

  void
  destroy(Legion::Context ctx, Legion::Runtime* rt);

  const std::unordered_map<std::string, Column>&
  columns() const {
    return m_columns;
  }

  // boolean values in 'column_modes': (read-only, restricted, mapped)
  PhysicalTable
  attach_columns(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const CXX_FILESYSTEM_NAMESPACE::path& file_path,
    const std::unordered_map<std::string, std::string>& column_paths,
    const std::unordered_map<std::string, std::tuple<bool, bool, bool>>&
      column_modes) const;

  static void
  preregister_tasks();

protected:

  friend class PhysicalTable;

  static const constexpr Legion::FieldID m_index_col_fid = 0;

  static const constexpr TypeTag m_index_col_dt = HYPERION_TYPE_BOOL;

  ColumnSpace m_index_col_cs;

  Legion::LogicalRegion m_index_col_region;

  Legion::LogicalRegion m_index_col_parent;

  std::unordered_map<std::string, Column> m_columns;

public:
// protected:

//   friend class Legion::LegionTaskWrapper;

  static bool
  is_conformant_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *rt);

  static ColumnSpacePartition
  partition_rows_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *rt);

  static add_columns_result_t
  add_columns_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *rt);

  static Table
  reindexed_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *rt);

  static Legion::IndexSpace
  reindex_column_space_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *rt);

  static Legion::TaskID is_conformant_task_id;

  static const char* is_conformant_task_name;

  static Legion::TaskID partition_rows_task_id;

  static const char* partition_rows_task_name;

  static Legion::TaskID add_columns_task_id;

  static const char* add_columns_task_name;

  static Legion::TaskID reindexed_task_id;

  static const char* reindexed_task_name;

  static Legion::TaskID reindex_column_space_task_id;

  static const char* reindex_column_space_task_name;

  size_t
  legion_buffer_size(void) const;

  size_t
  legion_serialize(void* buffer) const;

  size_t
  legion_deserialize(const void* buffer);
};

} // end namespace hyperion

#endif // HYPERION_COLUMN_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
