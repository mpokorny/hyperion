/*
 * Copyright 2019 Associated Universities, Inc. Washington DC, USA.
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
#ifndef HYPERION_X_TABLE_H_
#define HYPERION_X_TABLE_H_

#include <hyperion/hyperion.h>
#include <hyperion/x/Column.h>
#include <hyperion/x/ColumnSpace.h>
#include <hyperion/Table.h>

#pragma GCC visibility push(default)
# include <array>
# include <string>
# include <type_traits>
# include <utility>
# include <vector>
#pragma GCC visibility pop

namespace hyperion {
namespace x {

struct HYPERION_API Table {

  static const constexpr size_t MAX_COLUMNS = HYPERION_MAX_NUM_TABLE_COLUMNS;

  static const constexpr Legion::FieldID COLUMNS_NM_FID = 0;
  typedef hyperion::string COLUMNS_NM_TYPE;
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using ColumnsNameAccessor =
    Legion::FieldAccessor<
      MODE,
      COLUMNS_NM_TYPE,
      1,
      Legion::coord_t,
      Legion::AffineAccessor<COLUMNS_NM_TYPE, 1, Legion::coord_t>,
      CHECK_BOUNDS>;

  static const constexpr Legion::FieldID COLUMNS_DT_FID = 1;
  typedef hyperion::TypeTag COLUMNS_DT_TYPE;
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using ColumnsDatatypeAccessor =
    Legion::FieldAccessor<
      MODE,
      COLUMNS_DT_TYPE,
      1,
      Legion::coord_t,
      Legion::AffineAccessor<COLUMNS_DT_TYPE, 1, Legion::coord_t>,
      CHECK_BOUNDS>;

  static const constexpr Legion::FieldID COLUMNS_KW_FID = 2;
  typedef hyperion::Keywords COLUMNS_KW_TYPE;
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using ColumnsKeywordsAccessor =
    Legion::FieldAccessor<
      MODE,
      COLUMNS_KW_TYPE,
      1,
      Legion::coord_t,
      Legion::AffineAccessor<COLUMNS_KW_TYPE, 1, Legion::coord_t>,
      CHECK_BOUNDS>;

  static const constexpr Legion::FieldID COLUMNS_MR_FID = 3;
  typedef hyperion::MeasRef COLUMNS_MR_TYPE;
  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using ColumnsMeasRefAccessor =
    Legion::FieldAccessor<
      MODE,
      COLUMNS_MR_TYPE,
      1,
      Legion::coord_t,
      Legion::AffineAccessor<COLUMNS_MR_TYPE, 1, Legion::coord_t>,
      CHECK_BOUNDS>;

  Table() {}

  Table(
    const Legion::LogicalRegion& columns_lr,
    const Legion::LogicalRegion& values_lr,
    const ColumnSpace& column_space);

  static Table
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<std::pair<std::string, Column>>& columns,
    const ColumnSpace& column_space);

  typedef std::array<Table, MAX_COLUMNS> convert_result_t;
  // The following check is at the end of this file, after the Table class has
  // been defined --
  // static_assert(sizeof(convert_result_t) <= LEGION_MAX_RETURN_SIZE);

  static Legion::Future /* convert_result_t */
  convert(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const hyperion::Table& table,
    const std::unordered_map<std::string, Legion::FieldID> fids);

  static convert_result_t
  convert(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::unordered_map<std::string, Legion::FieldID>& fids,
    const std::vector<Legion::IndexSpace>& col_values_iss,
    const std::vector<
      std::tuple<
        Legion::PhysicalRegion,
        Legion::PhysicalRegion,
        std::optional<hyperion::MeasRef::DataRegions>,
        std::optional<hyperion::Keywords::pair<Legion::PhysicalRegion>>>>&
    col_prs);

  bool
  is_valid() const {
    return columns_lr != Legion::LogicalRegion::NO_REGION
      && values_lr != Legion::LogicalRegion::NO_REGION
      && column_space.is_valid();
  }

  void
  copy_values_from(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const hyperion::Table& table) const;

  static void
  copy_values_from(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const Legion::PhysicalRegion& columns_pr,
    const Legion::PhysicalRegion& values_pr,
    const std::vector<
      std::tuple<Legion::PhysicalRegion, Legion::PhysicalRegion>>& src_col_prs);

  void
  destroy(
    Legion::Context ctx,
    Legion::Runtime* rt,
    bool destroy_column_space=false,
    bool destroy_column_index_space=false);


  typedef std::array<hyperion::string, MAX_COLUMNS> column_names_result_t;
  static_assert(sizeof(column_names_result_t) <= LEGION_MAX_RETURN_SIZE);

  // names array provided in returned Future contains empty string values for
  // all indexes past the number of Columns in the Table
  Legion::Future /* column_names_result_t */
  column_names(Legion::Context ctx, Legion::Runtime *rt) const;

  // names array returned below contains empty string values for all indexes
  // past the number of Columns in the Table
  static column_names_result_t
  column_names(
    Legion::Runtime *rt,
    const Legion::PhysicalRegion& columns_pr);

  typedef Column column_result_t;

  // Column value provided in the returned Future should be checked for
  // validity, using Column::is_valid(); Column value will be invalid if no
  // Column in the Table has a matching name
  Legion::Future /* column_result_t */
  column(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::string& name) const;

  // Column value returned below should be checked for validity, using
  // Column::is_valid(); Column value will be invalid if no Column in the Table
  // has a matching name
  static column_result_t
  column(
    Legion::Runtime* rt,
    const Legion::PhysicalRegion& columns_pr,
    const Legion::FieldSpace& values_fs,
    const hyperion::string& name);

  static void
  preregister_tasks();

  Legion::LogicalRegion columns_lr;

  Legion::LogicalRegion values_lr;

  ColumnSpace column_space;

private:

  static Legion::TaskID init_task_id;

  static const char* init_task_name;

  static void
  init_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *rt);

  static Legion::TaskID column_task_id;

  static const char* column_task_name;

  static column_result_t
  column_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *rt);

  static Legion::TaskID column_names_task_id;

  static const char* column_names_task_name;

  static column_names_result_t
  column_names_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *rt);

  static Legion::TaskID convert_task_id;

  static const char* convert_task_name;

  static convert_result_t
  convert_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *rt);

  static Legion::TaskID copy_values_from_task_id;

  static const char* copy_values_from_task_name;

  static void
  copy_values_from_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *rt);
};

static_assert(sizeof(Table::convert_result_t) <= LEGION_MAX_RETURN_SIZE);

} // end namespace x
} // end namespace hyperion

#endif // HYPERION_X_COLUMN_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
