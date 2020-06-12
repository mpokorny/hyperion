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
#ifndef HYPERION_TABLE_READ_TASK_H_
#define HYPERION_TABLE_READ_TASK_H_

#include <hyperion/hyperion.h>
#include <hyperion/Table.h>
#include <hyperion/utility.h>
#include <hyperion/Column.h>

#include <array>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <unordered_map>

#include <casacore/tables/Tables.h>

namespace hyperion {

class HYPERION_EXPORT TableReadTask {
public:

  static std::tuple<
    std::vector<Legion::RegionRequirement>,
    std::vector<ColumnSpacePartition>,
    Table::Desc>
  requirements(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const PhysicalTable& table,
    const ColumnSpacePartition& table_partition = ColumnSpacePartition(),
    Legion::PrivilegeMode columns_privilege = WRITE_ONLY);

  static std::tuple<
    std::vector<Legion::RegionRequirement>,
    std::vector<ColumnSpacePartition>,
    Table::Desc>
  requirements(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const Table& table,
    const ColumnSpacePartition& table_partition = ColumnSpacePartition(),
    Legion::PrivilegeMode columns_privilege = WRITE_ONLY);

  struct Args {
    // path to MS table
    char table_path[1024];
    Table::Desc table_desc;
  };

  static void
  preregister_tasks();

  static Legion::TaskID TASK_ID;

  static constexpr const char* TASK_NAME = "TableReadTask";

  // read values from MS table for mapped regions in all mapped columns

  static void
  impl(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *runtime);
};

} // end namespace hyperion

#endif // HYPERION_TABLE_READ_TASK_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
