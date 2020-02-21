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
#ifndef HYPERION_TABLE_READ_TASK_H_
#define HYPERION_TABLE_READ_TASK_H_

#include <hyperion/hyperion.h>
#include <hyperion/Table.h>
#include <hyperion/utility.h>
#include <hyperion/Column.h>

#pragma GCC visibility push(default)
# include <array>
# include <cstring>
# include <memory>
# include <new>
# include <type_traits>
# include <unordered_map>

# include <casacore/tables/Tables.h>
#pragma GCC visibility pop

namespace hyperion {

class HYPERION_API TableReadTask {
public:

  static Legion::TaskID TASK_ID;
  static const char* TASK_NAME;

  TableReadTask(
    const std::string& table_path,
    const Table& table,
    size_t min_block_length)
    : m_table_path(table_path)
    , m_table(table)
    , m_min_block_length(min_block_length) {

    // FIXME: the following is insufficient in the case of multiple nodes
    // FIXME: is this necessary?
    casacore::Table tb(
      casacore::String(table_path),
      casacore::TableLock::PermanentLockingWait);
  }

  static void
  preregister_task();

  void
  dispatch(Legion::Context ctx, Legion::Runtime* rt);

  static void
  base_impl(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *runtime);

private:

  std::string m_table_path;

  Table m_table;

  size_t m_min_block_length;
};

} // end namespace hyperion

#endif // HYPERION_TABLE_READ_TASK_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
