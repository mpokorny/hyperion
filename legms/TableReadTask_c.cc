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
#include <legms/TableReadTask_c.h>
#include <legms/TableReadTask.h>
#include <legms/c_util.h>

#pragma GCC visibility push(default)
#include <legion/legion_c_util.h>
#pragma GCC visibility pop

using namespace legms;

// void
// table_block_read_task(
//   legion_context_t context,
//   legion_runtime_t runtime,
//   const char* path,
//   const table_t* table,
//   unsigned num_column_names,
//   const char** column_names,
//   size_t block_length) {

//   TableReadTask read_task(
//     path,
//     TableGenArgs(*table)(
//       Legion::CObjectWrapper::unwrap(context)->context(),
//       Legion::CObjectWrapper::unwrap(runtime)).get(),
//     column_names,
//     column_names + num_column_names,
//     block_length);
//   read_task.dispatch();
// }

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
