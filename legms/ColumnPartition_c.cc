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
#include <legms/c_util.h>
#include <legms/ColumnPartition_c.h>
#include <legms/ColumnPartition.h>

#pragma GCC visibility push(default)
#include <legion/legion_c_util.h>
#pragma GCC visibility pop

using namespace legms;
using namespace legms::CObjectWrapper;

const Legion::FieldID axes_uid_fs[1] = {ColumnPartition::AXES_UID_FID};
const Legion::FieldID axes_fs[1] = {ColumnPartition::AXES_FID};

const legion_field_id_t*
column_partition_axes_uid_fs() {
  return axes_uid_fs;
}

const legion_field_id_t*
column_partition_axes_fs() {
  return axes_fs;
}

legion_index_space_t
column_partition_color_space(legion_runtime_t rt, column_partition_t cp) {
  return
    Legion::CObjectWrapper::wrap(
      Legion::CObjectWrapper::unwrap(rt)
      ->get_index_partition_color_space_name(
        Legion::CObjectWrapper::unwrap(cp.index_partition)));
}

void
column_partition_destroy(
  legion_context_t ctx,
  legion_runtime_t rt,
  column_partition_t cp,
  int destroy_color_space) {
  unwrap(cp).destroy(
    Legion::CObjectWrapper::unwrap(ctx)->context(),
    Legion::CObjectWrapper::unwrap(rt),
    destroy_color_space);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
