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
#pragma GCC visibility push(default)
#include <vector>
#include <legion/legion_c_util.h>
#pragma GCC visibility pop

#include <hyperion/c_util.h>
#include <hyperion/Column_c.h>
#include <hyperion/Column.h>

using namespace hyperion;
using namespace hyperion::CObjectWrapper;

const Legion::FieldID metadata_fs[3] =
  {Column::METADATA_NAME_FID,
   Column::METADATA_AXES_UID_FID,
   Column::METADATA_DATATYPE_FID};
const Legion::FieldID axes_fs[1] = {Column::AXES_FID};
const Legion::FieldID values_fs[1] = {Column::VALUE_FID};

const legion_field_id_t*
column_metadata_fs() {
  return metadata_fs;
}

const legion_field_id_t*
column_axes_fs() {
  return axes_fs;
}

// values field types: [metadata[0][2]]
const legion_field_id_t*
column_values_fs() {
  return values_fs;
}

unsigned
column_rank(legion_runtime_t rt, column_t col) {
  return unwrap(col).rank(Legion::CObjectWrapper::unwrap(rt));
}

int
column_is_empty(column_t col) {
  return unwrap(col).is_empty();
}

column_partition_t
column_partition_on_axes(
  legion_context_t ctx,
  legion_runtime_t rt,
  column_t col,
  unsigned num_axes,
  const int* axes) {

  std::vector<int> ax;
  ax.reserve(num_axes);
  std::copy(axes, axes + num_axes, std::back_inserter(ax));

  return
    wrap(
      unwrap(col)
      .partition_on_axes(
        Legion::CObjectWrapper::unwrap(ctx)->context(),
        Legion::CObjectWrapper::unwrap(rt),
        ax));
}

column_partition_t
column_projected_column_partition(
  legion_context_t ctx,
  legion_runtime_t rt,
  column_t col,
  column_partition_t cp) {
  return
    wrap(
      unwrap(col)
      .projected_column_partition(
        Legion::CObjectWrapper::unwrap(ctx)->context(),
        Legion::CObjectWrapper::unwrap(rt),
        unwrap(cp)));
}

void
column_destroy(legion_context_t ctx, legion_runtime_t rt, column_t col) {
  unwrap(col).destroy(
    Legion::CObjectWrapper::unwrap(ctx)->context(),
    Legion::CObjectWrapper::unwrap(rt));
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
