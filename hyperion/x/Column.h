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
#ifndef HYPERION_X_COLUMN_H_
#define HYPERION_X_COLUMN_H_

#include <hyperion/hyperion.h>
#include <hyperion/utility.h>
#include <hyperion/Keywords.h>
#include <hyperion/MeasRef.h>

namespace hyperion {
namespace x {

struct HYPERION_API Column {

  Column() {}

  Column(
    TypeTag dt_,
    Legion::FieldID fid_,
    const hyperion::MeasRef& mr_,
    const hyperion::Keywords& kw_)
    : dt(dt_)
    , fid(fid_)
    , mr(mr_)
    , kw(kw_) {}

  bool
  is_valid() const {
    return dt != HYPERION_TYPE_NONE;
  }

  TypeTag dt;
  Legion::FieldID fid;
  hyperion::MeasRef mr;
  hyperion::Keywords kw;
};

} // end namespace x
} // end namespace hyperion

#endif // HYPERION_X_COLUMN_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
