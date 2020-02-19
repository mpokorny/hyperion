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
#ifndef HYPERION_X_TABLE_FIELD_H_
#define HYPERION_X_TABLE_FIELD_H_

#include <hyperion/hyperion.h>
#include <hyperion/utility.h>
#include <hyperion/Keywords.h>
#ifdef HYPERION_USE_CASACORE
# include <hyperion/MeasRef.h>
# pragma GCC visibility push(default)
#  include <optional>
#  include <string>
# pragma GCC visibility pop
#endif

namespace hyperion {
namespace x {

struct HYPERION_API TableField {

  TableField() {}

  TableField(
    TypeTag dt_,
    Legion::FieldID fid_,
#ifdef HYPERION_USE_CASACORE
    const hyperion::MeasRef& mr_,
    const std::optional<hyperion::string>& rc_,
#endif
    const hyperion::Keywords& kw_)
    : dt(dt_)
    , fid(fid_)
#ifdef HYPERION_USE_CASACORE
    , mr(mr_)
    , rc(rc_)
#endif
    , kw(kw_) {}

  TypeTag dt;
  Legion::FieldID fid;
#ifdef HYPERION_USE_CASACORE
  hyperion::MeasRef mr;
  std::optional<hyperion::string> rc;
#endif
  hyperion::Keywords kw;
};

} // end namespace x
} // end namespace hyperion

#endif // HYPERION_X_TABLE_FIELD_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
