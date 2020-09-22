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
#ifndef HYPERION_TABLE_FIELD_H_
#define HYPERION_TABLE_FIELD_H_

#include <hyperion/hyperion.h>
#include <hyperion/utility.h>
#include <hyperion/Keywords.h>
#ifdef HYPERION_USE_CASACORE
# include <hyperion/MeasRef.h>
#endif
#include CXX_OPTIONAL_HEADER
#include <vector>

namespace hyperion {

class HYPERION_EXPORT TableField {
public:

  TableField() {}

  TableField(
    TypeTag dt_,
    Legion::FieldID fid_,
    const Keywords& kw_ = Keywords()
#ifdef HYPERION_USE_CASACORE
    , const MeasRef& mr_ = MeasRef()
    , const CXX_OPTIONAL_NAMESPACE::optional<hyperion::string>& rc_ =
      CXX_OPTIONAL_NAMESPACE::nullopt
#endif
    )
  : dt(dt_)
  , fid(fid_)
  , kw(kw_)
#ifdef HYPERION_USE_CASACORE
  , mr(mr_)
  , rc(rc_)
#endif
  {}

  std::vector<Legion::RegionRequirement>
  requirements(
    Legion::Runtime* rt,
    Legion::PrivilegeMode privilege,
    bool mapped) const;

  TypeTag dt;
  Legion::FieldID fid;
  Keywords kw;
#ifdef HYPERION_USE_CASACORE
  MeasRef mr;
  CXX_OPTIONAL_NAMESPACE::optional<hyperion::string> rc;
#endif
};

} // end namespace hyperion

#endif // HYPERION_TABLE_FIELD_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
