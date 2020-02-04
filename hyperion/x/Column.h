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
#ifdef HYPERION_USE_CASACORE
# include <hyperion/MeasRef.h>
#endif
#include <hyperion/x/ColumnSpace.h>

namespace hyperion {
namespace x {

struct HYPERION_API Column {

  Column() {}

  Column(
    TypeTag dt_,
    Legion::FieldID fid_,
#ifdef HYPERION_USE_CASACORE
    const hyperion::MeasRef& mr_,
#endif
    const hyperion::Keywords& kw_,
    const ColumnSpace& csp_,
    const Legion::RegionRequirement& vreq_)
    : dt(dt_)
    , fid(fid_)
#ifdef HYPERION_USE_CASACORE
    , mr(mr_)
#endif
    , kw(kw_)
    , csp(csp_)
    , vreq(vreq_) {}

  bool
  is_valid() const {
    return csp.is_valid();
  }

  static constexpr const Legion::FieldID COLUMN_INDEX_VALUE_FID = 0;
  static constexpr const Legion::FieldID COLUMN_INDEX_ROWS_FID = 1;
  typedef std::vector<Legion::DomainPoint> COLUMN_INDEX_ROWS_TYPE;

  Legion::LogicalRegion
  create_index(Legion::Context ctx, Legion::Runtime* rt) const;

  static void
  preregister_tasks();

  TypeTag dt;
  Legion::FieldID fid;
#ifdef HYPERION_USE_CASACORE
  hyperion::MeasRef mr;
#endif
  hyperion::Keywords kw;
  ColumnSpace csp;
  Legion::RegionRequirement vreq;

  template <TypeTag DT>
  static acc_field_redop_rhs<typename DataType<DT>::ValueType>
  index_accumulate_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context,
    Legion::Runtime* rt);

private:

  static Legion::TaskID index_accumulate_task_id[HYPERION_NUM_TYPE_TAGS];

  static std::string index_accumulate_task_name[HYPERION_NUM_TYPE_TAGS];

  template <TypeTag DT>
  static void
  preregister_index_accumulate_task();
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
