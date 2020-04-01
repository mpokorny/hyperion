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
#ifndef HYPERION_COLUMN_H_
#define HYPERION_COLUMN_H_

#include <hyperion/hyperion.h>
#include <hyperion/utility.h>
#include <hyperion/Keywords.h>
#ifdef HYPERION_USE_CASACORE
# include <hyperion/MeasRef.h>
# pragma GCC visibility push(default)
#  include <optional>
# pragma GCC visibility pop
#endif
#pragma GCC visibility push(default)
# include <string>
#pragma GCC visibility pop
#include <hyperion/ColumnSpace.h>

namespace hyperion {

struct HYPERION_API Column {

  Column() {}

  Column(
    TypeTag dt_,
    Legion::FieldID fid_,
#ifdef HYPERION_USE_CASACORE
    const MeasRef& mr_,
    const std::optional<string>& rc_,
#endif
    const Keywords& kw_,
    const ColumnSpace& csp_,
    const Legion::LogicalRegion& vlr_)
  : dt(dt_)
  , fid(fid_)
#ifdef HYPERION_USE_CASACORE
  , mr(mr_)
  , rc(rc_)
#endif
  , kw(kw_)
  , csp(csp_)
  , vlr(vlr_) {}

  bool
  is_valid() const {
    return csp.is_valid();
  }

  struct Req {
    Legion::PrivilegeMode privilege;
    Legion::CoherenceProperty coherence;
    bool mapped;

    bool
    operator==(const Req& rhs) const {
      return privilege == rhs.privilege
      && coherence == rhs.coherence
      && mapped == rhs.mapped;
    }
  };

  struct Requirements {
    Req values;
    Req keywords;
    Req measref;
    Req column_space;
    Legion::MappingTagID tag;
  };

  static constexpr const Requirements default_requirements{
    Req{READ_ONLY, EXCLUSIVE, false},
    Req{READ_ONLY, EXCLUSIVE, true},
    Req{READ_ONLY, EXCLUSIVE, true},
    Req{READ_ONLY, EXCLUSIVE, true},
    0
  };

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
  MeasRef mr;
  std::optional<hyperion::string> rc;
#endif
  Keywords kw;
  ColumnSpace csp;
  Legion::LogicalRegion vlr;

// protected:

//   friend class Legion::LegionTaskWrapper;

  template <TypeTag DT>
  static acc_field_redop_rhs<typename DataType<DT>::ValueType>
  index_accumulate_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context,
    Legion::Runtime* rt);

  static Legion::TaskID index_accumulate_task_id[HYPERION_NUM_TYPE_TAGS];

  static std::string index_accumulate_task_name[HYPERION_NUM_TYPE_TAGS];

private:

  template <TypeTag DT>
  static void
  preregister_index_accumulate_task();
};
} // end namespace hyperion

#endif // HYPERION_COLUMN_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
