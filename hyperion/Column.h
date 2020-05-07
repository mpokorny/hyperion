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
#ifndef HYPERION_COLUMN_H_
#define HYPERION_COLUMN_H_

#include <hyperion/hyperion.h>
#include <hyperion/utility.h>
#include <hyperion/Keywords.h>
#include <hyperion/TableMapper.h>
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

/**
 * Logical table column
 *
 *  Maintains metadata about the column, including keywords, measures and
 *  Legion::LogicalRegion name.
 */
struct /*HYPERION_API*/ Column {

  /**
   * Create an empty Column
   */
  Column() {}

  /**
   * Create a Column
   *
   * Copies argument values
   */
  Column(
    hyperion::TypeTag dt_,
    Legion::FieldID fid_,
    const ColumnSpace& cs_,
    const Legion::LogicalRegion& region_,
    const Legion::LogicalRegion& parent_,
    const Keywords& kw_ = Keywords()
#ifdef HYPERION_USE_CASACORE
    , const MeasRef& mr_ = MeasRef()
    , const std::optional<hyperion::string>& rc_ = std::nullopt
#endif
    )
  : dt(dt_)
  , fid(fid_)
  , cs(cs_)
  , region(region_)
  , parent(parent_)
  , kw(kw_)
#ifdef HYPERION_USE_CASACORE
  , mr(mr_)
  , rc(rc_)
#endif
   {}

  /**
   * Is Column valid?
   *
   * @return true, if and only if the index space for the column is valid
   */
  bool
  is_valid() const {
    return cs.is_valid();
  }

  /**
   * Arguments for a region requirement
   */
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

  /**
   * Arguments for all region requirements
   *
   * Includes all regions referenced by the Column, as well as mapper arguments
   */
  struct Requirements {
    Req values; /**< values region */
    Req keywords; /**< keywords regions */
    Req measref; /**< measref regions */
    Req column_space; /**< column space region */
    Legion::MappingTagID tag; /**< values region mapping tag */
  };

  /**
   * Default Requirements
   */
  static constexpr const Requirements default_requirements{
    Req{READ_ONLY, EXCLUSIVE, false},
    Req{READ_ONLY, EXCLUSIVE, true},
    Req{READ_ONLY, EXCLUSIVE, true},
    Req{READ_ONLY, EXCLUSIVE, true},
    TableMapper::to_mapping_tag(TableMapper::default_column_layout_tag)
  };

  /**
   * Column descriptor
   *
   * Used in task arguments associated with Tables
   */
  struct Desc {
    hyperion::string name;
    hyperion::TypeTag dt;
    Legion::FieldID fid;
    Legion::LogicalRegion region;
    uint_least8_t n_kw;
#ifdef HYPERION_USE_CASACORE
    hyperion::string refcol;
    uint_least8_t n_mr;
#endif
  };

  /**
   * Create Desc for this Column with the given name
   */
  Desc
  desc(const std::string& name) const {
    return Desc{
      name,
        dt,
        fid,
        region,
        kw.num_regions()
#ifdef HYPERION_USE_CASACORE
        , rc.value_or(hyperion::string())
        , mr.num_regions()
#endif
        };
  }

  /**
   * Column index value field id
   */
  static constexpr const Legion::FieldID COLUMN_INDEX_VALUE_FID = 0;

  /**
   * Column index rows field id
   */
  static constexpr const Legion::FieldID COLUMN_INDEX_ROWS_FID = 1;
  typedef std::vector<Legion::DomainPoint> COLUMN_INDEX_ROWS_TYPE;

  /**
   * Create a column index Legion::LogicalRegion
   */
  Legion::LogicalRegion
  create_index(Legion::Context ctx, Legion::Runtime* rt) const;

  /**
   * Register (indexing) tasks
   *
   * Must be called be Legion runtime starts
   */
  static void
  preregister_tasks();

  hyperion::TypeTag dt; /**< value data type (as hyperion::TypeTag)*/
  Legion::FieldID fid; /**< value Legion::FieldID */
  ColumnSpace cs; /**< column ColumnSpace */
  Legion::LogicalRegion region; /**< column values region */
  Legion::LogicalRegion parent; /**< column values parent region */
  Keywords kw;/**< column keywords */
#ifdef HYPERION_USE_CASACORE
  MeasRef mr; /**< column MeasRef */
  std::optional<hyperion::string> rc; /**< measure reference column name */
#endif

// protected:

//   friend class Legion::LegionTaskWrapper;

  /**
   * Task to compute a column index
   */
  template <hyperion::TypeTag DT>
  static acc_field_redop_rhs<typename DataType<DT>::ValueType>
  index_accumulate_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context,
    Legion::Runtime* rt);

  /**
   * Legion::TaskIDs for typed index_accumulate_task<DT>
   */
  static Legion::TaskID index_accumulate_task_id[HYPERION_NUM_TYPE_TAGS];

  /**
   * Names of typed index_accumulate_task<DT>
   */
  static std::string index_accumulate_task_name[HYPERION_NUM_TYPE_TAGS];

private:

  /**
   * Register task for index_accumulate_task<DT>()
   */
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
