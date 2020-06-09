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
#include <hyperion/TableReadTask.h>
#include <hyperion/PhysicalTable.h>
#include <hyperion/PhysicalColumn.h>
#include <hyperion/TableMapper.h>

#include <casacore/casa/aipstype.h>
#include <casacore/casa/Arrays.h>

#include <algorithm>

using namespace hyperion;

using namespace Legion;

template <hyperion::TypeTag DT, int DIM>
static void
read_scalar_column(
  const casacore::Table& table,
  const casacore::ColumnDesc& col_desc,
  const PhysicalColumnTD<DT, 1, DIM, AffineAccessor>& column) {

  typedef typename DataType<DT>::CasacoreType CT;

  auto values = column.template accessor<WRITE_ONLY>();

  casacore::ScalarColumn<CT> col(table, col_desc.name());
  coord_t row_number;
  CT col_value;
  {
    PointInDomainIterator<DIM> pid(column.domain(), false);
    row_number = pid[0];
    col.get(row_number, col_value);
  }

  for (PointInDomainIterator<DIM> pid(column.domain(), false); pid(); pid++) {
    if (row_number != pid[0]) {
      row_number = pid[0];
      col.get(row_number, col_value);
    }
    DataType<DT>::from_casacore(values[*pid], col_value);
  }
}

template <hyperion::TypeTag DT, int DIM>
static void
read_array_column(
  const casacore::Table& table,
  const casacore::ColumnDesc& col_desc,
  const PhysicalColumnTD<DT, 1, DIM, AffineAccessor>& column) {

  typedef typename DataType<DT>::CasacoreType CT;

  auto values = column.template accessor<WRITE_ONLY>();

  casacore::ArrayColumn<CT> col(table, col_desc.name());
  coord_t row_number;
  unsigned array_cell_rank;
  {
    PointInDomainIterator<DIM> pid(column.domain(), false);
    row_number = pid[0];
    array_cell_rank = col.ndim(row_number);
  }

  casacore::Array<CT> col_array;
  col.get(row_number, col_array, true);
  switch (array_cell_rank) {
  case 1: {
    casacore::Vector<CT> col_vector;
    col_vector.reference(col_array);
    for (PointInDomainIterator<DIM> pid(column.domain(), false); pid(); pid++) {
      if (row_number != pid[0]) {
        row_number = pid[0];
        col.get(row_number, col_array, true);
        col_vector.reference(col_array);
      }
      DataType<DT>::from_casacore(values[*pid], col_vector[pid[DIM - 1]]);
    }
    break;
  }
  case 2: {
    casacore::IPosition ip(2);
    casacore::Matrix<CT> col_matrix;
    col_matrix.reference(col_array);
    for (PointInDomainIterator<DIM> pid(column.domain(), false); pid(); pid++) {
      if (row_number != pid[0]) {
        row_number = pid[0];
        col.get(row_number, col_array, true);
        col_matrix.reference(col_array);
      }
      ip[0] = pid[DIM - 1];
      ip[1] = pid[DIM - 2];
      DataType<DT>::from_casacore(values[*pid], col_matrix(ip));
    }
    break;
  }
  case 3: {
    casacore::IPosition ip(3);
    casacore::Cube<CT> col_cube;
    col_cube.reference(col_array);
    for (PointInDomainIterator<DIM> pid(column.domain(), false); pid(); pid++) {
      if (row_number != pid[0]) {
        row_number = pid[0];
        col.get(row_number, col_array, true);
        col_cube.reference(col_array);
      }
      ip[0] = pid[DIM - 1];
      ip[1] = pid[DIM - 2];
      ip[2] = pid[DIM - 3];
      DataType<DT>::from_casacore(values[*pid], col_cube(ip));
    }
    break;
  }
  default: {
    casacore::IPosition ip(array_cell_rank);
    for (PointInDomainIterator<DIM> pid(column.domain(), false); pid(); pid++) {
      if (row_number != pid[0]) {
        row_number = pid[0];
        col.get(row_number, col_array, true);
      }
      for (unsigned i = 0; i < array_cell_rank; ++i)
        ip[i] = pid[DIM - i - 1];
      DataType<DT>::from_casacore(values[*pid], col_array(ip));
    }
    break;
  }
  }
}

template <int DIM>
static void
read_column(
  const casacore::Table& table,
  const casacore::ColumnDesc& col_desc,
  const PhysicalColumn& column) {

  switch (column.dt()) {
#define READ_COL(DT)                                          \
    case DT:                                                  \
      switch (col_desc.trueDataType()) {                      \
      case DataType<DT>::CasacoreTypeTag: {                   \
        PhysicalColumnTD<DT, 1, DIM, AffineAccessor>          \
          col_td(column);                                     \
        read_scalar_column<DT, DIM>(table, col_desc, col_td); \
        break;                                                \
      }                                                       \
      case DataType<DT>::CasacoreArrayTypeTag: {              \
        PhysicalColumnTD<DT, 1, DIM, AffineAccessor>          \
          col_td(column);                                     \
        read_array_column<DT, DIM>(table, col_desc, col_td);  \
        break;                                                \
      }                                                       \
      default:                                                \
        assert(false);                                        \
      }                                                       \
      break;
    HYPERION_FOREACH_CC_DATATYPE(READ_COL);
#undef READ_COL
    default:
      assert(false);
      break;
  }
}

TaskID TableReadTask::TASK_ID;

void
TableReadTask::preregister_tasks() {
  {
    TASK_ID = Runtime::generate_static_task_id();
    TaskVariantRegistrar registrar(TASK_ID, TASK_NAME);
    registrar.add_constraint(ProcessorConstraint(Processor::IO_PROC));
    registrar.set_leaf();
    registrar.add_layout_constraint_set(
      TableMapper::to_mapping_tag(
        TableMapper::default_column_layout_tag),
      soa_row_major_layout);
    Runtime::preregister_task_variant<impl>(registrar, TASK_NAME);
  }
}

void
TableReadTask::impl(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *rt) {

  const Args* args = static_cast<const Args*>(task->args);

  auto ptcr =
    PhysicalTable::create(
      rt,
      args->table_desc,
      task->regions.begin(),
      task->regions.end(),
      regions.begin() ,
      regions.end()).value();
#if HAVE_CXX17
  auto& [table, rit, pit] = ptcr;
#else // !HAVE_CXX17
  auto& table = std::get<0>(ptcr);
  auto& rit = std::get<1>(ptcr);
  auto& pit = std::get<2>(ptcr);
#endif // HAVE_CXX17
  assert(rit == task->regions.end());
  assert(pit == regions.end());

  casacore::Table cctable(
    args->table_path,
    casacore::TableLock::PermanentLockingWait);
  auto tdesc = cctable.tableDesc();

  for (auto& nm_column : table.columns()) {
#if HAVE_CXX17
    auto& [nm, column] = nm_column;
#else // !HAVE_CXX17
    auto& nm = std::get<0>(nm_column);
    auto& column = std::get<1>(nm_column);
#endif // HAVE_CXX17
    if (nm != "") {
      switch (column->domain().get_dim()) {
      case 0: // for index column space
        break;
#define READ_COLUMN(D)                                  \
        case D:                                         \
          read_column<D>(cctable, tdesc[nm], *column);  \
          break;
#if LEGION_MAX_DIM >= 1
        READ_COLUMN(1);
#endif
#if LEGION_MAX_DIM >= 2
        READ_COLUMN(2);
#endif
#if LEGION_MAX_DIM >= 3
        READ_COLUMN(3);
#endif
#if LEGION_MAX_DIM >= 4
        READ_COLUMN(4);
#endif
      default:
        assert(false);
        break;
      }
    }
  }
}

static Column::Requirements
column_reqs(PrivilegeMode privilege) {
  return Column::Requirements{
    Column::Req{privilege, EXCLUSIVE, true},
    Column::default_requirements.keywords,
    Column::default_requirements.measref,
    Column::default_requirements.column_space,
    Column::default_requirements.tag};
};

std::tuple<
  std::vector<RegionRequirement>,
  std::vector<LogicalPartition>,
  Table::Desc>
TableReadTask::requirements(
  Context ctx,
  Runtime* rt,
  const PhysicalTable& table,
  const ColumnSpacePartition& table_partition,
  PrivilegeMode columns_privilege) {

  assert(
    columns_privilege == READ_WRITE
    || columns_privilege == WRITE_DISCARD
    || columns_privilege == WRITE_ONLY);

  return
    table.requirements(
      ctx,
      rt,
      table_partition,
      {},
      column_reqs(columns_privilege));
}

std::tuple<
  std::vector<RegionRequirement>,
  std::vector<LogicalPartition>,
  Table::Desc>
TableReadTask::requirements(
  Context ctx,
  Runtime* rt,
  const Table& table,
  const ColumnSpacePartition& table_partition,
  PrivilegeMode columns_privilege) {

  assert(
    columns_privilege == READ_WRITE
    || columns_privilege == WRITE_DISCARD
    || columns_privilege == WRITE_ONLY);

  return
    table.requirements(
      ctx,
      rt,
      table_partition,
      {},
      column_reqs(columns_privilege));
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
