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
#include <hyperion/TableReadTask.h>

#include <casacore/casa/aipstype.h>
#include <casacore/casa/Arrays.h>

#include <algorithm>

using namespace hyperion;

using namespace Legion;

template <typename T>
static inline void
init_rect(
  const casacore::Array<T> array,
  casacore::IPosition& start,
  casacore::IPosition& end) {

  start = 0;
  end = array.shape();
  auto ndim = array.shape().size();
  for (unsigned i = 0; i < ndim; ++i)
    end[i] -= 1;
}

template <int DIM, hyperion::TypeTag DT>
static void
read_scalar_column(
  const casacore::Table& table,
  const casacore::ColumnDesc& col_desc,
  DomainT<DIM> reg_domain,
  const PhysicalRegion& region,
  FieldID fid) {

  typedef typename DataType<DT>::ValueType T;
  typedef typename DataType<DT>::CasacoreType CT;

// FIXME: use GenericAccessor rather than AffineAccessor, or at least leave it
// as a parameter
  typedef FieldAccessor<
    WRITE_ONLY,
    T,
    DIM,
    coord_t,
    AffineAccessor<T, DIM, coord_t>,
    false> ValueAccessor;

  const ValueAccessor values(region, fid);

  casacore::ScalarColumn<CT> col(table, col_desc.name());
  coord_t row_number;
  CT col_value;
  {
    PointInDomainIterator<DIM> pid(reg_domain, false);
    row_number = pid[0];
    col.get(row_number, col_value);
  }

  for (PointInDomainIterator<DIM> pid(reg_domain, false);
       pid();
       pid++) {
    if (row_number != pid[0]) {
      row_number = pid[0];
      col.get(row_number, col_value);
    }
    DataType<DT>::from_casacore(values[*pid], col_value);
  }
}

template <int DIM, hyperion::TypeTag DT>
static void
read_array_column(
  const casacore::Table& table,
  const casacore::ColumnDesc& col_desc,
  DomainT<DIM> reg_domain,
  const PhysicalRegion& region,
  FieldID fid) {

  typedef typename DataType<DT>::ValueType T;
  typedef typename DataType<DT>::CasacoreType CT;

// FIXME: use GenericAccessor rather than AffineAccessor, or at least leave it
// as a parameter
  typedef FieldAccessor<
    WRITE_ONLY,
    T,
    DIM,
    coord_t,
    AffineAccessor<T, DIM, coord_t>,
    false> ValueAccessor;

  const ValueAccessor values(region, fid);

  casacore::ArrayColumn<CT> col(table, col_desc.name());
  coord_t row_number;
  unsigned array_cell_rank;
  {
    PointInDomainIterator<DIM> pid(reg_domain, false);
    row_number = pid[0];
    array_cell_rank = col.ndim(row_number);
  }

  casacore::Array<CT> col_array;
  col.get(row_number, col_array, true);
  switch (array_cell_rank) {
  case 1: {
    casacore::Vector<CT> col_vector;
    col_vector.reference(col_array);
    for (PointInDomainIterator<DIM> pid(reg_domain, false);
         pid();
         pid++) {
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
    for (PointInDomainIterator<DIM> pid(reg_domain, false);
         pid();
         pid++) {
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
    for (PointInDomainIterator<DIM> pid(reg_domain, false);
         pid();
         pid++) {
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
    for (PointInDomainIterator<DIM> pid(reg_domain, false);
         pid();
         pid++) {
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
  hyperion::TypeTag dt,
  DomainT<DIM> reg_domain,
  const PhysicalRegion& region,
  FieldID fid) {

  switch (dt) {
#define READ_COL(DT)                                                    \
    case DT:                                                            \
      switch (col_desc.trueDataType()) {                                \
      case DataType<DT>::CasacoreTypeTag:                               \
        read_scalar_column<DIM, DT>(table, col_desc, reg_domain, region, fid); \
        break;                                                          \
      case DataType<DT>::CasacoreArrayTypeTag:                          \
        read_array_column<DIM, DT>(table, col_desc, reg_domain, region, fid); \
        break;                                                          \
      default:                                                          \
        assert(false);                                                  \
      }                                                                 \
      break;
    HYPERION_FOREACH_DATATYPE(READ_COL);
#undef READ_COL
    default:
      assert(false);
      break;
  }
}

TaskID TableReadTask::TASK_ID = 0;
const char* TableReadTask::TASK_NAME = "TableReadTask";

void
TableReadTask::preregister_task() {
  TASK_ID = Runtime::generate_static_task_id();
  TaskVariantRegistrar registrar(TASK_ID, TASK_NAME, false);
  registrar.add_constraint(ProcessorConstraint(Processor::IO_PROC));
  registrar.set_leaf();
  Runtime::preregister_task_variant<base_impl>(registrar, TASK_NAME);
}

struct TableReadTaskArgs {
  char table_path[1024];
  FieldID fid;
  hyperion::TypeTag dt;
  hyperion::string nm;
};

void
TableReadTask::dispatch(Context ctx, Runtime* rt) {

  bool is_empty =
    Table::is_empty(
      m_table
      .index_column_space(ctx, rt)
      .get_result<Table::index_column_space_result_t>());
  if (!is_empty) {
    auto columns =
      m_table.columns(ctx, rt).get_result<Table::columns_result_t>();
    // N.B: MS table columns always have a ROW index, and tables have no index
    // columns
    auto i0dom =
      rt->get_index_space_domain(std::get<0>(columns.fields.front()).column_is);
    auto i0sz = i0dom.hi()[0] - i0dom.lo()[0] + 1;
    size_t num_subregions =
      std::max(
        rt->select_tunable_value(
          ctx,
          Mapping::DefaultMapper::DefaultTunables::DEFAULT_TUNABLE_GLOBAL_IOS)
        .get_result<size_t>(),
        1ul);
    num_subregions = min_divisor(i0sz, m_min_block_length, num_subregions);
    auto block_length = (i0sz + num_subregions - 1) / num_subregions;
    auto csps =
      m_table
      .partition_rows(ctx, rt, {std::make_optional(block_length)})
      .get_result<Table::partition_rows_result_t>();

    TableReadTaskArgs args;
    assert(m_table_path.size() < sizeof(args.table_path));
    std::strcpy(args.table_path, m_table_path.c_str());
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
    for (auto& [cs, ixfl, vlr, nm_tflds] : columns.fields)  {
#pragma GCC diagnostic pop
      auto csp = csps.find(cs).value(); // optional should not be empty; hard
                                        // fail o.w. is intentional
      auto lp = rt->get_logical_partition(ctx, vlr, csp.column_ip);
      auto launcher =
        IndexTaskLauncher(
          TASK_ID,
          rt->get_index_partition_color_space(csp.column_ip),
          TaskArgument(&args, sizeof(args)),
          ArgumentMap());
      for (auto& [nm, tfld] : nm_tflds) {
        launcher.region_requirements.clear();
        RegionRequirement req(lp, 0, WRITE_ONLY, EXCLUSIVE, vlr);
        req.add_field(tfld.fid);
        launcher.add_region_requirement(req);
        args.fid = tfld.fid;
        args.dt = tfld.dt;
        args.nm = nm;
        rt->execute_index_space(ctx, launcher);
      }
      rt->destroy_logical_partition(ctx, lp);
    }
    for (auto& csp : csps.partitions)
      csp.destroy(ctx, rt);
  }
}

void
TableReadTask::base_impl(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *rt) {

  const TableReadTaskArgs* args =
    static_cast<const TableReadTaskArgs*>(task->args);
  casacore::Table table(
    args->table_path,
    casacore::TableLock::PermanentLockingWait);
  auto tdesc = table.tableDesc();
  auto cdesc = tdesc[std::string(args->nm)];
  switch (regions[0].get_logical_region().get_index_space().get_dim()) {
#if LEGION_MAX_DIM >= 1
  case 1:
    read_column<1>(
      table,
      cdesc,
      args->dt,
      rt->get_index_space_domain(task->regions[0].region.get_index_space()),
      regions[0],
      args->fid);
    break;
#endif
#if LEGION_MAX_DIM >= 2
  case 2:
    read_column<2>(
      table,
      cdesc,
      args->dt,
      rt->get_index_space_domain(task->regions[0].region.get_index_space()),
      regions[0],
      args->fid);
    break;
#endif
#if LEGION_MAX_DIM >= 3
  case 3:
    read_column<3>(
      table,
      cdesc,
      args->dt,
      rt->get_index_space_domain(task->regions[0].region.get_index_space()),
      regions[0],
      args->fid);
    break;
#endif
#if LEGION_MAX_DIM >= 4
  case 4:
    read_column<4>(
      table,
      cdesc,
      args->dt,
      rt->get_index_space_domain(task->regions[0].region.get_index_space()),
      regions[0],
      args->fid);
    break;
#endif
  default:
    assert(false);
    break;
  }
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
