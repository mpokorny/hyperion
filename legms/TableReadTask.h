#ifndef LEGMS_TABLE_READ_TASK_H_
#define LEGMS_TABLE_READ_TASK_H_
#pragma GCC visibility push(default)
#include <array>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <unordered_map>
#pragma GCC visibility pop

#include "legms.h"

#ifdef LEGMS_USE_CASACORE

#include "Table.h"
#include "utility.h"
#include "Column.h"

#pragma GCC visibility push(default)
#include <casacore/casa/aipstype.h>
#include <casacore/casa/Arrays.h>
#include <casacore/tables/Tables.h>
#pragma GCC visibility pop

namespace legms {

struct LEGMS_API TableReadTaskArgs {
  char table_path[1024];
};

class LEGMS_API TableReadTask {
public:

  static Legion::TaskID TASK_ID;
  static const char* TASK_NAME;

  /*
   * TableReadTask constructor
   */
  template <typename Iter>
  TableReadTask(
    const std::string& table_path,
    const Table& table,
    Iter colname_iter,
    Iter end_colname_iter,
    size_t block_length)
    : m_table_path(table_path)
    , m_table(table)
    , m_block_length(block_length) {

    std::copy(colname_iter, end_colname_iter, std::back_inserter(m_colnames));

    // FIXME: the following is insufficient in the case of multiple nodes
    // FIXME: is this necessary?
    casacore::Table tb(
      casacore::String(table_path),
      casacore::TableLock::PermanentLockingWait);
  }

  static void
  preregister_task();

  void
  dispatch(Legion::Context ctx, Legion::Runtime* rt);

  static void
  base_impl(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *runtime);

  template <int DIM>
  static void
  read_column(
    const casacore::Table& table,
    const casacore::ColumnDesc& col_desc,
    TypeTag lr_datatype,
    Legion::DomainT<DIM> reg_domain,
    const Legion::PhysicalRegion& region) {

#define READ_COL(DT)                                                    \
    case DT:                                                            \
      switch (col_desc.trueDataType()) {                                \
      case DataType<DT>::CasacoreTypeTag:                               \
        read_scalar_column<DIM, DT>(table, col_desc, reg_domain, region); \
        break;                                                          \
      case DataType<DT>::CasacoreArrayTypeTag:                          \
        read_array_column<DIM, DT>(table, col_desc, reg_domain, region); \
        break;                                                          \
      default:                                                          \
        assert(false);                                                  \
      }                                                                 \
      break;

    switch (lr_datatype) {
      LEGMS_FOREACH_DATATYPE(READ_COL);
    default:
      assert(false);
    }
#undef READ_COL
  }

  template <int DIM, TypeTag DT>
  static void
  read_scalar_column(
    const casacore::Table& table,
    const casacore::ColumnDesc& col_desc,
    Legion::DomainT<DIM> reg_domain,
    const Legion::PhysicalRegion& region) {

    typedef typename DataType<DT>::ValueType T;
    typedef typename DataType<DT>::CasacoreType CT;

    typedef Legion::FieldAccessor<
      WRITE_ONLY,
      T,
      DIM,
      Legion::coord_t,
      Legion::AffineAccessor<T, DIM, Legion::coord_t>,
      false> ValueAccessor;

    const ValueAccessor values(region, Column::VALUE_FID);

    casacore::ScalarColumn<CT> col(table, col_desc.name());
    Legion::coord_t row_number;
    CT col_value;
    {
      Legion::PointInDomainIterator<DIM> pid(reg_domain, false);
      row_number = pid[0];
      col.get(row_number, col_value);
    }

    for (Legion::PointInDomainIterator<DIM> pid(reg_domain, false);
         pid();
         pid++) {
      if (row_number != pid[0]) {
        row_number = pid[0];
        col.get(row_number, col_value);
      }
      DataType<DT>::from_casacore(values[*pid], col_value);
    }
  }

  template <int DIM, TypeTag DT>
  static void
  read_array_column(
    const casacore::Table& table,
    const casacore::ColumnDesc& col_desc,
    Legion::DomainT<DIM> reg_domain,
    const Legion::PhysicalRegion& region) {

    typedef typename DataType<DT>::ValueType T;
    typedef typename DataType<DT>::CasacoreType CT;

    typedef Legion::FieldAccessor<
      WRITE_ONLY,
      T,
      DIM,
      Legion::coord_t,
      Legion::AffineAccessor<T, DIM, Legion::coord_t>,
      false> ValueAccessor;

    const ValueAccessor values(region, Column::VALUE_FID);

    casacore::ArrayColumn<CT> col(table, col_desc.name());
    Legion::coord_t row_number;
    unsigned array_cell_rank;
    {
      Legion::PointInDomainIterator<DIM> pid(reg_domain, false);
      row_number = pid[0];
      array_cell_rank = col.ndim(row_number);
    }

    casacore::Array<CT> col_array;
    col.get(row_number, col_array, true);
    switch (array_cell_rank) {
    case 1: {
      casacore::Vector<CT> col_vector;
      col_vector.reference(col_array);
      for (Legion::PointInDomainIterator<DIM> pid(reg_domain, false);
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
      for (Legion::PointInDomainIterator<DIM> pid(reg_domain, false);
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
      for (Legion::PointInDomainIterator<DIM> pid(reg_domain, false);
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
      for (Legion::PointInDomainIterator<DIM> pid(reg_domain, false);
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

private:

  std::string m_table_path;

  Table m_table;

  std::vector<std::string> m_colnames;

  size_t m_block_length;

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
};

} // end namespace legms

#endif // LEGMS_USE_CASACORE
#endif // LEGMS_TABLE_READ_TASK_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
