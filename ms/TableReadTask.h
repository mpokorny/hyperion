#ifndef LEGMS_MS_TABLE_READ_TASK_H_
#define LEGMS_MS_TABLE_READ_TASK_H_
#include <array>
#include <cstring>
#include <memory>
#include <new>
#include <optional>
#include <type_traits>
#include <unordered_map>

#include <casacore/casa/aipstype.h>
#include <casacore/casa/Arrays.h>
#include <casacore/tables/Tables.h>
#include "legion.h"
#include "Table.h"
#include "TableBuilder.h"
#include "utility.h"
#include "Column.h"
#include "ColumnHint.h"

namespace legms {
namespace ms {

struct TableReadTaskArgs {
  char table_path[80];
  char table_name[80];
  char column_name[20];
  unsigned column_rank;
  casacore::DataType column_datatype;
  ColumnHint column_hint;
  unsigned char ser_row_index_pattern[];
};

class TableReadTask {
public:

  static Legion::TaskID TASK_ID;
  static constexpr const char* TASK_NAME = "table_read_task";

  /*
   * TableReadTask constructor
   */
  template <typename Iter>
  TableReadTask(
    const std::string& table_path,
    const std::shared_ptr<const Table>& table,
    Iter colname_iter,
    Iter end_colname_iter,
    const std::unordered_map<std::string, ColumnHint>& column_hints,
    std::optional<size_t> block_length = std::nullopt,
    std::optional<Legion::IndexPartition> ipart = std::nullopt)
    : m_table_path(table_path)
    , m_table(table)
    , m_column_names(colname_iter, end_colname_iter)
    , m_block_length(block_length)
    , m_index_partition(ipart) {

    casacore::Table tb(
      casacore::String(table_path),
      casacore::TableLock::PermanentLockingWait);
    std::transform(
      colname_iter,
      end_colname_iter,
      std::inserter(m_column_hints, m_column_hints.end()),
      [&column_hints, tdesc=tb.tableDesc()](auto& nm) {
        return std::make_pair(
          nm,
          (column_hints.count(nm) > 0)
          ? column_hints.at(nm)
          : TableBuilder::inferred_column_hint(tdesc[nm]));
      });
  }

  static void
  register_task(Legion::Runtime* runtime);

  std::vector<Legion::LogicalRegion>
  dispatch();

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
    const ColumnHint& col_hint,
    const IndexTreeL& row_index_pattern,
    casacore::DataType lr_datatype,
    Legion::DomainT<DIM> reg_domain,
    const std::vector<Legion::PhysicalRegion>& regions) {

#define READ_COL(dt)                                                    \
    casacore::DataType::Tp##dt:                                         \
      switch (col_desc.trueDataType()) {                                \
      case casacore::DataType::Tp##dt:                                  \
        read_scalar_column<DIM, casacore::DataType::Tp##dt>(            \
          table, col_desc, row_index_pattern, reg_domain, regions);     \
        break;                                                          \
      case casacore::DataType::TpArray##dt:                             \
        read_array_column<DIM, casacore::DataType::Tp##dt>(             \
          table, col_desc, col_hint, row_index_pattern, reg_domain, regions); \
        break;                                                          \
      default:                                                          \
        assert(false);                                                  \
      }                                                                 \
    break;                                                              \
    case casacore::DataType::TpArray##dt:                               \
      switch (col_desc.trueDataType()) {                                \
      case casacore::DataType::TpArray##dt:                             \
        read_vector_column<DIM, casacore::DataType::Tp##dt>(            \
          table, col_desc, col_hint, row_index_pattern, reg_domain, regions); \
        break;                                                          \
      default:                                                          \
        assert(false);                                                  \
      }

    switch (lr_datatype) {
    case READ_COL(Bool)
      break;
    case READ_COL(Char)
      break;
    case READ_COL(UChar)
      break;
    case READ_COL(Short)
      break;
    case READ_COL(UShort)
      break;
    case READ_COL(Int)
      break;
    case READ_COL(UInt)
      break;
    // case READ_COL(Int64, casacore::Int64)
    //   break;
    case READ_COL(Float)
      break;
    case READ_COL(Double)
      break;
    case READ_COL(Complex)
      break;
    case READ_COL(DComplex)
      break;
    case READ_COL(String)
      break;
    default:
      assert(false);
    }
#undef READ_COL
  }

  template <int DIM, casacore::DataType DT>
  static void
  read_scalar_column(
    const casacore::Table& table,
    const casacore::ColumnDesc& col_desc,
    const IndexTreeL& row_index_pattern,
    Legion::DomainT<DIM> reg_domain,
    const std::vector<Legion::PhysicalRegion>& regions) {

    typedef typename DataType<DT>::ValueType T;

    typedef Legion::FieldAccessor<
      WRITE_DISCARD,
      T,
      DIM,
      Legion::coord_t,
      Legion::AffineAccessor<T, DIM, Legion::coord_t>,
      false> ValueAccessor;

    typedef Legion::FieldAccessor<
      READ_ONLY,
      Column::row_number_t,
      DIM,
      Legion::coord_t,
      Legion::AffineAccessor<Column::row_number_t, DIM, Legion::coord_t>,
      false> RowNumberAccessor;

    const ValueAccessor values(regions[0], Column::value_fid);
    const RowNumberAccessor row_numbers(regions[1], Column::row_number_fid);

    casacore::ScalarColumn<T> col(table, col_desc.name());
    Column::row_number_t row_number;
    T col_value;
    {
      Legion::PointInDomainIterator<DIM> pid(reg_domain, false);
      row_number = row_numbers[*pid];
      col.get(row_number, col_value);
    }

    for (Legion::PointInDomainIterator<DIM> pid(reg_domain, false);
         pid();
         pid++) {
      auto rn = row_numbers[*pid];
      if (row_number != rn) {
        row_number = rn;
        col.get(row_number, col_value);
      }
      // field_init() is a necessary because the memory in the allocated region
      // is uninitialized, which can be a problem with non-trivially copyable
      // types (such as strings)
      field_init(values.ptr(*pid));
      values[*pid] = col_value;
    }
  }

  template <int DIM, casacore::DataType DT>
  static void
  read_array_column(
    const casacore::Table& table,
    const casacore::ColumnDesc& col_desc,
    const ColumnHint& col_hint,
    const IndexTreeL& row_index_pattern,
    Legion::DomainT<DIM> reg_domain,
    const std::vector<Legion::PhysicalRegion>& regions) {

    typedef typename DataType<DT>::ValueType T;

    typedef Legion::FieldAccessor<
      WRITE_DISCARD,
      T,
      DIM,
      Legion::coord_t,
      Legion::AffineAccessor<T, DIM, Legion::coord_t>,
      false> ValueAccessor;

    typedef Legion::FieldAccessor<
      READ_ONLY,
      Column::row_number_t,
      DIM,
      Legion::coord_t,
      Legion::AffineAccessor<Column::row_number_t, DIM, Legion::coord_t>,
      false> RowNumberAccessor;

    const ValueAccessor values(regions[0], Column::value_fid);
    const RowNumberAccessor row_numbers(regions[1], Column::row_number_fid);

    casacore::ArrayColumn<T> col(table, col_desc.name());
    Column::row_number_t row_number;
    unsigned array_cell_rank;
    {
      Legion::PointInDomainIterator<DIM> pid(reg_domain, false);
      row_number = row_numbers[*pid];
      array_cell_rank = col.ndim(row_number);
    }

    casacore::Array<T> col_array;
    col.get(row_number, col_array, true);
    switch (array_cell_rank) {
    case 1: {
      casacore::Vector<T> col_vector;
      col_vector.reference(col_array);
      for (Legion::PointInDomainIterator<DIM> pid(reg_domain, false);
           pid();
           pid++) {
        auto rn = row_numbers[*pid];
        if (row_number != rn) {
          row_number = rn;
          col.get(row_number, col_array, true);
          col_vector.reference(col_array);
        }
        field_init(values.ptr(*pid));
        values[*pid] = col_vector[pid[DIM - 1]];
      }
      break;
    }
    case 2: {
      casacore::IPosition ip(2);
      casacore::Matrix<T> col_matrix;
      col_matrix.reference(col_array);
      for (Legion::PointInDomainIterator<DIM> pid(reg_domain, false);
           pid();
           pid++) {
        auto rn = row_numbers[*pid];
        if (row_number != rn) {
          row_number = rn;
          col.get(row_number, col_array, true);
          col_matrix.reference(col_array);
        }
        field_init(values.ptr(*pid));
        pt2ipos<2>(ip, col_hint.index_permutations, *pid);
        values[*pid] = col_matrix(ip);
      }
      break;
    }
    case 3: {
      casacore::IPosition ip(3);
      casacore::Cube<T> col_cube;
      col_cube.reference(col_array);
      for (Legion::PointInDomainIterator<DIM> pid(reg_domain, false);
           pid();
           pid++) {
        auto rn = row_numbers[*pid];
        if (row_number != rn) {
          row_number = rn;
          col.get(row_number, col_array, true);
          col_cube.reference(col_array);
        }
        field_init(values.ptr(*pid));
        pt2ipos<3>(ip, col_hint.index_permutations, *pid);
        values[*pid] = col_cube(ip);
      }
      break;
    }
    default: {
      casacore::IPosition ip(array_cell_rank);
      for (Legion::PointInDomainIterator<DIM> pid(reg_domain, false);
           pid();
           pid++) {
        auto rn = row_numbers[*pid];
        if (row_number != rn) {
          row_number = rn;
          col.get(row_number, col_array, true);
        }
        for (unsigned i = 0; i < array_cell_rank; ++i)
          ip[i] = pid[DIM - array_cell_rank + i];
        field_init(values.ptr(*pid));
        pt2ipos(ip, col_hint.index_permutations.data(), *pid);
        values[*pid] = col_array(ip);
      }
      break;
    }
    }
  }

  template <int DIM, casacore::DataType DT>
  static void
  read_vector_column(
    const casacore::Table& table,
    const casacore::ColumnDesc& col_desc,
    const ColumnHint& col_hint,
    const IndexTreeL& row_index_pattern,
    Legion::DomainT<DIM> reg_domain,
    const std::vector<Legion::PhysicalRegion>& regions) {

    typedef typename DataType<DT>::ValueType T;

    typedef Legion::FieldAccessor<
      WRITE_DISCARD,
      std::vector<T>,
      DIM,
      Legion::coord_t,
      Legion::AffineAccessor<std::vector<T>, DIM, Legion::coord_t>,
      false> ValueAccessor;

    typedef Legion::FieldAccessor<
      READ_ONLY,
      Column::row_number_t,
      DIM,
      Legion::coord_t,
      Legion::AffineAccessor<Column::row_number_t, DIM, Legion::coord_t>,
      false> RowNumberAccessor;

    const ValueAccessor values(regions[0], Column::value_fid);
    const RowNumberAccessor row_numbers(regions[1], Column::row_number_fid);

    casacore::ArrayColumn<T> col(table, col_desc.name());
    Column::row_number_t row_number;
    unsigned array_cell_rank;
    {
      Legion::PointInDomainIterator<DIM> pid(reg_domain, false);
      row_number = row_numbers[*pid];
      array_cell_rank = col.ndim(row_number);
    }
    assert(array_cell_rank == 1);

    casacore::Array<T> col_array;
    col.get(row_number, col_array, true);
    casacore::Vector<T> col_vector;
    col_vector.reference(col_array);
    for (Legion::PointInDomainIterator<DIM> pid(reg_domain, false);
         pid();
         pid++) {
      auto rn = row_numbers[*pid];
      if (row_number != rn) {
        row_number = rn;
        col.get(row_number, col_array, true);
        col_vector.reference(col_array);
      }
      std::vector<T> cv;
      col_vector.tovector(cv);
      field_init(values.ptr(*pid));
      values[*pid] = cv;
    }
  }

private:

  std::string m_table_path;

  const std::shared_ptr<const Table> m_table;

  std::vector<std::string> m_column_names;

  std::optional<size_t> m_block_length;

  std::optional<Legion::IndexPartition> m_index_partition;

  std::unordered_map<std::string, ColumnHint> m_column_hints;

  template <typename T>
  static inline
  typename std::enable_if_t<std::is_trivially_copyable_v<T>>
  field_init(T*) {}

  template <typename T>
  static inline
  typename std::enable_if_t<!std::is_trivially_copyable_v<T>>
  field_init(T* fld) {
    ::new (fld) T;
  }
};

} // end namespace ms
} // end namespace legms

#endif // LEGMS_MS_TABLE_READ_TASK_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
