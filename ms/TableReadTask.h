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
#include "utility.h"
#include "Column.h"

namespace legms {
namespace ms {

struct TableReadTaskArgs {
  char table_path[80];
  char table_name[80];
  char column_name[20];
  unsigned column_rank;
  casacore::DataType column_datatype;
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
    const Table* table,
    Iter colname_iter,
    Iter end_colname_iter,
    std::optional<size_t> block_length = std::nullopt)
    : m_context(table->context())
    , m_runtime(table->runtime())
    , m_table_path(table_path)
    , m_table_name(table->name()) {

    std::transform(
      colname_iter,
      end_colname_iter,
      std::back_inserter(m_columns),
      [table](const auto& nm) { return table->column(nm); });

    {
      auto nr = table->num_rows();
      auto bl = block_length.value_or(nr);
      std::vector<std::vector<Column::row_number_t>> rowp((nr + bl - 1) / bl);
      for (Column::row_number_t i = 0; i < nr; ++i)
        rowp[i / bl].push_back(i);
      m_blockp = table->row_partition(rowp, false, true);
    }

    casacore::Table tb(
      casacore::String(table_path),
      casacore::TableLock::PermanentLockingWait);
  }

  static void
  register_task(Legion::Runtime* runtime);

  void
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
    casacore::DataType lr_datatype,
    Legion::DomainT<DIM> reg_domain,
    const std::vector<Legion::PhysicalRegion>& regions) {

#define READ_COL(dt)                                                    \
    casacore::DataType::Tp##dt:                                         \
      switch (col_desc.trueDataType()) {                                \
      case casacore::DataType::Tp##dt:                                  \
        read_scalar_column<DIM, casacore::DataType::Tp##dt>(            \
          table, col_desc, reg_domain, regions);                        \
        break;                                                          \
      case casacore::DataType::TpArray##dt:                             \
        read_array_column<DIM, casacore::DataType::Tp##dt>(             \
          table, col_desc, reg_domain, regions);                        \
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
        ip[0] = pid[DIM - 1];
        ip[1] = pid[DIM - 2];
        field_init(values.ptr(*pid));
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
        ip[0] = pid[DIM - 1];
        ip[1] = pid[DIM - 2];
        ip[2] = pid[DIM - 3];
        field_init(values.ptr(*pid));
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
          ip[i] = pid[DIM - i - 1];
        field_init(values.ptr(*pid));
        values[*pid] = col_array(ip);
      }
      break;
    }
    }
  }

private:

  Legion::Context m_context;

  Legion::Runtime *m_runtime;

  std::string m_table_path;

  std::string m_table_name;

  std::vector<std::shared_ptr<Column>> m_columns;

  std::unique_ptr<ColumnPartition> m_blockp;

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

} // end namespace ms
} // end namespace legms

#endif // LEGMS_MS_TABLE_READ_TASK_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
