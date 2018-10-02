#ifndef LEGMS_MS_TABLE_READ_TASK_H_
#define LEGMS_MS_TABLE_READ_TASK_H_
#include <array>
#include <cstring>
#include <memory>
#include <new>
#include <optional>

#include <casacore/casa/aipstype.h>
#include <casacore/casa/Arrays.h>
#include <casacore/tables/Tables.h>
#include "legion.h"
#include "Table.h"
#include "utility.h"

namespace legms {
namespace ms {

struct TableReadTaskArgs {
  char table_path[80];
  char table_name[80];
  char column_name[20];
  unsigned column_rank;
  casacore::DataType column_datatype;
  unsigned char ser_row_index_shape[];
};

class TableReadTask {
public:

  static Legion::TaskID TASK_ID;
  static constexpr const char* TASK_NAME = "table_read_task";

  template <typename Iter, typename EndIter>
  TableReadTask(
    const std::string& table_path,
    const std::shared_ptr<Table>& table,
    Iter colname_iter,
    EndIter end_colname_iter,
    size_t block_length,
    std::optional<Legion::IndexPartition> ipart = std::nullopt)
    : m_table_path(table_path)
    , m_table(table)
    , m_column_names(colname_iter, end_colname_iter)
    , m_block_length(block_length)
    , m_index_partition(ipart) {
  }

  static void
  register_task(Legion::Runtime* runtime);

  std::vector<std::tuple<Legion::LogicalRegion, Legion::FieldID>>
  dispatch(Legion::Context ctx, Legion::Runtime* runtime);

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
    const IndexTreeL& row_index_shape,
    casacore::DataType lr_datatype,
    Legion::DomainT<DIM> reg_domain,
    const Legion::PhysicalRegion& region) {

#define READ_COL(dt)                                                    \
    casacore::DataType::Tp##dt:                                         \
      switch (col_desc.trueDataType()) {                                \
      case casacore::DataType::Tp##dt:                                  \
        read_scalar_column<DIM, DataType<casacore::DataType::Tp##dt>::  \
                           ValueType>(                                  \
          table, col_desc, row_index_shape, reg_domain, region); \
        break;                                                          \
      case casacore::DataType::TpArray##dt:                             \
        read_array_column<DIM, DataType<casacore::DataType::Tp##dt>::   \
                          ValueType>(                                   \
          table, col_desc, row_index_shape, reg_domain, region); \
        break;                                                          \
      default:                                                          \
        assert(false);                                                  \
      }                                                                 \
    break;                                                              \
    case casacore::DataType::TpArray##dt:                               \
      switch (col_desc.trueDataType()) {                                \
      case casacore::DataType::TpArray##dt:                             \
        read_vector_column<DIM, DataType<casacore::DataType::Tp##dt>::  \
                           ValueType>(                                  \
          table, col_desc, row_index_shape, reg_domain, region); \
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

  template <int DIM, typename T>
  static void
  read_scalar_column(
    const casacore::Table& table,
    const casacore::ColumnDesc& col_desc,
    const IndexTreeL& row_index_shape,
    Legion::DomainT<DIM> reg_domain,
    const Legion::PhysicalRegion& region) {

    typedef Legion::FieldAccessor<
      WRITE_DISCARD,
      T,
      DIM,
      Legion::coord_t,
      Legion::AffineAccessor<T, DIM, Legion::coord_t>,
      false> WDAccessor;

    std::vector<Legion::FieldID> fids;
    region.get_fields(fids);
    assert(fids.size() == 1);
    const WDAccessor values(region, fids[0]);

    casacore::ScalarColumn<T> col(table, col_desc.name());
    std::array<Legion::coord_t, DIM> pt;
    size_t row_number;
    T col_value;
    {
      Legion::PointInDomainIterator<DIM> pid(reg_domain, false);
      for (size_t i = 0; i < DIM; ++i)
        pt[i] = pid[i];
      row_number = Table::row_number(row_index_shape, pt.begin(), pt.end());
      col.get(row_number, col_value);
    }

    for (Legion::PointInDomainIterator<DIM> pid(reg_domain, false);
         pid();
         pid++) {
      for (size_t i = 0; i < DIM; ++i)
        pt[i] = pid[i];
      auto rn = Table::row_number(row_index_shape, pt.begin(), pt.end());
      if (row_number != rn) {
        row_number = rn;
        col.get(row_number, col_value);
      }
      // the call to field_init() is a workaround, done to avoid a segfault that
      // occurs when a zero-length string is assigned to a field; TODO: is there
      // a better solution?
      field_init(values[*pid], col_value);
      values[*pid] = col_value;
    }
  }

  template <int DIM, typename T>
  static void
  read_array_column(
    const casacore::Table& table,
    const casacore::ColumnDesc& col_desc,
    const IndexTreeL& row_index_shape,
    Legion::DomainT<DIM> reg_domain,
    const Legion::PhysicalRegion& region) {

    typedef Legion::FieldAccessor<
      WRITE_DISCARD,
      T,
      DIM,
      Legion::coord_t,
      Legion::AffineAccessor<T, DIM, Legion::coord_t>,
      false> WDAccessor;

    std::vector<Legion::FieldID> fids;
    region.get_fields(fids);
    assert(fids.size() == 1);
    const WDAccessor values(region, fids[0]);

    casacore::ArrayColumn<T> col(table, col_desc.name());
    std::array<Legion::coord_t, DIM> pt;
    size_t row_number;
    unsigned array_cell_rank;
    {
      Legion::PointInDomainIterator<DIM> pid(reg_domain, false);
      for (size_t i = 0; i < DIM; ++i)
        pt[i] = pid[i];
      row_number = Table::row_number(row_index_shape, pt.begin(), pt.end());
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
        for (size_t i = 0; i < DIM; ++i)
          pt[i] = pid[i];
        auto rn = Table::row_number(row_index_shape, pt.begin(), pt.end());
        if (row_number != rn) {
          row_number = rn;
          col.get(row_number, col_array, true);
          col_vector.reference(col_array);
        }
        values[*pid] = col_vector[pid[DIM - 1]];
      }
      break;
    }
    case 2: {
      casacore::Matrix<T> col_matrix;
      col_matrix.reference(col_array);
      for (Legion::PointInDomainIterator<DIM> pid(reg_domain, false);
           pid();
           pid++) {
        for (size_t i = 0; i < DIM; ++i)
          pt[i] = pid[i];
        auto rn = Table::row_number(row_index_shape, pt.begin(), pt.end());
        if (row_number != rn) {
          row_number = rn;
          col.get(row_number, col_array, true);
          col_matrix.reference(col_array);
        }
        values[*pid] = col_matrix(pid[DIM - 2], pid[DIM - 1]);
      }
      break;
    }
    case 3: {
      casacore::Cube<T> col_cube;
      col_cube.reference(col_array);
      for (Legion::PointInDomainIterator<DIM> pid(reg_domain, false);
           pid();
           pid++) {
        for (size_t i = 0; i < DIM; ++i)
          pt[i] = pid[i];
        auto rn = Table::row_number(row_index_shape, pt.begin(), pt.end());
        if (row_number != rn) {
          row_number = rn;
          col.get(row_number, col_array, true);
          col_cube.reference(col_array);
        }
        values[*pid] = col_cube(pid[DIM - 3], pid[DIM - 2], pid[DIM - 1]);
      }
      break;
    }
    default: {
      casacore::IPosition ip(array_cell_rank);
      for (Legion::PointInDomainIterator<DIM> pid(reg_domain, false);
           pid();
           pid++) {
        for (size_t i = 0; i < DIM; ++i)
          pt[i] = pid[i];
        auto rn = Table::row_number(row_index_shape, pt.begin(), pt.end());
        if (row_number != rn) {
          row_number = rn;
          col.get(row_number, col_array, true);
        }
        for (unsigned i = 0; i < array_cell_rank; ++i)
          ip[i] = pid[DIM - array_cell_rank + i];
        values[*pid] = col_array(ip);
      }
      break;
    }
    }
  }

  template <int DIM, typename T>
  static void
  read_vector_column(
    const casacore::Table& table,
    const casacore::ColumnDesc& col_desc,
    const IndexTreeL& row_index_shape,
    Legion::DomainT<DIM> reg_domain,
    const Legion::PhysicalRegion& region) {

    typedef Legion::FieldAccessor<
      WRITE_DISCARD,
      std::vector<T>,
      DIM,
      Legion::coord_t,
      Legion::AffineAccessor<std::vector<T>, DIM, Legion::coord_t>,
      false> WDAccessor;

    std::vector<Legion::FieldID> fids;
    region.get_fields(fids);
    assert(fids.size() == 1);
    const WDAccessor values(region, fids[0]);

    casacore::ArrayColumn<T> col(table, col_desc.name());
    std::array<Legion::coord_t, DIM> pt;
    size_t row_number;
    unsigned array_cell_rank;
    {
      Legion::PointInDomainIterator<DIM> pid(reg_domain, false);
      for (size_t i = 0; i < DIM; ++i)
        pt[i] = pid[i];
      row_number = Table::row_number(row_index_shape, pt.begin(), pt.end());
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
      for (size_t i = 0; i < DIM; ++i)
        pt[i] = pid[i];
      auto rn = Table::row_number(row_index_shape, pt.begin(), pt.end());
      if (row_number != rn) {
        row_number = rn;
        col.get(row_number, col_array, true);
        col_vector.reference(col_array);
      }
      std::vector<T> cv;
      col_vector.tovector(cv);
      values[*pid] = cv;
    }
  }

private:

  std::string m_table_path;

  const std::shared_ptr<Table> m_table;

  std::vector<std::string> m_column_names;

  size_t m_block_length;

  std::optional<Legion::IndexPartition> m_index_partition;

  template <typename T>
  static inline void field_init(T&, const T&) {}

};

template <> inline
void
TableReadTask::field_init<casacore::String>(
  casacore::String& fld, const casacore::String& val) {
  fld.resize(std::max(val.size(), static_cast<casacore::String::size_type>(1)));
}

} // end namespace ms
} // end namespace legms

#endif // LEGMS_MS_TABLE_READ_TASK_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:

