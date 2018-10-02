#include <unordered_set>

#include "utility.h"
#include "ReadOnlyTable.h"
#include "IndexTree.h"

using namespace legms::ms;

TableBuilder
ReadOnlyTable::builder(const std::experimental::filesystem::path& path) {

  casacore::Table table(
    casacore::String(path),
    casacore::TableLock::NoLocking);

  {
    auto kws = table.keywordSet();
    auto sort_columns_fn = kws.fieldNumber("SORT_COLUMNS");
    if (sort_columns_fn != -1)
      std::cerr << "Sorted table optimization unimplemented" << std::endl;
  }

  TableBuilder result(path.filename(), IndexTreeL(1));
  std::unordered_set<std::string> array_names;

#define COL(tp) casacore::DataType::Tp##tp:                             \
  result.                                                               \
    add_scalar_column<DataType<casacore::DataType::Tp##tp>::ValueType>(nm); \
  break;                                                                \
  case casacore::DataType::TpArray##tp:                                 \
    switch (col.ndim()) {                                             \
    case 1:                                                           \
      result.add_array_column<\
        1,\
        DataType<casacore::DataType::Tp##tp>::ValueType>( \
        nm, size<1>);                                               \
      array_names.insert(nm);                                          \
      break;                                                          \
    case 2:                                                           \
      result.add_array_column<\
        2,\
        DataType<casacore::DataType::Tp##tp>::ValueType>( \
        nm, size<2>);                                               \
      array_names.insert(nm);                                          \
      break;                                                          \
    case 3:                                                           \
      result.add_array_column<\
        3,\
        DataType<casacore::DataType::Tp##tp>::ValueType>( \
        nm, size<3>);                                               \
      array_names.insert(nm);                                          \
      break;                                                          \
    default:                                                          \
      assert(false);                                                  \
      break;                                                          \
    }

  auto tdesc = table.tableDesc();
  auto column_names = tdesc.columnNames();
  auto end_column_names =
    std::remove_if(
      column_names.begin(),
      column_names.end(),
      [&tdesc](auto& nm) {
        return !tdesc.isColumn(nm) || tdesc[nm].ndim() == -1;
      });
  std::for_each(
    column_names.begin(),
    end_column_names,
    [&result, &tdesc, &array_names](auto& nm) {
      auto col = tdesc[nm];
      switch (col.trueDataType()) {
      case COL(Bool)
        break;
      case COL(Char)
        break;
      case COL(UChar)
        break;
      case COL(Short)
        break;
      case COL(UShort)
        break;
      case COL(Int)
        break;
      case COL(UInt)
        break;
      case COL(Float)
        break;
      case COL(Double)
        break;
      case COL(Complex)
        break;
      case COL(DComplex)
        break;
      case COL(String)
        break;
      default:
        assert(false);
        break;
      }
    });

  std::unordered_map<std::string, std::any> args;
  std::transform(
    array_names.begin(),
    array_names.end(),
    std::inserter(args, args.end()),
    [&table, &tdesc](auto& nm) {
      SizeArgs sa;
      const casacore::IPosition& shp = tdesc[nm].shape();
      if (shp.empty()) {
        sa.tcol = std::make_shared<casacore::TableColumn>(table, nm);
        sa.row = 0;
      } else {
        sa.shape = shp;
      }
      return std::make_pair(nm, sa);
    });

  auto nrow = table.nrow();
  for (unsigned i = 0; i < nrow; ++i) {
    for (auto& arg : args) {
      auto ap = &std::get<1>(arg);
      auto sap = std::any_cast<SizeArgs>(ap);
      if (sap->tcol)
        sap->row = i;
    }
    result.add_row(args);
  }

  return result;
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
