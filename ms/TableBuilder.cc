#include "TableBuilder.h"
#include "utility.h"
#include "IndexTree.h"

using namespace legms::ms;

struct SizeArgs {
  std::shared_ptr<casacore::TableColumn> tcol;
  unsigned row;
  casacore::IPosition shape;
};

template <int DIM>
static std::array<size_t, DIM>
size(const std::any& args) {
  std::array<size_t, DIM> result;
  auto sa = std::any_cast<SizeArgs>(args);
  const casacore::IPosition& shape =
    (sa.tcol ? sa.tcol->shape(sa.row) : sa.shape);
  assert(shape.size() == DIM);
  shape.copy(result.begin());
  return result;
}

template <casacore::DataType DT>
void
scalar_column(TableBuilder& tb, const std::string& nm) {
  tb.add_scalar_column<typename legms::DataType<DT>::ValueType>(nm);
}

template <casacore::DataType DT>
void
array_column(
  TableBuilder& tb,
  const std::string& nm,
  int ndim,
  std::unordered_set<std::string>& array_names) {

  switch (ndim) {
  case 1:
    tb.add_array_column<1, typename legms::DataType<DT>::ValueType>(
      nm,
      size<1>);
    array_names.insert(nm);
    break;

  case 2:
    tb.add_array_column<2, typename legms::DataType<DT>::ValueType>(
      nm,
      size<2>);
    array_names.insert(nm);
    break;

  case 3:
    tb.add_array_column<3, typename legms::DataType<DT>::ValueType>(
      nm,
      size<3>);
    array_names.insert(nm);
    break;

  case -1:
    break;

  default:
    assert(false);
    break;
  }
}

TableBuilder
TableBuilder::from_casacore_table(
  const std::experimental::filesystem::path& path,
  const std::unordered_set<std::string>& column_selection) {

  casacore::Table table(
    casacore::String(path),
    casacore::TableLock::PermanentLockingWait);

  {
    auto kws = table.keywordSet();
    auto sort_columns_fn = kws.fieldNumber("SORT_COLUMNS");
    if (sort_columns_fn != -1)
      std::cerr << "Sorted table optimization unimplemented" << std::endl;
  }

  TableBuilder result(path.filename(), IndexTreeL(1));
  std::unordered_set<std::string> array_names;

  bool select_all = column_selection.count("*") > 0;
  auto tdesc = table.tableDesc();
  auto column_names = tdesc.columnNames();
  std::for_each(
    column_names.begin(),
    column_names.end(),
    [&result, &tdesc, &array_names, &column_selection, &select_all](auto& nm) {
      if (tdesc.isColumn(nm)
          && (select_all || column_selection.count(nm) > 0)) {
        auto col = tdesc[nm];
        switch (col.trueDataType()) {
        case casacore::DataType::TpBool:
          scalar_column<casacore::DataType::TpBool>(result, nm);
          break;

        case casacore::DataType::TpArrayBool:
          array_column<casacore::DataType::TpBool>(
            result,
            nm,
            col.ndim(),
            array_names);
          break;

        case casacore::DataType::TpChar:
          scalar_column<casacore::DataType::TpChar>(result, nm);
          break;

        case casacore::DataType::TpArrayChar:
          array_column<casacore::DataType::TpChar>(
            result,
            nm,
            col.ndim(),
            array_names);
          break;

        case casacore::DataType::TpUChar:
          scalar_column<casacore::DataType::TpUChar>(result, nm);
          break;

        case casacore::DataType::TpArrayUChar:
          array_column<casacore::DataType::TpUChar>(
            result,
            nm,
            col.ndim(),
            array_names);
          break;

        case casacore::DataType::TpShort:
          scalar_column<casacore::DataType::TpShort>(result, nm);
          break;

        case casacore::DataType::TpArrayShort:
          array_column<casacore::DataType::TpShort>(
            result,
            nm,
            col.ndim(),
            array_names);
          break;

        case casacore::DataType::TpUShort:
          scalar_column<casacore::DataType::TpUShort>(result, nm);
          break;

        case casacore::DataType::TpArrayUShort:
          array_column<casacore::DataType::TpUShort>(
            result,
            nm,
            col.ndim(),
            array_names);
          break;

        case casacore::DataType::TpInt:
          scalar_column<casacore::DataType::TpInt>(result, nm);
          break;

        case casacore::DataType::TpArrayInt:
          array_column<casacore::DataType::TpInt>(
            result,
            nm,
            col.ndim(),
            array_names);
          break;

        case casacore::DataType::TpUInt:
          scalar_column<casacore::DataType::TpUInt>(result, nm);
          break;

        case casacore::DataType::TpArrayUInt:
          array_column<casacore::DataType::TpUInt>(
            result,
            nm,
            col.ndim(),
            array_names);
          break;

        case casacore::DataType::TpFloat:
          scalar_column<casacore::DataType::TpFloat>(result, nm);
          break;

        case casacore::DataType::TpArrayFloat:
          array_column<casacore::DataType::TpFloat>(
            result,
            nm,
            col.ndim(),
            array_names);
          break;

        case casacore::DataType::TpDouble:
          scalar_column<casacore::DataType::TpDouble>(result, nm);
          break;

        case casacore::DataType::TpArrayDouble:
          array_column<casacore::DataType::TpDouble>(
            result,
            nm,
            col.ndim(),
            array_names);
          break;

        case casacore::DataType::TpComplex:
          scalar_column<casacore::DataType::TpComplex>(result, nm);
          break;

        case casacore::DataType::TpArrayComplex:
          array_column<casacore::DataType::TpComplex>(
            result,
            nm,
            col.ndim(),
            array_names);
          break;

        case casacore::DataType::TpDComplex:
          scalar_column<casacore::DataType::TpDComplex>(result, nm);
          break;

        case casacore::DataType::TpArrayDComplex:
          array_column<casacore::DataType::TpDComplex>(
            result,
            nm,
            col.ndim(),
            array_names);
          break;

        case casacore::DataType::TpString:
          scalar_column<casacore::DataType::TpString>(result, nm);
          break;

        case casacore::DataType::TpArrayString:
          array_column<casacore::DataType::TpString>(
            result,
            nm,
            col.ndim(),
            array_names);
          break;

        default:
          assert(false);
          break;
        }
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
    std::for_each(
      args.begin(),
      args.end(),
      [i](auto& arg) {
        auto ap = &std::get<1>(arg);
        auto sap = std::any_cast<SizeArgs>(ap);
        if (sap->tcol)
          sap->row = i;
      });
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
