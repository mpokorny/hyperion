#include <functional>

#include "TableBuilder.h"
#include "utility.h"
#include "IndexTree.h"

using namespace legms::ms;

template <typename HINT_MAP>
static ColumnHint
get_column_hint(
  const casacore::TableDesc& tdesc,
  HINT_MAP column_hints,
  const std::string& column_name) {

  if (column_hints.count(column_name) > 0)
    return column_hints.at(column_name);
  return TableBuilder::inferred_column_hint(tdesc[column_name]);
}

ColumnHint
TableBuilder::inferred_column_hint(const casacore::ColumnDesc& coldesc) {

  if (coldesc.isScalar())
    return ColumnHint{0, 0, {}};
  unsigned ndim = coldesc.ndim();
  ColumnHint ch;
  ch.ms_value_rank = ndim;
  ch.index_rank = ndim;
  // make a guess here about the index permutations, using the order that
  // is most common in the MS definition...this is not be optimal in all
  // cases, but hopefully allows the user to provide fewer ColumnHints
  for (unsigned i = 0; i < ndim; ++i)
    ch.index_permutations[i] = i;
  return ch;
}

struct SizeArgs {
  std::shared_ptr<casacore::TableColumn> tcol;
  unsigned *row;
  casacore::IPosition shape;
  std::array<unsigned, ColumnHint::MAX_RANK> index_permutations;
};

template <int DIM>
static std::array<size_t, DIM>
size(const std::any& args) {
  static_assert(DIM <= ColumnHint::MAX_RANK);
  std::array<size_t, DIM> result;
  auto sa = std::any_cast<SizeArgs>(args);
  const casacore::IPosition& shape =
    (sa.tcol ? sa.tcol->shape(*sa.row) : sa.shape);
  assert(shape.size() >= DIM);
  for (size_t i = 0; i < DIM; ++i)
    result[i] = shape[sa.index_permutations[i]];
  return result;
}

template <casacore::DataType DT>
void
addcol(
  TableBuilder& tb,
  const std::string& nm,
  const ColumnHint& hint,
  std::unordered_set<std::string>& array_names) {

  typedef typename legms::DataType<DT>::ValueType VT;

  assert(hint.ms_value_rank >= hint.index_rank);

  switch (hint.index_rank) {
  case 0:
    if (hint.ms_value_rank == 0)
      tb.add_scalar_column<VT>(nm);
    else
      tb.add_scalar_column<std::vector<VT>>(nm);
    break;

  case 1: {
    if (hint.ms_value_rank == 1)
      tb.add_array_column<1, VT>(nm, size<1>);
    else
      tb.add_array_column<1, std::vector<VT>>(nm, size<1>);
    array_names.insert(nm);
    break;
  }
  case 2: {
    if (hint.ms_value_rank == 2)
      tb.add_array_column<2, VT>(nm, size<2>);
    else
      tb.add_array_column<2, std::vector<VT>>(nm, size<2>);
    array_names.insert(nm);
    break;
  }
  case 3: {
    if (hint.ms_value_rank == 3)
      tb.add_array_column<3, VT>(nm, size<3>);
    else
      tb.add_array_column<3, std::vector<VT>>(nm, size<3>);
    array_names.insert(nm);
    break;
  }
  default:
    assert(false);
    break;
  }
}

TableBuilder
TableBuilder::from_casacore_table(
  const std::experimental::filesystem::path& path,
  const std::unordered_set<std::string>& column_selections,
  const std::unordered_map<std::string, ColumnHint>& column_hints) {

  casacore::Table table(
    casacore::String(path),
    casacore::TableLock::PermanentLockingWait);

  {
    auto kws = table.keywordSet();
    auto sort_columns_fn = kws.fieldNumber("SORT_COLUMNS");
    if (sort_columns_fn != -1)
      std::cerr << "Sorted table optimization unimplemented" << std::endl;
  }

  std::string table_name = path.filename();
  if (table_name == ".")
    table_name = "MAIN";
  TableBuilder result(table_name, IndexTreeL(1));
  std::unordered_set<std::string> array_names;

  // expand wildcard column selection, get selected columns that exist in the
  // table
  auto tdesc = table.tableDesc();
  auto column_names = tdesc.columnNames();
  std::unordered_set<std::string> actual_column_selections;
  bool select_all = column_selections.count("*") > 0;
  for (auto& nm : column_names) {
    if (tdesc.isColumn(nm)
        && (select_all || column_selections.count(nm) > 0)) {
      auto col = tdesc[nm];
      if (col.isScalar() || (col.isArray() && col.ndim() >= 0))
        actual_column_selections.insert(nm);
    }
  }
  // using ColumnHints, add a column to TableBuilder for each of the selected
  // columns
  std::for_each(
    actual_column_selections.begin(),
    actual_column_selections.end(),
    [&result, &tdesc, &column_hints, &array_names](auto& nm) {
      ColumnHint hint = get_column_hint(tdesc, column_hints, nm);
      switch (tdesc[nm].dataType()) {
      case casacore::DataType::TpBool:
        addcol<casacore::DataType::TpBool>(result, nm, hint, array_names);
        break;

      case casacore::DataType::TpChar:
        addcol<casacore::DataType::TpChar>(result, nm, hint, array_names);
        break;

      case casacore::DataType::TpUChar:
        addcol<casacore::DataType::TpUChar>(result, nm, hint, array_names);
        break;

      case casacore::DataType::TpShort:
        addcol<casacore::DataType::TpShort>(result, nm, hint, array_names);
        break;

      case casacore::DataType::TpUShort:
        addcol<casacore::DataType::TpUShort>(result, nm, hint, array_names);
        break;

      case casacore::DataType::TpInt:
        addcol<casacore::DataType::TpInt>(result, nm, hint, array_names);
        break;

      case casacore::DataType::TpUInt:
        addcol<casacore::DataType::TpUInt>(result, nm, hint, array_names);
        break;

      case casacore::DataType::TpFloat:
        addcol<casacore::DataType::TpFloat>(result, nm, hint, array_names);
        break;

      case casacore::DataType::TpDouble:
        addcol<casacore::DataType::TpDouble>(result, nm, hint, array_names);
        break;

      case casacore::DataType::TpComplex:
        addcol<casacore::DataType::TpComplex>(result, nm, hint, array_names);
        break;

      case casacore::DataType::TpDComplex:
        addcol<casacore::DataType::TpDComplex>(result, nm, hint, array_names);
        break;

      case casacore::DataType::TpString:
        addcol<casacore::DataType::TpString>(result, nm, hint, array_names);
        break;

      default:
        assert(false);
        break;
      }
    });

  std::unordered_map<std::string, std::any> args;
  unsigned row; // local variable to hold row number, args values contain
                // pointer to this variable
  std::transform(
    array_names.begin(),
    array_names.end(),
    std::inserter(args, args.end()),
    [&column_hints, &table, &tdesc, &row](auto& nm) {
      SizeArgs sa;
      sa.row = &row;
      sa.index_permutations =
        get_column_hint(tdesc, column_hints, nm).index_permutations;
      const casacore::IPosition& shp = tdesc[nm].shape();
      if (shp.empty())
        sa.tcol = std::make_shared<casacore::TableColumn>(table, nm);
      else
        sa.shape = shp;
      return std::make_pair(nm, sa);
    });

  auto nrow = table.nrow();
  for (row = 0; row < nrow; ++row)
    result.add_row(args);

  return result;
}

const std::unordered_map<std::string, ColumnHint>&
TableBuilder::ms_column_hints(const std::string& table) {
  if (m_ms_column_hints.count(table) > 0)
    return m_ms_column_hints.at(table);
  return m_empty_column_hints;
}

const std::unordered_map<std::string, ColumnHint>
TableBuilder::m_empty_column_hints;

const std::unordered_map<
  std::string,
  std::unordered_map<std::string, ColumnHint>>
TableBuilder::m_ms_column_hints = {
  {"MAIN",
   {{"UVW", {1, 0, {}}},
    {"UVW2", {1, 0, {}}},
   }},
  {"ANTENNA",
   {{"POSITION", {1, 0, {}}},
    {"OFFSET", {1, 0, {}}},
    {"MEAN_ORBIT", {1, 0, {}}},
   }},
  {"FEED",
   {{"BEAM_OFFSET", {2, 1, {1}}},
    {"POL_RESPONSE", {2, 1, {1}}},
    {"POSITION", {1, 0, {}}},
   }},
  {"HISTORY",
   {{"CLI_COMMAND", {1, 0, {}}},
    {"APP_PARAMS", {1, 0, {}}},
   }},
  {"OBSERVATION",
   {{"TIME_RANGE", {1, 0, {}}},
    {"LOG", {1, 0, {}}},
    {"SCHEDULE", {1, 0, {}}},
   }},
  {"POLARIZATION",
   {{"CORR_PRODUCT", {2, 2, {1,0}}},
    // alternative: {"CORR_PRODUCT", {2, 1, {1}}},
   }},
  {"SOURCE",
   {{"DIRECTION", {1, 0, {}}},
    {"POSITION", {1, 0, {}}},
    {"PROPER_MOTION", {1, 0, {}}},
   }},
  {"SPECTRAL_WINDOW",
   {{"ASSOC_SPW_ID", {1, 0, {}}},
    {"ASSOC_NATURE", {1, 0, {}}},
   }},
};

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
