#ifndef LEGMS_TABLE_BUILDER_H_
#define LEGMS_TABLE_BUILDER_H_

#include <algorithm>
#include <cassert>
#include <experimental/filesystem>
#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "legion.h"

#include <casacore/casa/aipstype.h>
#include <casacore/casa/BasicSL/String.h>
#include <casacore/tables/Tables.h>

#include "WithKeywordsBuilder.h"
#include "ColumnBuilder.h"
#include "Column.h"
#include "IndexTree.h"
#include "MSTable.h"

namespace legms {

template <typename D>
class TableT;

template <typename D>
class TableBuilderT
  : public WithKeywordsBuilder {

  friend class TableT<D>;

public:

  TableBuilderT(
    const std::string& name,
    unsigned full_rank,
    const std::vector<D>& row_axes,
    IndexTreeL row_index_pattern)
    : WithKeywordsBuilder()
    , m_name(name)
    , m_row_axes(row_axes)
    , m_row_index_pattern(row_index_pattern) {

    // number of axes must match rank of row_index_pattern
    assert(m_row_axes.size() == m_row_index_pattern.rank().value());
    // axes must be unique in row_axes
    assert(has_unique_values(m_row_axes));
  }

  const std::string&
  name() const {
    return m_name;
  }

  const std::vector<D>&
  row_axes() const {
    return m_row_axes;
  }

  template <typename T>
  void
  add_scalar_column(const std::string& name) {

    add_column(ScalarColumnBuilder<D>::template generator<T>(name)(
                 m_row_axes, m_row_index_pattern));
  }

  template <typename T, int DIM>
  void
  add_array_column(
    const std::string& name,
    const std::vector<D>& element_axes,
    std::function<std::array<size_t, DIM>(const std::any&)> element_shape) {

    std::vector<D> axes = m_row_axes;
    std::copy(
      element_axes.begin(),
      element_axes.end(),
      std::back_inserter(axes));
    add_column(
      ArrayColumnBuilder<D, DIM>::template generator<T>(name, element_shape)(
        axes,
        m_row_index_pattern));
  }

  void
  add_row(const std::unordered_map<std::string, std::any>& args) {
    std::for_each(
      m_columns.begin(),
      m_columns.end(),
      [&args](auto& nm_col) {
        const std::string& nm = std::get<0>(nm_col);
        auto nm_arg = args.find(nm);
        std::any arg = ((nm_arg != args.end()) ? nm_arg->second : std::any());
        std::get<1>(nm_col)->add_row(arg);
      });
  }

  void
  add_row() {
    add_row(std::unordered_map<std::string, std::any>());
  }

  std::unordered_set<std::string>
  column_names() const {
    std::unordered_set<std::string> result;
    std::transform(
      m_columns.begin(),
      m_columns.end(),
      std::inserter(result, result.end()),
      [](auto& nm_col) {
        return std::get<0>(nm_col);
      });
    return result;
  }

  std::vector<typename ColumnT<D>::Generator>
  column_generators() const {
    std::vector<typename ColumnT<D>::Generator> result;
    std::transform(
      m_columns.begin(),
      m_columns.end(),
      std::back_inserter(result),
      [](auto& cb) {
        return
          [cb](Legion::Context ctx, Legion::Runtime *runtime) {
            return std::make_shared<ColumnT<D>>(ctx, runtime, *cb.second);
          };
      });
    return result;
  }

protected:

  struct SizeArgs {
    std::shared_ptr<casacore::TableColumn> tcol;
    unsigned *row;
    casacore::IPosition shape;
  };

  template <int DIM>
  static std::array<size_t, DIM>
  size(const std::any& args) {
    std::array<size_t, DIM> result;
    auto sa = std::any_cast<SizeArgs>(args);
    std::optional<casacore::IPosition> shape;
    if (sa.tcol) {
      if (sa.tcol->hasContent(*sa.row))
        shape = sa.tcol->shape(*sa.row);
    } else {
      shape = sa.shape;
    }
    if (shape) {
      auto shp = shape.value();
      assert(shp.size() == DIM);
      for (size_t i = 0; i < DIM; ++i)
        result[i] = shp[DIM - 1 - i];
    } else {
      for (size_t i = 0; i < DIM; ++i)
        result[i] = 0;
    }
    return result;
  }

  template <casacore::DataType DT>
  void
  add_from_table_column(
    const std::string& nm,
    const std::vector<D>& element_axes,
    std::unordered_set<std::string>& array_names) {

    typedef typename legms::DataType<DT>::ValueType VT;

    switch (element_axes.size()) {
    case 0:
      add_scalar_column<VT>(nm);
      break;

    case 1:
      add_array_column<VT, 1>(nm, element_axes, size<1>);
      array_names.insert(nm);
      break;

    case 2:
      add_array_column<VT, 2>(nm, element_axes, size<2>);
      array_names.insert(nm);
      break;

    case 3:
      add_array_column<VT, 3>(nm, element_axes, size<3>);
      array_names.insert(nm);
      break;

    default:
      assert(false);
      break;
    }
  }

  void
  add_column(std::unique_ptr<ColumnBuilder<D>>&& col) {
    std::shared_ptr<ColumnBuilder<D>> scol = std::move(col);
    std::vector<D> col_row_axes;
    std::copy_n(
      scol->axes().begin(),
      m_row_axes.size(),
      std::back_inserter(col_row_axes));
    assert(col_row_axes == m_row_axes);
    assert(scol->row_index_pattern() == m_row_index_pattern);
    assert(m_columns.count(scol->name()) == 0);
    m_columns[scol->name()] = scol;
  }

  std::string m_name;

  std::vector<D> m_row_axes;

  std::unordered_map<std::string, std::shared_ptr<ColumnBuilder<D>>> m_columns;

  IndexTreeL m_row_index_pattern;

public:

  static TableBuilderT
  from_casacore_table(
    const std::experimental::filesystem::path& path,
    const std::unordered_set<std::string>& column_selections,
    const std::unordered_map<std::string, std::vector<D>>& element_axes) {

    casacore::Table table(
      casacore::String((path.filename() == "MAIN") ? path.parent_path() : path),
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
    TableBuilderT
      result(
        table_name,
        static_cast<int>(D::last) + 1,
        {D::row},
        IndexTreeL(1));
    std::unordered_set<std::string> array_names;

    // expand wildcard column selection, get selected columns that exist in the
    // table
    auto tdesc = table.tableDesc();
    auto column_names = tdesc.columnNames();
    std::unordered_set<std::string> actual_column_selections;
    bool select_all = column_selections.count("*") > 0;
    for (auto& nm : column_names) {
      if (tdesc.isColumn(nm)
          && element_axes.count(nm) > 0
          && (select_all || (column_selections.count(nm) > 0))) {
        auto col = tdesc[nm];
        if (col.isScalar() || (col.isArray() && col.ndim() >= 0))
          actual_column_selections.insert(nm);
      }
    }
    // add a column to TableBuilderT for each of the selected columns
    std::for_each(
      actual_column_selections.begin(),
      actual_column_selections.end(),
      [&result, &tdesc, &element_axes, &array_names](auto& nm) {
        auto axes = element_axes.at(nm);
        switch (tdesc[nm].dataType()) {
        case casacore::DataType::TpBool:
          result.template add_from_table_column<casacore::DataType::TpBool>(
            nm, axes, array_names);
          break;

        case casacore::DataType::TpChar:
          result.template add_from_table_column<casacore::DataType::TpChar>(
            nm, axes, array_names);
          break;

        case casacore::DataType::TpUChar:
          result.template add_from_table_column<casacore::DataType::TpUChar>(
            nm, axes, array_names);
          break;

        case casacore::DataType::TpShort:
          result.template add_from_table_column<casacore::DataType::TpShort>(
            nm, axes, array_names);
          break;

        case casacore::DataType::TpUShort:
          result.template add_from_table_column<casacore::DataType::TpUShort>(
            nm, axes, array_names);
          break;

        case casacore::DataType::TpInt:
          result.template add_from_table_column<casacore::DataType::TpInt>(
            nm, axes, array_names);
          break;

        case casacore::DataType::TpUInt:
          result.template add_from_table_column<casacore::DataType::TpUInt>(
            nm, axes, array_names);
          break;

        case casacore::DataType::TpFloat:
          result.template add_from_table_column<casacore::DataType::TpFloat>(
            nm, axes, array_names);
          break;

        case casacore::DataType::TpDouble:
          result.template add_from_table_column<casacore::DataType::TpDouble>(
            nm, axes, array_names);
          break;

        case casacore::DataType::TpComplex:
          result.template add_from_table_column<casacore::DataType::TpComplex>(
            nm, axes, array_names);
          break;

        case casacore::DataType::TpDComplex:
          result.template add_from_table_column<casacore::DataType::TpDComplex>(
            nm, axes, array_names);
          break;

        case casacore::DataType::TpString:
          result.template add_from_table_column<casacore::DataType::TpString>(
            nm, axes, array_names);
          break;

        default:
          assert(false);
          break;
        }
      });

    // scan rows to get shapes for all selected array columns
    //
    std::unordered_map<std::string, std::any> args;
    // local variable to hold row number, args values contain
    // pointer to this variable
    unsigned row;
    std::transform(
      array_names.begin(),
      array_names.end(),
      std::inserter(args, args.end()),
      [&table, &tdesc, &row](auto& nm) {
        SizeArgs sa;
        sa.row = &row;
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
};

struct TableBuilder {

  template <MSTables T>
  static TableBuilderT<typename MSTable<T>::Axes>
  from_ms(
    const std::experimental::filesystem::path& path,
    const std::unordered_set<std::string>& column_selections) {

    return
      TableBuilderT<typename MSTable<T>::Axes>::from_casacore_table(
        ((path.filename() == MSTable<T>::name)
         ? path
         : (path / MSTable<T>::name)),
        column_selections,
        MSTable<T>::element_axes);
  }
};

} // end namespace legms

#endif // LEGMS_TABLE_BUILDER_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
