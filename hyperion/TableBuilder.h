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
#ifndef HYPERION_TABLE_BUILDER_H_
#define HYPERION_TABLE_BUILDER_H_

#include <hyperion/hyperion.h>
#include <hyperion/KeywordsBuilder.h>
#include <hyperion/ColumnBuilder.h>
#include <hyperion/ColumnSpace.h>
#include <hyperion/Table.h>
#include <hyperion/IndexTree.h>
#include <hyperion/MSTable.h>
#include <hyperion/MSTableColumns.h>
#include <hyperion/MeasRef.h>
#include <hyperion/Measures.h>
#include <hyperion/tree_index_space.h>

#include <algorithm>
#include CXX_ANY_HEADER
#include <cassert>
#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <casacore/casa/aipstype.h>
#include <casacore/casa/BasicSL/String.h>
#include <casacore/tables/Tables.h>

namespace hyperion {

template <MSTables D>
class TableBuilderT
  : public KeywordsBuilder {

  friend class Table;

public:

  typedef typename MSTable<D>::Axes Axes;

  TableBuilderT(const std::string& name)
    : KeywordsBuilder()
    , m_name(name)
    , m_num_rows(0) {
  }

  const std::string&
  name() const {
    return m_name;
  }

  size_t
  num_rows() const {
    return m_num_rows;
  }

  template <typename T>
  CXX_OPTIONAL_NAMESPACE::optional<std::string>
  add_scalar_column(
    const casacore::Table& table,
    const std::string& name,
    unsigned fid) {

    return
      add_column(
        table,
        ScalarColumnBuilder<D>::template generator<T>(name, fid)());
  }

  template <typename T, int DIM>
  CXX_OPTIONAL_NAMESPACE::optional<std::string>
  add_array_column(
    const casacore::Table& table,
    const std::string& name,
    unsigned fid,
    const std::vector<Axes>& element_axes,
    std::function<std::array<size_t, DIM>(const CXX_ANY_NAMESPACE::any&)>
      element_shape) {

    std::vector<Axes> axes {MSTable<D>::ROW_AXIS};
    std::copy(
      element_axes.begin(),
      element_axes.end(),
      std::back_inserter(axes));
    return
      add_column(
        table,
        ArrayColumnBuilder<D, DIM>::template generator<T>(
          name,
          fid,
          element_shape)(axes));
  }

  void
  add_row(const std::unordered_map<std::string, CXX_ANY_NAMESPACE::any>& args) {
    std::for_each(
      m_columns.begin(),
      m_columns.end(),
      [&args](auto& nm_col) {
        const std::string& nm = std::get<0>(nm_col);
        auto nm_arg = args.find(nm);
        CXX_ANY_NAMESPACE::any arg =
          ((nm_arg != args.end()) ? nm_arg->second : CXX_ANY_NAMESPACE::any());
        std::get<1>(nm_col)->add_row(arg);
      });
    ++m_num_rows;
  }

  void
  add_row() {
    add_row(std::unordered_map<std::string, CXX_ANY_NAMESPACE::any>());
  }

  std::vector<
    std::tuple<ColumnSpace, std::vector<std::pair<std::string, TableField>>>>
  columns(Legion::Context ctx, Legion::Runtime* rt) const {
    // sort the ColumnArgs in order to put instances with common index_tree
    // values next to one another
    std::vector<ColumnArgs> col_args;
    for (auto& nm_cb : m_columns) {
      // the body of this loop is big enough that I don't want to repeat it, so
      // just use non-c++17 construct here
      std::string nm;
      std::shared_ptr<ColumnBuilder<D>> cb;
      std::tie(nm, cb) = nm_cb;
      auto ca = cb->column(ctx, rt);
      if (ca.index_tree != IndexTreeL()) {
        auto it_match =
          std::find_if(
            col_args.begin(),
            col_args.end(),
            [&ca](auto& cca) {
              return ca.index_tree == cca.index_tree;
            });
        col_args.insert(it_match, ca);
      }
    }

    // collect TableFields with common ColumnSpaces by iterating through
    // "col_args" in its sort order
    std::vector<
      std::tuple<ColumnSpace, std::vector<std::pair<std::string, TableField>>>>
      result;
    IndexTreeL prev;
    for (auto& ca : col_args) {
      assert(ca.index_tree != IndexTreeL());
      if (ca.index_tree != prev) {
        prev = ca.index_tree;
        auto csp =
          ColumnSpace::create(
            ctx,
            rt,
            ca.axes,
            ca.axes_uid,
            tree_index_space(ca.index_tree, ctx, rt),
            false);
        result.emplace_back(
          csp,
          std::vector<std::pair<std::string, TableField>>());
      }
      std::get<1>(result.back()).emplace_back(ca.name, ca.tf);
    }
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
  size(const CXX_ANY_NAMESPACE::any& args) {
    std::array<size_t, DIM> result;
    auto sa = CXX_ANY_NAMESPACE::any_cast<SizeArgs>(args);
    CXX_OPTIONAL_NAMESPACE::optional<casacore::IPosition> shape;
    if (sa.tcol) {
      if (sa.tcol->hasContent(*sa.row)){
        auto rsh = sa.tcol->shape(*sa.row);
        if (std::count(rsh.begin(), rsh.end(), 0) == 0)
          shape = rsh;
      }
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

  template <hyperion::TypeTag DT>
  CXX_OPTIONAL_NAMESPACE::optional<std::string>
  add_from_table_column(
    const casacore::Table& table,
    const std::string& nm,
    unsigned fid,
    const std::vector<Axes>& element_axes,
    std::unordered_set<std::string>& array_names) {

    typedef typename hyperion::DataType<DT>::ValueType VT;

    CXX_OPTIONAL_NAMESPACE::optional<std::string> result;
    switch (element_axes.size()) {
    case 0:
      result = add_scalar_column<VT>(table, nm, fid);
      break;

    case 1:
      result = add_array_column<VT, 1>(table, nm, fid, element_axes, size<1>);
      array_names.insert(nm);
      break;

    case 2:
      result = add_array_column<VT, 2>(table, nm, fid, element_axes, size<2>);
      array_names.insert(nm);
      break;

    case 3:
      result = add_array_column<VT, 3>(table, nm, fid, element_axes, size<3>);
      array_names.insert(nm);
      break;

    default:
      assert(false);
      break;
    }
    return result;
  }

  CXX_OPTIONAL_NAMESPACE::optional<std::string>
  add_column(
    const casacore::Table& table,
    std::unique_ptr<ColumnBuilder<D>>&& col) {

    CXX_OPTIONAL_NAMESPACE::optional<std::string> result;
    std::shared_ptr<ColumnBuilder<D>> scol = std::move(col);
    assert(scol->num_rows() == m_num_rows);
    if (m_columns.count(scol->name()) > 0) {
      // this can happen if columns share a measure reference column
      return CXX_OPTIONAL_NAMESPACE::nullopt;
    }
    m_columns[scol->name()] = scol;
    auto mr = get_meas_refs(table, scol->name());
    if (mr) {
      if (std::get<2>(mr.value()))
        result = std::get<2>(mr.value()).value();
      scol->set_meas_record(std::move(mr.value()));
    }
    auto tcol = casacore::TableColumn(table, scol->name());
    auto kws = tcol.keywordSet();
    auto nf = kws.nfields();
    for (unsigned f = 0; f < nf; ++f) {
      std::string name = kws.name(f);
      auto dt = kws.dataType(f);
      if (name != "MEASINFO" && name != "QuantumUnits") {
        switch (dt) {
#define ADD_KW(DT)                              \
          case DataType<DT>::CasacoreTypeTag:   \
            scol->add_keyword(name, DT);        \
            break;
          HYPERION_FOREACH_CC_DATATYPE(ADD_KW);
#undef ADD_KW
        default:
          // ignore other kw types, like Table

          // TODO: support for Array<String> could be useful (e.g, in
          // FLAG_CATEGORY)
          break;
        }
      }
    }
    return result;
  }

  std::string m_name;

  std::unordered_map<std::string, std::shared_ptr<ColumnBuilder<D>>> m_columns;

  size_t m_num_rows;

public:

  static TableBuilderT
  from_casacore_table(
    const CXX_FILESYSTEM_NAMESPACE::path& path,
    const std::unordered_set<std::string>& column_selections,
    const std::unordered_map<std::string, std::vector<Axes>>& element_axes) {

    casacore::Table table(
      casacore::String((path.filename() == "MAIN") ? path.parent_path() : path),
      casacore::TableLock::PermanentLockingWait);

    std::string table_name = path.filename();
    if (table_name == ".")
      table_name = "MAIN";
    TableBuilderT result(table_name);
    std::unordered_set<std::string> array_names;

    // expand wildcard column selection, get selected columns that exist in the
    // table
    auto tdesc = table.tableDesc();
    auto column_names = tdesc.columnNames();
    std::unordered_set<std::string> actual_column_selections;
    std::copy_if(
      column_names.begin(),
      column_names.end(),
      std::inserter(actual_column_selections, actual_column_selections.end()),
      [&tdesc, &column_selections, select_all=column_selections.count("*") > 0]
      (auto& nm) {
        return tdesc.isColumn(nm)
          && (select_all || (column_selections.count(nm) > 0));
      });

    // FIXME: awaiting Table keyword support
#if NEVER_DEFINED_HAVE_TABLE_KEYWORD_SUPPORT
    // get table keyword names and types
    {
      auto kws = table.keywordSet();
      auto nf = kws.nfields();
      for (unsigned f = 0; f < nf; ++f) {
        std::string name = kws.name(f);
        auto dt = kws.dataType(f);
        if (name != "MEASINFO" && name != "QuantumUnits") {
          switch (dt) {
#define ADD_KW(DT)                                  \
            case DataType<DT>::CasacoreTypeTag:     \
              result.add_keyword(name, DT);         \
              break;
            HYPERION_FOREACH_DATATYPE(ADD_KW);
#undef ADD_KW
          default:
            // ignore other kw types, like Table
            break;
          }
        }
      }
    }
#endif // NEVER_DEFINED_HAVE_TABLE_KEYWORD_SUPPORT

    // add a column to TableBuilderT for each of the selected columns
    //
    unsigned unreserved_fid = MSTableColumns<D>::column_names.size();
    std::for_each(
      actual_column_selections.begin(),
      actual_column_selections.end(),
      [&table, &result, &tdesc, &element_axes, &array_names, &unreserved_fid]
      (auto& nm) {
        CXX_OPTIONAL_NAMESPACE::optional<std::string> refcol;
        auto c = MSTableColumns<D>::lookup_col(nm);
        // "c" will be empty if "nm" belongs to a measure reference column (all
        // of which either are undocumented in the MS standard and therefore are
        // unknown to hyperion, or have been intentionally omitted in
        // MSTableColumns data values)
        if (c) {
          auto axes = element_axes.at(nm);
          auto cdesc = tdesc[nm];
          unsigned fid = MSTableColumns<D>::fid(c.value());
          switch (cdesc.dataType()) {
#define ADD_FROM_TCOL(DT)                                   \
            case DataType<DT>::CasacoreTypeTag:             \
              refcol =                                      \
                result.template add_from_table_column<DT>(  \
                  table, nm, fid, axes, array_names);       \
              break;
            HYPERION_FOREACH_CC_DATATYPE(ADD_FROM_TCOL);
#undef ADD_FROM_TCOL
          default:
            assert(false);
            break;
          }
        }
        if (refcol) {
          auto rc =
            result.template add_from_table_column<HYPERION_TYPE_INT>(
              table,
              refcol.value(),
              unreserved_fid++,
              {},
              array_names);
          assert(!rc);
        }
      });

    // scan rows to get shapes for all selected array columns
    //
    std::unordered_map<std::string, CXX_ANY_NAMESPACE::any> args;
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

struct HYPERION_EXPORT TableBuilder {

  template <MSTables T>
  static TableBuilderT<T>
  from_ms(
    const CXX_FILESYSTEM_NAMESPACE::path& path,
    const std::unordered_set<std::string>& column_selections) {

    return
      TableBuilderT<T>::from_casacore_table(
        ((path.filename() == MSTable<T>::name)
         ? path
         : (path / MSTable<T>::name)),
        column_selections,
        MSTable<T>::element_axes);
  }
};

#if NEVER_DEFINED_HAVE_TABLE_KEYWORD_SUPPORT
void
initialize_keywords_from_ms(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const CXX_FILESYSTEM_NAMESPACE::path& path,
  Table& table);
#endif // NEVER_DEFINED_HAVE_TABLE_KEYWORD_SUPPORT

template <MSTables T>
std::tuple<std::string, ColumnSpace, Table::fields_t>
from_ms(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const CXX_FILESYSTEM_NAMESPACE::path& path,
  const std::unordered_set<std::string>& column_selections) {

  const CXX_FILESYSTEM_NAMESPACE::path& table_path =
    ((path.filename() == MSTable<T>::name)
     ? path
     : (path / MSTable<T>::name));

  auto builder = TableBuilder::from_ms<T>(table_path, column_selections);

  Legion::IndexSpace column_is =
    rt->create_index_space(ctx, Legion::Rect<1>(0, builder.num_rows() - 1));
  std::vector<typename MSTable<T>::Axes> axes = {MSTable<T>::ROW_AXIS};
  auto index_cs = ColumnSpace::create(ctx, rt, axes, column_is, false);

  // FIXME: awaiting keyword support in Tables
  //initialize_keywords_from_ms(ctx, rt, table_path, result);

  return std::make_tuple(builder.name(), index_cs, builder.columns(ctx, rt));
}

} // end namespace hyperion

#endif // HYPERION_TABLE_BUILDER_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
