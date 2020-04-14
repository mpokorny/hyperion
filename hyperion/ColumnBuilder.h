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
#ifndef HYPERION_COLUMN_BUILDER_H_
#define HYPERION_COLUMN_BUILDER_H_

#include <hyperion/hyperion.h>
#include <hyperion/utility.h>
#include <hyperion/IndexTree.h>
#include <hyperion/TableField.h>
#include <hyperion/KeywordsBuilder.h>
#include <hyperion/MSTable.h>
#include <hyperion/MeasRef.h>

#pragma GCC visibility push(default)
# include <any>
# include <cassert>
# include <functional>
# include <memory>
# include <numeric>
# include <optional>
# include <vector>
#pragma GCC visibility pop

#include <casacore/casa/BasicSL/String.h>
#include <casacore/casa/Containers/Record.h>
#include <casacore/measures/Measures/MeasureHolder.h>

namespace hyperion {

/**
 *  Values sufficient for constructing a Column.
 *
 *  Returned from ColumnBuilder::column()
 */
struct ColumnArgs {
  std::string name;
  TableField tf;
  std::string axes_uid;
  std::vector<int> axes;
  IndexTreeL index_tree;
};

/**
 * Build a Column from a MeasurementSet column
 *
 * Provides a means of creating the arguments required for a Column instance
 * by scanning a MeasurementSet table column row by row *via* a TableBuilder.
 */
template <MSTables D>
class ColumnBuilder
  : public KeywordsBuilder {
public:

  typedef typename MSTable<D>::Axes AxesT;

  /**
   * Construct a Column *via* TableBuilder
   */
  ColumnBuilder(
    const std::string& name, /**< [in] column name */
    hyperion::TypeTag datatype, /**< [in] column data type */
    unsigned fid, /**< [in] column values field id */
    const std::vector<AxesT>& axes /**< [in] column axes */)
    : KeywordsBuilder()
    , m_name(name)
    , m_datatype(datatype)
    , m_fid(fid)
    , m_axes(axes)
    , m_num_rows(0) {

    assert(axes[0] == MSTable<D>::ROW_AXIS);
    assert(has_unique_values(m_axes));
  }

  virtual ~ColumnBuilder() {}

  /**
   * Column name
   */
  const std::string&
  name() const {
    return m_name;
  }

  /**
   * Column data type
   *
   * Column data type (as hyperion::TypeTag)
   */
  hyperion::TypeTag
  datatype() const {
    return m_datatype;
  }

  /**
   * Column axes
   */
  const std::vector<AxesT>&
  axes() const {
    return m_axes;
  }

  /**
   * Column index space
   *
   * Column index space (as IndexTreeL).
   */
  const IndexTreeL&
  index_tree() const {
    return m_index_tree;
  }

  /**
   * Column rank
   */
  unsigned
  rank() const {
    return m_axes.size();
  }

  /**
   * Number of column rows
   */
  size_t
  num_rows() const {
    return m_num_rows;
  }

  /**
   * Is the column empty?
   *
   * @return true if and only if the column index space is empty
   */
  bool
  empty() const {
    return index_tree().size() == 0;
  }

  /**
   * Column measure
   *
   * Values associated with a casacore::Measure that is in the MeasurementSet
   * column
   *
   * @return std::optional of a std::tuple of elements representing a column
   * measure (including row-based measures)
   */
  const std::optional<
    std::tuple<
      hyperion::MClass,
      std::vector<std::tuple<std::unique_ptr<casacore::MRBase>, unsigned>>,
      std::optional<std::string>>>&
  meas_record() const {
    return m_meas_record;
  }

  /**
   * Set values for the column measure
   */
  void
  set_meas_record(
    std::tuple<
      hyperion::MClass,
      std::vector<
      std::tuple<std::unique_ptr<casacore::MRBase>, unsigned>>,
      std::optional<std::string>>&& rec) {
    m_meas_record = std::move(rec);
  }

  /**
   * Add a row
   *
   * @param[in] any The shape of the row element
   */
  virtual void
  add_row(const std::any&) = 0;

  /**
   * Construct the ColumnArgs
   *
   * Construct the ColumnArgs value needed to, in turn, construct a Column.
   */
  ColumnArgs
  column(Legion::Context ctx, Legion::Runtime* rt) const {

    IndexTreeL itree;
    auto itrank = index_tree().rank();
    if (itrank && itrank.value() == rank())
      itree = index_tree();
    MeasRef mr;
    std::optional<std::string> ref_column;
    if (m_meas_record) {
      std::tuple<MClass, std::vector<std::tuple<casacore::MRBase*, unsigned>>>
        mrec;
      std::get<0>(mrec) = std::get<0>(m_meas_record.value());
      for (auto& [mrb, c] : std::get<1>(m_meas_record.value()))
        std::get<1>(mrec).emplace_back(mrb.get(), c);
      mr = std::get<1>(create_named_meas_refs(ctx, rt, {mrec}));
    }
    return ColumnArgs{
      name(),
      TableField(
        datatype(),
        m_fid,
        Keywords::create(ctx, rt, keywords()),
        mr,
        ref_column),
      Axes<AxesT>::uid,
      map_to_int(axes()),
      itree
    };
  }

protected:

  /**
   * Increase the index space by one row
   *
   * Used by add_row() to append one row to the column index space
   *
   * @param[in] element_tree IndexTreeL representing the element shape
   */
  void
  set_next_row(const IndexTreeL& element_tree) {
    auto row_index = m_num_rows++;
    m_index_tree =
      m_index_tree.merged_with(IndexTreeL({{row_index, 1, element_tree}}));
  }

private:

  std::string m_name;

  TypeTag m_datatype;

  unsigned m_fid;

  std::vector<AxesT> m_axes;

  size_t m_num_rows;

  IndexTreeL m_index_tree;

  std::optional<
    std::tuple<
      hyperion::MClass,
      std::vector<std::tuple<std::unique_ptr<casacore::MRBase>, unsigned>>,
      std::optional<std::string>>>
  m_meas_record;
};

/**
 * ColumnBuilder for a scalar MS column
 */
template <MSTables D>
class ScalarColumnBuilder
  : public ColumnBuilder<D> {
public:

  ScalarColumnBuilder(const std::string& name, TypeTag datatype, unsigned fid)
    : ColumnBuilder<D>(name, datatype, fid, {MSTable<D>::ROW_AXIS}) {
  }

  /**
   * Create a ScalarColumnBuilder generator function
   *
   *  @return lambda that returns a std::unique_ptr<ScalarColumnBuilder<D>>
   */
  template <typename T>
  static auto
  generator(
    const std::string& name /**< [in] column name */,
    unsigned fid /**< values field id */) {
    return
      [=]() {
        return
          std::make_unique<ScalarColumnBuilder<D>>(
            name,
            ValueType<T>::DataType,
            fid);
      };
  }

  virtual ~ScalarColumnBuilder() {}

  void
  add_row(const std::any&) override {
    ColumnBuilder<D>::set_next_row(IndexTreeL());
  }
};

/**
 * ColumnBuilder for an array MS column
 */
template <MSTables D, int ARRAYDIM>
class ArrayColumnBuilder
  : public ColumnBuilder<D> {
public:

  typedef typename MSTable<D>::Axes AxesT;

  ArrayColumnBuilder(
    const std::string& name,
    TypeTag datatype,
    unsigned fid,
    const std::vector<AxesT>& axes,
    std::function<std::array<size_t, ARRAYDIM>(const std::any&)> element_shape)
    : ColumnBuilder<D>(name, datatype, fid, axes)
    , m_element_shape(element_shape) {

    assert(axes.size() > 0);
  }

  virtual ~ArrayColumnBuilder() {}

  /**
   * Create an ArrayColumnBuilder generator function
   *
   *  @return lambda that takes a vector of AxesT, and returns a
   *  std::unique_ptr<ArrayColumnBuilder<D>>
   */
  template <typename T>
  static auto
  generator(
    const std::string& name, /**< [in] column name */
    unsigned fid, /**< [in] column value field id */
    std::function<std::array<size_t, ARRAYDIM>(const std::any&)>
    element_shape /**< [in] conversion from cell shape to array of sizes */) {

    return
      [=](const std::vector<AxesT>& axes) {
        return std::make_unique<ArrayColumnBuilder<D, ARRAYDIM>>(
          name,
          ValueType<T>::DataType,
          fid,
          axes,
          element_shape);
      };
  }

  void
  add_row(const std::any& args) override {
    auto ary = m_element_shape(args);
    IndexTreeL t =
      std::accumulate(
        ary.rbegin(),
        ary.rend(),
        IndexTreeL(),
        [](const auto& t, const auto& d) {
          return IndexTreeL({{d, t}});
        });
    ColumnBuilder<D>::set_next_row(t);
  }

private:

  std::function<std::array<size_t, ARRAYDIM>(const std::any&)> m_element_shape;
};

} // end namespace hyperion

#endif // HYPERION_COLUMN_BUILDER_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
