/*
 * Copyright 2019 Associated Universities, Inc. Washington DC, USA.
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
# include <hyperion/IndexTree.h>
# include <hyperion/TableField.h>
#ifdef HYPERION_USE_CASACORE
# include <hyperion/utility.h>
# include <hyperion/KeywordsBuilder.h>
# include <hyperion/MSTable.h>
# include <hyperion/MeasRef.h>

# pragma GCC visibility push(default)
#  include <any>
#  include <cassert>
#  include <functional>
#  include <memory>
#  include <numeric>
#  include <optional>
#  include <vector>
# pragma GCC visibility pop

# include <casacore/casa/BasicSL/String.h>
# include <casacore/casa/Containers/Record.h>
# include <casacore/measures/Measures/MeasureHolder.h>

namespace hyperion {

struct ColumnArgs {
  std::string name;
  TableField tf;
  std::string axes_uid;
  std::vector<int> axes;
  IndexTreeL index_tree;
};

template <MSTables D>
class ColumnBuilder
  : public KeywordsBuilder {
public:

  typedef typename MSTable<D>::Axes AxesT;

  ColumnBuilder(
    const std::string& name,
    TypeTag datatype,
    unsigned fid,
    const std::vector<AxesT>& axes)
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

  const std::string&
  name() const {
    return m_name;
  }

  TypeTag
  datatype() const {
    return m_datatype;
  }

  const std::vector<AxesT>&
  axes() const {
    return m_axes;
  }

  const IndexTreeL&
  index_tree() const {
    return m_index_tree;
  }

  unsigned
  rank() const {
    return m_axes.size();
  }

  size_t
  num_rows() const {
    return m_num_rows;
  }

  bool
  empty() const {
    return index_tree().size() == 0;
  }

  const std::optional<
    std::tuple<
      hyperion::MClass,
      std::vector<std::tuple<std::unique_ptr<casacore::MRBase>, unsigned>>,
      std::optional<std::string>>>&
  meas_record() const {
    return m_meas_record;
  }

  void
  set_meas_record(
    std::tuple<
      hyperion::MClass,
      std::vector<
      std::tuple<std::unique_ptr<casacore::MRBase>, unsigned>>,
      std::optional<std::string>>&& rec) {
    m_meas_record = std::move(rec);
  }

  virtual void
  add_row(const std::any&) = 0;

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
        Keywords::create(ctx, rt, keywords())
#ifdef HYPERION_USE_CASACORE
        , mr
        , ref_column
#endif
        ),
      Axes<AxesT>::uid,
      map_to_int(axes()),
      itree
    };
  }

protected:

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

template <MSTables D>
class ScalarColumnBuilder
  : public ColumnBuilder<D> {
public:

  ScalarColumnBuilder(const std::string& name, TypeTag datatype, unsigned fid)
    : ColumnBuilder<D>(name, datatype, fid, {MSTable<D>::ROW_AXIS}) {
  }

  template <typename T>
  static auto
  generator(const std::string& name, unsigned fid) {
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

  template <typename T>
  static auto
  generator(
    const std::string& name,
    unsigned fid,
    std::function<std::array<size_t, ARRAYDIM>(const std::any&)>
    element_shape) {

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

#endif // HYPERION_USE_CASACORE

#endif // HYPERION_COLUMN_BUILDER_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
