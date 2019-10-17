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

#ifdef HYPERION_USE_CASACORE

#pragma GCC visibility push(default)
#include <any>
#include <cassert>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>
#pragma GCC visibility pop

#include <hyperion/hyperion.h>
#include <hyperion/utility.h>
#include <hyperion/KeywordsBuilder.h>
#include <hyperion/IndexTree.h>
#include <hyperion/Column.h>
#include <hyperion/MSTable.h>

#include <casacore/casa/BasicSL/String.h>
#include <casacore/casa/Containers/Record.h>
#include <casacore/measures/Measures/MeasureHolder.h>

namespace hyperion {

template <MSTables D>
class ColumnBuilder
  : public KeywordsBuilder {
public:

  typedef typename MSTable<D>::Axes Axes;

  ColumnBuilder(
    const std::string& name,
    TypeTag datatype,
    const std::vector<Axes>& axes)
    : KeywordsBuilder()
    , m_name(name)
    , m_datatype(datatype)
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

  const std::vector<Axes>&
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

  void
  add_meas_record(const casacore::Record& rec) {
    m_meas_records.push_back(rec);
  }

  virtual void
  add_row(const std::any&) = 0;

  Column
  column(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::string& name_prefix,
    const MeasRefContainer& inherited_meas_ref) const {

    IndexTreeL itree;
    auto itrank = index_tree().rank();
    if (itrank && itrank.value() == rank())
      itree = index_tree();
    std::vector<MeasRef> meas_refs;
    std::for_each(
      m_meas_records.begin(),
      m_meas_records.end(),
      [&meas_refs, &ctx, rt](const casacore::RecordInterface& rec) {
        casacore::MeasureHolder mh;
        casacore::String err;
        auto converted = mh.fromType(err, rec);
        if (converted) {
          if (false) {}
#define MK_MR(MC)                                 \
          else if (MClassT<MC>::holds(mh)) {      \
            auto m = MClassT<MC>::get(mh);        \
            meas_refs.push_back(                  \
              MeasRef::create<MClassT<MC>::type>( \
                ctx,                              \
                rt,                               \
                MClassT<MC>::name,                \
                m.getRef()));                     \
          }
          HYPERION_FOREACH_MCLASS(MK_MR)
#undef MK_MR
          else { assert(false); }
        }
      });
    return
      Column::create(
        ctx,
        rt,
        name(),
        axes(),
        datatype(),
        itree,
#ifdef HYPERION_USE_CASACORE
        MeasRefContainer::create(ctx, rt, meas_refs, inherited_meas_ref),
#endif
        keywords(),
        name_prefix);
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

  std::vector<Axes> m_axes;

  size_t m_num_rows;

  IndexTreeL m_index_tree;

  std::vector<casacore::Record> m_meas_records;
};

template <MSTables D>
class ScalarColumnBuilder
  : public ColumnBuilder<D> {
public:

  ScalarColumnBuilder(
    const std::string& name,
    TypeTag datatype)
    : ColumnBuilder<D>(name, datatype, {MSTable<D>::ROW_AXIS}) {
  }

  template <typename T>
  static auto
  generator(const std::string& name) {
    return
      [=]() {
        return std::make_unique<ScalarColumnBuilder<D>>(
          name,
          ValueType<T>::DataType);
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

  typedef typename MSTable<D>::Axes Axes;

  ArrayColumnBuilder(
    const std::string& name,
    TypeTag datatype,
    const std::vector<Axes>& axes,
    std::function<std::array<size_t, ARRAYDIM>(const std::any&)> element_shape)
    : ColumnBuilder<D>(name, datatype, axes)
    , m_element_shape(element_shape) {

    assert(axes.size() > 0);
  }

  virtual ~ArrayColumnBuilder() {}

  template <typename T>
  static auto
  generator(
    const std::string& name,
    std::function<std::array<size_t, ARRAYDIM>(const std::any&)>
    element_shape) {

    return
      [=](const std::vector<Axes>& axes) {
        return std::make_unique<ArrayColumnBuilder<D, ARRAYDIM>>(
          name,
          ValueType<T>::DataType,
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