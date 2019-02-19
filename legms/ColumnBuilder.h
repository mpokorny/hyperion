#ifndef LEGMS_COLUMN_BUILDER_H_
#define LEGMS_COLUMN_BUILDER_H_

#include <any>
#include <cassert>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>

#include <casacore/casa/aipstype.h>
#include <casacore/casa/Utilities/DataType.h>
#include "legion.h"

#include "utility.h"
#include "WithKeywordsBuilder.h"
#include "IndexTree.h"

namespace legms {

template <typename D>
class ColumnBuilder
  : public WithKeywordsBuilder {
public:

  ColumnBuilder(
    const std::string& name,
    casacore::DataType datatype,
    const std::vector<D>& axes)
    : WithKeywordsBuilder()
    , m_name(name)
    , m_datatype(datatype)
    , m_axes(axes)
    , m_num_rows(0) {

    assert(axes[0] == D::ROW);
    assert(has_unique_values(m_axes));
  }

  virtual ~ColumnBuilder() {}

  const std::string&
  name() const {
    return m_name;
  }

  casacore::DataType
  datatype() const {
    return m_datatype;
  }

  const std::vector<D>&
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

  virtual void
  add_row(const std::any&) = 0;

protected:

  void
  set_next_row(const IndexTreeL& element_tree) {
    auto row_index = m_num_rows++;
    m_index_tree =
      m_index_tree.merged_with(IndexTreeL({{row_index, 1, element_tree}}));
  }

private:

  std::string m_name;

  casacore::DataType m_datatype;

  std::vector<D> m_axes;

  size_t m_num_rows;

  IndexTreeL m_index_tree;
};

template <typename D>
class ScalarColumnBuilder
  : public ColumnBuilder<D> {
public:

  ScalarColumnBuilder(
    const std::string& name,
    casacore::DataType datatype)
    : ColumnBuilder<D>(name, datatype, {D::ROW}) {
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

template <typename D, int ARRAYDIM>
class ArrayColumnBuilder
  : public ColumnBuilder<D> {
public:

  ArrayColumnBuilder(
    const std::string& name,
    casacore::DataType datatype,
    const std::vector<D>& axes,
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
      [=](const std::vector<D>& axes) {
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

} // end namespace legms

#endif // LEGMS_COLUMN_BUILDER_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
