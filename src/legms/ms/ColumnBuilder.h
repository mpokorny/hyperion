#ifndef LEGMS_MS_COLUMN_BUILDER_H_
#define LEGMS_MS_COLUMN_BUILDER_H_

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
namespace ms {

template <typename D>
class ColumnBuilder
  : public WithKeywordsBuilder {
public:

  ColumnBuilder(
    const std::string& name,
    casacore::DataType datatype,
    const std::vector<D>& axes,
    const IndexTreeL& row_index_pattern)
    : WithKeywordsBuilder()
    , m_name(name)
    , m_datatype(datatype)
    , m_axes(axes)
    , m_num_rows(0)
    , m_row_index_pattern(row_index_pattern)
    , m_row_index_iterator(row_index_pattern) {

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

  const IndexTreeL&
  row_index_pattern() const {
    return m_row_index_pattern;
  }

  unsigned
  rank() const {
    return m_axes.size();
  }

  unsigned
  row_rank() const {
    return m_row_index_pattern.rank().value();
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
    auto row_index = *m_row_index_iterator;
    ++m_row_index_iterator;
    ++m_num_rows;
    IndexTreeL result =
      std::accumulate(
        row_index.rbegin(),
        row_index.rend(),
        element_tree,
        [](const auto& t, const auto& i) {
          return IndexTreeL({{i, 1, t}});
        });
    m_index_tree = m_index_tree.merged_with(result);
  }

private:

  std::string m_name;

  casacore::DataType m_datatype;

  std::vector<D> m_axes;

  size_t m_num_rows;

  IndexTreeL m_row_index_pattern;

  IndexTreeIterator<Legion::coord_t> m_row_index_iterator;

  IndexTreeL m_index_tree;
};

template <typename D>
class ScalarColumnBuilder
  : public ColumnBuilder<D> {
public:

    ScalarColumnBuilder(
      const std::string& name,
      casacore::DataType datatype,
      const std::vector<D>& axes,
      const IndexTreeL& row_index_pattern)
      : ColumnBuilder<D>(name, datatype, axes, row_index_pattern) {

      assert(axes.size() == row_index_pattern.rank().value());
    }

  template <typename T>
  static auto
  generator(const std::string& name) {
    return
      [=](const std::vector<D>& axes, const IndexTreeL& row_index_pattern) {
      return std::make_unique<ScalarColumnBuilder<D>>(
        name,
        ValueType<T>::DataType,
        axes,
        row_index_pattern);
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
    const IndexTreeL& row_index_pattern,
    std::function<std::array<size_t, ARRAYDIM>(const std::any&)> element_shape)
    : ColumnBuilder<D>(name, datatype, axes, row_index_pattern)
    , m_element_shape(element_shape) {

    assert(axes.size() > row_index_pattern.rank().value());
  }

  virtual ~ArrayColumnBuilder() {}

  template <typename T>
  static auto
  generator(
    const std::string& name,
    std::function<std::array<size_t, ARRAYDIM>(const std::any&)>
    element_shape) {

    return
      [=](const std::vector<D>& axes, const IndexTreeL& row_index_pattern) {
        return std::make_unique<ArrayColumnBuilder<D, ARRAYDIM>>(
          name,
          ValueType<T>::DataType,
          axes,
          row_index_pattern,
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

} // end namespace ms
} // end namespace legms

#endif // LEGMS_MS_COLUMN_BUILDER_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
