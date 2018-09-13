#ifndef LEGMS_MS_COLUMN_BUILDER_H_
#define LEGMS_MS_COLUMN_BUILDER_H_

#include <any>
#include <cassert>
#include <numeric>
#include <vector>

#include <casacore/casa/Utilities/DataType.h>
#include "legion.h"

#include "utility.h"
#include "WithKeywordsBuilder.h"
#include "IndexTree.h"

namespace legms {
namespace ms {

class ColumnBuilder
  : public WithKeywordsBuilder {
public:

  ColumnBuilder(
    const std::string& name,
    casacore::DataType datatype,
    unsigned row_rank,
    unsigned element_rank,
    const IndexTreeL& row_index_shape,
    const IndexTreeL& index_tree)
    : WithKeywordsBuilder()
    , m_name(name)
    , m_datatype(datatype)
    , m_row_rank(row_rank)
    , m_rank(row_rank + element_rank)
    , m_row_index_shape(row_index_shape)
    , m_row_index_iterator(row_index_shape)
    , m_index_tree(index_tree) {

    assert(index_tree == IndexTreeL()
           || index_tree.rank().value_or(0) == m_rank);
    assert(row_index_shape.rank().value_or(0) == row_rank);
  }

  const std::string&
  name() const {
    return m_name;
  }

  casacore::DataType
  datatype() const {
    return m_datatype;
  }

  const IndexTreeL
  row_index_tree() const {
    return m_index_tree.pruned(m_row_rank - 1);
  }

  const IndexTreeL&
  index_tree() const {
    return m_index_tree;
  }

  const IndexTreeL&
  row_index_shape() const {
    return m_row_index_shape;
  }

  unsigned
  rank() const {
    return m_rank;
  }

  unsigned
  row_rank() const {
    return m_row_rank;
  }

  virtual void
  add_row(const std::any&) = 0;

protected:

  void
  set_next_row(const IndexTreeL& element_tree) {
    auto row_index = *m_row_index_iterator;
    ++m_row_index_iterator;
    IndexTreeL result =
      std::accumulate(
        row_index.rend(),
        row_index.rbegin(),
        element_tree,
        [](const auto& t, const auto& i) {
          return IndexTreeL({{i, 1, t}});
        });
    m_index_tree = std::move(m_index_tree.merged_with(result));
  }

private:

  std::string m_name;

  casacore::DataType m_datatype;

  unsigned m_row_rank;

  unsigned m_rank;

  IndexTreeL m_row_index_shape;

  IndexTreeIterator<Legion::coord_t> m_row_index_iterator;

  IndexTreeL m_index_tree;
};

template <int ROWDIM>
class ScalarColumnBuilder
  : public ColumnBuilder {
public:

  ScalarColumnBuilder(
    const std::string& name,
    casacore::DataType datatype,
    const IndexTreeL& row_index_shape,
    const IndexTreeL& index_tree)
    : ColumnBuilder(name, datatype, ROWDIM, 0, row_index_shape, index_tree) {
  }

  void
  add_row(const std::any&) override {
    set_next_row(IndexTreeL());
  }
};

template <int ROWDIM, int ARRAYDIM>
class ArrayColumnBuilder
  : public ColumnBuilder {
public:

  ArrayColumnBuilder(
    const std::string& name,
    casacore::DataType datatype,
    const IndexTreeL& row_index_shape,
    const IndexTreeL& index_tree)
    : ColumnBuilder(
      name,
      datatype,
      ROWDIM,
      ARRAYDIM,
      row_index_shape,
      index_tree) {
  }

  void
  add_row(const std::any& args) override {
    auto ary = row_dimensions(args);
    auto a = ary.rbegin();
    IndexTreeL t =
      std::accumulate(
        a + 1,
        ary.rend(),
        *a,
        [](const auto& t, const auto& d) {
          return IndexTreeL({{d, t}});
        });
    set_next_row(t);
  }

protected:

  virtual std::array<size_t, ARRAYDIM>
  row_dimensions(const std::any&) = 0;
};

} // end namespace ms
} // end namespace legms

#endif // LEGMS_MS_COLUMN_BUILDER_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
