#ifndef LEGMS_MS_COLUMN_BUILDER_H_
#define LEGMS_MS_COLUMN_BUILDER_H_

#include <any>
#include <cassert>
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

class ColumnBuilder
  : public WithKeywordsBuilder {
public:

  ColumnBuilder(
    const std::string& name,
    casacore::DataType datatype,
    unsigned row_rank,
    unsigned element_rank,
    const IndexTreeL& row_index_shape,
    std::optional<Legion::FieldID> fid = std::nullopt)
    : WithKeywordsBuilder()
    , m_name(name)
    , m_datatype(datatype)
    , m_row_rank(row_rank)
    , m_rank(row_rank + element_rank)
    , m_row_index_shape(row_index_shape)
    , m_row_index_iterator(row_index_shape)
    , m_fid(fid) {

    assert(row_index_shape.rank().value_or(0) == row_rank);
  }

  virtual ~ColumnBuilder() {
  }

  const std::string&
  name() const {
    return m_name;
  }

  casacore::DataType
  datatype() const {
    return m_datatype;
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

  std::optional<Legion::FieldID>
  field_id() const {
    return m_fid;
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

  unsigned m_row_rank;

  unsigned m_rank;

  IndexTreeL m_row_index_shape;

  IndexTreeIterator<Legion::coord_t> m_row_index_iterator;

  IndexTreeL m_index_tree;

  std::optional<Legion::FieldID> m_fid;
};

template <int ROWDIM>
class ScalarColumnBuilder
  : public ColumnBuilder {
public:

  ScalarColumnBuilder(
    const std::string& name,
    casacore::DataType datatype,
    const IndexTreeL& row_index_shape,
    std::optional<Legion::FieldID> fid = std::nullopt)
    : ColumnBuilder(
      name,
      datatype,
      ROWDIM,
      0,
      row_index_shape,
      fid) {
  }

  virtual ~ScalarColumnBuilder() {}

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
    std::optional<Legion::FieldID> fid = std::nullopt)
    : ColumnBuilder(
      name,
      datatype,
      ROWDIM,
      ARRAYDIM,
      row_index_shape,
      fid) {
  }

  virtual ~ArrayColumnBuilder() {}

  void
  add_row(const std::any& args) override {
    auto ary = row_dimensions(args);
    IndexTreeL t =
      std::accumulate(
        ary.rbegin(),
        ary.rend(),
        IndexTreeL(),
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

