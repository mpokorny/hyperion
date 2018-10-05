#ifndef LEGMS_MS_COLUMN_H_
#define LEGMS_MS_COLUMN_H_

#include <casacore/casa/aipstype.h>
#include <casacore/casa/Utilities/DataType.h>
#include "legion.h"

#include "utility.h"
#include "tree_index_space.h"
#include "WithKeywords.h"
#include "IndexTree.h"
#include "ColumnBuilder.h"

namespace legms {
namespace ms {

class Column
  : public WithKeywords {
public:

  Column(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const ColumnBuilder& builder)
    : WithKeywords(builder.keywords())
    , m_name(builder.name())
    , m_datatype(builder.datatype())
    , m_num_rows(builder.num_rows())
    , m_row_index_shape(builder.row_index_shape())
    , m_index_tree(builder.index_tree())
    , m_context(ctx)
    , m_runtime(runtime) {
  }

  Column(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const ColumnBuilder& builder,
    const IndexTreeL& index_tree)
    : WithKeywords(builder.keywords())
    , m_name(builder.name())
    , m_datatype(builder.datatype())
    , m_row_rank(builder.row_rank())
    , m_rank(builder.rank())
    , m_num_rows(builder.num_rows())
    , m_row_index_shape(builder.row_index_shape())
    , m_index_tree(index_tree)
    , m_context(ctx)
    , m_runtime(runtime) {
  }

  virtual ~Column() {
    if (m_index_space)
      m_runtime->destroy_index_space(m_context, m_index_space.value());
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
  row_rank() const {
    return m_row_index_shape.rank().value();
  }

  unsigned
  rank() const {
    return m_index_tree.rank().value();
  }

  size_t
  num_rows() const {
    return m_num_rows;
  }

  std::optional<Legion::IndexSpace>
  index_space() const {
    if (!m_index_space)
      m_index_space =
        legms::tree_index_space(m_index_tree, m_context, m_runtime);
    return m_index_space;
  }

  Legion::FieldID
  add_field(Legion::FieldSpace fs, Legion::FieldAllocator fa) const {

    auto result = legms::add_field(m_datatype, fa);
    m_runtime->attach_name(fs, result, name().c_str());
    return result;
  }

private:

  std::string m_name;

  casacore::DataType m_datatype;

  size_t m_num_rows;

  IndexTreeL m_row_index_shape;

  IndexTreeL m_index_tree;

  Legion::Context m_context;

  Legion::Runtime* m_runtime;

  mutable std::optional<Legion::IndexSpace> m_index_space;
};

} // end namespace ms
} // end namespace legms

#endif // LEGMS_MS_COLUMN_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
