#ifndef LEGMS_MS_COLUMN_H_
#define LEGMS_MS_COLUMN_H_

#include <cassert>
#include <functional>
#include <mutex>
#include <tuple>
#include <unordered_map>

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

  typedef std::function<
    std::shared_ptr<Column>(Legion::Context, Legion::Runtime*)> Generator;

  Column(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const ColumnBuilder& builder)
    : WithKeywords(builder.keywords())
    , m_name(builder.name())
    , m_datatype(builder.datatype())
    , m_num_rows(builder.num_rows())
    , m_row_index_pattern(builder.row_index_pattern())
    , m_index_tree(builder.index_tree())
    , m_context(ctx)
    , m_runtime(runtime) {
  }

  Column(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    casacore::DataType datatype,
    const IndexTreeL& row_index_pattern,
    const IndexTreeL& index_tree,
    const std::unordered_map<std::string, casacore::DataType>& kws =
      std::unordered_map<std::string, casacore::DataType>())
    : WithKeywords(kws)
    , m_name(name)
    , m_datatype(datatype)
    , m_num_rows(nr(row_index_pattern, index_tree).value())
    , m_row_index_pattern(row_index_pattern)
    , m_index_tree(index_tree)
    , m_context(ctx)
    , m_runtime(runtime) {
  }

  Column(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    casacore::DataType datatype,
    const IndexTreeL& row_index_pattern,
    unsigned num_rows,
    const std::unordered_map<std::string, casacore::DataType>& kws =
      std::unordered_map<std::string, casacore::DataType>())
    : WithKeywords(kws)
    , m_name(name)
    , m_datatype(datatype)
    , m_num_rows(num_rows)
    , m_row_index_pattern(row_index_pattern)
    , m_index_tree(ixt(row_index_pattern, num_rows))
    , m_context(ctx)
    , m_runtime(runtime) {
  }

  Column(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    casacore::DataType datatype,
    const IndexTreeL& row_index_pattern,
    const IndexTreeL& row_pattern,
    unsigned num_rows,
    const std::unordered_map<std::string, casacore::DataType>& kws =
      std::unordered_map<std::string, casacore::DataType>())
    : WithKeywords(kws)
    , m_name(name)
    , m_datatype(datatype)
    , m_num_rows(num_rows)
    , m_row_index_pattern(row_index_pattern)
    , m_index_tree(
      ixt(
        row_pattern,
        num_rows * row_pattern.size() / row_index_pattern.size()))
    , m_context(ctx)
    , m_runtime(runtime) {

    assert(pattern_matches(row_index_pattern, row_pattern));
  }

  virtual ~Column() {
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    if (m_index_space)
      m_runtime->destroy_index_space(m_context, m_index_space.value());
    if (m_logical_region)
      m_runtime->destroy_logical_region(m_context, m_logical_region.value());
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
  row_index_pattern() const {
    return m_row_index_pattern;
  }

  unsigned
  row_rank() const {
    return m_row_index_pattern.rank().value();
  }

  unsigned
  rank() const {
    return m_index_tree.rank().value();
  }

  size_t
  num_rows() const {
    return m_num_rows;
  }

  Legion::IndexSpace
  index_space() const {
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    if (!m_index_space)
      m_index_space =
        legms::tree_index_space(m_index_tree, m_context, m_runtime);
    return m_index_space.value();
  }

  Legion::FieldID
  add_field(Legion::FieldSpace fs, Legion::FieldAllocator fa) const {

    auto result = legms::add_field(m_datatype, fa);
    m_runtime->attach_name(fs, result, name().c_str());
    return result;
  }

  Legion::IndexPartition
  projected_index_partition(const Legion::IndexPartition&) const;

  std::tuple<Legion::LogicalRegion, Legion::FieldID>
  logical_region() const {
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    if (!m_logical_region) {
      Legion::FieldSpace fs = m_runtime->create_field_space(m_context);
      auto fa = m_runtime->create_field_allocator(m_context, fs);
      m_field_id = add_field(fs, fa);
      m_logical_region =
        m_runtime->create_logical_region(m_context, index_space(), fs);
    }
    return std::make_tuple(m_logical_region.value(), m_field_id);
  }

  static Generator
  generator(
    const std::string& name,
    casacore::DataType datatype,
    const IndexTreeL& row_index_pattern,
    const IndexTreeL& index_tree,
    const std::unordered_map<std::string, casacore::DataType>& kws =
      std::unordered_map<std::string, casacore::DataType>()) {

    return
      [=](Legion::Context ctx, Legion::Runtime* runtime) {
        return
          std::make_shared<Column>(
            ctx,
            runtime,
            name,
            datatype,
            row_index_pattern,
            index_tree,
            kws);
      };
  }

  static Generator
  generator(
    const std::string& name,
    casacore::DataType datatype,
    const IndexTreeL& row_index_pattern,
    unsigned num_rows,
    const std::unordered_map<std::string, casacore::DataType>& kws =
      std::unordered_map<std::string, casacore::DataType>()) {

    return
      [=](Legion::Context ctx, Legion::Runtime* runtime) {
        return
          std::make_shared<Column>(
            ctx,
            runtime,
            name,
            datatype,
            row_index_pattern,
            num_rows,
            kws);
      };
  }

  static Generator
  generator(
    const std::string& name,
    casacore::DataType datatype,
    const IndexTreeL& row_index_pattern,
    const IndexTreeL& row_pattern,
    unsigned num_rows,
    const std::unordered_map<std::string, casacore::DataType>& kws =
      std::unordered_map<std::string, casacore::DataType>()) {

    return
      [=](Legion::Context ctx, Legion::Runtime* runtime) {
        return
          std::make_shared<Column>(
            ctx,
            runtime,
            name,
            datatype,
            row_index_pattern,
            row_pattern,
            num_rows,
            kws);
      };
  }

private:

  static std::optional<size_t>
  nr(
    const IndexTreeL& row_pattern,
    const IndexTreeL& full_shape,
    bool cycle = true);

  static bool
  pattern_matches(const IndexTreeL& pattern, const IndexTreeL& shape);

  static IndexTreeL
  ixt(const IndexTreeL& row_pattern, size_t num);

  std::string m_name;

  casacore::DataType m_datatype;

  size_t m_num_rows;

  IndexTreeL m_row_index_pattern;

  IndexTreeL m_index_tree;

  Legion::Context m_context;

  Legion::Runtime* m_runtime;

  mutable std::recursive_mutex m_mutex;

  mutable std::optional<Legion::IndexSpace> m_index_space;

  mutable std::optional<Legion::LogicalRegion> m_logical_region;

  mutable Legion::FieldID m_field_id;
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
