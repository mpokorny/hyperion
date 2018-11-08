#ifndef LEGMS_MS_COLUMN_H_
#define LEGMS_MS_COLUMN_H_

#include <cassert>
#include <functional>

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

  typedef casacore::uInt row_number_t;

  Column(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const ColumnBuilder& builder);

  Column(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    casacore::DataType datatype,
    const IndexTreeL& row_index_pattern,
    const IndexTreeL& index_tree,
    const std::unordered_map<std::string, casacore::DataType>& kws =
      std::unordered_map<std::string, casacore::DataType>());

  Column(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    casacore::DataType datatype,
    const IndexTreeL& row_index_pattern,
    unsigned num_rows,
    const std::unordered_map<std::string, casacore::DataType>& kws =
      std::unordered_map<std::string, casacore::DataType>());

  Column(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    casacore::DataType datatype,
    const IndexTreeL& row_index_pattern,
    const IndexTreeL& row_pattern,
    unsigned num_rows,
    const std::unordered_map<std::string, casacore::DataType>& kws =
      std::unordered_map<std::string, casacore::DataType>());

  virtual ~Column() {
    m_runtime->destroy_index_space(m_context, m_index_space);
    m_runtime->destroy_logical_region(m_context, m_logical_region);
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

  const Legion::IndexSpace&
  index_space() const {
    return m_index_space;
  }

  Legion::IndexPartition
  projected_index_partition(const Legion::IndexPartition&) const;

  const Legion::LogicalRegion&
  logical_region() const {
    return m_logical_region;
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

  static constexpr Legion::FieldID value_fid = 0;

  static constexpr Legion::FieldID row_number_fid = 1;

  static void
  register_tasks(Legion::Runtime *runtime);

protected:

  void
  init();

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

  Legion::IndexSpace m_index_space;

  Legion::LogicalRegion m_logical_region;
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
