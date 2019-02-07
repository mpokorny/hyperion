#ifndef LEGMS_COLUMN_H_
#define LEGMS_COLUMN_H_

#include <cassert>
#include <functional>
#include <memory>
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
#include "ColumnPartition.h"

namespace legms {

class Column
  : public WithKeywords {
public:

  typedef casacore::uInt row_number_t;

  virtual ~Column() {
    if (m_index_space != Legion::IndexSpace::NO_SPACE)
      m_runtime->destroy_index_space(m_context, m_index_space);
    if (m_logical_region != Legion::LogicalRegion::NO_REGION)
      m_runtime->destroy_logical_region(m_context, m_logical_region);
  };

  const std::string&
  name() const {
    return m_name;
  }

  unsigned
  row_rank() const {
    return m_row_index_pattern.rank().value();
  }

  unsigned
  rank() const {
    return m_rank;
  }

  size_t
  num_rows() const {
    return m_num_rows;
  }

  const IndexTreeL&
  index_tree() const {
    return m_index_tree;
  }

  const IndexTreeL&
  row_index_pattern() const {
    return m_row_index_pattern;
  }

  casacore::DataType
  datatype() const {
    return m_datatype;
  }

  const Legion::IndexSpace&
  index_space() const {
    return m_index_space;
  }

  const Legion::LogicalRegion&
  logical_region() const {
    return m_logical_region;
  }

  virtual std::unique_ptr<ColumnPartition>
  projected_column_partition(const ColumnPartition* cp) const = 0;

  static constexpr Legion::FieldID value_fid = 0;

  static constexpr Legion::FieldID row_number_fid = 1;

  static void
  register_tasks(Legion::Runtime *runtime);

protected:

  Column(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    casacore::DataType datatype,
    unsigned num_rows,
    const IndexTreeL& row_index_pattern,
    const IndexTreeL& index_tree,
    const std::unordered_map<std::string, casacore::DataType>& kws)
    : WithKeywords(kws)
    , m_context(ctx)
    , m_runtime(runtime)
    , m_name(name)
    , m_datatype(datatype)
    , m_rank(index_tree.rank().value())
    , m_num_rows(num_rows)
    , m_row_index_pattern(row_index_pattern)
    , m_index_tree(index_tree) {

    init();
  }

  static bool
  pattern_matches(const IndexTreeL& pattern, const IndexTreeL& shape);

  Legion::Context m_context;

  Legion::Runtime* m_runtime;

private:

  void
  init();

  std::string m_name;

  casacore::DataType m_datatype;

  unsigned m_rank;

  size_t m_num_rows;

  IndexTreeL m_row_index_pattern;

  IndexTreeL m_index_tree;

  Legion::IndexSpace m_index_space;

  Legion::LogicalRegion m_logical_region;
};

template <typename D>
class ColumnT
  : public Column {
public:

  typedef std::function<
  std::shared_ptr<ColumnT<D>>(Legion::Context, Legion::Runtime*)> Generator;

  ColumnT(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const ColumnBuilder<D>& builder)
    : Column(
      ctx,
      runtime,
      builder.name(),
      builder.datatype(),
      builder.num_rows(),
      builder.row_index_pattern(),
      builder.index_tree(),
      builder.keywords())
    , m_axes(builder.axes()) {}

  ColumnT(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    casacore::DataType datatype,
    const std::vector<D>& axes,
    const IndexTreeL& row_index_pattern,
    const IndexTreeL& index_tree_,
    const std::unordered_map<std::string, casacore::DataType>& kws =
    std::unordered_map<std::string, casacore::DataType>())
    : Column(
      ctx,
      runtime,
      name,
      datatype,
      index_tree_.num_repeats(row_index_pattern).value(),
      row_index_pattern,
      index_tree_,
      kws)
    , m_axes(axes) {}

  virtual ~ColumnT() {}

  const std::vector<D>&
  axes() const {
    return m_axes;
  }

  std::unique_ptr<ColumnPartition>
  projected_column_partition(const ColumnPartition* cp) const override {

    const ColumnPartitionT<D>* cpt =
      dynamic_cast<const ColumnPartitionT<D>*>(cp);

    if (index_space() == Legion::IndexSpace::NO_SPACE)
      return
        std::make_unique<ColumnPartitionT<D>>(
          m_context,
          m_runtime,
          Legion::IndexPartition::NO_PART,
          m_axes);

    std::vector<int> dmap = dimensions_map(m_axes, cpt->axes());

    switch (cpt->axes().size()) {
#if MAX_DIM >= 1
    case 1:
      switch (rank()) {
#if MAX_DIM >= 1
      case 1:
        return
          std::make_unique<ColumnPartitionT<D>>(
            m_context,
            m_runtime,
            legms::projected_index_partition<1, 1>(
              m_context,
              m_runtime,
              Legion::IndexPartitionT<1>(cpt->index_partition()),
              Legion::IndexSpaceT<1>(index_space()),
              {dmap[0]}),
            m_axes);
        break;
#endif
#if MAX_DIM >= 2
      case 2:
        return
          std::make_unique<ColumnPartitionT<D>>(
            m_context,
            m_runtime,
            legms::projected_index_partition<1, 2>(
              m_context,
              m_runtime,
              Legion::IndexPartitionT<1>(cpt->index_partition()),
              Legion::IndexSpaceT<2>(index_space()),
              {dmap[0], dmap[1]}),
            m_axes);
        break;
#endif
#if MAX_DIM >= 3
      case 3:
        return
          std::make_unique<ColumnPartitionT<D>>(
            m_context,
            m_runtime,
            legms::projected_index_partition<1, 3>(
              m_context,
              m_runtime,
              Legion::IndexPartitionT<1>(cpt->index_partition()),
              Legion::IndexSpaceT<3>(index_space()),
              {dmap[0], dmap[1], dmap[2]}),
            m_axes);
        break;
#endif
#if MAX_DIM >= 4
      case 4:
        return
          std::make_unique<ColumnPartitionT<D>>(
            m_context,
            m_runtime,
            legms::projected_index_partition<1, 4>(
              m_context,
              m_runtime,
              Legion::IndexPartitionT<1>(cpt->index_partition()),
              Legion::IndexSpaceT<4>(index_space()),
              {dmap[0], dmap[1], dmap[2], dmap[3]}),
            m_axes);
        break;
#endif
      default:
        assert(false);
        break;
      }
      break;
#endif
#if MAX_DIM >= 2
    case 2:
      switch (rank()) {
#if MAX_DIM >= 1
      case 1:
        return
          std::make_unique<ColumnPartitionT<D>>(
            m_context,
            m_runtime,
            legms::projected_index_partition<2, 1>(
              m_context,
              m_runtime,
              Legion::IndexPartitionT<2>(cpt->index_partition()),
              Legion::IndexSpaceT<1>(index_space()),
              {dmap[0]}),
            m_axes);
        break;
#endif
#if MAX_DIM >= 2
      case 2:
        return
          std::make_unique<ColumnPartitionT<D>>(
            m_context,
            m_runtime,
            legms::projected_index_partition<2, 2>(
              m_context,
              m_runtime,
              Legion::IndexPartitionT<2>(cpt->index_partition()),
              Legion::IndexSpaceT<2>(index_space()),
              {dmap[0], dmap[1]}),
            m_axes);
        break;
#endif
#if MAX_DIM >= 3
      case 3:
        return
          std::make_unique<ColumnPartitionT<D>>(
            m_context,
            m_runtime,
            legms::projected_index_partition<2, 3>(
              m_context,
              m_runtime,
              Legion::IndexPartitionT<2>(cpt->index_partition()),
              Legion::IndexSpaceT<3>(index_space()),
              {dmap[0], dmap[1], dmap[2]}),
            m_axes);
        break;
#endif
#if MAX_DIM >= 4
      case 4:
        return
          std::make_unique<ColumnPartitionT<D>>(
            m_context,
            m_runtime,
            legms::projected_index_partition<2, 4>(
              m_context,
              m_runtime,
              Legion::IndexPartitionT<2>(cpt->index_partition()),
              Legion::IndexSpaceT<4>(index_space()),
              {dmap[0], dmap[1], dmap[2], dmap[3]}),
            m_axes);
        break;
#endif
      default:
        assert(false);
        break;
      }
      break;
#endif
#if MAX_DIM >= 3
    case 3:
      switch (rank()) {
#if MAX_DIM >= 1
      case 1:
        return
          std::make_unique<ColumnPartitionT<D>>(
            m_context,
            m_runtime,
            legms::projected_index_partition<3, 1>(
              m_context,
              m_runtime,
              Legion::IndexPartitionT<3>(cpt->index_partition()),
              Legion::IndexSpaceT<1>(index_space()),
              {dmap[0]}),
            m_axes);
        break;
#endif
#if MAX_DIM >= 2
      case 2:
        return
          std::make_unique<ColumnPartitionT<D>>(
            m_context,
            m_runtime,
            legms::projected_index_partition<3, 2>(
              m_context,
              m_runtime,
              Legion::IndexPartitionT<3>(cpt->index_partition()),
              Legion::IndexSpaceT<2>(index_space()),
              {dmap[0], dmap[1]}),
            m_axes);
        break;
#endif
#if MAX_DIM >= 3
      case 3:
        return
          std::make_unique<ColumnPartitionT<D>>(
            m_context,
            m_runtime,
            legms::projected_index_partition<3, 3>(
              m_context,
              m_runtime,
              Legion::IndexPartitionT<3>(cpt->index_partition()),
              Legion::IndexSpaceT<3>(index_space()),
              {dmap[0], dmap[1], dmap[2]}),
            m_axes);
        break;
#endif
#if MAX_DIM >= 4
      case 4:
        return
          std::make_unique<ColumnPartitionT<D>>(
            m_context,
            m_runtime,
            legms::projected_index_partition<3, 4>(
              m_context,
              m_runtime,
              Legion::IndexPartitionT<3>(cpt->index_partition()),
              Legion::IndexSpaceT<4>(index_space()),
              {dmap[0], dmap[1], dmap[2], dmap[3]}),
            m_axes);
        break;
#endif
      default:
        assert(false);
        break;
      }
      break;
#endif
#if MAX_DIM >= 4
    case 4:
      switch (rank()) {
#if MAX_DIM >= 1
      case 1:
        return
          std::make_unique<ColumnPartitionT<D>>(
            m_context,
            m_runtime,
            legms::projected_index_partition<4, 1>(
              m_context,
              m_runtime,
              Legion::IndexPartitionT<4>(cpt->index_partition()),
              Legion::IndexSpaceT<1>(index_space()),
              {dmap[0]}),
            m_axes);
        break;
#endif
#if MAX_DIM >= 2
      case 2:
        return
          std::make_unique<ColumnPartitionT<D>>(
            m_context,
            m_runtime,
            legms::projected_index_partition<4, 2>(
              m_context,
              m_runtime,
              Legion::IndexPartitionT<4>(cpt->index_partition()),
              Legion::IndexSpaceT<2>(index_space()),
              {dmap[0], dmap[1]}),
            m_axes);
        break;
#endif
#if MAX_DIM >= 3
      case 3:
        return
          std::make_unique<ColumnPartitionT<D>>(
            m_context,
            m_runtime,
            legms::projected_index_partition<4, 3>(
              m_context,
              m_runtime,
              Legion::IndexPartitionT<4>(cpt->index_partition()),
              Legion::IndexSpaceT<3>(index_space()),
              {dmap[0], dmap[1], dmap[2]}),
            m_axes);
        break;
#endif
#if MAX_DIM >= 4
      case 4:
        return
          std::make_unique<ColumnPartitionT<D>>(
            m_context,
            m_runtime,
            legms::projected_index_partition<4, 4>(
              m_context,
              m_runtime,
              Legion::IndexPartitionT<4>(cpt->index_partition()),
              Legion::IndexSpaceT<4>(index_space()),
              {dmap[0], dmap[1], dmap[2], dmap[3]}),
            m_axes);
        break;
#endif
      default:
        assert(false);
        break;
      }
      break;
#endif
    default:
      assert(false);
      break;
    }

    return
      std::make_unique<ColumnPartitionT<D>>(
        m_context,
        m_runtime,
        Legion::IndexPartition::NO_PART,
        m_axes);
  }

  static Generator
  generator(
    const std::string& name,
    casacore::DataType datatype,
    const std::vector<D>& axes,
    const IndexTreeL& row_index_pattern,
    const IndexTreeL& index_tree,
    const std::unordered_map<std::string, casacore::DataType>& kws =
    std::unordered_map<std::string, casacore::DataType>()) {

    return
      [=](Legion::Context ctx, Legion::Runtime* runtime) {
        return
          std::make_shared<ColumnT<D>>(
            ctx,
            runtime,
            name,
            datatype,
            axes,
            row_index_pattern,
            index_tree,
            kws);
      };
  }

  static Generator
  generator(
    const std::string& name,
    casacore::DataType datatype,
    const std::vector<D>& axes,
    const IndexTreeL& row_index_pattern,
    unsigned num_rows,
    const std::unordered_map<std::string, casacore::DataType>& kws =
    std::unordered_map<std::string, casacore::DataType>()) {

    return
      generator(
        name,
        datatype,
        axes,
        row_index_pattern,
        IndexTreeL(row_index_pattern, num_rows),
        kws);
  }

  static Generator
  generator(
    const std::string& name,
    casacore::DataType datatype,
    const std::vector<D>& axes,
    const IndexTreeL& row_index_pattern,
    const IndexTreeL& row_pattern,
    unsigned num_rows,
    const std::unordered_map<std::string, casacore::DataType>& kws =
    std::unordered_map<std::string, casacore::DataType>()) {

    return
      generator(
        name,
        datatype,
        axes,
        row_index_pattern,
        IndexTreeL(
          row_pattern,
          num_rows * row_pattern.size() / row_index_pattern.size()),
        kws);
  }

private:

  std::vector<D> m_axes;
};

} // end namespace legms

#endif // LEGMS_COLUMN_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
