#ifndef LEGMS_COLUMN_H_
#define LEGMS_COLUMN_H_

#include <cassert>
#include <functional>
#include <memory>
#include <tuple>
#include <unordered_map>

#include <casacore/casa/aipstype.h>
#include <casacore/casa/Utilities/DataType.h>

#include "legms.h"
#include "utility.h"
#include "WithKeywords.h"
#include "IndexTree.h"
#include "ColumnBuilder.h"
#include "ColumnPartition.h"

namespace legms {

template <typename T>
class ColumnT;

struct ColumnGenArgs {
  // TODO: should I add a type tag here to catch errors in calling () with the
  // wrong type?
  std::string name;
  casacore::DataType datatype;
  std::vector<int> axes;
  Legion::LogicalRegion values;
  Legion::LogicalRegion keywords;

  template <typename D>
  std::unique_ptr<ColumnT<D>>
  operator()(Legion::Context ctx, Legion::Runtime* runtime) const;

  size_t
  legion_buffer_size(void) const;

  size_t
  legion_serialize(void *buffer) const;

  size_t
  legion_deserialize(const void *buffer);
};

class Column
  : public WithKeywords {
public:

  virtual ~Column() {
    if (m_logical_region != Legion::LogicalRegion::NO_REGION)
      m_runtime->destroy_logical_region(m_context, m_logical_region);
  };

  const std::string&
  name() const {
    return m_name;
  }

  virtual std::vector<int>
  axes() const = 0;

  unsigned
  rank() const {
    return m_rank;
  }

  const IndexTreeL&
  index_tree() const {
    return m_index_tree;
  }

  casacore::DataType
  datatype() const {
    return m_datatype;
  }

  Legion::IndexSpace
  index_space() const {
    return (
      (m_logical_region == Legion::LogicalRegion::NO_REGION)
      ? Legion::IndexSpace::NO_SPACE
      : m_logical_region.get_index_space());
  }

  Legion::LogicalRegion
  logical_region() const {
    return m_logical_region;
  }

  virtual std::unique_ptr<ColumnPartition>
  partition_on_axes(const std::vector<AxisPartition<int>>& axes) const = 0;

  std::unique_ptr<ColumnPartition>
  partition_on_axes(
    const std::vector<std::tuple<int, Legion::coord_t>>& axes) const;

  std::unique_ptr<ColumnPartition>
  partition_on_axes(const std::vector<int>& axes) const;

  virtual std::unique_ptr<ColumnPartition>
  projected_column_partition(const ColumnPartition* cp) const = 0;

  virtual ColumnGenArgs
  generator_args() const = 0;

  Legion::Runtime*
  runtime() const {
    return m_runtime;
  }

  Legion::Context&
  context() const {
    return m_context;
  }

  static constexpr Legion::FieldID value_fid = 0;

protected:

  Column(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    casacore::DataType datatype,
    const IndexTreeL& index_tree,
    const std::unordered_map<std::string, casacore::DataType>& kws)
    : WithKeywords(ctx, runtime, kws)
    , m_context(ctx)
    , m_runtime(runtime)
    , m_name(name)
    , m_datatype(datatype)
    , m_rank(index_tree.rank().value())
    , m_index_tree(index_tree) {

    init();
  }

  Column(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    casacore::DataType datatype,
    Legion::LogicalRegion values,
    Legion::LogicalRegion keywords)
    : WithKeywords(ctx, runtime, keywords)
    , m_context(ctx)
    , m_runtime(runtime)
    , m_name(name)
    , m_datatype(datatype)
    , m_rank(
      static_cast<decltype(m_rank)>(values.get_index_space().get_dim())) {

    init(values);
  }

  mutable Legion::Context m_context;

  mutable Legion::Runtime* m_runtime;

private:

  void
  init();

  void
  init(Legion::LogicalRegion region);

  std::string m_name;

  casacore::DataType m_datatype;

  unsigned m_rank;

  IndexTreeL m_index_tree;

  Legion::LogicalRegion m_logical_region;
};

template <typename D> class ColumnT;

template <typename D>
class ColumnT
  : public Column {
public:

  typedef std::function<
  std::unique_ptr<ColumnT<D>>(Legion::Context, Legion::Runtime*)> Generator;

  ColumnT(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const ColumnBuilder<D>& builder)
    : Column(
      ctx,
      runtime,
      builder.name(),
      builder.datatype(),
      builder.index_tree(),
      builder.keywords())
    , m_axes(builder.axes()) {}

  ColumnT(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    casacore::DataType datatype,
    const std::vector<D>& axes,
    const IndexTreeL& index_tree_,
    const std::unordered_map<std::string, casacore::DataType>& kws =
    std::unordered_map<std::string, casacore::DataType>())
    : Column(
      ctx,
      runtime,
      name,
      datatype,
      index_tree_,
      kws)
    , m_axes(axes) {}

  ColumnT(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    casacore::DataType datatype,
    const std::vector<D>& axes,
    Legion::LogicalRegion values,
    Legion::LogicalRegion keywords)
    : Column(
      ctx,
      runtime,
      name,
      datatype,
      values,
      keywords)
    , m_axes(axes) {}

  virtual ~ColumnT() {}

  std::vector<int>
  axes() const override {
    std::vector<int> result;
    result.reserve(m_axes.size());
    std::transform(
      m_axes.begin(),
      m_axes.end(),
      std::back_inserter(result),
      [](auto& d) {return static_cast<int>(d);});
    return result;
  }

  const std::vector<D>&
  axesT() const {
    return m_axes;
  }

  ColumnGenArgs
  generator_args() const {
    return
      ColumnGenArgs {
      name(),
        datatype(),
        axes(),
        logical_region(),
        keywords_region()};
  }

  std::unique_ptr<ColumnPartition>
  partition_on_axes(const std::vector<AxisPartition<int>>& parts)
    const override {

    std::vector<AxisPartition<D>> dparts;
    dparts.reserve(parts.size());
    std::transform(
      parts.begin(),
      parts.end(),
      std::back_inserter(dparts),
      [](auto& part) {
        return
          AxisPartition<D>{
          static_cast<D>(part.dim),
            part.stride,
            part.offset,
            part.lo,
            part.hi};
      });
    return partition_on_axes(dparts);
  }

  std::unique_ptr<ColumnPartition>
  partition_on_axes(const std::vector<D>& ds) const {
    std::vector<AxisPartition<D>> parts;
    parts.reserve(ds.size());
    std::transform(
      ds.begin(),
      ds.end(),
      std::back_inserter(parts),
      [](auto& d) { return AxisPartition<D>{d, 1, 0, 0, 0}; });
    return partition_on_axes(parts);
  }

  std::unique_ptr<ColumnPartition>
  partition_on_axes(const std::vector<std::tuple<D, Legion::coord_t>>& dss)
    const {

    std::vector<AxisPartition<D>> parts;
    parts.reserve(dss.size());
    std::transform(
      dss.begin(),
      dss.end(),
      std::back_inserter(parts),
      [](auto& d_s) {
        auto& [d, s] = d_s;
        return AxisPartition<D>{d, s, 0, 0, s - 1};
      });
    return partition_on_axes(parts);
  }

  std::unique_ptr<ColumnPartition>
  partition_on_axes(const std::vector<AxisPartition<D>>& parts) const {

    // All variations of partition_on_axes() in the Column and ColumnT classes
    // should ultimately call this method, which takes care of the change in
    // semantics of the "dim" field of the AxisPartition structure, as needed by
    // create_partition_on_axes(). For all such methods in Column and ColumnT,
    // the "dim" field simply names an axis, whereas for
    // create_partition_on_axes(), "dim" is a mapping from a named axis to a
    // Column axis (i.e, an axis in the Table index space to an axis in the
    // Column index space).
    std::vector<D> ds;
    ds.reserve(parts.size());
    std::transform(
      parts.begin(),
      parts.end(),
      std::back_inserter(ds),
      [](auto& part){ return part.dim; });
    auto dm = dimensions_map(ds, axesT());
    std::vector<AxisPartition<int>> iparts;
    iparts.reserve(dm.size());
    for (size_t i = 0; i < dm.size(); ++i) {
      auto& part = parts[i];
      iparts.push_back(
        AxisPartition<int>{dm[i], part.stride, part.offset, part.lo, part.hi});
    }
    return
      std::make_unique<ColumnPartitionT<D>>(
        m_context,
        m_runtime,
        create_partition_on_axes(m_context, m_runtime, index_space(), iparts),
        ds);
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

    std::vector<int> dmap = dimensions_map(axesT(), cpt->axesT());

    switch (cpt->axes().size()) {
#if LEGMS_MAX_DIM >= 1
    case 1:
      switch (rank()) {
#if LEGMS_MAX_DIM >= 1
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
#if LEGMS_MAX_DIM >= 2
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
#if LEGMS_MAX_DIM >= 3
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
#if LEGMS_MAX_DIM >= 4
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
#if LEGMS_MAX_DIM >= 2
    case 2:
      switch (rank()) {
#if LEGMS_MAX_DIM >= 1
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
#if LEGMS_MAX_DIM >= 2
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
#if LEGMS_MAX_DIM >= 3
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
#if LEGMS_MAX_DIM >= 4
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
#if LEGMS_MAX_DIM >= 3
    case 3:
      switch (rank()) {
#if LEGMS_MAX_DIM >= 1
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
#if LEGMS_MAX_DIM >= 2
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
#if LEGMS_MAX_DIM >= 3
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
#if LEGMS_MAX_DIM >= 4
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
#if LEGMS_MAX_DIM >= 4
    case 4:
      switch (rank()) {
#if LEGMS_MAX_DIM >= 1
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
#if LEGMS_MAX_DIM >= 2
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
#if LEGMS_MAX_DIM >= 3
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
#if LEGMS_MAX_DIM >= 4
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
    const IndexTreeL& index_tree,
    const std::unordered_map<std::string, casacore::DataType>& kws =
    std::unordered_map<std::string, casacore::DataType>()) {

    return
      [=](Legion::Context ctx, Legion::Runtime* runtime) {
        return
          std::make_unique<ColumnT<D>>(
            ctx,
            runtime,
            name,
            datatype,
            axes,
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
        IndexTreeL(
          row_pattern,
          num_rows * row_pattern.size() / row_index_pattern.size()),
        kws);
  }

  static Generator
  generator(const ColumnGenArgs& genargs) {

    std::vector<D> axes;
    std::transform(
      genargs.axes.begin(),
      genargs.axes.end(),
      std::back_inserter(axes),
      [](auto& i) { return static_cast<D>(i); });
    return
      [=](Legion::Context ctx, Legion::Runtime* runtime) {
      return
        std::make_unique<ColumnT<D>>(
          ctx,
          runtime,
          genargs.name,
          genargs.datatype,
          axes,
          genargs.values,
          genargs.keywords);
    };
  }

private:

  std::vector<D> m_axes;
};

template <typename D>
std::unique_ptr<ColumnT<D>>
ColumnGenArgs::operator()(
  Legion::Context ctx,
  Legion::Runtime* runtime) const {

  return ColumnT<D>::generator(*this)(ctx, runtime);
}

} // end namespace legms

#endif // LEGMS_COLUMN_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
