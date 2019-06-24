#ifndef LEGMS_COLUMN_H_
#define LEGMS_COLUMN_H_

#include <cassert>
#include <functional>
#include <memory>
#include <tuple>
#include <unordered_map>

#include "legms.h"
#include "utility.h"

#include "Column_c.h"

#include "WithKeywords.h"
#include "IndexTree.h"
#include "ColumnPartition.h"

#include "c_util.h"

namespace legms {

class Column;

struct ColumnGenArgs {
  std::string name;
  std::string axes_uid;
  TypeTag datatype;
  std::vector<int> axes;
  Legion::LogicalRegion values;
  Legion::LogicalRegion keywords;
  std::vector<TypeTag> keyword_datatypes;

  std::unique_ptr<Column>
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

  typedef std::function<
  std::unique_ptr<Column>(Legion::Context, Legion::Runtime*)> Generator;

  Column(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    const std::string& axes_uid,
    const std::vector<int>& axes,
    TypeTag datatype,
    const IndexTreeL& index_tree,
    const kw_desc_t& kws = kw_desc_t())
    : WithKeywords(ctx, runtime, kws)
    , m_name(name)
    , m_axes_uid(axes_uid)
    , m_axes(axes)
    , m_datatype(datatype)
    , m_rank(index_tree.rank().value())
    , m_index_tree(index_tree) {

    assert(m_rank == m_axes.size());
    init();
  }

  Column(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    const std::string& axes_uid,
    const std::vector<int>& axes,
    TypeTag datatype,
    Legion::LogicalRegion values,
    Legion::LogicalRegion keywords,
    const std::vector<TypeTag>& kw_datatypes)
    : WithKeywords(ctx, runtime, keywords, kw_datatypes)
    , m_name(name)
    , m_axes_uid(axes_uid)
    , m_axes(axes)
    , m_datatype(datatype)
    , m_rank(
      static_cast<decltype(m_rank)>(values.get_index_space().get_dim())) {

    assert(m_rank == m_axes.size());
    init(values);
  }

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  Column(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    const std::vector<D>& axes,
    TypeTag datatype,
    const IndexTreeL& index_tree,
    const kw_desc_t& kws = kw_desc_t())
    : Column(
      ctx,
      runtime,
      name,
      Axes<D>::uid,
      map_to_int(axes),
      datatype,
      index_tree,
      kws) {}

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  Column(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& name,
    const std::vector<D>& axes,
    TypeTag datatype,
    Legion::LogicalRegion values,
    Legion::LogicalRegion keywords,
    const std::vector<TypeTag>& kw_datatypes)
    : Column(
      ctx,
      runtime,
      name,
      Axes<D>::uid,
      map_to_int(axes),
      datatype,
      values,
      keywords,
      kw_datatypes) {}

  virtual ~Column() {
    // TODO: ???
    // if (m_logical_region != Legion::LogicalRegion::NO_REGION)
    //   m_runtime->destroy_logical_region(m_context, m_logical_region);
  };

  const std::string&
  name() const {
    return m_name;
  }

  const std::string&
  axes_uid() const {
    return m_axes_uid;
  }

  const std::vector<int>&
  axes() const {
    return m_axes;
  }

  unsigned
  rank() const {
    return m_rank;
  }

  const IndexTreeL&
  index_tree() const {
    return m_index_tree;
  }

  TypeTag
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

  std::unique_ptr<ColumnPartition>
  partition_on_axes(const std::vector<AxisPartition>& parts) const;

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  std::unique_ptr<ColumnPartition>
  partition_on_axes(const std::vector<D>& ds) const {
    assert(Axes<D>::uid == m_axes_uid);
    return partition_on_iaxes(map_to_int(ds));
  }

  std::unique_ptr<ColumnPartition>
  partition_on_axes(const std::vector<int>& ds) const {
    assert(AxesRegistrar::in_range(axes_uid(), ds));
    return partition_on_iaxes(ds);
  }

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  std::unique_ptr<ColumnPartition>
  partition_on_axes(const std::vector<std::tuple<D, Legion::coord_t>>& ds)
    const {
    assert(Axes<D>::uid == m_axes_uid);
    std::vector<std::tuple<int, Legion::coord_t>> is =
      map(
        ds,
        [](const auto& d) {
          return
            std::make_tuple(static_cast<int>(std::get<0>(d)), std::get<1>(d));
        });
    return partition_on_iaxes(is);
  }

  std::unique_ptr<ColumnPartition>
  partition_on_axes(const std::vector<std::tuple<int, Legion::coord_t>>& ds)
    const {
    std::vector<int> is = map(ds, [](const auto& d) { return std::get<0>(d); });
    assert(AxesRegistrar::in_range(axes_uid(), is));
    return partition_on_iaxes(ds);
  }

  std::unique_ptr<ColumnPartition>
  projected_column_partition(const ColumnPartition* cp) const;

  ColumnGenArgs
  generator_args() const {
    return
      ColumnGenArgs {
      name(),
        axes_uid(),
        datatype(),
        axes(),
        logical_region(),
        keywords_region(),
        keywords_datatypes()};
  }

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  static Generator
  generator(
    const std::string& name,
    const std::vector<D>& axes,
    TypeTag datatype,
    const IndexTreeL& index_tree,
    const std::unordered_map<std::string, TypeTag>& kws =
    std::unordered_map<std::string, TypeTag>()) {

    return
      [=](Legion::Context ctx, Legion::Runtime* runtime) {
        return
          std::make_unique<Column>(
            ctx,
            runtime,
            name,
            axes,
            datatype,
            index_tree,
            kws);
      };
  }

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  static Generator
  generator(
    const std::string& name,
    const std::vector<D>& axes,
    TypeTag datatype,
    const IndexTreeL& row_index_pattern,
    unsigned num_rows,
    const std::unordered_map<std::string, TypeTag>& kws =
    std::unordered_map<std::string, TypeTag>()) {

    return
      generator(
        name,
        axes,
        datatype,
        IndexTreeL(row_index_pattern, num_rows),
        kws);
  }

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  static Generator
  generator(
    const std::string& name,
    const std::vector<D>& axes,
    TypeTag datatype,
    const IndexTreeL& row_index_pattern,
    const IndexTreeL& row_pattern,
    unsigned num_rows,
    const std::unordered_map<std::string, TypeTag>& kws =
    std::unordered_map<std::string, TypeTag>()) {

    return
      generator(
        name,
        axes,
        datatype,
        IndexTreeL(
          row_pattern,
          num_rows * row_pattern.size() / row_index_pattern.size()),
        kws);
  }

  static Generator
  generator(const ColumnGenArgs& genargs) {
    return
      [=](Legion::Context ctx, Legion::Runtime* runtime) {
        return
          std::make_unique<Column>(
            ctx,
            runtime,
            genargs.name,
            genargs.axes_uid,
            genargs.axes,
            genargs.datatype,
            genargs.values,
            genargs.keywords,
            genargs.keyword_datatypes);
      };
  }

  static constexpr Legion::FieldID value_fid = 0;

protected:

  std::unique_ptr<ColumnPartition>
  partition_on_iaxes(const std::vector<int>& ds) const;

  std::unique_ptr<ColumnPartition>
  partition_on_iaxes(const std::vector<std::tuple<int, Legion::coord_t>>& ds)
    const;

private:

  void
  init();

  void
  init(Legion::LogicalRegion region);

  std::string m_name;

  std::string m_axes_uid;

  std::vector<int> m_axes;

  TypeTag m_datatype;

  unsigned m_rank;

  IndexTreeL m_index_tree;

  Legion::LogicalRegion m_logical_region;
};

template <>
struct CObjectWrapper::SharedWrapper<Column> {
  typedef column_t type_t;
};

template <>
struct CObjectWrapper::SharedWrapped<column_t> {
  typedef Column type_t;
  typedef std::shared_ptr<type_t> impl_t;
};

} // end namespace legms

#endif // LEGMS_COLUMN_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
