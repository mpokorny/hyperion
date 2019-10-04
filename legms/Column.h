#ifndef LEGMS_COLUMN_H_
#define LEGMS_COLUMN_H_

#pragma GCC visibility push(default)
#include <cassert>
#include <functional>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#pragma GCC visibility pop

#include <legms/legms.h>
#include <legms/utility.h>

#include <legms/Column_c.h>

#include <legms/Keywords.h>
#include <legms/IndexTree.h>
#include <legms/ColumnPartition.h>

#include <legms/c_util.h>

#ifdef LEGMS_USE_HDF5
# include <legms/hdf5.h>
#endif // LEGMS_USE_HDF5

#ifdef LEGMS_USE_CASACORE
# include <legms/MeasRefContainer.h>
# include <legms/MeasRefDict.h>
#endif

namespace legms {

class LEGMS_API Column
#ifdef LEGMS_USE_CASACORE
  : public MeasRefContainer
#endif
{

public:

  static const constexpr Legion::FieldID METADATA_NAME_FID = 0;
  static const constexpr Legion::FieldID METADATA_AXES_UID_FID = 1;
  static const constexpr Legion::FieldID METADATA_DATATYPE_FID = 2;
  Legion::LogicalRegion metadata_lr;
  static const constexpr Legion::FieldID AXES_FID = 0;
  Legion::LogicalRegion axes_lr;
  static const constexpr Legion::FieldID VALUE_FID = 0;
  Legion::LogicalRegion values_lr;
  Keywords keywords;

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using NameAccessor =
    Legion::FieldAccessor<
    MODE,
    legms::string,
    1,
    Legion::coord_t,
    Legion::AffineAccessor<legms::string, 1, Legion::coord_t>,
    CHECK_BOUNDS>;

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using AxesUidAccessor =
    Legion::FieldAccessor<
    MODE,
    legms::string,
    1,
    Legion::coord_t,
    Legion::AffineAccessor<legms::string, 1, Legion::coord_t>,
    CHECK_BOUNDS>;

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using DatatypeAccessor =
    Legion::FieldAccessor<
    MODE,
    legms::TypeTag,
    1,
    Legion::coord_t,
    Legion::AffineAccessor<legms::TypeTag, 1, Legion::coord_t>,
    CHECK_BOUNDS>;

  typedef std::function<
    Column(Legion::Context, Legion::Runtime*, const std::string&)>
  Generator;

  template <legion_privilege_mode_t MODE, bool CHECK_BOUNDS=false>
  using AxesAccessor =
    Legion::FieldAccessor<
    MODE,
    int,
    1,
    Legion::coord_t,
    Legion::AffineAccessor<int, 1, Legion::coord_t>,
    CHECK_BOUNDS>;

  Column() {}

#ifdef LEGMS_USE_CASACORE
  Column(
    Legion::LogicalRegion metadata,
    Legion::LogicalRegion axes,
    Legion::LogicalRegion values,
    const std::vector<MeasRef>& new_meas_refs,
    const MeasRefContainer& inherited_meas_refs,
    const Keywords& keywords)
    : MeasRefContainer(new_meas_refs, inherited_meas_refs)
    , metadata_lr(metadata)
    , axes_lr(axes)
    , values_lr(values)
    , keywords(keywords) {
  }

  Column(
    Legion::LogicalRegion metadata,
    Legion::LogicalRegion axes,
    Legion::LogicalRegion values,
    const std::vector<MeasRef>& new_meas_refs,
    const MeasRefContainer& inherited_meas_refs,
    Keywords&& keywords)
    : MeasRefContainer(new_meas_refs, inherited_meas_refs)
    , metadata_lr(metadata)
    , axes_lr(axes)
    , values_lr(values)
    , keywords(std::move(keywords)) {
  }

#else // !LEGMS_USE_CASACORE

  Column(
    Legion::LogicalRegion metadata,
    Legion::LogicalRegion axes,
    Legion::LogicalRegion values,
    const Keywords& keywords)
    : metadata_lr(metadata)
    , axes_lr(axes)
    , values_lr(values)
    , keywords(keywords) {
  }

  Column(
    Legion::LogicalRegion metadata,
    Legion::LogicalRegion axes,
    Legion::LogicalRegion values,
    Keywords&& keywords)
    : metadata_lr(metadata)
    , axes_lr(axes)
    , values_lr(values)
    , keywords(std::move(keywords)) {
}
#endif

  static Column
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::string& name,
    const std::string& axes_uid,
    const std::vector<int>& axes,
    legms::TypeTag datatype,
    const IndexTreeL& index_tree,
#ifdef LEGMS_USE_CASACORE
    const std::vector<MeasRef>& new_meas_refs,
    const MeasRefContainer& inherited_meas_refs,
#endif
    const Keywords::kw_desc_t& kws = Keywords::kw_desc_t(),
    const std::string& name_prefix = "");

  void
  destroy(Legion::Context ctx, Legion::Runtime* rt);

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  static Column
  create(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::string& name,
    const std::vector<D>& axes,
    legms::TypeTag datatype,
    const IndexTreeL& index_tree,
#ifdef LEGMS_USE_CASACORE
    const std::vector<MeasRef>& new_meas_refs,
    const MeasRefContainer& meas_refs,
#endif
    const Keywords::kw_desc_t& kws = Keywords::kw_desc_t(),
    const std::string& name_prefix = "") {
    return
      create(
        ctx,
        rt,
        name,
        Axes<D>::uid,
        map_to_int(axes),
        datatype,
        index_tree,
#ifdef LEGMS_USE_CASACORE
        new_meas_refs,
        meas_refs,
#endif
        kws,
        name_prefix);
  }

  std::string
  name(Legion::Context ctx, Legion::Runtime* rt) const;

  static const char*
  name(const Legion::PhysicalRegion& metadata);

  std::string
  axes_uid(Legion::Context ctx, Legion::Runtime* rt) const;

  static const char*
  axes_uid(const Legion::PhysicalRegion& metadata);

  legms::TypeTag
  datatype(Legion::Context ctx, Legion::Runtime* rt) const;

  static legms::TypeTag
  datatype(const Legion::PhysicalRegion& metadata);

  std::vector<int>
  axes(Legion::Context ctx, Legion::Runtime* rt) const;

  unsigned
  rank(Legion::Runtime* rt) const;

  bool
  is_empty() const;

  IndexTreeL
  index_tree(Legion::Runtime* rt) const;

#ifdef LEGMS_USE_CASACORE

  MeasRefDict
  get_measure_references_dictionary(
    Legion::Context ctx,
    Legion::Runtime* rt) const;

#endif // LEGMS_USE_CASACORE

#ifdef LEGMS_USE_HDF5
  template <
    typename FN,
    std::enable_if_t<
      !std::is_void_v<
        std::invoke_result_t<FN, Legion::Context, Legion::Runtime*, Column&>>,
      int> = 0>
  std::invoke_result_t<FN, Legion::Context, Legion::Runtime*, Column&>
  with_attached(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const LEGMS_FS::path& file_path,
    const std::string& table_root,
    FN f,
    bool mapped = false,
    bool read_write = false) {

    typedef
      std::invoke_result_t<FN, Legion::Context, Legion::Runtime*, Column&> RET;

    Legion::PhysicalRegion pr =
      with_attached_prologue(
        ctx,
        rt,
        file_path,
        table_root,
        mapped,
        read_write);
    RET result = f(ctx, rt, *this);
    with_attached_epilogue(ctx, rt, pr);
    return result;
  }

  template <
    typename FN,
    std::enable_if_t<
      std::is_void_v<
      std::invoke_result_t<FN, Legion::Context, Legion::Runtime*, Column&>>,
      int> = 0>
  void
  with_attached(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const LEGMS_FS::path& file_path,
    const std::string& table_root,
    FN f,
    bool mapped = false,
    bool read_write = false) {

    Legion::PhysicalRegion pr =
      with_attached_prologue(
        ctx,
        rt,
        file_path,
        table_root,
        mapped,
        read_write);
    f(ctx, rt, *this);
    with_attached_epilogue(ctx, rt, pr);
  }
#endif // LEGMS_USE_HDF5

  ColumnPartition
  partition_on_axes(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<AxisPartition>& parts) const;

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  ColumnPartition
  partition_on_axes(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<D>& ds) const {
    assert(axes_uid(ctx, rt) == Axes<D>::uid);
    return partition_on_iaxes(ctx, rt, map_to_int(ds));
  }

  ColumnPartition
  partition_on_axes(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<int>& ds) const {
    assert(AxesRegistrar::in_range(axes_uid(ctx, rt), ds));
    return partition_on_iaxes(ctx, rt, ds);
  }

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  ColumnPartition
  partition_on_axes(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<std::tuple<D, Legion::coord_t>>& ds)
    const {
    assert(axes_uid(ctx, rt) == Axes<D>::uid);
    std::vector<std::tuple<int, Legion::coord_t>> is =
      map(
        ds,
        [](const auto& d) {
          return
            std::make_tuple(static_cast<int>(std::get<0>(d)), std::get<1>(d));
        });
    return partition_on_iaxes(ctx, rt, is);
  }

  ColumnPartition
  partition_on_axes(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<std::tuple<int, Legion::coord_t>>& ds)
    const {
    std::vector<int> is = map(ds, [](const auto& d) { return std::get<0>(d); });
    assert(AxesRegistrar::in_range(axes_uid(ctx, rt), is));
    return partition_on_iaxes(ctx, rt, ds);
  }

  ColumnPartition
  projected_column_partition(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const ColumnPartition& cp) const;

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  static Generator
  generator(
    const std::string& name,
    const std::vector<D>& axes,
    legms::TypeTag datatype,
    const IndexTreeL& index_tree,
#ifdef LEGMS_USE_CASACORE
    const std::vector<MeasRef>& new_meas_refs,
    const MeasRefContainer& inherited_meas_refs,
#endif
    const Keywords::kw_desc_t& kws = Keywords::kw_desc_t()) {

    return
      [=]
      (Legion::Context ctx,
       Legion::Runtime* rt,
       const std::string& name_prefix) {
        return
          create(
            ctx,
            rt,
            name,
            axes,
            datatype,
            index_tree,
#ifdef LEGMS_USE_CASACORE
            new_meas_refs,
            inherited_meas_refs,
#endif
            kws,
            name_prefix);
      };
  }

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  static Generator
  generator(
    const std::string& name,
    const std::vector<D>& axes,
    legms::TypeTag datatype,
    const IndexTreeL& row_index_pattern,
    unsigned num_rows,
#ifdef LEGMS_USE_CASACORE
    const std::vector<MeasRef>& new_meas_refs,
    const MeasRefContainer& inherited_meas_refs,
#endif
    const Keywords::kw_desc_t& kws = Keywords::kw_desc_t()) {

    return
      generator(
        name,
        axes,
        datatype,
        IndexTreeL(row_index_pattern, num_rows),
#ifdef LEGMS_USE_CASACORE
        new_meas_refs,
        inherited_meas_refs,
#endif
        kws);
  }

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  static Generator
  generator(
    const std::string& name,
    const std::vector<D>& axes,
    legms::TypeTag datatype,
    const IndexTreeL& row_index_pattern,
    const IndexTreeL& row_pattern,
    unsigned num_rows,
#ifdef LEGMS_USE_CASACORE
    const std::vector<MeasRef>& new_meas_refs,
    const MeasRefContainer& inherited_meas_refs,
#endif
    const Keywords::kw_desc_t& kws = Keywords::kw_desc_t()) {

    return
      generator(
        name,
        axes,
        datatype,
        IndexTreeL(
          row_pattern,
          num_rows * row_pattern.size() / row_index_pattern.size()),
#ifdef LEGMS_USE_CASACORE
        new_meas_refs,
        inherited_meas_refs,
#endif
        kws);
  }

protected:
#ifdef LEGMS_USE_HDF5
  Legion::PhysicalRegion
  with_attached_prologue(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const LEGMS_FS::path& file_path,
    const std::string& table_root,
    bool mapped,
    bool read_write);

  void
  with_attached_epilogue(
    Legion::Context ctx,
    Legion::Runtime* rt,
    Legion::PhysicalRegion pr);
#endif // LEGMS_USE_HDF5

  ColumnPartition
  partition_on_iaxes(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<int>& ds) const;

  ColumnPartition
  partition_on_iaxes(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<std::tuple<int, Legion::coord_t>>& ds) const;
};

template <>
struct CObjectWrapper::Wrapper<Column> {

  typedef column_t t;
  static column_t
  wrap(const Column& c) {
    return
      column_t{
      Legion::CObjectWrapper::wrap(c.metadata_lr),
        Legion::CObjectWrapper::wrap(c.axes_lr),
        Legion::CObjectWrapper::wrap(c.values_lr),
        Legion::CObjectWrapper::wrap(c.keywords.type_tags_lr),
        Legion::CObjectWrapper::wrap(c.keywords.values_lr)};
  }
};

template <>
struct CObjectWrapper::Unwrapper<column_t> {

  typedef Column t;
  static Column
  unwrap(const column_t& c) {
    return
      Column(
        Legion::CObjectWrapper::unwrap(c.metadata),
        Legion::CObjectWrapper::unwrap(c.axes),
        Legion::CObjectWrapper::unwrap(c.values),
        {}, // FIXME
        MeasRefContainer(), // FIXME
        Keywords(
          Keywords::pair<Legion::LogicalRegion>{
            Legion::CObjectWrapper::unwrap(c.keyword_type_tags),
              Legion::CObjectWrapper::unwrap(c.keyword_values)}));
  }
};

} // end namespace legms

#endif // LEGMS_COLUMN_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
