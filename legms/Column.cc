#pragma GCC visibility push(default)
#include <cassert>
#pragma GCC visibility pop

#include "legms.h"
#include "tree_index_space.h"
#include "legion/legion_c_util.h"
#include "Column.h"
#include "Table.h"

using namespace legms;
using namespace Legion;

ColumnGenArgs::ColumnGenArgs(const column_t& col)
  : name(col.name)
  , axes_uid(col.axes_uid)
  , datatype(col.datatype)
  , values(Legion::CObjectWrapper::unwrap(col.values))
  , keywords(Legion::CObjectWrapper::unwrap(col.keywords)) {

  axes.resize(col.num_axes);
  for (unsigned i = 0; i < col.num_axes; ++i)
    axes[i] = col.axes[i];

  keyword_datatypes.resize(col.num_keywords);
  for (unsigned i = 0; i < col.num_keywords; ++i)
    keyword_datatypes[i] = col.keyword_datatypes[i];
}

Column::Column(
  LogicalRegion metadata_,
  LogicalRegion axes_,
  LogicalRegion values_,
  const Keywords& keywords_)
  : metadata(metadata_)
  , axes(axes_)
  , values(values_)
  , keywords(keywords_) {
}

Column::Column(
  LogicalRegion metadata_,
  LogicalRegion axes_,
  LogicalRegion values_,
  Keywords&& keywords_)
  : metadata(metadata_)
  , axes(axes_)
  , values(values_)
  , keywords(std::move(keywords_)) {
}

Column
Column::create(
  Context ctx,
  Runtime* rt,
  const std::string& name,
  const std::string& axes_uid,
  const std::vector<int>& axes,
  legms::TypeTag datatype,
  const IndexTreeL& index_tree,
  const Keywords::kw_desc_t& kws = Keywords::kw_desc_t()) {

  LogicalRegion metadata;
  {
    IndexSpace is = rt->create_index_space(ctx, Rect<1>(0, 0));
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    fa.allocate_field(sizeof(legms::string), METADATA_NAME_FID);
    fa.allocate_field(sizeof(legms::string), METADATA_AXES_UID_FID);
    fa.allocate_field(sizeof(legms::TypeTag), METADATA_DATATYPE_FID);
    metadata = rt->create_logical_region(ctx, is, fs);
    {
      RegionRequirement req(metadata, WRITE_ONLY, EXCLUSIVE, metadata);
      req.add_field(METADATA_NAME_FID);
      req.add_field(METADATA_AXES_UID_FID);
      req.add_field(METADATA_DATATYPE_FID);
      PhysicalRegion pr = rt->map_region(ctx, req);
      const NameAccessor<WRITE_ONLY> nm(pr, METADATA_NAME_FID);
      const AxesUidAccessor<WRITE_ONLY> au(pr, METADATA_AXES_UID_FID);
      const DatatypeAccessor<WRITE_ONLY> dt(pr, METADATA_DATATYPE_FID);
      std::strncpy(&nm[0].val, name, sizeof(nm[0].val));
      std::strncpy(&au[0].val, axes_uid, sizeof(au[0].val));
      dt[0] = datatype;
      rt->unmap_region(ctx, pr);
    }
  }
  LogicalRegion axes;
  {
    Rect<1> rect(0, axes.size() - 1);
    IndexSpace is = rt->create_index_space(ctx, rect);
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    add_field(ValueType<int>::DataType, fa, AXES_FID);
    axes = rt->create_logical_region(ctx, is, fs);
    {
      RegionRequirement(axes, WRITE_ONLY, EXCLUSIVE, axes);
      req.add_field(AXES_FID);
      PhysicalRegion pr = rt->map_region(ctx, req);
      const AxesAccessor<WRITE_ONLY> ax(pr, AXES_FID);
      for (PointInRectIterator<1> pir(rect); pir(); pir++)
        ax[*pir] = axes[pir[0]];
      rt->unmap_region(ctx, pr);
    }
  }
  LogicalRegion values;
  if (index_tree.size() > 0) {
    IndexSpace is = tree_index_space(index_tree, ctx, rt);
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    add_field(datatype, fa, VALUES_FID);
    values = rt->create_logical_region(ctx, is, fs);
  }
  return Column(metadata, axes, values, Keywords::create(ctx, rt, kws));
}

void
Column::destroy(Context ctx, Runtime* rt) {
  if (metadata != LogicalRegion::NO_REGION) {
    assert(axes != LogicalRegion::NO_REGION);
    assert(values != LogicalRegion::NO_REGION);
    std::vector<LogicalRegion&> lrs{metadata, axes, values};
    for (auto& lr : lrs)
      rt->destroy_field_space(ctx, lr.get_field_space());
    for (auto& lr : lrs)
      rt->destroy_index_space(ctx, lr.get_index_space());
    for (auto& lr : lrs) {
      rt->destroy_logical_region(ctx, lr);
      lr = LogicalRegion::NO_REGION;
    }
  }
  keywords.destroy(ctx, rt);
}

std::string
Column::name(Context ctx, Runtime* rt) const {
  RegionRequirement req(metadata, READ_ONLY, EXCLUSIVE, metadata);
  req.add_field(METADATA_NAME_FID);
  auto pr = rt->map_region(ctx, req);
  std::string result(name(pr));
  rt->unmap_region(ctx, pr);
  return result;
}

const char*
Column::name(const PhysicalRegion& metadata) {
  const NameAccessor<READ_ONLY> name(metadata, METADATA_NAME_FID);
  return name[0].val;
}

std::string
Column::axes_uid(Context ctx, Runtime* rt) const {
  RegionRequirement req(metadata, READ_ONLY, EXCLUSIVE, metadata);
  req.add_field(METADATA_AXES_UID_FID);
  auto pr = rt->map_region(ctx, req);
  std::string result(axes_uid(pr));
  rt->unmap_region(ctx, pr);
  return result;
}

const char*
Column::axes_uid(const PhysicalRegion& metadata) {
  const AxesUidAccessor<READ_ONLY> axes_uid(metadata, METADATA_AXES_UID_FID);
  return axes_uid[0].val;
}

legms::TypeTag
Column::datatype() const {
  RegionRequirement req(metadata, READ_ONLY, EXCLUSIVE, metadata);
  req.add_field(METADATA_DATATYPE_FID);
  auto pr = rt->map_region(ctx, req);
  legms::TypeTag result = datatype(pr);
  rt->unmap_region(ctx, pr);
  return result;  
}

legms::TypeTag
Column::datatype(const PhysicalRegion& metadata) {
  const DatatypeAccessor<READ_ONLY> datatype(metadata, METADATA_DATATYPE_FID);
  return datatype[0];
}

std::vector<int>
Column::axes(Context ctx, Runtime* rt) const {
  RegionRequirement req(axes, READ_ONLY, EXCLUSIVE, axes);
  req.add_field(AXES_FID);
  auto pr = rt->map_region(ctx, req);
  IndexSpaceT<1> is(axes.get_index_space());
  DomainT<1> dom = rt->get_index_space_domain(is);
  std::vector<int> result(dom.hi.x + 1);
  const AxesAccessor<READ_ONLY> ax(pr, AXES_FID);
  for (PointInRectIterator<1> pir(dom); pir(); pir++)
    result[pir[0]] = ax[*pir];
  rt->unmap_region(ctx, pr);
  return result;
}

unsigned
Column::rank() const {
  return axes.get_index_space().get_dim();
}

IndexTreeL
Column::index_tree(Runtime* rt) const {

  IndexTreeL result;
  Domain dom = rt->get_index_space_domain(values.get_index_space());
  switch (dom.get_dim()) {
#define TREE(N)                                         \
    case (N): {                                         \
      RectInDomainIterator<N> rid(dom);                 \
      while (rid()) {                                   \
        IndexTreeL t;                                   \
        for (size_t i = N; i > 0; --i) {                \
          t =                                           \
            IndexTreeL({                                \
                std::make_tuple(                        \
                  rid->lo[i - 1],                       \
                  rid->hi[i - 1] - rid->lo[i - 1] + 1,  \
                  t)});                                 \
        }                                               \
        result = result.merged_with(t);                 \
        rid++;                                          \
      }                                                 \
      break;                                            \
    }
    LEGMS_FOREACH_N(TREE)
#undef TREE
  default:
      assert(false);
    break;
  }
  return result;
}

ColumnPartition
Column::partition_on_axes(
  Context ctx,
  Runtime* rt,
  const std::vector<AxisPartition>& parts) const {

  std::string auid = axes_uid(ctx, rt);

  assert(
    std::all_of(
      parts.begin(),
      parts.end(),
      [this, &auid](auto& p) { return p.axes_uid == auid; }));

  // All variations of partition_on_axes() in the Column class should ultimately
  // call this method, which takes care of the change in semantics of the "dim"
  // field of the AxisPartition structure, as needed by
  // ColumnPartition::create(). For all such methods in Column, the "dim" field
  // simply names an axis, whereas for ColumnPartition::create(), "dim" is a
  // mapping from a named axis to a Column axis (i.e, an axis in the Table index
  // space to an axis in the Column index space).
  std::vector<int> ds = legms::map(parts, [](const auto& p) { return p.dim; });
  auto dm = dimensions_map(ds, axes());
  std::vector<AxisPartition> iparts;
  iparts.reserve(dm.size());
  for (size_t i = 0; i < dm.size(); ++i) {
    auto& part = parts[i];
    iparts.push_back(
      AxisPartition{part.axes_uid, dm[i], part.stride, part.offset,
                    part.lo, part.hi});
  }
  return
    ColumnPartition::create(ctx, rt, auid, ds, values.get_index_space(), iparts);
}

ColumnPartition
Column::partition_on_iaxes(
  Context ctx,
  Runtime*rt,
  const std::vector<int>& ds) const {

  // TODO: verify values in 'ds' are valid axes
  std::vector<AxisPartition> parts =
    legms::map(
      ds,
      [au=axes_uid()](const auto& d) {
        return AxisPartition{au, d, 1, 0, 0, 0};
      });
  return partition_on_axes(ctx, rt, parts);
}

ColumnPartition
Column::partition_on_iaxes(
  Context ctx,
  Runtime* rt,
  const std::vector<std::tuple<int, coord_t>>& dss) const {

  // TODO: verify axis values in 'dss' are valid axes
  std::vector<AxisPartition> parts =
    legms::map(
      dss,
      [au=axes_uid()](const auto& d_s) {
        auto& [d, s] = d_s;
        return AxisPartition{au, d, s, 0, 0, s - 1};
      });
  return partition_on_axes(ctx, rt, parts);
}

ColumnPartition
Column::projected_column_partition(
  Context ctx,
  Runtime* rt,
  const ColumnPartition& cp) const {

  std::string auid = axes_uid(ctx, rt);
  assert(cp.axes_uid(ctx, rt) == auid);

  auto ax = axes(ctx, rt);
  if (values.get_index_space() == IndexSpace::NO_SPACE)
    return ColumnPartition::create(ctx, rt, auid, IndexPartition::NO_PART, ax);

  auto cpax = cp.axes(ctx, rt);
  std::vector<int> dmap = dimensions_map(ax, cpax);

  switch (cp.index_partition.get_dim() * LEGION_MAX_DIM + ax.size()) {
#define CP(I, P)                                      \
    case (I * LEGION_MAX_DIM + P):                    \
      return                                          \
        ColumnPartition::create(                      \
          ctx,                                        \
          rt,                                         \
          auid,                                       \
          ax,                                         \
          legms::projected_index_partition<I, P>(     \
            ctx,                                      \
            rt,                                       \
            IndexPartitionT<I>(cp.index_partition),   \
            IndexSpaceT<P>(values.get_index_space()), \
            dmap));                                   \
      break;
    LEGMS_FOREACH_NN(CP);
#undef CP
  default:
    assert(false);
    // keep compiler happy
    return ColumnPartition::create(ctx, rt, auid, IndexPartition::NO_PART, ax);
    break;
  }
}

std::unique_ptr<Column>
ColumnGenArgs::operator()(Context ctx, Runtime* runtime) const {

  return Column::generator(*this)(ctx, runtime);
}

size_t
ColumnGenArgs::legion_buffer_size(void) const {
  return
    name.size() * sizeof(decltype(name)::value_type) + 1
    + axes_uid.size() * sizeof(decltype(axes_uid)::value_type) + 1
    + sizeof(datatype)
    + vector_serdez<int>::serialized_size(axes)
    + 2 * sizeof(LogicalRegion)
    + vector_serdez<TypeTag>::serialized_size(keyword_datatypes);
}

size_t
ColumnGenArgs::legion_serialize(void *buffer) const {
  char* buff = static_cast<char *>(buffer);

  size_t s = name.size() * sizeof(decltype(name)::value_type) + 1;
  memcpy(buff, name.c_str(), s);
  buff += s;

  s = axes_uid.size() * sizeof(decltype(axes_uid)::value_type) + 1;
  memcpy(buff, axes_uid.c_str(), s);
  buff += s;

  s = sizeof(datatype);
  memcpy(buff, &datatype, s);
  buff += s;

  buff += vector_serdez<int>::serialize(axes, buff);

  s = sizeof(LogicalRegion);
  memcpy(buff, &values, s);
  buff += s;

  memcpy(buff, &keywords, s);
  buff += s;

  buff += vector_serdez<TypeTag>::serialize(keyword_datatypes, buff);

  return buff - static_cast<char *>(buffer);
}

size_t
ColumnGenArgs::legion_deserialize(const void *buffer) {
  const char *buff = static_cast<const char*>(buffer);

  name = buff;
  buff += name.size() * sizeof(decltype(name)::value_type) + 1;

  axes_uid = buff;
  buff += axes_uid.size() * sizeof(decltype(axes_uid)::value_type) + 1;

  datatype = *reinterpret_cast<const decltype(datatype) *>(buff);
  buff += sizeof(datatype);

  buff += vector_serdez<int>::deserialize(axes, buff);

  values = *reinterpret_cast<const decltype(values) *>(buff);
  buff += sizeof(values);

  keywords = *reinterpret_cast<const decltype(values) *>(buff);
  buff += sizeof(keywords);

  buff += vector_serdez<TypeTag>::deserialize(keyword_datatypes, buff);

  return buff - static_cast<const char*>(buffer);
}

column_t
ColumnGenArgs::to_column_t() const {
  column_t result;
  std::strncpy(result.name, name.c_str(), sizeof(result.name));
  result.name[sizeof(result.name) - 1] = '\0';
  std::strncpy(result.axes_uid, axes_uid.c_str(), sizeof(result.axes_uid));
  result.axes_uid[sizeof(result.axes_uid) - 1] = '\0';
  result.datatype = datatype;
  result.num_axes = axes.size();
  std::memcpy(result.axes, axes.data(), axes.size() * sizeof(int));
  result.values = CObjectWrapper::wrap(values);
  result.num_keywords = keyword_datatypes.size();
  std::memcpy(
    result.keyword_datatypes,
    keyword_datatypes.data(),
    keyword_datatypes.size() * sizeof(type_tag_t));
  result.keywords = CObjectWrapper::wrap(keywords);
  return result;
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
