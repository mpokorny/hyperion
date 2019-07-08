#include <cassert>

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

void
Column::init() {
  if (m_index_tree.size() > 0) {
    IndexSpace index_space =
      legms::tree_index_space(m_index_tree, m_context, m_runtime);
    FieldSpace fs = m_runtime->create_field_space(m_context);
    auto fa = m_runtime->create_field_allocator(m_context, fs);
    legms::add_field(m_datatype, fa, value_fid);
    m_runtime->attach_name(fs, value_fid, name().c_str());
    m_logical_region =
      m_runtime->create_logical_region(m_context, index_space, fs);
    // TODO: keep?
    // m_runtime->destroy_index_space(m_context, index_space);
  } else {
    m_logical_region = LogicalRegion::NO_REGION;
  }
}

void
Column::init(LogicalRegion region) {

  m_logical_region = region;

  Domain dom = m_runtime->get_index_space_domain(region.get_index_space());

#define TREE(N)                                               \
  case (N): {                                                 \
    m_index_tree = IndexTreeL();                              \
    RectInDomainIterator<N> rid(dom);                         \
    while (rid()) {                                           \
      IndexTreeL t;                                           \
      for (size_t i = N; i > 0; --i) {                        \
        t =                                                   \
          IndexTreeL({                                        \
              std::make_tuple(                                \
                rid->lo[i - 1],                               \
                rid->hi[i - 1] - rid->lo[i - 1] + 1,          \
                t)});                                         \
      }                                                       \
      m_index_tree = m_index_tree.merged_with(t);             \
      rid++;                                                  \
    }                                                         \
    break;                                                    \
  }

  switch (dom.get_dim()) {
    LEGMS_FOREACH_N(TREE)
  default:
      assert(false);
    break;
  }
#undef TREE
}

std::unique_ptr<ColumnPartition>
Column::partition_on_axes(const std::vector<AxisPartition>& parts) const {

  assert(
    std::all_of(
      parts.begin(),
      parts.end(),
      [this](auto& p) { return p.axes_uid == axes_uid(); }));

  // All variations of partition_on_axes() in the Column class should
  // ultimately call this method, which takes care of the change in semantics
  // of the "dim" field of the AxisPartition structure, as needed by
  // create_partition_on_axes(). For all such methods in Column, the "dim"
  // field simply names an axis, whereas for create_partition_on_axes(), "dim"
  // is a mapping from a named axis to a Column axis (i.e, an axis in the
  // Table index space to an axis in the Column index space).
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
    std::make_unique<ColumnPartition>(
      m_context,
      m_runtime,
      axes_uid(),
      create_partition_on_axes(m_context, m_runtime, index_space(), iparts),
      ds);
}

std::unique_ptr<ColumnPartition>
Column::partition_on_iaxes(const std::vector<int>& ds) const {

  // TODO: verify values in 'ds' are valid axes
  std::vector<AxisPartition> parts =
    legms::map(
      ds,
      [au=axes_uid()](const auto& d) {
        return AxisPartition{au, d, 1, 0, 0, 0};
      });
  return partition_on_axes(parts);
}

std::unique_ptr<ColumnPartition>
Column::partition_on_iaxes(
  const std::vector<std::tuple<int, Legion::coord_t>>& dss) const {

  // TODO: verify axis values in 'dss' are valid axes
  std::vector<AxisPartition> parts =
    legms::map(
      dss,
      [au=axes_uid()](const auto& d_s) {
        auto& [d, s] = d_s;
        return AxisPartition{au, d, s, 0, 0, s - 1};
      });
  return partition_on_axes(parts);
}

std::unique_ptr<ColumnPartition>
Column::projected_column_partition(const ColumnPartition* cp) const {

  assert(cp->axes_uid() == axes_uid());

  if (index_space() == Legion::IndexSpace::NO_SPACE)
    return
      std::make_unique<ColumnPartition>(
        m_context,
        m_runtime,
        axes_uid(),
        Legion::IndexPartition::NO_PART,
        m_axes);

  std::vector<int> dmap = dimensions_map(axes(), cp->axes());

#define CP(I, P)                                              \
  case (I * LEGION_MAX_DIM + P):                              \
    return                                                    \
      std::make_unique<ColumnPartition>(                      \
        m_context,                                            \
        m_runtime,                                            \
        axes_uid(),                                           \
        legms::projected_index_partition<I, P>(               \
          m_context,                                          \
          m_runtime,                                          \
          Legion::IndexPartitionT<I>(cp->index_partition()),  \
          Legion::IndexSpaceT<P>(index_space()),              \
          dmap),                                              \
        m_axes);                                              \
    break;

  switch (cp->axes().size() * LEGION_MAX_DIM + rank()) {
    LEGMS_FOREACH_NN(CP);
  default:
    assert(false);
    // keep compiler happy
    return
      std::make_unique<ColumnPartition>(
        m_context,
        m_runtime,
        axes_uid(),
        Legion::IndexPartition::NO_PART,
        m_axes);
    break;
  }
}

std::unique_ptr<Column>
ColumnGenArgs::operator()(Legion::Context ctx, Legion::Runtime* runtime) const {

  return Column::generator(*this)(ctx, runtime);
}

size_t
ColumnGenArgs::legion_buffer_size(void) const {
  return
    name.size() * sizeof(decltype(name)::value_type) + 1
    + axes_uid.size() * sizeof(decltype(axes_uid)::value_type) + 1
    + sizeof(datatype)
    + vector_serdez<int>::serialized_size(axes)
    + 2 * sizeof(Legion::LogicalRegion)
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

  s = sizeof(Legion::LogicalRegion);
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
  result.values = Legion::CObjectWrapper::wrap(values);
  result.num_keywords = keyword_datatypes.size();
  std::memcpy(
    result.keyword_datatypes,
    keyword_datatypes.data(),
    keyword_datatypes.size() * sizeof(type_tag_t));
  result.keywords = Legion::CObjectWrapper::wrap(keywords);
  return result;
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
