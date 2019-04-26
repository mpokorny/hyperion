#include <cassert>

#include "legms.h"
#include "tree_index_space.h"
#include "Column.h"
#include "Table.h"

using namespace legms;

using namespace Legion;

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
    m_runtime->destroy_index_space(m_context, index_space);
  } else {
    m_logical_region = LogicalRegion::NO_REGION;
  }
}

void
Column::init(LogicalRegion region) {

  m_logical_region = region;

  Domain dom = m_runtime->get_index_space_domain(region.get_index_space());
  assert(dom.dense()); // FIXME: remove

#define TREE(N)                                     \
  case (N): {                                       \
    Rect<N> rect(dom);                              \
    m_index_tree = IndexTreeL();                    \
    for (size_t i = N; i > 0; --i) {                \
      m_index_tree =                                \
        IndexTreeL({                                \
            std::make_tuple(                        \
              rect.lo[i - 1],                       \
              rect.hi[i - 1] - rect.lo[i - 1] + 1,  \
              m_index_tree)});                      \
    }                                               \
    break;                                          \
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
Column::partition_on_axes(
  const std::vector<std::tuple<int, Legion::coord_t>>& axes) const {
  std::vector<AxisPartition<int>> parts;
  parts.reserve(axes.size());
  std::transform(
    axes.begin(),
    axes.end(),
    std::back_inserter(parts),
    [](auto& d_s) {
      auto& [d, s] = d_s;
      return AxisPartition<int> {d, s, 0, 0, s - 1}; });
  return partition_on_axes(parts);
}

std::unique_ptr<ColumnPartition>
Column::partition_on_axes(const std::vector<int>& axes) const {
  std::vector<AxisPartition<int>> parts;
  parts.reserve(axes.size());
  std::transform(
    axes.begin(),
    axes.end(),
    std::back_inserter(parts),
    [](auto& d) { return AxisPartition<int> {d, 1, 0, 0, 0}; });
  return partition_on_axes(parts);
}

size_t
ColumnGenArgs::legion_buffer_size(void) const {
  return
    name.size() * sizeof(decltype(name)::value_type) + 1
    + axes_uid.size() * sizeof(decltype(axes_uid)::value_type) + 1
    + sizeof(datatype)
    + vector_serdez<int>::serialized_size(axes)
    + 2 * sizeof(Legion::LogicalRegion)
    + vector_serdez<casacore::DataType>::serialized_size(keyword_datatypes);
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

  buff += vector_serdez<casacore::DataType>::serialize(keyword_datatypes, buff);

  return buff - static_cast<char *>(buffer);
}

size_t
ColumnGenArgs::legion_deserialize(const void *buffer) {
  const char *buff = static_cast<const char*>(buffer);

  name = *buff;
  buff += name.size() * sizeof(decltype(name)::value_type) + 1;

  axes_uid = *buff;
  buff += axes_uid.size() * sizeof(decltype(axes_uid)::value_type) + 1;

  datatype = *reinterpret_cast<const decltype(datatype) *>(buff);
  buff += sizeof(datatype);

  buff += vector_serdez<int>::deserialize(axes, buff);

  values = *reinterpret_cast<const decltype(values) *>(buff);
  buff += sizeof(values);

  keywords = *reinterpret_cast<const decltype(values) *>(buff);
  buff += sizeof(keywords);

  buff +=
    vector_serdez<casacore::DataType>::deserialize(keyword_datatypes, buff);

  return buff - static_cast<const char*>(buffer);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
