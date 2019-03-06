#include <cassert>

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
  m_index_tree =
    IndexTreeL(
      m_runtime->get_index_space_domain(m_logical_region.get_index_space())
      .hi()[0] + 1);
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
    + sizeof(datatype)
    + sizeof(size_t) + axes.size() * sizeof(decltype(axes)::value_type)
    + 2 * sizeof(Legion::LogicalRegion);
}

size_t
ColumnGenArgs::legion_serialize(void *buffer) const {
  char* buff = static_cast<char *>(buffer);

  size_t s = name.size() * sizeof(decltype(name)::value_type) + 1;
  memcpy(buff, name.c_str(), s);
  buff += s;

  s = sizeof(datatype);
  memcpy(buff, &datatype, s);
  buff += s;

  size_t asz = axes.size();
  s = sizeof(asz);
  memcpy(buff, &asz, s);
  buff += s;

  s = asz * sizeof(decltype(axes)::value_type);
  memcpy(buff, axes.data(), s);
  buff += s;

  s = sizeof(Legion::LogicalRegion);
  memcpy(buff, &values, s);
  buff += s;

  memcpy(buff, &keywords, s);
  buff += s;

  return buff - static_cast<char *>(buffer);
}

size_t
ColumnGenArgs::legion_deserialize(const void *buffer) {
  const char *buff = static_cast<const char*>(buffer);

  name = *buff;
  buff += name.size() * sizeof(decltype(name)::value_type) + 1;

  datatype = *reinterpret_cast<const decltype(datatype) *>(buff);
  buff += sizeof(datatype);

  axes.resize(*reinterpret_cast<const size_t *>(buff));
  buff += sizeof(size_t);

  memcpy(axes.data(), buff, axes.size() * sizeof(decltype(axes)::value_type));
  buff += axes.size() * sizeof(decltype(axes)::value_type);

  values = *reinterpret_cast<const decltype(values) *>(buff);
  buff += sizeof(values);

  keywords = *reinterpret_cast<const decltype(values) *>(buff);
  buff += sizeof(keywords);

  return buff - static_cast<const char*>(buffer);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End: