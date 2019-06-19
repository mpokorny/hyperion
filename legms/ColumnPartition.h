#ifndef LEGMS_COLUMN_PARTITION_H_
#define LEGMS_COLUMN_PARTITION_H_

#include <algorithm>
#include <vector>

#include "legms.h"

#include "ColumnPartition_c.h"

#include "c_util.h"

namespace legms {

class ColumnPartition {
public:

  Legion::IndexPartition
  index_partition() const {
    return m_index_partition;
  }

  const std::vector<int>&
  axes() const {
    return m_axes;
  }

  virtual ~ColumnPartition() {
    // FIXME:
    // if (m_index_partition != Legion::IndexPartition::NO_PART)
    //   m_runtime->destroy_index_partition(m_context, m_index_partition);
  }

protected:

  ColumnPartition(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const Legion::IndexPartition& ip,
    const std::vector<int>& axes)
    : m_context(ctx)
    , m_runtime(runtime)
    , m_index_partition(ip)
    , m_axes(axes) {
  }

  Legion::Context m_context;

  Legion::Runtime* m_runtime;

  Legion::IndexPartition m_index_partition;

  std::vector<int> m_axes;
};

template <typename D>
class ColumnPartitionT
  : public ColumnPartition {
public:

  ColumnPartitionT(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const Legion::IndexPartition& ip,
    const std::vector<D>& axes)
    : ColumnPartition(
      ctx,
      runtime,
      ip,
      std::vector<int>(axes.size())) {

    for (size_t i = 0; i < axes.size(); ++i)
      m_axes[i] = static_cast<int>(axes[i]);
  }

  std::vector<D>
  axesT() const {
    std::vector<D> result;
    result.reserve(m_axes.size());
    std::transform(
      m_axes.begin(),
      m_axes.end(),
      std::back_inserter(result),
      [](auto& i) { return static_cast<D>(i); });
    return result;
  }
};

template <>
struct CObjectWrapper::UniqueWrapper<ColumnPartition> {
  typedef legms_column_partition_t type_t;
};

template <>
struct CObjectWrapper::UniqueWrapped<legms_column_partition_t> {
  typedef ColumnPartition type_t;
  typedef std::unique_ptr<type_t> impl_t;
};

} // end namespace legms

#endif // LEGMS_COLUMN_PARTITION_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
