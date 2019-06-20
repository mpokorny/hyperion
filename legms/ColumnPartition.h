#ifndef LEGMS_COLUMN_PARTITION_H_
#define LEGMS_COLUMN_PARTITION_H_

#include <algorithm>
#include <vector>

#include "legms.h"
#include "utility.h"
#include "ColumnPartition_c.h"

#include "c_util.h"

namespace legms {

class ColumnPartition {
public:

  ColumnPartition(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const std::string& axes_uid,
    const Legion::IndexPartition& ip,
    const std::vector<int>& axes)
    : m_context(ctx)
    , m_runtime(runtime)
    , m_axes_uid(axes_uid)
    , m_index_partition(ip)
    , m_axes(axes) {
  }

  template <typename D, std::enable_if_t<!std::is_same_v<D, int>, int> = 0>
  ColumnPartition(
    Legion::Context ctx,
    Legion::Runtime* runtime,
    const Legion::IndexPartition& ip,
    const std::vector<D>& axes)
    : m_context(ctx)
    , m_runtime(runtime)
    , m_axes_uid(Axes<D>::uid)
    , m_index_partition(ip)
    , m_axes(map_to_int(axes)) {
  }

  const std::string&
  axes_uid() const {
    return m_axes_uid;

  }
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

  Legion::Context m_context;

  Legion::Runtime* m_runtime;

  std::string m_axes_uid;

  Legion::IndexPartition m_index_partition;

  std::vector<int> m_axes;
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
