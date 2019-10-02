#ifndef LEGMS_TREE_INDEX_SPACE_H_
#define LEGMS_TREE_INDEX_SPACE_H_

#pragma GCC visibility push(default)
#include <algorithm>
#include <limits>
#include <ostream>
#include <vector>
#pragma GCC visibility pop

#include <legms/legms.h>
#include <legms/utility.h>
#include <legms/IndexTree.h>

namespace legms {

typedef IndexTree<Legion::coord_t> IndexTreeL;

class LEGMS_API TreeIndexSpaceTask {
public:

  static Legion::TaskID TASK_ID;
  static constexpr const char* const TASK_NAME = "TreeIndexSpaceTask";

  TreeIndexSpaceTask(const IndexTreeL& it);

  Legion::Future /* Legion::IndexSpace */
  dispatch(Legion::Context context, Legion::Runtime* runtime);

  static Legion::IndexSpace
  base_impl(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context context,
    Legion::Runtime *runtime);

  static void
  preregister_task();

private:

  std::vector<Legion::coord_t> m_blocks;

  std::vector<IndexTreeL> m_trees;
};

LEGMS_API Legion::IndexSpace
tree_index_space(
  const IndexTreeL& tree,
  Legion::Context ctx,
  Legion::Runtime* runtime);

} // end namespace legms

#endif // LEGMS_TREE_INDEX_SPACE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
