#include "tree_index_space.h"

using namespace legms;

Legion::TaskID TreeIndexSpaceTask<1>::TASK_ID;

void
TreeIndexSpace::register_tasks(Legion::Runtime* runtime) {
#define REG_TASK(D)                             \
  TreeIndexSpaceTask<D>::register_task(         \
    runtime,                                    \
    TreeIndexSpace::task_id<D>(runtime));

  LEGMS_FOREACH_N(REG_TASK);

#undef REG_TASK
}

Legion::IndexSpace
legms::tree_index_space(
  const IndexTreeL& tree,
  Legion::Context ctx,
  Legion::Runtime* runtime) {

  auto rank = tree.rank();
  assert(rank);

#define TIS(N)                                              \
  case N: {                                                 \
    auto result = tree_index_space<N>(tree, ctx, runtime);  \
    return result;                                          \
  }

  switch(rank.value()) {
    LEGMS_FOREACH_N(TIS);
  default:
    assert(false);
  }
#undef TIS
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
