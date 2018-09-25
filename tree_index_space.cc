#include "tree_index_space.h"

using namespace legms;

Legion::TaskID legms::TreeIndexSpaceTask<1>::TASK_ID;

void
legms::TreeIndexSpace::register_tasks(Legion::Runtime* runtime) {
    // TreeIndexSpaceTask<8>::register_task(
    //   runtime,
    //   TreeIndexSpace::task_id<8>(runtime));
    // TreeIndexSpaceTask<7>::register_task(
    //   runtime,
    //   TreeIndexSpace::task_id<7>(runtime));
    // TreeIndexSpaceTask<6>::register_task(
    //   runtime,
    //   TreeIndexSpace::task_id<6>(runtime));
    // TreeIndexSpaceTask<5>::register_task(
    //   runtime,
    //   TreeIndexSpace::task_id<5>(runtime));
    // TreeIndexSpaceTask<4>::register_task(
    //   runtime,
    //   TreeIndexSpace::task_id<4>(runtime));
    TreeIndexSpaceTask<3>::register_task(
        runtime,
        TreeIndexSpace::task_id<3>(runtime));
    TreeIndexSpaceTask<2>::register_task(
        runtime,
        TreeIndexSpace::task_id<2>(runtime));
    TreeIndexSpaceTask<1>::register_task(
        runtime,
        TreeIndexSpace::task_id<1>(runtime));
}

Legion::IndexSpace
legms::tree_index_space(
    const IndexTreeL& tree,
    Legion::Context ctx,
    Legion::Runtime* runtime) {

    auto rank = tree.rank();
    assert(rank);
    switch(rank.value()) {
    case 1: {
        auto result = tree_index_space<1>(tree, ctx, runtime);
        return result;
    }
    case 2: {
        auto result = tree_index_space<2>(tree, ctx, runtime);
        return result;
    }
    case 3: {
        auto result = tree_index_space<3>(tree, ctx, runtime);
        return result;
    }
    // case 4:
    //   return tree_index_space<4>(tree, ctx, runtime);
    // case 5:
    //   return tree_index_space<5>(tree, ctx, runtime);
    // case 6:
    //   return tree_index_space<6>(tree, ctx, runtime);
    // case 7:
    //   return tree_index_space<7>(tree, ctx, runtime);
    // case 8:
    //   return tree_index_space<8>(tree, ctx, runtime);
    default:
        assert(false);
    }
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
