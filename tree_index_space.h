#ifndef LEGMS_TREE_INDEX_SPACE_H_
#define LEGMS_TREE_INDEX_SPACE_H_

#include <algorithm>
#include <limits>
#include <ostream>
#include <vector>

#include "legion.h"
#include "utility.h"
#include "IndexTree.h"

namespace legms {

typedef IndexTree<Legion::coord_t> IndexTreeL;

class TreeIndexSpace {
public:

  constexpr static const int MAX_DIM = 3;

  enum {
    FID_ENVELOPE,
    FID_TREE,
    FID_INDEX_SPACE,
    FID_PART_COLOR,
  };

  template <int DIM>
  static Legion::TaskID
  task_id(Legion::Runtime* runtime) {
    static_assert(DIM <= MAX_DIM);
    return runtime->
      generate_library_task_ids("legms::TreeIndexSpaceTask", MAX_DIM) + DIM - 1;
  }

  static void
  register_tasks(Legion::Runtime* runtime);
};

template <int DIM, typename CB>
Legion::IndexSpaceT<DIM>
offspring_index_space(
  ssize_t n,
  CB cbfn,
  Legion::Context ctx,
  Legion::Runtime* runtime);

template <int DIM>
class TreeIndexSpaceTask
  : public Legion::IndexTaskLauncher {
public:

  constexpr static const char * const TASK_NAME = "tree_space";
  static Legion::TaskID TASK_ID;

  TreeIndexSpaceTask(
    Legion::LogicalPartition child_input_lp,
    Legion::LogicalPartition child_output_lp,
    Legion::LogicalRegion child_input_lr,
    Legion::LogicalRegion child_output_lr,
    const Legion::IndexSpace& launch_space)
    : Legion::IndexTaskLauncher(
      TASK_ID,
      launch_space,
      Legion::TaskArgument(),
      Legion::ArgumentMap()) {

    Legion::RegionRequirement child_input_req(
      child_input_lp,
      0,
      READ_ONLY,
      EXCLUSIVE,
      child_input_lr);
    child_input_req.add_field(TreeIndexSpace::FID_ENVELOPE);
    child_input_req.add_field(TreeIndexSpace::FID_TREE);
    add_region_requirement(child_input_req);
    Legion::RegionRequirement child_output_req(
      child_output_lp,
      0,
      WRITE_DISCARD,
      EXCLUSIVE,
      child_output_lr);
    child_output_req.add_field(TreeIndexSpace::FID_INDEX_SPACE);
    add_region_requirement(child_output_req);
  }

  void
  dispatch(Legion::Context ctx, Legion::Runtime *runtime) {
    runtime->execute_index_space(ctx, *this);
  }

  static void
  base_impl(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *runtime) {

    const Legion::FieldAccessor<
      READ_ONLY,
      Legion::Rect<DIM>,
      1,
      Legion::coord_t,
      Realm::AffineAccessor<Legion::Rect<DIM>,1,Legion::coord_t>,
      false> envelopes(regions[0], TreeIndexSpace::FID_ENVELOPE);
    const Legion::FieldAccessor<
      READ_ONLY,
      IndexTreeL,
      1,
      Legion::coord_t,
      Realm::AffineAccessor<IndexTreeL,1,Legion::coord_t>,
      false> trees(regions[0], TreeIndexSpace::FID_TREE);
    const Legion::FieldAccessor<
      WRITE_DISCARD,
      Legion::IndexSpaceT<DIM>,
      1,
      Legion::coord_t,
      Realm::AffineAccessor<Legion::IndexSpaceT<DIM>,1,Legion::coord_t>,
      false> ispaces(regions[1], TreeIndexSpace::FID_INDEX_SPACE);

    Legion::Rect<1> rect = runtime->get_index_space_domain(
      ctx,
      task->regions[0].region.get_index_space());

    for (Legion::PointInRectIterator<1> pir(rect); pir(); pir++) {
      const Legion::Rect<DIM>& envelope = envelopes[*pir];
      const IndexTreeL& tree = trees[*pir];

      auto tree_rect = tree.envelope();
      assert(tree_rect.size() <= DIM);
      Legion::coord_t tree_rect_lo[DIM], tree_rect_hi[DIM];
      size_t fixd = DIM - tree_rect.size();
      for (size_t i = 0; i < fixd; ++i) {
        tree_rect_lo[i] = envelope.lo[i];
        tree_rect_hi[i] = envelope.hi[i];
      }
      for (size_t i = 0; i < tree_rect.size(); ++i)
        std::tie(tree_rect_lo[i + fixd], tree_rect_hi[i + fixd]) = tree_rect[i];
      if (tree.is_array()) {
        ispaces[*pir] = runtime->create_index_space(
          ctx,
          Legion::Rect<DIM>(
            Legion::Point<DIM>(tree_rect_lo),
            Legion::Point<DIM>(tree_rect_hi))) ;
      } else {
        auto children = tree.children();
        ispaces[*pir] = offspring_index_space<DIM>(
          children.size(),
          [&](Legion::Point<1> chidx, Legion::Rect<DIM>& rect, IndexTreeL& tr) {
            Legion::coord_t i, n;
            IndexTreeL t;
            std::tie(i, n, t) = children[chidx];
            tree_rect_lo[fixd] = i;
            tree_rect_hi[fixd] = i + n - 1;
            rect =
              Legion::Rect<DIM>(
                Legion::Point<DIM>(tree_rect_lo),
                Legion::Point<DIM>(tree_rect_hi));
            tr = t;
          },
          ctx,
          runtime);
      }
    }
  }

  static void
  register_task(Legion::Runtime* runtime, Legion::TaskID tid) {
    TASK_ID = tid;
    Legion::TaskVariantRegistrar registrar(TASK_ID, TASK_NAME);
    registrar.add_constraint(
      Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    runtime->register_task_variant<base_impl>(registrar);
  }
};

template <>
class TreeIndexSpaceTask<1>
  : public Legion::IndexTaskLauncher {
public:

  constexpr static const char * const TASK_NAME = "tree_space";
  static Legion::TaskID TASK_ID;

  TreeIndexSpaceTask(
    Legion::LogicalPartition child_input_lp,
    Legion::LogicalPartition child_output_lp,
    Legion::LogicalRegion child_input_lr,
    Legion::LogicalRegion child_output_lr,
    const Legion::IndexSpace& launch_space)
    : Legion::IndexTaskLauncher(
      TASK_ID,
      launch_space,
      Legion::TaskArgument(),
      Legion::ArgumentMap()) {

    Legion::RegionRequirement child_input_req(
      child_input_lp,
      0,
      READ_ONLY,
      EXCLUSIVE,
      child_input_lr);
    child_input_req.add_field(TreeIndexSpace::FID_ENVELOPE);
    child_input_req.add_field(TreeIndexSpace::FID_TREE);
    add_region_requirement(child_input_req);
    Legion::RegionRequirement child_output_req(
      child_output_lp,
      0,
      WRITE_DISCARD,
      EXCLUSIVE,
      child_output_lr);
    child_output_req.add_field(TreeIndexSpace::FID_INDEX_SPACE);
    add_region_requirement(child_output_req);
  }

  void
  dispatch(Legion::Context ctx, Legion::Runtime *runtime) {
    runtime->execute_index_space(ctx, *this);
  }

  static void
  base_impl(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *runtime) {

    const Legion::FieldAccessor<
      READ_ONLY,
      Legion::Rect<1>,
      1,
      Legion::coord_t,
      Realm::AffineAccessor<Legion::Rect<1>,1,Legion::coord_t>,
      false> envelopes(regions[0], TreeIndexSpace::FID_ENVELOPE);
    const Legion::FieldAccessor<
      READ_ONLY,
      IndexTreeL,
      1,
      Legion::coord_t,
      Realm::AffineAccessor<IndexTreeL,1,Legion::coord_t>,
      false> trees(regions[0], TreeIndexSpace::FID_TREE);
    const Legion::FieldAccessor<
      WRITE_DISCARD,
      Legion::IndexSpaceT<1>,
      1,
      Legion::coord_t,
      Realm::AffineAccessor<Legion::IndexSpaceT<1>,1,Legion::coord_t>,
      false> ispaces(regions[1], TreeIndexSpace::FID_INDEX_SPACE);

    Legion::Rect<1> rect = runtime->get_index_space_domain(
      ctx,
      task->regions[0].region.get_index_space());

    for (Legion::PointInRectIterator<1> pir(rect); pir(); pir++) {
      const IndexTreeL& tree = trees[*pir];

      auto tree_rect = tree.envelope();
      assert(tree_rect.size() == 1);
      Legion::coord_t tree_rect_lo, tree_rect_hi;
      std::tie(tree_rect_lo, tree_rect_hi) = tree_rect[0];
      if (tree.is_array()) {
        ispaces[*pir] = runtime->create_index_space(
          ctx,
          Legion::Rect<1>(
            Legion::Point<1>(tree_rect_lo),
            Legion::Point<1>(tree_rect_hi))) ;
      } else {
        auto children = tree.children();
        ispaces[*pir] = offspring_index_space<1>(
          children.size(),
          [&](Legion::Point<1> chidx, Legion::Rect<1>& rect, IndexTreeL& tr) {
            Legion::coord_t i, n;
            IndexTreeL t;
            std::tie(i, n, t) = children[chidx];
            tree_rect_lo = i;
            tree_rect_hi = i + n - 1;
            rect = Legion::Rect<1>(
              Legion::Point<1>(tree_rect_lo),
              Legion::Point<1>(tree_rect_hi));
            tr = t;
          },
          ctx,
          runtime);
      }
    }
  }

  static void
  register_task(Legion::Runtime* runtime, Legion::TaskID tid) {
    TASK_ID = tid;
    Legion::TaskVariantRegistrar registrar(TASK_ID, TASK_NAME);
    registrar.add_constraint(
      Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    runtime->register_task_variant<base_impl>(registrar);
  }
};

template <int DIM>
Legion::TaskID TreeIndexSpaceTask<DIM>::TASK_ID;

template <int DIM>
class ChildInputInitTask
  : Legion::InlineLauncher {
public:

  ChildInputInitTask(Legion::LogicalRegion lr, ssize_t n)
    : Legion::InlineLauncher(
      Legion::RegionRequirement(
        lr,
        {TreeIndexSpace::FID_ENVELOPE, TreeIndexSpace::FID_TREE},
        {TreeIndexSpace::FID_ENVELOPE, TreeIndexSpace::FID_TREE},
        WRITE_DISCARD,
        EXCLUSIVE,
        lr)),
      m_nch(n) {
  }

  template <typename CB>
  void
  dispatch(CB cbfn, Legion::Context ctx, Legion::Runtime* runtime) {
    Legion::PhysicalRegion region = runtime->map_region(ctx, *this);
    const Legion::FieldAccessor<
      WRITE_DISCARD,
      Legion::Rect<DIM>,
      1,
      Legion::coord_t,
      Realm::AffineAccessor<Legion::Rect<DIM>,1,Legion::coord_t>,
      false> acc_rect(region, TreeIndexSpace::FID_ENVELOPE);
    const Legion::FieldAccessor<
      WRITE_DISCARD,
      IndexTreeL,
      1,
      Legion::coord_t,
      Realm::AffineAccessor<IndexTreeL,1,Legion::coord_t>,
      false> acc_tree(region, TreeIndexSpace::FID_TREE);
    for (Legion::coord_t i = 0; i < m_nch; ++i) {
      ::new (acc_tree.ptr(i)) IndexTreeL;
      cbfn(i, acc_rect[i], acc_tree[i]);
    }
    runtime->unmap_region(ctx, region);
  }

  ssize_t m_nch;
};

template <int DIM>
class ChildOutputCollectTask
  : Legion::InlineLauncher {
public:

  ChildOutputCollectTask(Legion::LogicalRegion lr, ssize_t n)
    : Legion::InlineLauncher(
      Legion::RegionRequirement(
        lr,
        {TreeIndexSpace::FID_INDEX_SPACE},
        {TreeIndexSpace::FID_INDEX_SPACE},
        READ_ONLY,
        EXCLUSIVE,
        lr)),
      m_nch(n) {
  }

  Legion::IndexSpaceT<DIM>
  dispatch(Legion::Context ctx, Legion::Runtime* runtime) {
    Legion::PhysicalRegion region = runtime->map_region(ctx, *this);
    const Legion::FieldAccessor<
      READ_ONLY,
      Legion::IndexSpaceT<DIM>,
      1,
      Legion::coord_t,
      Realm::AffineAccessor<Legion::IndexSpaceT<DIM>,1,Legion::coord_t>,
      false> acc_ispace(region, TreeIndexSpace::FID_INDEX_SPACE);

    std::vector<Legion::IndexSpaceT<DIM>> ispaces;
    for (Legion::coord_t i = 0; i < m_nch; ++i)
      ispaces.push_back(acc_ispace[i]);
    runtime->unmap_region(ctx, region);
    Legion::IndexSpaceT<DIM> result = runtime->union_index_spaces(ctx, ispaces);
    std::for_each(
      ispaces.begin(),
      ispaces.end(),
      [&ctx, runtime](auto& is) {
        runtime->destroy_index_space(ctx, is);
      });
    return result;
  }

  ssize_t m_nch;
};

template <int DIM, int COLOR_DIM>
class PartitionIndexTreeTask
  : public Legion::InlineLauncher {
public:

  PartitionIndexTreeTask(
    Legion::LogicalRegionT<DIM> lr,
    Legion::IndexSpaceT<COLOR_DIM> colors)
    : Legion::InlineLauncher(
      Legion::RegionRequirement(
        lr,
        {TreeIndexSpace::FID_PART_COLOR},
        {TreeIndexSpace::FID_PART_COLOR},
        WRITE_DISCARD,
        EXCLUSIVE,
        lr))
    , m_lr(lr)
    , m_colors(colors) {
  }

  template <typename CB>
  Legion::IndexPartitionT<DIM>
  dispatch(CB cbfn, Legion::Context ctx, Legion::Runtime* runtime) {
    Legion::PhysicalRegion region = runtime->map_region(ctx, *this);
    const Legion::FieldAccessor<
      WRITE_DISCARD,
      Legion::Point<COLOR_DIM>,
      DIM,
      Legion::coord_t,
      Realm::AffineAccessor<Legion::Point<COLOR_DIM>,DIM,Legion::coord_t>,
      false> part_colors(region, TreeIndexSpace::FID_PART_COLOR);
    Legion::DomainT<DIM> domain =
      runtime->get_index_space_domain(ctx, m_lr.get_index_space());
    for (Legion::PointInDomainIterator<DIM> pid(domain); pid(); pid++)
      part_colors[*pid] = cbfn(pid);
    Legion::IndexPartitionT<DIM> result =
      runtime->create_partition_by_field<DIM>(
        ctx,
        m_lr,
        m_lr,
        TreeIndexSpace::FID_PART_COLOR,
        m_colors);
    runtime->unmap_region(ctx, region);
    return result;
  }

  Legion::LogicalRegionT<DIM> m_lr;
  Legion::IndexSpaceT<1> m_colors;
};

template <int DIM, typename CB>
Legion::IndexSpaceT<DIM>
offspring_index_space(
  ssize_t n,
  CB cbfn,
  Legion::Context ctx,
  Legion::Runtime* runtime) {

  Legion::Rect<1> child_is_rect =
    Legion::Rect<1>(Legion::Point<1>(0), Legion::Point<1>(n - 1));
  Legion::IndexSpaceT<1> child_is =
    runtime->create_index_space(ctx, child_is_rect);

  Legion::FieldSpace child_input_fs = runtime->create_field_space(ctx);
  {
    auto fa = runtime->create_field_allocator(ctx, child_input_fs);
    fa.allocate_field(sizeof(Legion::Rect<DIM>), TreeIndexSpace::FID_ENVELOPE);
    fa.allocate_field(
      sizeof(IndexTreeL),
      TreeIndexSpace::FID_TREE,
      SerdezManager::INDEX_TREE_SID);
  }
  Legion::LogicalRegion child_input_lr =
    runtime->create_logical_region(ctx, child_is, child_input_fs);

  ChildInputInitTask<DIM> init(child_input_lr, n);
  init.dispatch(cbfn, ctx, runtime);

  Legion::FieldSpace child_output_fs = runtime->create_field_space(ctx);
  {
    auto fa = runtime->create_field_allocator(ctx, child_output_fs);
    fa.allocate_field(
      sizeof(Legion::IndexSpaceT<DIM>),
      TreeIndexSpace::FID_INDEX_SPACE);
  }
  Legion::LogicalRegion child_output_lr =
    runtime->create_logical_region(ctx, child_is, child_output_fs);

  auto child_ip = runtime->create_equal_partition(ctx, child_is, child_is);
  Legion::LogicalPartition child_input_lp =
    runtime->get_logical_partition(ctx, child_input_lr, child_ip);
  Legion::LogicalPartition child_output_lp =
    runtime->get_logical_partition(ctx, child_output_lr, child_ip);
  TreeIndexSpaceTask<DIM> tree_space(
    child_input_lp,
    child_output_lp,
    child_input_lr,
    child_output_lr,
    child_is);
  tree_space.dispatch(ctx, runtime);

  runtime->destroy_index_partition(ctx, child_ip);
  runtime->destroy_logical_partition(ctx, child_input_lp);
  runtime->destroy_logical_partition(ctx, child_output_lp);

  ChildOutputCollectTask<DIM> collect(child_output_lr, n);
  Legion::IndexSpaceT<DIM> result = collect.dispatch(ctx, runtime);

  runtime->destroy_logical_region(ctx, child_input_lr);
  runtime->destroy_logical_region(ctx, child_output_lr);
  runtime->destroy_field_space(ctx, child_input_fs);
  runtime->destroy_field_space(ctx, child_output_fs);
  runtime->destroy_index_space(ctx, child_is);
  return result;
}

template <int DIM>
Legion::IndexSpaceT<DIM>
tree_index_space(
  const IndexTreeL& tree,
  Legion::Context ctx,
  Legion::Runtime* runtime) {

  auto rank = tree.rank();
  assert(rank);
  assert(rank.value() == DIM);
  return
    offspring_index_space<DIM>(
      1,
      [&](Legion::Point<1>, Legion::Rect<DIM>& rect, IndexTreeL& tr) {
        auto env = tree.envelope();
        Legion::coord_t tree_rect_lo[DIM], tree_rect_hi[DIM];
        for (size_t i = 0; i < DIM; ++i)
          std::tie(tree_rect_lo[i], tree_rect_hi[i]) = env[i];
        rect = Legion::Rect<DIM>(
          Legion::Point<DIM>(tree_rect_lo),
          Legion::Point<DIM>(tree_rect_hi));
        tr = tree;
      },
      ctx,
      runtime);
}

template <> inline
Legion::IndexSpaceT<1>
tree_index_space<1>(
  const IndexTreeL& tree,
  Legion::Context ctx,
  Legion::Runtime* runtime) {

  auto rank = tree.rank();
  assert(rank);
  assert(rank.value() == 1);
  return
    offspring_index_space<1>(
      1,
      [&](Legion::Point<1>, Legion::Rect<1>& rect, IndexTreeL& tr) {
        auto env = tree.envelope();
        auto& [tree_rect_lo, tree_rect_hi] = env[0];
        rect = Legion::Rect<1>(
          Legion::Point<1>(tree_rect_lo),
          Legion::Point<1>(tree_rect_hi));
        tr = tree;
      },
      ctx,
      runtime);
}

Legion::IndexSpace
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
