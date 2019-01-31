#include <algorithm>
#include <limits>
#include <ostream>
#include <vector>

#include "legion.h"
#include "IndexTree.h"

using namespace legms;
using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  TREE_SPACE_TASK_ID,
  REPORT_INDEXES_TASK_ID,
};

enum FieldIDs {
  FID_ENVELOPE,
  FID_TREE,
  FID_INDEX_SPACE,

  FID_PART_COLOR,
};

enum SerdezIDs {
  SID_TREE = 1,
};

typedef IndexTree<coord_t> IndexTreeL;

class tree_serdez {
public:
  typedef IndexTreeL FIELD_TYPE;

  static const size_t MAX_SERIALIZED_SIZE = std::numeric_limits<size_t>::max();

  static size_t
  serialized_size(const IndexTreeL& val) {
    return val.serialized_size();
  }

  static size_t
  serialize(const IndexTreeL& val, void *buffer) {
    return val.serialize(reinterpret_cast<char*>(buffer));
  }

  static size_t
  deserialize(IndexTreeL& val, const void *buffer) {
    val = IndexTreeL::deserialize(static_cast<const char*>(buffer));
    return *reinterpret_cast<const size_t *>(buffer);
  }

  static void
  destroy(IndexTreeL&) {
  }
};

template <int DIM, typename CB>
IndexSpaceT<DIM>
offspring_index_space(ssize_t n, CB cbfn, Context ctx, Runtime* runtime);

template <int DIM>
class IndexTreeSpaceTask
  : public IndexTaskLauncher {
public:

  constexpr static const char * const TASK_NAME = "tree_space";
  static const int TASK_ID = TREE_SPACE_TASK_ID;

  IndexTreeSpaceTask(
    LogicalPartition child_input_lp,
    LogicalPartition child_output_lp,
    LogicalRegion child_input_lr,
    LogicalRegion child_output_lr,
    const IndexSpace& launch_space)
    : IndexTaskLauncher(TASK_ID, launch_space, TaskArgument(), ArgumentMap()) {

    RegionRequirement child_input_req(
      child_input_lp,
      0,
      READ_ONLY,
      EXCLUSIVE,
      child_input_lr);
    child_input_req.add_field(FID_ENVELOPE);
    child_input_req.add_field(FID_TREE);
    add_region_requirement(child_input_req);
    RegionRequirement child_output_req(
      child_output_lp,
      0,
      WRITE_DISCARD,
      EXCLUSIVE,
      child_output_lr);
    child_output_req.add_field(FID_INDEX_SPACE);
    add_region_requirement(child_output_req);
  }

  void
  dispatch(Context ctx, Runtime *runtime) {
    runtime->execute_index_space(ctx, *this);
  }

  static void
  base_impl(
    const Task* task,
    const std::vector<PhysicalRegion>& regions,
    Context ctx,
    Runtime *runtime) {

    const FieldAccessor<
      READ_ONLY,
      Rect<DIM>,
      1,
      coord_t,
      Realm::AffineAccessor<Rect<DIM>,1,coord_t>,
      false> envelopes(regions[0], FID_ENVELOPE);
    const FieldAccessor<
      READ_ONLY,
      IndexTreeL,
      1,
      coord_t,
      Realm::AffineAccessor<IndexTreeL,1,coord_t>,
      false> trees(regions[0], FID_TREE);
    const FieldAccessor<
      WRITE_DISCARD,
      IndexSpaceT<DIM>,
      1,
      coord_t,
      Realm::AffineAccessor<IndexSpaceT<DIM>,1,coord_t>,
      false> ispaces(regions[1], FID_INDEX_SPACE);

    Rect<1> rect = runtime->get_index_space_domain(
      ctx,
      task->regions[0].region.get_index_space());

    for (PointInRectIterator<1> pir(rect); pir(); pir++) {
      const Rect<DIM>& envelope = envelopes[*pir];
      const IndexTreeL& tree = trees[*pir];

      auto tree_rect = tree.envelope();
      assert(tree_rect.size() <= DIM);
      coord_t tree_rect_lo[DIM], tree_rect_hi[DIM];
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
          Rect<DIM>(Point<DIM>(tree_rect_lo), Point<DIM>(tree_rect_hi))) ;
      } else {
        auto children = tree.children();
        ispaces[*pir] = offspring_index_space<DIM>(
          children.size(),
          [&](Point<1> chidx, Rect<DIM>& rect, IndexTreeL& tr) {
            coord_t i, n;
            IndexTreeL t;
            std::tie(i, n, t) = children[chidx];
            tree_rect_lo[fixd] = i;
            tree_rect_hi[fixd] = i + n - 1;
            rect = Rect<DIM>(Point<DIM>(tree_rect_lo), Point<DIM>(tree_rect_hi));
            tr = t;
          },
          ctx,
          runtime);
      }
    }
  }

  static void
  register_task() {
    TaskVariantRegistrar registrar(TASK_ID, TASK_NAME);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<base_impl>(registrar, TASK_NAME);
  }
};

template <int DIM>
class ChildInputInitTask
  : InlineLauncher {
public:

  ChildInputInitTask(LogicalRegion lr, ssize_t n)
    : InlineLauncher(
      RegionRequirement(
        lr,
        {FID_ENVELOPE, FID_TREE},
        {FID_ENVELOPE, FID_TREE},
        WRITE_DISCARD,
        EXCLUSIVE,
        lr)),
      m_nch(n) {
  }

  template <typename CB>
  void
  dispatch(CB cbfn, Context ctx, Runtime* runtime) {
    PhysicalRegion region = runtime->map_region(ctx, *this);
    const FieldAccessor<
      WRITE_DISCARD,
      Rect<DIM>,
      1,
      coord_t,
      Realm::AffineAccessor<Rect<DIM>,1,coord_t>,
      false> acc_rect(region, FID_ENVELOPE);
    const FieldAccessor<
      WRITE_DISCARD,
      IndexTreeL,
      1,
      coord_t,
      Realm::AffineAccessor<IndexTreeL,1,coord_t>,
      false> acc_tree(region, FID_TREE);
    for (coord_t i = 0; i < m_nch; ++i) {
      ::new (acc_tree.ptr(i)) IndexTreeL;
      cbfn(i, acc_rect[i], acc_tree[i]);
    }
    runtime->unmap_region(ctx, region);
  }

  ssize_t m_nch;
};

template <int DIM>
class ChildOutputCollectTask
  : InlineLauncher {
public:

  ChildOutputCollectTask(LogicalRegion lr, ssize_t n)
    : InlineLauncher(
      RegionRequirement(
        lr,
        {FID_INDEX_SPACE},
        {FID_INDEX_SPACE},
        READ_ONLY,
        EXCLUSIVE,
        lr)),
      m_nch(n) {
  }

  IndexSpaceT<DIM>
  dispatch(Context ctx, Runtime* runtime) {
    PhysicalRegion region = runtime->map_region(ctx, *this);
    const FieldAccessor<
      READ_ONLY,
      IndexSpaceT<DIM>,
      1,
      coord_t,
      Realm::AffineAccessor<IndexSpaceT<DIM>,1,coord_t>,
      false> acc_ispace(region, FID_INDEX_SPACE);

    std::vector<IndexSpaceT<DIM>> ispaces;
    for (coord_t i = 0; i < m_nch; ++i)
      ispaces.push_back(acc_ispace[i]);
    runtime->unmap_region(ctx, region);
    return runtime->union_index_spaces(ctx, ispaces);;
  }

  ssize_t m_nch;
};

template <int DIM, int COLOR_DIM>
class ReportIndexesTask
  : public IndexTaskLauncher {
public:

  constexpr static const char * const TASK_NAME = "report_indexes";
  static const int TASK_ID = REPORT_INDEXES_TASK_ID;

  ReportIndexesTask(
    LogicalPartition lp,
    LogicalRegion lr,
    const IndexSpace& launch_space):
    IndexTaskLauncher(TASK_ID, launch_space, TaskArgument(), ArgumentMap()) {

    RegionRequirement req(lp, 0, READ_ONLY, EXCLUSIVE, lr);
    req.add_field(FID_PART_COLOR);
    add_region_requirement(req);
  }

  void
  dispatch(Context ctx, Runtime *runtime) {
    runtime->execute_index_space(ctx, *this);
  }

  static void
  base_impl(
    const Task* task,
    const std::vector<PhysicalRegion>&,
    Context ctx,
    Runtime *runtime) {

    DomainT<DIM> domain = runtime->get_index_space_domain(
      ctx,
      task->regions[0].region.get_index_space());
    std::ostringstream oss;
    oss << "task (" << task->index_point.point_data[0];
    for (size_t i = 1; i < COLOR_DIM; ++i)
      oss << "," << task->index_point.point_data[i];
    oss << "): [";
    for (PointInDomainIterator<DIM> pid(domain); pid(); pid++) {
      oss << "(" << pid[0];
      for (size_t i = 1; i < DIM; ++i)
        oss << "," << pid[i];
      oss << ")";
    }
    oss << "]" << std::endl;
    std::cout << oss.str();
  }

  static void
  register_task() {
    TaskVariantRegistrar registrar(TASK_ID, TASK_NAME);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<base_impl>(registrar, TASK_NAME);
  }
};

template <int DIM, int COLOR_DIM>
class PartitionIndexTreeTask
  : public InlineLauncher {
public:

  PartitionIndexTreeTask(LogicalRegionT<DIM> lr, IndexSpaceT<COLOR_DIM> colors)
    : InlineLauncher(
      RegionRequirement(
        lr,
        {FID_PART_COLOR},
        {FID_PART_COLOR},
        WRITE_DISCARD,
        EXCLUSIVE,
        lr))
    , m_lr(lr)
    , m_colors(colors) {
  }

  template <typename CB>
  IndexPartitionT<DIM>
  dispatch(CB cbfn, Context ctx, Runtime* runtime) {
    PhysicalRegion region = runtime->map_region(ctx, *this);
    const FieldAccessor<
      WRITE_DISCARD,
      Point<COLOR_DIM>,
      DIM,
      coord_t,
      Realm::AffineAccessor<Point<COLOR_DIM>,DIM,coord_t>,
      false> part_colors(region, FID_PART_COLOR);
    DomainT<DIM> domain =
      runtime->get_index_space_domain(ctx, m_lr.get_index_space());
    for (PointInDomainIterator<DIM> pid(domain); pid(); pid++)
      part_colors[*pid] = cbfn(pid);
    IndexPartitionT<DIM> result = runtime->create_partition_by_field<DIM>(
      ctx,
      m_lr,
      m_lr,
      FID_PART_COLOR,
      m_colors);
    runtime->unmap_region(ctx, region);
    return result;
  }

  LogicalRegionT<DIM> m_lr;
  IndexSpaceT<1> m_colors;
};

template <int DIM, typename CB>
IndexSpaceT<DIM>
offspring_index_space(ssize_t n, CB cbfn, Context ctx, Runtime* runtime) {

  Rect<1> child_is_rect = Rect<1>(Point<1>(0), Point<1>(n - 1));
  IndexSpaceT<1> child_is =
    runtime->create_index_space(ctx, child_is_rect);

  FieldSpace child_input_fs = runtime->create_field_space(ctx);
  {
    auto fa = runtime->create_field_allocator(ctx, child_input_fs);
    fa.allocate_field(sizeof(Rect<DIM>), FID_ENVELOPE);
    fa.allocate_field(sizeof(IndexTreeL), FID_TREE, SID_TREE);
  }
  LogicalRegion child_input_lr =
    runtime->create_logical_region(ctx, child_is, child_input_fs);

  ChildInputInitTask<DIM> init(child_input_lr, n);
  init.dispatch(cbfn, ctx, runtime);

  FieldSpace child_output_fs = runtime->create_field_space(ctx);
  {
    auto fa = runtime->create_field_allocator(ctx, child_output_fs);
    fa.allocate_field(sizeof(IndexSpaceT<DIM>), FID_INDEX_SPACE);
  }
  LogicalRegion child_output_lr =
    runtime->create_logical_region(ctx, child_is, child_output_fs);

  auto child_ip = runtime->create_equal_partition(ctx, child_is, child_is);
  LogicalPartition child_input_lp =
    runtime->get_logical_partition(ctx, child_input_lr, child_ip);
  LogicalPartition child_output_lp =
    runtime->get_logical_partition(ctx, child_output_lr, child_ip);
  IndexTreeSpaceTask<DIM> tree_space(
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
  IndexSpaceT<DIM> result = collect.dispatch(ctx, runtime);

  runtime->destroy_logical_region(ctx, child_input_lr);
  runtime->destroy_logical_region(ctx, child_output_lr);
  runtime->destroy_field_space(ctx, child_input_fs);
  runtime->destroy_field_space(ctx, child_output_fs);
  runtime->destroy_index_space(ctx, child_is);
  return result;
}

template <int DIM>
IndexSpaceT<DIM>
tree_index_space(const IndexTreeL& tree, Context ctx, Runtime* runtime) {
  auto rank = tree.rank();
  assert(rank);
  assert(rank.value() == DIM);
  return
    offspring_index_space<DIM>(
      1,
      [&](Point<1>, Rect<DIM>& rect, IndexTreeL& tr) {
        auto env = tree.envelope();
        coord_t tree_rect_lo[DIM], tree_rect_hi[DIM];
        for (size_t i = 0; i < DIM; ++i)
          std::tie(tree_rect_lo[i], tree_rect_hi[i]) = env[i];
        rect = Rect<DIM>(Point<DIM>(tree_rect_lo), Point<DIM>(tree_rect_hi));
        tr = tree;
      },
      ctx,
      runtime);
}

template <>
IndexSpaceT<1>
tree_index_space(const IndexTreeL& tree, Context ctx, Runtime* runtime) {
  auto rank = tree.rank();
  assert(rank);
  assert(rank.value() == 1);
  return
    offspring_index_space<1>(
      1,
      [&](Point<1>, Rect<1>& rect, IndexTreeL& tr) {
        auto env = tree.envelope();
        coord_t tree_rect_lo, tree_rect_hi;
        std::tie(tree_rect_lo, tree_rect_hi) = env[0];
        rect = Rect<1>(Point<1>(tree_rect_lo), Point<1>(tree_rect_hi));
        tr = tree;
      },
      ctx,
      runtime);
}

IndexSpace
tree_index_space(const IndexTreeL& tree, Context ctx, Runtime* runtime) {
  auto rank = tree.rank();
  assert(rank);
  switch(rank.value()) {
  case 1:
    return tree_index_space<1>(tree, ctx, runtime);
  case 2:
    return tree_index_space<2>(tree, ctx, runtime);
  case 3:
    return tree_index_space<3>(tree, ctx, runtime);
  case 4:
    return tree_index_space<4>(tree, ctx, runtime);
  case 5:
    return tree_index_space<5>(tree, ctx, runtime);
  case 6:
    return tree_index_space<6>(tree, ctx, runtime);
  case 7:
    return tree_index_space<7>(tree, ctx, runtime);
  case 8:
    return tree_index_space<8>(tree, ctx, runtime);
  default:
    assert(false);
  }
}

class TopLevelTask {
public:

  constexpr static const char * const TASK_NAME = "top_level";
  static const int TASK_ID = TOP_LEVEL_TASK_ID;

  static void
  base_impl(
    const Task*,
    const std::vector<PhysicalRegion>&,
    Context ctx,
    Runtime *runtime) {

    IndexTreeL t02(2);
    IndexTreeL t03(3);
    IndexTreeL t1({{0, 3, t02}, {6, 2, t03}});
    IndexTreeL tree({{10, 2, t1}, {20, 1, t1}});

    IndexSpaceT<3> index_space = tree_index_space<3>(tree, ctx, runtime);
    DomainT<3> domain =
      runtime->get_index_space_domain(ctx, index_space);
    for (PointInDomainIterator<3> pid(domain); pid(); pid++)
      std::cout << "(" << pid[0]
                << "," << pid[1]
                << "," << pid[2]
                << ")" << std::endl;

    FieldSpace fs = runtime->create_field_space(ctx);
    {
      auto fa = runtime->create_field_allocator(ctx, fs);
      fa.allocate_field(sizeof(Point<1>), FID_PART_COLOR);
    }
    LogicalRegionT<3> lr =
      runtime->create_logical_region(ctx, index_space, fs);

    IndexSpaceT<1> colors = runtime->create_index_space(ctx, Rect<1>{0, 1});

    PartitionIndexTreeTask partitioner(lr, colors);
    IndexPartitionT<3> partition =
      partitioner.dispatch(
        [](PointInDomainIterator<3>& pid) {
          return Point<1>((pid[1] < 2) ? 0 : 1);
        },
        ctx,
        runtime);
    LogicalPartition lp = runtime->get_logical_partition(ctx, lr, partition);

    ReportIndexesTask<3, 1> report(lp, lr, colors);
    report.dispatch(ctx, runtime);

    runtime->destroy_index_partition(ctx, partition);
    runtime->destroy_index_space(ctx, colors);
    runtime->destroy_index_space(ctx, index_space);
  }

  static void
  register_task() {
    TaskVariantRegistrar registrar(TASK_ID, TASK_NAME);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<base_impl>(registrar, TASK_NAME);
  }
};

int
main(int argc, char* argv[]) {

  Runtime::set_top_level_task_id(TopLevelTask::TASK_ID);
  TopLevelTask::register_task();
  IndexTreeSpaceTask<3>::register_task();
  ReportIndexesTask<3, 1>::register_task();

  Runtime::register_custom_serdez_op<tree_serdez>(SID_TREE);

  return Runtime::start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
