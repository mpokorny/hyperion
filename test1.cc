#include <algorithm>
#include <limits>
#include <ostream>
#include <vector>

#include "legion.h"
#include "Tree.h"

using namespace grdly;
using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  TREE_SPACE_TASK_ID,
  REPORT_INDEXES_TASK_ID,
};

enum FieldIDs {
  FID_ENVELOPE,
  FID_TREE,
  FID_VRECT,

  FID_PART_COLOR,
};

enum SerdezIDs {
  SID_TREE = 1,
  SID_VRECT = 2,
};

typedef Tree<coord_t> TreeL;

class tree_serdez {
public:
  typedef TreeL FIELD_TYPE;

  static const size_t MAX_SERIALIZED_SIZE = std::numeric_limits<size_t>::max();

  static size_t
  serialized_size(const TreeL& val) {
    return val.serialized_size();
  }

  static size_t
  serialize(const TreeL& val, void *buffer) {
    return val.serialize(reinterpret_cast<char*>(buffer));
  }

  static size_t
  deserialize(TreeL& val, const void *buffer) {
    val = TreeL::deserialize(static_cast<const char*>(buffer));
    return *reinterpret_cast<const size_t *>(buffer);
  }

  static void
  destroy(TreeL&) {
  }
};

template <int DIM>
class vrect_serdez {
public:
  typedef std::vector<Rect<DIM>> FIELD_TYPE;

  static const size_t MAX_SERIALIZED_SIZE = std::numeric_limits<size_t>::max();

  static size_t
  serialized_size(const std::vector<Rect<DIM>>& val) {
    return val.size() * sizeof(Rect<DIM>) + sizeof(size_t);
  }

  static size_t
  serialize(const std::vector<Rect<DIM>>& val, void *buffer) {
    unsigned char *b = reinterpret_cast<unsigned char *>(buffer);
    size_t vlen = val.size();
    *reinterpret_cast<size_t *>(b) = vlen;
    b += sizeof(vlen);
    size_t rectsz = sizeof(Rect<DIM>);
    for (size_t i = 0; i < vlen; ++i) {
      memcpy(b, &val[i], rectsz);
      b += rectsz;
    }
    return serialized_size(val);
  }

  static size_t
  deserialize(std::vector<Rect<DIM>>& val, const void *buffer) {
    const unsigned char *b = reinterpret_cast<const unsigned char *>(buffer);
    size_t vlen = *reinterpret_cast<const size_t *>(b);
    b += sizeof(vlen);
    size_t rectsz = sizeof(Rect<DIM>);
    val.clear();
    for (size_t i = 0; i < vlen; ++i) {
      val.push_back(*reinterpret_cast<const Rect<DIM> *>(b));
      b += rectsz;
    }
    return serialized_size(val);
  }

  static void
  destroy(std::vector<Rect<DIM>>&) {
  }
};

template <int DIM, typename CB>
std::vector<Rect<DIM>>
offspring_rects(ssize_t n, CB cbfn, Context ctx, Runtime* runtime);

template <int DIM>
class TreeSpaceTask
  : public IndexTaskLauncher {
public:

  constexpr static const char * const TASK_NAME = "tree_space";
  static const int TASK_ID = TREE_SPACE_TASK_ID;

  TreeSpaceTask(
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
    child_output_req.add_field(FID_VRECT);
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
      TreeL,
      1,
      coord_t,
      Realm::AffineAccessor<TreeL,1,coord_t>,
      false> trees(regions[0], FID_TREE);
    const FieldAccessor<
      WRITE_DISCARD,
      std::vector<Rect<DIM>>,
      1,
      coord_t,
      Realm::AffineAccessor<std::vector<Rect<DIM>>,1,coord_t>,
      false> domains(regions[1], FID_VRECT);

    Rect<1> rect = runtime->get_index_space_domain(
      ctx,
      task->regions[0].region.get_index_space());

    for (PointInRectIterator<1> pir(rect); pir(); pir++) {
      const Rect<DIM>& envelope = envelopes[*pir];
      const TreeL& tree = trees[*pir];

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
        domains[*pir] =
          {Rect<DIM>(Point<DIM>(tree_rect_lo), Point<DIM>(tree_rect_hi))};
      } else {
        auto children = tree.children();
        domains[*pir] = offspring_rects<DIM>(
          children.size(),
          [&](Point<1> chidx, Rect<DIM>& rect, TreeL& tr) {
            coord_t i, n;
            TreeL t;
            std::tie(i, n, t) = children[chidx];
            tree_rect_lo[fixd] = i;
            tree_rect_hi[fixd] = i + n - 1;
            rect =
              Rect<DIM>(Point<DIM>(tree_rect_lo), Point<DIM>(tree_rect_hi));
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
      TreeL,
      1,
      coord_t,
      Realm::AffineAccessor<TreeL,1,coord_t>,
      false> acc_tree(region, FID_TREE);
    for (coord_t i = 0; i < m_nch; ++i)
      cbfn(i, acc_rect[i], acc_tree[i]);
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
        {FID_VRECT},
        {FID_VRECT},
        READ_ONLY,
        EXCLUSIVE,
        lr)),
      m_nch(n) {
  }

  std::vector<Rect<DIM>>
  dispatch(Context ctx, Runtime* runtime) {
    PhysicalRegion region = runtime->map_region(ctx, *this);
    const FieldAccessor<
      READ_ONLY,
      std::vector<Rect<DIM>>,
      1,
      coord_t,
      Realm::AffineAccessor<std::vector<Rect<DIM>>,1,coord_t>,
      false> acc_rect(region, FID_VRECT);

    std::vector<Rect<DIM>> result;
    auto result_in = std::back_inserter(result);
    for (coord_t i = 0; i < m_nch; ++i) {
      const std::vector<Rect<DIM>>& vr = acc_rect[i];
      std::copy(
        vr.begin(),
        vr.end(),
        result_in);
    }
    runtime->unmap_region(ctx, region);
    return result;
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
class PartitionTreeTask
  : public InlineLauncher {
public:

  PartitionTreeTask(LogicalRegionT<DIM> lr, IndexSpaceT<COLOR_DIM> colors)
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
std::vector<Rect<DIM>>
offspring_rects(ssize_t n, CB cbfn, Context ctx, Runtime* runtime) {

  Rect<1> child_is_rect = Rect<1>(Point<1>(0), Point<1>(n - 1));
  IndexSpaceT<1> child_is =
    runtime->create_index_space(ctx, child_is_rect);

  FieldSpace child_input_fs = runtime->create_field_space(ctx);
  {
    auto fa = runtime->create_field_allocator(ctx, child_input_fs);
    fa.allocate_field(sizeof(Rect<DIM>), FID_ENVELOPE);
    fa.allocate_field(sizeof(TreeL), FID_TREE, SID_TREE);
  }
  LogicalRegion child_input_lr =
    runtime->create_logical_region(ctx, child_is, child_input_fs);

  FieldSpace child_output_fs = runtime->create_field_space(ctx);
  {
    auto fa = runtime->create_field_allocator(ctx, child_output_fs);
    fa.allocate_field(
      sizeof(std::vector<Rect<DIM>>),
      FID_VRECT,
      SID_VRECT);
  }
  LogicalRegion child_output_lr =
    runtime->create_logical_region(ctx, child_is, child_output_fs);

  ChildInputInitTask<DIM> init(child_input_lr, n);
  init.dispatch(cbfn, ctx, runtime);

  auto child_cs = runtime->create_index_space(ctx, child_is_rect);
  auto child_ip = runtime->create_equal_partition(ctx, child_is, child_cs);
  LogicalPartition child_input_lp =
    runtime->get_logical_partition(ctx, child_input_lr, child_ip);
  LogicalPartition child_output_lp =
    runtime->get_logical_partition(ctx, child_output_lr, child_ip);
  TreeSpaceTask<DIM> tree_space(
    child_input_lp,
    child_output_lp,
    child_input_lr,
    child_output_lr,
    child_cs);
  tree_space.dispatch(ctx, runtime);

  runtime->destroy_index_partition(ctx, child_ip);
  runtime->destroy_logical_partition(ctx, child_input_lp);
  runtime->destroy_logical_partition(ctx, child_output_lp);
  runtime->destroy_index_space(ctx, child_cs);

  ChildOutputCollectTask<DIM> collect(child_output_lr, n);
  std::vector<Rect<DIM>> result = collect.dispatch(ctx, runtime);

  runtime->destroy_logical_region(ctx, child_input_lr);
  runtime->destroy_logical_region(ctx, child_output_lr);
  runtime->destroy_field_space(ctx, child_input_fs);
  runtime->destroy_field_space(ctx, child_output_fs);
  runtime->destroy_index_space(ctx, child_is);
  return result;
}

template <int DIM>
IndexSpaceT<DIM>
tree_index_space(const TreeL& tree, Context ctx, Runtime* runtime) {
  auto rank = tree.rank();
  assert(rank);
  assert(rank.value() == DIM);
  std::vector<Rect<DIM>> domain =
    offspring_rects<DIM>(
      1,
      [&](Point<1>, Rect<DIM>& rect, TreeL& tr) {
        auto env = tree.envelope();
        coord_t tree_rect_lo[DIM], tree_rect_hi[DIM];
        for (size_t i = 0; i < DIM; ++i)
          std::tie(tree_rect_lo[i], tree_rect_hi[i]) = env[i];
        rect = Rect<DIM>(Point<DIM>(tree_rect_lo), Point<DIM>(tree_rect_hi));
        tr = tree;
      },
      ctx,
      runtime);
  return runtime->create_index_space(ctx, domain);
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

    TreeL t02(2);
    TreeL t03(3);
    TreeL t1({{0, 3, t02}, {6, 2, t03}});
    TreeL tree({{10, 2, t1}, {20, 1, t1}});

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

    PartitionTreeTask partitioner(lr, colors);
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
  TreeSpaceTask<3>::register_task();
  ReportIndexesTask<3, 1>::register_task();

  Runtime::register_custom_serdez_op<tree_serdez>(SID_TREE);
  Runtime::register_custom_serdez_op<vrect_serdez<3>>(SID_VRECT);

  return Runtime::start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
