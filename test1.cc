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
};

enum FieldIDs {
  FID_ENVELOPE,
  FID_TREE,
  FID_VRECT,
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
offspring_rects(
  ssize_t n,
  CB cbfn,
  Context ctx,
  Runtime* runtime) {

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

  RegionRequirement input_req(
    child_input_lr,
    WRITE_DISCARD,
    EXCLUSIVE,
    child_input_lr);
  input_req.add_field(FID_ENVELOPE);
  input_req.add_field(FID_TREE);
  InlineLauncher input_launcher(input_req);
  PhysicalRegion input_region = runtime->map_region(ctx, input_launcher);
  const FieldAccessor<
    WRITE_DISCARD,
    Rect<DIM>,
    1,
    coord_t,
    Realm::AffineAccessor<Rect<DIM>,1,coord_t>,
    false> acc_rect(input_region, FID_ENVELOPE);
  const FieldAccessor<
    WRITE_DISCARD,
    TreeL,
    1,
    coord_t,
    Realm::AffineAccessor<TreeL,1,coord_t>,
    false> acc_tree(input_region, FID_TREE);
  for (coord_t i = 0; i < n; ++i)
    cbfn(i, acc_rect[i], acc_tree[i]);

  auto child_cs = runtime->create_index_space(ctx, child_is_rect);
  auto child_ip = runtime->create_equal_partition(ctx, child_is, child_cs);
  ArgumentMap arg_map;
  IndexTaskLauncher child_launcher(
    TREE_SPACE_TASK_ID,
    child_cs,
    TaskArgument(NULL, 0),
    arg_map);
  LogicalPartition child_input_lp =
    runtime->get_logical_partition(ctx, child_input_lr, child_ip);
  RegionRequirement child_input_req(
    child_input_lp,
    0,
    READ_ONLY,
    EXCLUSIVE,
    child_input_lr);
  child_input_req.add_field(FID_ENVELOPE);
  child_input_req.add_field(FID_TREE);
  child_launcher.add_region_requirement(child_input_req);
  LogicalPartition child_output_lp =
    runtime->get_logical_partition(ctx, child_output_lr, child_ip);
  RegionRequirement child_output_req(
    child_output_lp,
    0,
    WRITE_DISCARD,
    EXCLUSIVE,
    child_output_lr);
  child_output_req.add_field(FID_VRECT);
  child_launcher.add_region_requirement(child_output_req);
  runtime->execute_index_space(ctx, child_launcher);
  runtime->destroy_index_partition(ctx, child_ip);
  runtime->destroy_logical_partition(ctx, child_input_lp);
  runtime->destroy_logical_partition(ctx, child_output_lp);
  runtime->destroy_index_space(ctx, child_cs);

  std::vector<Rect<DIM>> result;
  RegionRequirement output_req(
    child_output_lr,
    READ_ONLY,
    EXCLUSIVE,
    child_output_lr);
  output_req.add_field(FID_VRECT);
  InlineLauncher output_launcher(output_req);
  PhysicalRegion output_region = runtime->map_region(ctx, output_launcher);
  const FieldAccessor<
    READ_ONLY,
    std::vector<Rect<DIM>>,
    1,
    coord_t,
    Realm::AffineAccessor<std::vector<Rect<DIM>>,1,coord_t>,
    false> acc_is(output_region, FID_VRECT);

  auto result_in = std::back_inserter(result);
  for (coord_t i = 0; i < n; ++i) {
    const std::vector<Rect<DIM>>& vr = acc_is[i];
    std::copy(
      vr.begin(),
      vr.end(),
      result_in);
  }
  runtime->unmap_region(ctx, input_region);
  runtime->unmap_region(ctx, output_region);
  runtime->destroy_logical_region(ctx, child_input_lr);
  runtime->destroy_logical_region(ctx, child_output_lr);
  runtime->destroy_field_space(ctx, child_input_fs);
  runtime->destroy_field_space(ctx, child_output_fs);
  runtime->destroy_index_space(ctx, child_is);
  return result;
}

template <int DIM>
void
tree_space_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* runtime) {

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
          rect = Rect<DIM>(Point<DIM>(tree_rect_lo), Point<DIM>(tree_rect_hi));
          tr = t;
        },
        ctx,
        runtime);
    }
  }
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

void
top_level_task(
  const Task*,
  const std::vector<PhysicalRegion>&,
  Context ctx,
  Runtime* runtime) {

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

  runtime->destroy_index_space(ctx, index_space);
}

int
main(int argc, char* argv[]) {

  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(TREE_SPACE_TASK_ID, "tree_space");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<tree_space_task<3>>(registrar, "tree_space");
  }

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
