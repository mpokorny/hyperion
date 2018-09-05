#include <ostream>
#include <vector>

#include "Grids.h"
#include "Tree.h"

using namespace legms;
using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  COLOR2_TASK_ID,
  COLOR3_TASK_ID,
  REPORT_COLORS_TASK_ID,
  REPORT_TREE_TASK_ID,
};

enum FieldIDs {
  FID_COLORS,
};

#define MAX_COLORS 16
struct colors {
  unsigned length;
  coord_t values[3 * MAX_COLORS];
};

class RecordColor {
public:
  typedef colors LHS;
  typedef colors RHS;

  static const colors identity;

  template <bool EXCLUSIVE> static void apply(colors &lhs, colors rhs);

  template <bool EXCLUSIVE> static void fold(colors &rhs1, colors rhs2);
};

const colors RecordColor::identity{0, {}};

template<>
void
RecordColor::apply<true>(colors& lhs, colors rhs) {
  for (unsigned i = 0; i < rhs.length; ++i)
    lhs.values[lhs.length++] = rhs.values[i];
}

template <>
void
RecordColor::fold<true>(colors& rhs1, RHS rhs2) {
  apply<true>(rhs1, rhs2);
}

template<>
void
RecordColor::apply<false>(colors& lhs, colors rhs) {
  RecordColor::apply<true>(lhs, rhs);
}

template <>
void
RecordColor::fold<false>(colors& rhs1, colors rhs2) {
  RecordColor::fold<true>(rhs1, rhs2);
}

enum ReduceIds {
  RECORD_COLOR_ID = 1,
};

// (index space, tree) => index space
enum ISFields {
  IS_FID,
};

template <int DIM, typename COORD_T>
struct tree_space_args {
  Rect<DIM, COORD_T> envelope;
  COORD_T tree[];
};
#define SIZEOF_TREE_SPACE_ARGS(d, n) (\
    sizeof(tree_space_args<d>) + n * sizeof(size_t))
// FIXME: parametrize index size of Tree

template <int DIM>
void
tree_space_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* runtime) {

  const struct tree_space_args<DIM, COORD_T> *args =
    (const tree_space_args<DIM, COORD_T>*)task->args;
  Tree<COORD_T> tree = Tree<COORD_T>::deserialize(args->tree);
  if (tree.is_array()) {
    auto tree_rect = tree.envelope();
    assert(tree_rect.size() <= DIM);
    COORD_T tree_rect_lo[DIM], tree_rect_hi[DIM];
    auto fixd = DIM - tree_rect.size();
    for (COORD_T i = 0; i < fixd; ++i) {
      tree_rect_lo[i] = args->envelope.lo[i];
      tree_rect_hi[i] = args->envelope.hi[i];
    }
    for (COORD_T i = fixd; i < tree_rect.size(); ++i)
      std::tie(tree_rect_lo[i], tree_rect_hi[i]) = tree_rect[i];
    Rect<DIM, COORD_T> array(
      Point<DIM, COORD_T>(tree_rect_lo),
      Point<DIM, COORD_T>(tree_rect_hi));
    IndexSpace<DIM, COORD_T> is = runtime->create_index_space(ctx, array);

  } else {

  }
}

void
top_level_task(
  const Task*,
  const std::vector<PhysicalRegion>&,
  Context ctx,
  Runtime* runtime) {

  const Rect<2> grid(Point<2>(0, 0), Point<2>(5, 9));
  const Point<2> num_blocks(2, 3);
  const Point<2> border(1, 2);

  IndexSpaceT<2> grid_is = runtime->create_index_space(ctx, grid);
  FieldSpace grid_fs = runtime->create_field_space(ctx);
  {
    auto fa = runtime->create_field_allocator(ctx, grid_fs);
    fa.allocate_field(sizeof(struct colors), FID_COLORS);
  }
  LogicalRegion grid_lr = runtime->create_logical_region(ctx, grid_is, grid_fs);

  IndexPartition disjoint_ip, extended_ip;
  std::tie(disjoint_ip, extended_ip) =
    block_partition_and_extend<2>(grid_is, num_blocks, border, ctx, runtime);
  LogicalPartition disjoint_lp =
    runtime->get_logical_partition(ctx, grid_lr, disjoint_ip);
  LogicalPartition extended_lp =
    runtime->get_logical_partition(ctx, grid_lr, extended_ip);

  auto disjoint_color_space =
    runtime->get_index_partition_color_space_name(ctx, disjoint_ip);
  auto extended_color_space =
    runtime->get_index_partition_color_space_name(ctx, extended_ip);

  ArgumentMap arg_map;
  int disjoint_color_offset = 0;
  IndexTaskLauncher disjoint_launcher(
    COLOR2_TASK_ID,
    disjoint_color_space,
    TaskArgument(&disjoint_color_offset, sizeof(disjoint_color_offset)),
    arg_map);
  disjoint_launcher.add_region_requirement(
    RegionRequirement(disjoint_lp, 0, RECORD_COLOR_ID, ATOMIC, grid_lr));
  disjoint_launcher.add_field(0, FID_COLORS);
  runtime->execute_index_space(ctx, disjoint_launcher);

  int extended_color_offset = 100;
  IndexTaskLauncher extended_launcher(
    COLOR2_TASK_ID,
    extended_color_space,
    TaskArgument(&extended_color_offset, sizeof(extended_color_offset)),
    arg_map);
  extended_launcher.add_region_requirement(
    RegionRequirement(extended_lp, 0, RECORD_COLOR_ID, ATOMIC, grid_lr));
  extended_launcher.add_field(0, FID_COLORS);
  runtime->execute_index_space(ctx, extended_launcher);

  TaskLauncher report_colors_launcher(REPORT_COLORS_TASK_ID, TaskArgument(NULL, 0));
  report_colors_launcher.add_region_requirement(
    RegionRequirement(grid_lr, READ_ONLY, EXCLUSIVE, grid_lr));
  report_colors_launcher.add_field(0, FID_COLORS);
  runtime->execute_task(ctx, report_colors_launcher);

  runtime->destroy_logical_region(ctx, grid_lr);
  runtime->destroy_field_space(ctx, grid_fs);
  runtime->destroy_index_space(ctx, grid_is);
  runtime->destroy_index_space(ctx, disjoint_color_space);

  // tree index space test
  Tree t = Tree({{2, Tree({{1, Tree(2)}, {2, Tree(3)}})}});
  IndexSpace tree_is = tree_index_space<3>(t, ctx, runtime);
  FieldSpace tree_fs = runtime->create_field_space(ctx);
  {
    auto fa = runtime->create_field_allocator(ctx, tree_fs);
    fa.allocate_field(sizeof(struct colors), FID_COLORS);
  }
  LogicalRegion tree_lr = runtime->create_logical_region(ctx, tree_is, tree_fs);
  IndexPartition tree_ip =
    runtime->create_partition_by_blockify(ctx, tree_is, Point<3>(1, 2, 2));
  LogicalPartition tree_lp =
    runtime->get_logical_partition(ctx, tree_lr, tree_ip);

  int tree_color_offset = 0;
  auto tree_color_space =
    runtime->get_index_partition_color_space_name(ctx, tree_ip);
  IndexTaskLauncher tree_launcher(
    COLOR3_TASK_ID,
    tree_color_space,
    TaskArgument(&tree_color_offset, sizeof(tree_color_offset)),
    arg_map);
  tree_launcher.add_region_requirement(
    RegionRequirement(tree_lp, 0, RECORD_COLOR_ID, ATOMIC, tree_lr));
  tree_launcher.add_field(0, FID_COLORS);
  runtime->execute_index_space(ctx, tree_launcher);

  TaskLauncher report_tree_launcher(REPORT_TREE_TASK_ID, TaskArgument(NULL, 0));
  report_tree_launcher.add_region_requirement(
    RegionRequirement(tree_lr, READ_ONLY, EXCLUSIVE, tree_lr));
  report_tree_launcher.add_field(0, FID_COLORS);
  runtime->execute_task(ctx, report_tree_launcher);

  runtime->destroy_logical_region(ctx, tree_lr);
  runtime->destroy_field_space(ctx, tree_fs);
  runtime->destroy_index_space(ctx, tree_is);
  runtime->destroy_index_space(ctx, tree_color_space);
}

void
color2_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* runtime) {

  const int* color_offset = (const int *)task->args;
  colors c{
    2,
    {*color_offset + task->index_point.point_data[0],
     *color_offset + task->index_point.point_data[1]}
  };
  const ReductionAccessor<
    RecordColor,
    true,
    2,
    coord_t,
    Realm::AffineAccessor<colors,2,coord_t>,
    false> acc_colors(regions[0], FID_COLORS, RECORD_COLOR_ID);
  Rect<2> rect = runtime->get_index_space_domain(
    ctx,
    task->regions[0].region.get_index_space());
  for (PointInRectIterator<2> pir(rect); pir(); ++pir)
    acc_colors[*pir] <<= c;
}

void
color3_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* runtime) {

  const int* color_offset = (const int *)task->args;
  colors c{
    3,
    {*color_offset + task->index_point.point_data[0],
        *color_offset + task->index_point.point_data[1],
        *color_offset + task->index_point.point_data[2]}
  };
  const ReductionAccessor<
    RecordColor,
    true,
    3,
    coord_t,
    Realm::AffineAccessor<colors,3,coord_t>,
    false> acc_colors(regions[0], FID_COLORS, RECORD_COLOR_ID);
  DomainT<3> domain = runtime->get_index_space_domain(
    ctx,
    task->regions[0].region.get_index_space());
  for (PointInDomainIterator<3> pir(domain); pir(); ++pir)
    acc_colors[*pir] <<= c;
}

void
report_colors_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* runtime) {

  const FieldAccessor<
    READ_ONLY,
    colors,
    2,
    coord_t,
    Realm::AffineAccessor<colors,2,coord_t>,
    false> acc_colors(regions[0], FID_COLORS);
  Rect<2> rect = runtime->get_index_space_domain(
    ctx,
    task->regions[0].region.get_index_space());
  for (PointInRectIterator<2> pir(rect); pir(); ++pir) {
    std::ostringstream oss;
    oss << "(" << pir[0] << "," << pir[1] << "):";
    const colors& c = acc_colors[*pir];
    const char* sep = " ";
    for (unsigned i = 0; i < c.length; i += 2) {
      oss << sep << "(" << c.values[i] << "," << c.values[i + 1] << ")";
      sep = ",";
    }
    oss << std::endl;
    std::cout << oss.str();
  }
}

void
report_tree_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* runtime) {

  const FieldAccessor<
    READ_ONLY,
    colors,
    3,
    coord_t,
    Realm::AffineAccessor<colors,3,coord_t>,
    false> acc_colors(regions[0], FID_COLORS);
  DomainT<3> domain = runtime->get_index_space_domain(
    ctx,
    task->regions[0].region.get_index_space());
  for (PointInDomainIterator<3> pid(domain); pid(); ++pid) {
    std::ostringstream oss;
    oss << "(" << pid[0] << "," << pid[1] << "," << pid[2] << "): ";
    const colors& c = acc_colors[*pid];
    const char* sep = " ";
    for (unsigned i = 0; i < c.length; i += 3) {
      oss << sep << "(" << c.values[i]
          << "," << c.values[i + 1]
          << "," << c.values[i + 2] << ")";
      sep = ",";
    }
    oss << std::endl;
    std::cout << oss.str();
  }
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
    TaskVariantRegistrar registrar(COLOR2_TASK_ID, "color2");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<color2_task>(registrar, "color2");
  }

  {
    TaskVariantRegistrar registrar(COLOR3_TASK_ID, "color3");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<color3_task>(registrar, "color3");
  }

  {
    TaskVariantRegistrar registrar(REPORT_COLORS_TASK_ID, "report_colors");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<report_colors_task>(registrar, "report_colors");
  }

  {
    TaskVariantRegistrar registrar(REPORT_TREE_TASK_ID, "report_tree");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<report_tree_task>(registrar, "report_tree");
  }

  Runtime::register_reduction_op<RecordColor>(RECORD_COLOR_ID);

  return Runtime::start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
