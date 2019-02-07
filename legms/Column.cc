#include <cassert>

#include "Column.h"
#include "Table.h"

using namespace legms;
using namespace legms::ms;

using namespace Legion;

class FillRowNumbersTask {
public:

  static TaskID TASK_ID;
  constexpr static const char* TASK_NAME = "FillRowNumbersTask";

  FillRowNumbersTask(
    LogicalRegion lr,
    const IndexTreeL& row_index_pattern) {

    auto arg_size = row_index_pattern.serialized_size();
    m_arg_buffer = std::make_unique<char[]>(arg_size);
    row_index_pattern.serialize(m_arg_buffer.get());
    m_launcher =
      TaskLauncher(TASK_ID, TaskArgument(m_arg_buffer.get(), arg_size));
    m_launcher.add_region_requirement(
      RegionRequirement(lr, WRITE_DISCARD, EXCLUSIVE, lr));
    m_launcher.add_field(0, Column::row_number_fid);
  }

  void
  dispatch(Context ctx, Runtime* runtime) {
    runtime->execute_task(ctx, m_launcher);
  }

  template <int DIM>
  static void
  impl_n(
    const PhysicalRegion& pr,
    DomainT<DIM, coord_t> domain,
    const IndexTreeL& row_index_pattern) {

    FieldAccessor<
      WRITE_DISCARD,
      Column::row_number_t,
      DIM,
      coord_t,
      AffineAccessor<Column::row_number_t, DIM, coord_t>,
      false>
      row_numbers(pr, Column::row_number_fid);
    for (PointInDomainIterator pid(domain, false); pid(); pid++) {
      std::array<coord_t, DIM> p;
      for (size_t i = 0; i < DIM; ++i)
        p[i] = pid[i];
      row_numbers[*pid] =
        Table::row_number(row_index_pattern, p.begin(), p.end());
    }
  }

  static void
  base_impl(
    const Task* task,
    const std::vector<PhysicalRegion>& regions,
    Context,
    Runtime *runtime) {

    IndexTreeL row_index_pattern =
      IndexTreeL::deserialize(static_cast<const char*>(task->args));
    switch (task->regions[0].region.get_dim()) {
    case 1:
      impl_n<1>(
        regions[0],
        runtime->get_index_space_domain(
          task->regions[0].region.get_index_space()),
        row_index_pattern);
      break;
    case 2:
      impl_n<2>(
        regions[0],
        runtime->get_index_space_domain(
          task->regions[0].region.get_index_space()),
        row_index_pattern);
      break;
    case 3:
      impl_n<3>(
        regions[0],
        runtime->get_index_space_domain(
          task->regions[0].region.get_index_space()),
        row_index_pattern);
      break;
    default:
      assert(false);
      break;
    }
  }

  static void
  register_task(Runtime* runtime) {
    TASK_ID =
      runtime->generate_library_task_ids("legms::FillRowNumbersTask", 1);
    TaskVariantRegistrar registrar(TASK_ID, TASK_NAME);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    registrar.set_idempotent();
    runtime->register_task_variant<base_impl>(registrar);
  }

private:

  std::unique_ptr<char[]> m_arg_buffer;

  TaskLauncher m_launcher;
};

TaskID FillRowNumbersTask::TASK_ID;

template <int PREFIXLEN, int LEN>
bool
same_prefix_index(
  const PointInDomainIterator<LEN>& pid0,
  const PointInDomainIterator<LEN>& pid1) {

  bool result = pid0() && pid1();
  for (size_t i = 0; result && i < PREFIXLEN; ++i)
    result = pid0[i] == pid1[i];
  return result;
}

void
Column::init() {
  if (m_index_tree.size() > 0) {
    m_index_space = legms::tree_index_space(m_index_tree, m_context, m_runtime);
    
    FieldSpace fs = m_runtime->create_field_space(m_context);
    auto fa = m_runtime->create_field_allocator(m_context, fs);
    legms::add_field(m_datatype, fa, value_fid);
    m_runtime->attach_name(fs, value_fid, name().c_str());
    legms::add_field(ValueType<row_number_t>::DataType, fa, row_number_fid);
    m_runtime->attach_name(fs, row_number_fid, "rownr");
    m_logical_region =
      m_runtime->create_logical_region(m_context, m_index_space, fs);

    FillRowNumbersTask(m_logical_region, row_index_pattern()).
      dispatch(m_context, m_runtime);
  } else {
    m_index_space = IndexSpace::NO_SPACE;
    m_logical_region = LogicalRegion::NO_REGION;
  }
}

void
Column::register_tasks(Runtime *runtime) {
  FillRowNumbersTask::register_task(runtime);
}

bool
Column::pattern_matches(const IndexTreeL& pattern, const IndexTreeL& shape) {
  return shape.num_repeats(pattern).has_value();
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
