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
    Context ctx,
    Runtime *runtime) {

    IndexTreeL row_index_pattern =
      IndexTreeL::deserialize(static_cast<const char*>(task->args));
    switch (task->regions[0].region.get_dim()) {
    case 1:
      impl_n<1>(
        regions[0],
        runtime->get_index_space_domain(
          ctx,
          task->regions[0].region.get_index_space()),
        row_index_pattern);
      break;
    case 2:
      impl_n<2>(
        regions[0],
        runtime->get_index_space_domain(
          ctx,
          task->regions[0].region.get_index_space()),
        row_index_pattern);
      break;
    case 3:
      impl_n<3>(
        regions[0],
        runtime->get_index_space_domain(
          ctx,
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
}

void
Column::register_tasks(Runtime *runtime) {
  FillRowNumbersTask::register_task(runtime);
}

std::optional<size_t>
Column::nr(
  const IndexTreeL& row_pattern,
  const IndexTreeL& full_shape,
  bool cycle) {

  if (row_pattern.rank().value() > full_shape.rank().value())
    return std::nullopt;
  auto pruned_shape = full_shape.pruned(row_pattern.rank().value() - 1);
  auto p_iter = pruned_shape.children().begin();
  auto p_end = pruned_shape.children().end();
  coord_t i0 = std::get<0>(row_pattern.index_range());
  size_t result = 0;
  coord_t pi, pn;
  IndexTreeL pt;
  std::tie(pi, pn, pt) = *p_iter;
  while (p_iter != p_end) {
    auto r_iter = row_pattern.children().begin();
    auto r_end = row_pattern.children().end();
    coord_t i, n;
    IndexTreeL t;
    while (p_iter != p_end && r_iter != r_end) {
      std::tie(i, n, t) = *r_iter;
      if (i + i0 != pi) {
        return std::nullopt;
      } else if (t == pt) {
        auto m = std::min(n, pn);
        result += m * t.size();
        pi += m;
        pn -= m;
        if (pn == 0) {
          ++p_iter;
          if (p_iter != p_end)
            std::tie(pi, pn, pt) = *p_iter;
        }
      } else {
        ++p_iter;
        if (p_iter != p_end)
          return std::nullopt;
        auto chnr = nr(t, pt, false);
        if (chnr)
          result += chnr.value();
        else
          return std::nullopt;
      }
      ++r_iter;
    }
    i0 = i + n;
    if (!cycle && p_iter != p_end && r_iter == r_end)
      return std::nullopt;
  }
  return result;
}

bool
Column::pattern_matches(const IndexTreeL& pattern, const IndexTreeL& shape) {
  return nr(pattern, shape).has_value();
}

IndexTreeL
Column::ixt(const IndexTreeL& row_pattern, size_t num) {
  std::vector<std::tuple<coord_t, coord_t, IndexTreeL>> ch;
  auto pattern_n = row_pattern.size();
  auto pattern_rep = num / pattern_n;
  auto pattern_rem = num % pattern_n;
  assert(std::get<0>(row_pattern.index_range()) == 0);
  auto stride = std::get<1>(row_pattern.index_range()) + 1;
  coord_t offset = 0;
  if (row_pattern.children().size() == 1) {
    coord_t i;
    IndexTreeL t;
    std::tie(i, std::ignore, t) = row_pattern.children()[0];
    offset += pattern_rep * stride;
    ch.emplace_back(i, offset, t);
  } else {
    for (size_t r = 0; r < pattern_rep; ++r) {
      std::transform(
        row_pattern.children().begin(),
        row_pattern.children().end(),
        std::back_inserter(ch),
        [&offset](auto& c) {
          auto& [i, n, t] = c;
          return std::make_tuple(i + offset, n, t);
        });
      offset += stride;
    }
  }
  auto rch = row_pattern.children().begin();
  auto rch_end = row_pattern.children().end();
  while (pattern_rem > 0 && rch != rch_end) {
    auto& [i, n, t] = *rch;
    auto tsz = t.size();
    if (pattern_rem >= tsz) {
      auto nt = std::min(pattern_rem / tsz, static_cast<size_t>(n));
      ch.emplace_back(i + offset, nt, t);
      pattern_rem -= nt * tsz;
      if (nt == static_cast<size_t>(n))
        ++rch;
    } else /* pattern_rem < tsz */ {
      auto pt = ixt(t, pattern_rem);
      pattern_rem = 0;
      ch.emplace_back(i + offset, 1, pt);
    }
  }
  auto result = IndexTreeL(ch);
  assert(result.size() == num);
  return result;
}


// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
