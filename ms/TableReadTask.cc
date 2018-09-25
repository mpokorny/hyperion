#include "TableReadTask.h"

using namespace legms::ms;

Legion::TaskID TableReadTask::TASK_ID = 0;

void
TableReadTask::register_task(Legion::Runtime* runtime) {
  TASK_ID = runtime->generate_library_task_ids("legms::TableReadTask", 1);
  Legion::TaskVariantRegistrar registrar(TASK_ID, TASK_NAME);
  registrar.add_constraint(
    Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
  runtime->register_task_variant<base_impl>(registrar);
}

std::vector<std::tuple<Legion::LogicalRegion, Legion::FieldID>>
TableReadTask::dispatch(Legion::Context ctx, Legion::Runtime* runtime) {

  size_t ser_row_index_shape_size =
    index_tree_serdez::serialized_size(m_table.row_index_shape());
  size_t args_size = sizeof(TableReadTaskArgs) + ser_row_index_shape_size;
  std::unique_ptr<TableReadTaskArgs> args(
    static_cast<TableReadTaskArgs*>(::operator new(args_size)));
  assert(m_table_path.size() < sizeof(args->table_path));
  std::strcpy(args->table_path, m_table_path.c_str());
  assert(m_table.name().size() < sizeof(args->table_name));
  std::strcpy(args->table_name, m_table.name().c_str());
  assert(m_column_names.size()
         < sizeof(args->column_names) / sizeof(args->column_names[0]));
  for (size_t i = 0; i < m_column_names.size(); ++i) {
    assert(m_column_names[i].size() < sizeof(args->column_names[0]));
    std::strcpy(args->column_names[i], m_column_names[i].c_str());
    auto col = m_table.column(m_column_names[i]);
    args->column_ranks[i] = col->rank();
    args->column_datatypes[i] = col->datatype();
  }
  index_tree_serdez::serialize(
    m_table.row_index_shape(),
    args->ser_row_index_shape);

  auto result = m_table.logical_regions(ctx, runtime, m_column_names);
  if (m_index_partition) {
    auto ip = m_index_partition.value();
    auto ips =
      m_table.index_partitions(ctx, runtime, ip, m_column_names);
    std::vector<Legion::LogicalPartition> lps;
    for (size_t i = 0; i < result.size(); ++i)
      lps.push_back(
        runtime->get_logical_partition(
          ctx,
          std::get<0>(result[i]), ips[i]));

    auto launcher = Legion::IndexTaskLauncher(
      TASK_ID,
      runtime->get_index_partition_color_space(ctx, ip),
      Legion::TaskArgument(args.get(), args_size),
      Legion::ArgumentMap());
    for (size_t i = 0; i < lps.size(); ++i) {
      auto& [lr, fid] = result[i];
      Legion::RegionRequirement req(lps[i], 0, WRITE_DISCARD, EXCLUSIVE, lr);
      req.add_field(fid);
      launcher.add_region_requirement(req);
    }
    runtime->execute_index_space(ctx, launcher);
  } else {
    auto launcher = Legion::TaskLauncher(
      TASK_ID,
      Legion::TaskArgument(args.get(), args_size));
    for (size_t i = 0; i < result.size(); ++i) {
      auto& [lr, fid] = result[i];
      Legion::RegionRequirement req(lr, WRITE_DISCARD, EXCLUSIVE, lr);
      req.add_field(fid);
      launcher.add_region_requirement(req);
    }
    runtime->execute_task(ctx, launcher);
  }
  return result;
}

void
TableReadTask::base_impl(
  const Legion::Task* task,
  const std::vector<Legion::PhysicalRegion>& regions,
  Legion::Context ctx,
  Legion::Runtime *runtime) {

  const TableReadTaskArgs* args =
    static_cast<const TableReadTaskArgs*>(task->args);
  IndexTreeL row_index_shape;
  index_tree_serdez::deserialize(row_index_shape, args->ser_row_index_shape);
  casacore::Table table(args->table_path, casacore::TableLock::NoLocking);
  auto tdesc = table.tableDesc();
  // TODO: one task per column?
  for (size_t i = 0; i < regions.size(); ++i) {
    auto cdesc = tdesc[args->column_names[i]];
    switch (args->column_ranks[i]) {
    case 1:
      read_column<1>(
        table,
        cdesc,
        row_index_shape,
        args->column_datatypes[i],
        runtime->get_index_space_domain(
          ctx,
          task->regions[i].region.get_index_space()),
        regions[i]);
      break;
    case 2:
      read_column<2>(
        table,
        cdesc,
        row_index_shape,
        args->column_datatypes[i],
        runtime->get_index_space_domain(
          ctx,
          task->regions[i].region.get_index_space()),
        regions[i]);
      break;
    case 3:
      read_column<3>(
        table,
        cdesc,
        row_index_shape,
        args->column_datatypes[i],
        runtime->get_index_space_domain(
          ctx,
          task->regions[i].region.get_index_space()),
        regions[i]);
      break;
    default:
      assert(false);
      break;
    }
  }
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
