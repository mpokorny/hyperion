#include <algorithm>
#include "TableReadTask.h"

using namespace legms::ms;

using namespace Legion;

TaskID TableReadTask::TASK_ID = 0;

void
TableReadTask::register_task(Runtime* runtime) {
  TASK_ID = runtime->generate_library_task_ids("legms::TableReadTask", 1);
  TaskVariantRegistrar registrar(TASK_ID, TASK_NAME);
  registrar.add_constraint(ProcessorConstraint(Processor::IO_PROC));
  registrar.set_leaf(true);
  runtime->register_task_variant<base_impl>(registrar);
}

std::vector<LogicalRegion>
TableReadTask::dispatch() {

  TableReadTaskArgs arg_template;
  assert(m_table_path.size() < sizeof(arg_template.table_path));
  std::strcpy(arg_template.table_path, m_table_path.c_str());
  assert(m_table->name().size() < sizeof(arg_template.table_name));
  std::strcpy(arg_template.table_name, m_table->name().c_str());

  std::vector<TableReadTaskArgs> args;
  std::transform(
    m_column_names.begin(),
    m_column_names.end(),
    std::back_inserter(args),
    [this, &arg_template](auto& nm) {
      TableReadTaskArgs result = arg_template;
      assert(nm.size() < sizeof(result.column_name));
      std::strcpy(result.column_name, nm.c_str());
      auto col = m_table->column(nm);
      result.column_rank = col->rank();
      result.column_datatype = col->datatype();
      result.column_hint = m_column_hints.at(nm);
      return result;
    });

  Context ctx = m_table->context();
  Runtime* runtime = m_table->runtime();
  auto result = m_table->logical_regions(m_column_names);
  auto ips =
    m_table->row_block_index_partitions(
      m_index_partition,
      m_column_names,
      m_block_length.value_or(m_table->num_rows()));
  for (size_t i = 0; i < result.size(); ++i) {
    auto lr = result[i];
    auto lp = runtime->get_logical_partition(ctx, lr, ips[i]);
    auto cs = runtime->get_index_partition_color_space(ctx, ips[i]);
    auto launcher =
      IndexTaskLauncher(
        TASK_ID,
        cs,
        TaskArgument(&args[i], sizeof(TableReadTaskArgs)),
        ArgumentMap());
    launcher.add_region_requirement(
      RegionRequirement(lp, 0, WRITE_DISCARD, EXCLUSIVE, lr));
    launcher.add_field(0, Column::value_fid);
    launcher.add_region_requirement(
      RegionRequirement(lp, 0, READ_ONLY, EXCLUSIVE, lr));
    launcher.add_field(1, Column::row_number_fid);
    runtime->execute_index_space(ctx, launcher);
  }
  std::for_each(
    ips.begin(),
    ips.end(),
    [runtime, ctx](auto p) {
        runtime->destroy_index_partition(ctx, p);
    });
  return result;
}

void
TableReadTask::base_impl(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *runtime) {

  const TableReadTaskArgs* args =
    static_cast<const TableReadTaskArgs*>(task->args);
  casacore::Table table(
    args->table_path,
    casacore::TableLock::PermanentLockingWait);
  auto tdesc = table.tableDesc();
  auto cdesc = tdesc[args->column_name];
  switch (args->column_rank) {
  case 1:
    read_column<1>(
      table,
      cdesc,
      args->column_hint,
      args->column_datatype,
      runtime->get_index_space_domain(
        ctx,
        task->regions[0].region.get_index_space()),
      regions);
    break;
  case 2:
    read_column<2>(
      table,
      cdesc,
      args->column_hint,
      args->column_datatype,
      runtime->get_index_space_domain(
        ctx,
        task->regions[0].region.get_index_space()),
      regions);
    break;
  case 3:
    read_column<3>(
      table,
      cdesc,
      args->column_hint,
      args->column_datatype,
      runtime->get_index_space_domain(
        ctx,
        task->regions[0].region.get_index_space()),
      regions);
    break;
  default:
    assert(false);
    break;
  }
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
