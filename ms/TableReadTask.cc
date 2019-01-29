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
  runtime->register_task_variant<base_impl>(registrar);
}

void
TableReadTask::dispatch() {

  TableReadTaskArgs arg_template;
  assert(m_table_path.size() < sizeof(arg_template.table_path));
  std::strcpy(arg_template.table_path, m_table_path.c_str());
  assert(m_table_name.size() < sizeof(arg_template.table_name));
  std::strcpy(arg_template.table_name, m_table_name.c_str());

  std::vector<TableReadTaskArgs> args;
  std::transform(
    m_columns.begin(),
    m_columns.end(),
    std::back_inserter(args),
    [this, &arg_template](const auto& col) {
      TableReadTaskArgs result = arg_template;
      assert(col->name().size() < sizeof(result.column_name));
      std::strcpy(result.column_name, col->name().c_str());
      result.column_rank = col->rank();
      result.column_datatype = col->datatype();
      return result;
    });

  for (size_t i = 0; i < m_columns.size(); ++i) {
    auto col = m_columns[i];
    auto lr = col->logical_region();
    if (lr != LogicalRegion::NO_REGION) {
      auto cp = col->projected_column_partition(m_blockp.get());
      auto lp =
        m_runtime->get_logical_partition(m_context, lr, cp->index_partition());
      auto launcher =
        IndexTaskLauncher(
          TASK_ID,
          m_runtime->get_index_partition_color_space(cp->index_partition()),
          TaskArgument(&args[i], sizeof(TableReadTaskArgs)),
          ArgumentMap());
      launcher.add_region_requirement(
        RegionRequirement(lp, 0, WRITE_DISCARD, EXCLUSIVE, lr));
      launcher.add_field(0, Column::value_fid);
      launcher.add_region_requirement(
        RegionRequirement(lp, 0, READ_ONLY, EXCLUSIVE, lr));
      launcher.add_field(1, Column::row_number_fid);
      m_runtime->execute_index_space(m_context, launcher);
    }
  }
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
      args->column_datatype,
      runtime->get_index_space_domain(
        ctx,
        task->regions[0].region.get_index_space()),
      regions);
    break;
  case 4:
    read_column<4>(
      table,
      cdesc,
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
