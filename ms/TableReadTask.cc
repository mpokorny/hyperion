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

void
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
      return result;
    });

  Context ctx = m_table->context();
  Runtime* runtime = m_table->runtime();

  std::vector<Legion::IndexPartition> blockp;
  {
    auto nr = m_table->num_rows();
    auto block_length = m_block_length.value_or(nr);
    std::vector<std::vector<Column::row_number_t>>
      rowp((nr + block_length - 1) / block_length);
    for (Column::row_number_t i = 0; i < nr; ++i)
      rowp[i / block_length].push_back(i);
    blockp =
      m_table->projected_row_partitions(m_column_names, rowp, false, true);
  }

  for (size_t i = 0; i < m_column_names.size(); ++i) {
    auto lr = m_table->column(m_column_names[i])->logical_region();
    if (lr != LogicalRegion::NO_REGION) {
      auto lp = runtime->get_logical_partition(ctx, lr, blockp[i]);
      auto launcher =
        IndexTaskLauncher(
          TASK_ID,
          runtime->get_index_partition_color_space(blockp[i]),
          TaskArgument(&args[i], sizeof(TableReadTaskArgs)),
          ArgumentMap());
      launcher.add_region_requirement(
        RegionRequirement(lp, 0, WRITE_DISCARD, EXCLUSIVE, lr));
      launcher.add_field(0, Column::value_fid);
      launcher.add_region_requirement(
        RegionRequirement(lp, 0, READ_ONLY, EXCLUSIVE, lr));
      launcher.add_field(1, Column::row_number_fid);
      runtime->execute_index_space(ctx, launcher);
      runtime->destroy_index_partition(ctx, blockp[i]);
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
