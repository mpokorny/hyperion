#include <algorithm>
#include "TableReadTask.h"

using namespace legms::ms;

using namespace Legion;

TaskID TableReadTask::TASK_ID = 0;

void
TableReadTask::register_task(Runtime* runtime) {
  TASK_ID = runtime->generate_library_task_ids("legms::TableReadTask", 1);
  TaskVariantRegistrar registrar(TASK_ID, TASK_NAME);
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  runtime->register_task_variant<base_impl>(registrar);
}

std::vector<std::tuple<LogicalRegion, FieldID>>
TableReadTask::dispatch() {

  size_t ser_row_index_shape_size =
    index_tree_serdez::serialized_size(m_table->row_index_shape());
  size_t args_size = sizeof(TableReadTaskArgs) + ser_row_index_shape_size;
  std::unique_ptr<TableReadTaskArgs> arg_template(
    static_cast<TableReadTaskArgs*>(::operator new(args_size)));
  assert(m_table_path.size() < sizeof(arg_template->table_path));
  std::strcpy(arg_template->table_path, m_table_path.c_str());
  assert(m_table->name().size() < sizeof(arg_template->table_name));
  std::strcpy(arg_template->table_name, m_table->name().c_str());
  index_tree_serdez::serialize(
    m_table->row_index_shape(),
    arg_template->ser_row_index_shape);

  std::vector<std::unique_ptr<TableReadTaskArgs>> args;
  std::transform(
    m_column_names.begin(),
    m_column_names.end(),
    std::back_inserter(args),
    [this, &arg_template, args_size](auto& nm) {
      std::unique_ptr<TableReadTaskArgs> result(
        static_cast<TableReadTaskArgs*>(::operator new(args_size)));
      memcpy(result.get(), arg_template.get(), args_size);
      assert(nm.size() < sizeof(result->column_name));
      std::strcpy(result->column_name, nm.c_str());
      auto col = m_table->column(nm);
      result->column_rank = col->rank();
      result->column_datatype = col->datatype();
      return result;
    });

  auto result = m_table->logical_regions(m_column_names);
  auto [ips, ip] =
    m_table->row_block_index_partitions(
      m_index_partition,
      m_column_names,
      m_block_length);
  auto cs =
    m_table->runtime()->get_index_partition_color_space(m_table->context(), ip);
  for (size_t i = 0; i < result.size(); ++i) {
    auto launcher = IndexTaskLauncher(
      TASK_ID,
      cs,
      TaskArgument(args[i].get(), args_size),
      ArgumentMap());
    auto& [lr, fid] = result[i];
    LogicalPartition lp =
      m_table->runtime()->get_logical_partition(m_table->context(), lr, ips[i]);
    RegionRequirement req(lp, 0, WRITE_DISCARD, EXCLUSIVE, lr);
    req.add_field(fid);
    launcher.add_region_requirement(req);
    m_table->runtime()->execute_index_space(m_table->context(), launcher);
  }
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
  IndexTreeL row_index_shape;
  index_tree_serdez::deserialize(row_index_shape, args->ser_row_index_shape);
  casacore::Table table(args->table_path, casacore::TableLock::NoLocking);
  auto tdesc = table.tableDesc();
  auto cdesc = tdesc[args->column_name];
  switch (args->column_rank) {
  case 1:
    read_column<1>(
      table,
      cdesc,
      row_index_shape,
      args->column_datatype,
      runtime->get_index_space_domain(
        ctx,
        task->regions[0].region.get_index_space()),
      regions[0]);
    break;
  case 2:
    read_column<2>(
      table,
      cdesc,
      row_index_shape,
      args->column_datatype,
      runtime->get_index_space_domain(
        ctx,
        task->regions[0].region.get_index_space()),
      regions[0]);
    break;
  case 3:
    read_column<3>(
      table,
      cdesc,
      row_index_shape,
      args->column_datatype,
      runtime->get_index_space_domain(
        ctx,
        task->regions[0].region.get_index_space()),
      regions[0]);
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
