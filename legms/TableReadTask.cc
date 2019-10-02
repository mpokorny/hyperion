#pragma GCC visibility push(default)
#include <algorithm>
#pragma GCC visibility pop

#include <legms/TableReadTask.h>

using namespace legms;

using namespace Legion;

TaskID TableReadTask::TASK_ID = 0;
const char* TableReadTask::TASK_NAME = "TableReadTask";

void
TableReadTask::preregister_task() {
  TASK_ID = Runtime::generate_static_task_id();
  TaskVariantRegistrar registrar(TASK_ID, TASK_NAME, false);
  registrar.add_constraint(ProcessorConstraint(Processor::IO_PROC));
  registrar.set_leaf();
  Runtime::preregister_task_variant<base_impl>(registrar, TASK_NAME);
}

void
TableReadTask::dispatch(Context ctx, Runtime* rt) {

  RegionRequirement
    req(m_table.columns_lr, READ_ONLY, EXCLUSIVE, m_table.columns_lr);
  req.add_field(Table::COLUMNS_FID);
  auto tcol = rt->map_region(ctx, req);

  if (!Table::is_empty(ctx, rt, tcol)) {

    auto c = Table::min_rank_column(ctx, rt, tcol);
    auto blockp =
      c.partition_on_axes(ctx, rt, {std::make_tuple(0, m_block_length)});

    TableReadTaskArgs args;
    assert(m_table_path.size() < sizeof(args.table_path));
    std::strcpy(args.table_path, m_table_path.c_str());

    for (auto& colname : m_colnames)  {
      auto column = Table::column(ctx, rt, tcol, colname);
      if (!column.is_empty()) {
        auto cp = column.projected_column_partition(ctx, rt, blockp);
        auto lp =
          rt->get_logical_partition(ctx, column.values_lr, cp.index_partition);
        auto launcher =
          IndexTaskLauncher(
            TASK_ID,
            rt->get_index_partition_color_space(cp.index_partition),
            TaskArgument(&args, sizeof(args)),
            ArgumentMap());
        {
          RegionRequirement
            req(lp, 0, WRITE_ONLY, EXCLUSIVE, column.values_lr);
          req.add_field(Column::VALUE_FID);
          launcher.add_region_requirement(req);
        }
        {
          RegionRequirement
            req(column.metadata_lr, READ_ONLY, EXCLUSIVE, column.metadata_lr);
          req.add_field(Column::METADATA_NAME_FID);
          req.add_field(Column::METADATA_DATATYPE_FID);
          launcher.add_region_requirement(req);
        }
        rt->execute_index_space(ctx, launcher);
        // FIXME: enable
        // rt->destroy_logical_partition(ctx, lp);
        // cp.destroy(ctx, rt);
      }
    }
    blockp.destroy(ctx, rt);
  }
  rt->unmap_region(ctx, tcol);
}

void
TableReadTask::base_impl(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *rt) {

  const TableReadTaskArgs* args =
    static_cast<const TableReadTaskArgs*>(task->args);
  casacore::Table table(
    args->table_path,
    casacore::TableLock::PermanentLockingWait);
  auto tdesc = table.tableDesc();
  const Column::NameAccessor<READ_ONLY>
    name(regions[1], Column::METADATA_NAME_FID);
  const Column::DatatypeAccessor<READ_ONLY>
    datatype(regions[1], Column::METADATA_DATATYPE_FID);
  auto cdesc = tdesc[std::string(name[0])];
  switch (regions[0].get_logical_region().get_index_space().get_dim()) {
#if LEGION_MAX_DIM >= 1
  case 1:
    read_column<1>(
      table,
      cdesc,
      datatype[0],
      rt->get_index_space_domain(task->regions[0].region.get_index_space()),
      regions[0]);
    break;
#endif
#if LEGION_MAX_DIM >= 2
  case 2:
    read_column<2>(
      table,
      cdesc,
      datatype[0],
      rt->get_index_space_domain(task->regions[0].region.get_index_space()),
      regions[0]);
    break;
#endif
#if LEGION_MAX_DIM >= 3
  case 3:
    read_column<3>(
      table,
      cdesc,
      datatype[0],
      rt->get_index_space_domain(task->regions[0].region.get_index_space()),
      regions[0]);
    break;
#endif
#if LEGION_MAX_DIM >= 4
  case 4:
    read_column<4>(
      table,
      cdesc,
      datatype[0],
      rt->get_index_space_domain(task->regions[0].region.get_index_space()),
      regions[0]);
    break;
#endif
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
