#include "TableReadTask_c.h"
#include "TableReadTask.h"
#include "c_util.h"

#include "legion/legion_c_util.h"

using namespace legms::ms;
using namespace legms::ms::CObjectWrapper;

static const char**
find_null_ptr_value(const char** column_names) {
  while (*column_names != NULL)
    ++column_names;
  return column_names;
}

void
table_read_task(
  const char* path,
  table_t table,
  const char** column_names) {

  TableReadTask read_task(
    path,
    unwrap(table),
    column_names,
    find_null_ptr_value(column_names));
  read_task.dispatch();
}

void
table_block_read_task(
  const char* path,
  table_t table,
  const char** column_names,
  size_t block_length) {

  TableReadTask read_task(
    path,
    unwrap(table),
    column_names,
    find_null_ptr_value(column_names),
    block_length);
  read_task.dispatch();
}

void
table_read_task_register(legion_runtime_t runtime) {
  TableReadTask::register_task(Legion::CObjectWrapper::unwrap(runtime));
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
