#include "TableReadTask_c.h"
#include "TableReadTask.h"
#include "c_util.h"

#include "legion/legion_c_util.h"

using namespace legms;
using namespace legms::CObjectWrapper;

void
table_block_read_task(
  const char* path,
  table_t table,
  unsigned num_column_names,
  const char** column_names,
  size_t block_length) {

  TableReadTask read_task(
    path,
    unwrap(table),
    column_names,
    column_names + num_column_names,
    block_length);
  read_task.dispatch();
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
