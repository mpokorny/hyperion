#include "TableReadTask_c.h"
#include "TableReadTask.h"
#include "c_util.h"

#pragma GCC visibility push(default)
#include "legion/legion_c_util.h"
#pragma GCC visibility pop

using namespace legms;

// void
// table_block_read_task(
//   legion_context_t context,
//   legion_runtime_t runtime,
//   const char* path,
//   const table_t* table,
//   unsigned num_column_names,
//   const char** column_names,
//   size_t block_length) {

//   TableReadTask read_task(
//     path,
//     TableGenArgs(*table)(
//       Legion::CObjectWrapper::unwrap(context)->context(),
//       Legion::CObjectWrapper::unwrap(runtime)).get(),
//     column_names,
//     column_names + num_column_names,
//     block_length);
//   read_task.dispatch();
// }

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
