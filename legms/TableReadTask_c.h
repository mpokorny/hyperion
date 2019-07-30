#ifndef LEGMS_TABLE_READ_TASK_C_H_
#define LEGMS_TABLE_READ_TASK_C_H_

#include "utility_c.h"
#include "Table_c.h"

#ifdef __cplusplus
extern "C" {
#endif

LEGMS_API void
table_block_read_task(
  legion_context_t context,
  legion_runtime_t runtime,
  const char* path,
  const table_t* table,
  unsigned num_column_names,
  const char** column_names,
  size_t block_length);

#ifdef __cplusplus
}
#endif

#endif // LEGMS_TABLE_READ_TASK_C_H_

// Local Variables:
// mode: c
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
