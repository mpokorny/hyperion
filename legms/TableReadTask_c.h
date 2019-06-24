#ifndef LEGMS_TABLE_READ_TASK_C_H_
#define LEGMS_TABLE_READ_TASK_C_H_

#include "utility_c.h"
#include "Table_c.h"

#ifdef __cplusplus
extern "C" {
#endif

void
table_block_read_task(
  const char* path,
  table_t table,
  // NULL-terminated array of string pointers
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
