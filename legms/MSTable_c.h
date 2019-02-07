#ifndef LEGMS_MS_TABLE_C_H_
#define LEGMS_MS_TABLE_C_H_

#include "c_util.h"

#ifdef __cplusplus
extern "C" {
#endif

struct legms_axes_t {
  const char* column;
  const int* axes;
  unsigned num_axes;
};

#define TABLE_FUNCTION_DECLS(t)                 \
const char*                                     \
t##_table_name();                               \
                                                \
const struct legms_axes_t*                      \
t##_table_element_axes();                       \
                                                \
unsigned                                        \
t##_table_num_columns();                        \
                                                \
const char* const*                              \
t##_table_axis_names();                         \
                                                \
unsigned                                        \
t##_table_num_axes()

FOREACH_MS_TABLE_t(TABLE_FUNCTION_DECLS);

#ifdef __cplusplus
}
#endif

#endif // LEGMS_MS_TABLE_C_H_

// Local Variables:
// mode: c
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
