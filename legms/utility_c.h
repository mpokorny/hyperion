#ifndef LEGMS_UTILITY_C_H_
#define LEGMS_UTILITY_C_H_

#include "legms_c.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum legms_type_tag_t {
  LEGMS_TYPE_BOOL,
  LEGMS_TYPE_CHAR,
  LEGMS_TYPE_UCHAR,
  LEGMS_TYPE_SHORT,
  LEGMS_TYPE_USHORT,
  LEGMS_TYPE_INT,
  LEGMS_TYPE_UINT,
  LEGMS_TYPE_FLOAT,
  LEGMS_TYPE_DOUBLE,
  LEGMS_TYPE_COMPLEX,
  LEGMS_TYPE_DCOMPLEX,
  LEGMS_TYPE_STRING
} legms_type_tag_t;

void
legms_preregister_all();

void
legms_register_tasks(legion_runtime_t runtime);

#ifdef __cplusplus
}
#endif

#endif // LEGMS_UTILITY_C_H_

// Local Variables:
// mode: c
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
