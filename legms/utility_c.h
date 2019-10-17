/*
 * Copyright 2019 Associated Universities, Inc. Washington DC, USA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef LEGMS_UTILITY_C_H_
#define LEGMS_UTILITY_C_H_

#include <legms/legms_c.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum type_tag_t {
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
} type_tag_t;

LEGMS_API void
preregister_all();

LEGMS_API void
register_tasks(legion_context_t context, legion_runtime_t runtime);

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
