/*
 * Copyright 2020 Associated Universities, Inc. Washington DC, USA.
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
#ifndef HYPERION_UTILITY_C_H_
#define HYPERION_UTILITY_C_H_

#include <hyperion/hyperion_c.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum type_tag_t {
  HYPERION_TYPE_BOOL,
  HYPERION_TYPE_CHAR,
  HYPERION_TYPE_UCHAR,
  HYPERION_TYPE_SHORT,
  HYPERION_TYPE_USHORT,
  HYPERION_TYPE_INT,
  HYPERION_TYPE_UINT,
  HYPERION_TYPE_FLOAT,
  HYPERION_TYPE_DOUBLE,
  HYPERION_TYPE_COMPLEX,
  HYPERION_TYPE_DCOMPLEX,
  HYPERION_TYPE_STRING,
  HYPERION_TYPE_RECT2,
  HYPERION_TYPE_RECT3,
  HYPERION_NUM_TYPE_TAGS
} type_tag_t;

HYPERION_EXPORT void
preregister_all();

#ifdef __cplusplus
}
#endif

#endif // HYPERION_UTILITY_C_H_

// Local Variables:
// mode: c
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
