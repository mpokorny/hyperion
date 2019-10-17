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
#ifndef LEGMS_LEGMS_H_
#define LEGMS_LEGMS_H_

#pragma GCC visibility push(default)
#include <legion.h>
#pragma GCC visibility pop

#include <legms/legms_config.h>

#define LEGMS_API __attribute__((visibility("default")))
#define LEGMS_LOCAL __attribute__((visibility("hidden")))

#if GCC_VERSION >= 90000
# define LEGMS_FS std::filesystem
#else
# define LEGMS_FS std::experimental::filesystem
#endif

#endif // LEGMS_LEGMS_H_

// Local Variables:
// mode: c
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
