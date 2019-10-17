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
#ifndef HYPERION_HYPERION_H_
#define HYPERION_HYPERION_H_

#pragma GCC visibility push(default)
#include <legion.h>
#pragma GCC visibility pop

#include <hyperion/hyperion_config.h>

#define HYPERION_API __attribute__((visibility("default")))
#define HYPERION_LOCAL __attribute__((visibility("hidden")))

#if GCC_VERSION >= 90000
# define HYPERION_FS std::filesystem
#else
# define HYPERION_FS std::experimental::filesystem
#endif

#endif // HYPERION_HYPERION_H_

// Local Variables:
// mode: c
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End: