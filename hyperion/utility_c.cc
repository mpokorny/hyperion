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
#include <hyperion/utility_c.h>
#include <hyperion/utility.h>

#pragma GCC visibility push(default)
# include <legion/legion_c_util.h>
#pragma GCC visibility pop

void
preregister_all() {
  hyperion::preregister_all();
}

void
register_tasks(legion_context_t context, legion_runtime_t runtime) {
  hyperion::register_tasks(
    Legion::CObjectWrapper::unwrap(context)->context(),
    Legion::CObjectWrapper::unwrap(runtime));
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
