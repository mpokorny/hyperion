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
#ifndef HYPERION_C_UTIL_H_
#define HYPERION_C_UTIL_H_

#include <hyperion/hyperion_c.h>

#ifdef __cplusplus
#include <memory>
#ifdef __clang__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif // __clang__
#include <legion/legion_c_util.h>
#ifdef __clang__
#pragma GCC diagnostic pop
#endif // __clang__

namespace hyperion {
namespace CObjectWrapper {

template <typename Unwrapped>
struct Wrapper {
  // typedef ... t
  // static t wrap(const Unwrapped&);
};

template <typename Wrapped>
struct Unwrapper {
  // typedef ... t
  // static t unwrap(const Wrapper&)
};

template <typename Unwrapped>
static typename Wrapper<Unwrapped>::t
wrap(const Unwrapped& u) {
  return Wrapper<Unwrapped>::wrap(u);
}

template <typename Wrapped>
static typename Unwrapper<Wrapped>::t
unwrap(const Wrapped& w) {
  return Unwrapper<Wrapped>::unwrap(w);
}

}
}
#endif // __cplusplus

#endif // HYPERION_C_UTIL_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
