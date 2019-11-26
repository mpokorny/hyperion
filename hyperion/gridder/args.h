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
#include <hyperion/hyperion.h>

#include <hyperion/gridder/gridder.h>

#pragma GCC visibility push(default)
# include CXX_FILESYSTEM_HEADER
# include <optional>
# include <string>
# include <vector>

# include <yaml-cpp/yaml.h>
#pragma GCC visibility pop

namespace hyperion {
namespace gridder {

typedef enum {
  VALUE_ARGS,
  STRING_ARGS,
  OPT_VALUE_ARGS,
  OPT_STRING_ARGS} args_t;

template <args_t G>
struct ArgsCompletion {
  static const args_t val;
};
template <>
struct ArgsCompletion<VALUE_ARGS> {
  static const constexpr args_t val = VALUE_ARGS;
};
template <>
struct ArgsCompletion<STRING_ARGS> {
  static const constexpr args_t val = STRING_ARGS;
};
template <>
struct ArgsCompletion<OPT_VALUE_ARGS> {
  static const constexpr args_t val = VALUE_ARGS;
};
template <>
struct ArgsCompletion<OPT_STRING_ARGS> {
  static const constexpr args_t val = STRING_ARGS;
};

template <typename T, bool OPT, const char* const* TAG, args_t G>
struct ArgType {
  static const constexpr char* tag = *TAG;
};
template <typename T, const char* const* TAG>
struct ArgType<T, false, TAG, VALUE_ARGS> {
  typedef T type;

  T val;

  static const constexpr char* tag = *TAG;

  T
  value() const {
    return val;
  }

  void
  operator=(const T& t) {
    val = t;
  }

  operator bool() const {
    return true;
  }
};
template <typename T, const char* const* TAG>
struct ArgType<T, true, TAG, VALUE_ARGS> {
  typedef std::optional<T> type;

  std::optional<T> val;

  static const constexpr char* tag = *TAG;

  T
  value() const {
    return val.value();
  }

  void
  operator=(const T& t) {
    val = t;
  }

  void
  operator=(const std::optional<T>& ot) {
    val = ot;
  }

  operator bool() const {
    return val.has_value();
  }
};
template <typename T, const char* const* TAG>
struct ArgType<T, false, TAG, STRING_ARGS> {
  typedef std::string type;

  std::string val;

  static const constexpr char* tag = *TAG;

  std::string
  value() const {
    return val;
  }

  void
  operator=(const std::string& str) {
    val = str;
  }

  operator bool() const {
    return true;
  }
};
template <typename T, const char* const* TAG>
struct ArgType<T, true, TAG, STRING_ARGS> {
  typedef std::optional<std::string> type;

  std::optional<std::string> val;

  static const constexpr char* tag = *TAG;

  std::string
  value() const {
    return val.value();
  }

  void
  operator=(const std::string& str) {
    val = str;
  }

  void
  operator=(const std::optional<std::string>& ostr) {
    val = ostr;
  }

  operator bool() const {
    return val.has_value();
  }
};
template <typename T, const char* const* TAG>
struct ArgType<T, false, TAG, OPT_VALUE_ARGS> {
  typedef std::optional<T> type;

  std::optional<T> val;

  static const constexpr char* tag = *TAG;

  T
  value() const {
    return val;
  }

  void
  operator=(const T& t) {
    val = t;
  }

  void
  operator=(const std::optional<T>& ot) {
    val = ot;
  }

  operator bool() const {
    return val.has_value();
  }
};
template <typename T, const char* const* TAG>
struct ArgType<T, true, TAG, OPT_VALUE_ARGS> {
  typedef std::optional<T> type;

  std::optional<T> val;

  static const constexpr char* tag = *TAG;

  T
  value() const {
    return val.value();
  }

  void
  operator=(const T& t) {
    val = t;
  }

  void
  operator=(const std::optional<T>& ot) {
    val = ot;
  }

  operator bool() const {
    return val.has_value();
  }
};
template <typename T, const char* const* TAG>
struct ArgType<T, false, TAG, OPT_STRING_ARGS> {
  typedef std::optional<std::string> type;

  std::optional<std::string> val;

  static const constexpr char* tag = *TAG;

  std::string
  value() const {
    return val.value();
  }

  void
  operator=(const std::string& str) {
    val = str;
  }

  void
  operator=(const std::optional<std::string>& ostr) {
    val = ostr;
  }

  operator bool() const {
    return val.has_value();
  }
};
template <typename T, const char* const* TAG>
struct ArgType<T, true, TAG, OPT_STRING_ARGS> {
  typedef std::optional<std::string> type;

  std::optional<std::string> val;

  static const constexpr char* tag = *TAG;

  std::string
  value() const {
    return val.value();
  }

  void
  operator=(const std::string& str) {
    val = str;
  }

  void
  operator=(const std::optional<std::string>& ostr) {
    val = ostr;
  }

  operator bool() const {
    return val.has_value();
  }
};

struct ArgsBase {
  static const constexpr char* h5_path_tag = "h5";
  static const constexpr char* config_path_tag = "configuration";
  static const constexpr char* echo_tag = "echo";
  static const constexpr char* min_block_tag = "min_block";
  static const constexpr char* pa_step_tag = "pa_step";
  static const constexpr char* w_planes_tag = "w_proj_planes";

  static const std::vector<std::string>&
  tags() {
    static const std::vector<std::string> result{
      h5_path_tag,
      config_path_tag,
      echo_tag,
      min_block_tag,
      pa_step_tag,
      w_planes_tag
    };
    return result;
  }
};

template <args_t G>
struct Args
  : public ArgsBase {

  ArgType<FS::path, false, &h5_path_tag, G> h5_path;
  ArgType<FS::path, true, &config_path_tag, G> config_path;
  ArgType<bool, false, &echo_tag, G> echo;
  ArgType<size_t, false, &min_block_tag, G> min_block;
  ArgType<PARALLACTIC_ANGLE_TYPE, false, &pa_step_tag, G> pa_step;
  ArgType<int, false, &w_planes_tag, G> w_planes;

  Args() {}

  Args(
    const typename decltype(h5_path)::type& h5_path_,
    const typename decltype(config_path)::type& config_path_,
    const typename decltype(echo)::type& echo_,
    const typename decltype(min_block)::type& min_block_,
    const typename decltype(pa_step)::type& pa_step_,
    const typename decltype(w_planes)::type& w_planes_) {

    h5_path = h5_path_;
    config_path = config_path_;
    echo = echo_;
    min_block = min_block_;
    pa_step = pa_step_;
    w_planes = w_planes_;
  }

  bool
  is_complete() const {
    return
      h5_path
      && echo
      && min_block
      && pa_step
      && w_planes;
  }

  std::optional<Args<ArgsCompletion<G>::val>>
  as_complete() const {
    std::optional<Args<ArgsCompletion<G>::val>> result;
    if (is_complete())
      result =
        std::make_optional<Args<ArgsCompletion<G>::val>>(
          h5_path.value(),
          (config_path
           ? config_path.value()
           : std::optional<std::string>()),
          echo.value(),
          min_block.value(),
          pa_step.value(),
          w_planes.value());
    return result;
  }

  YAML::Node
  as_node() const {
    YAML::Node result;
    if (h5_path)
      result[h5_path.tag] = h5_path.value().c_str();
    if (config_path)
      result[config_path.tag] = config_path.value().c_str();
    if (echo)
      result[echo.tag] = echo.value();
    if (min_block)
      result[min_block.tag] = min_block.value();
    if (pa_step)
      result[pa_step.tag] = pa_step.value();
    if (w_planes)
      result[w_planes.tag] = w_planes.value();
    return result;
  }
};

Args<VALUE_ARGS>
as_args(YAML::Node&& node);

bool
get_args(const Legion::InputArgs& args, Args<OPT_STRING_ARGS>& gridder_args);

std::optional<std::string>
validate_args(const Args<VALUE_ARGS>& args);

} // end namespace gridder
} // end namespace hyperion


// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
