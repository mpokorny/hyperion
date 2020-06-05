/* Copyright 2020 Associated Universities, Inc. Washington DC, USA.
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
#include <hyperion/gridder/args.h>
#include <hyperion/utility.h>

#include <forward_list>
#include <sstream>
#if __cplusplus >= 201703L
#include <variant>
#endif

using namespace hyperion::gridder;

#if !HAVE_CXX17
const constexpr char* ArgsBase::h5_path_tag;
const constexpr char* ArgsBase::h5_path_desc;

const constexpr char* ArgsBase::config_path_tag;
const constexpr char* ArgsBase::config_path_desc;

const constexpr char* ArgsBase::echo_tag;
const constexpr char* ArgsBase::echo_desc;

const constexpr char* ArgsBase::min_block_tag;
const constexpr char* ArgsBase::min_block_desc;

const constexpr char* ArgsBase::pa_step_tag;
const constexpr char* ArgsBase::pa_step_desc;

const constexpr char* ArgsBase::pa_block_tag;
const constexpr char* ArgsBase::pa_block_desc;

const constexpr char* ArgsBase::w_planes_tag;
const constexpr char* ArgsBase::w_planes_desc;

const constexpr args_t ArgsCompletion<VALUE_ARGS>::val;
const constexpr args_t ArgsCompletion<STRING_ARGS>::val;
const constexpr args_t ArgsCompletion<OPT_VALUE_ARGS>::val;
const constexpr args_t ArgsCompletion<OPT_STRING_ARGS>::val;

#endif // !HAVE_CXX17

#if HAVE_CXX17
typedef std::variant<std::forward_list<std::string>, Args<OPT_STRING_ARGS>>
read_config_result_t;
#else // !HAVE_CXX17
typedef struct {
  bool is_list;
  std::forward_list<std::string> strings;
  Args<OPT_STRING_ARGS> args;
} read_config_result_t;
#endif // HAVE_CXX17

read_config_result_t
read_config(const YAML::Node& config) {

  std::forward_list<std::string> invalid_tags;
  Args<OPT_STRING_ARGS> args;
  for (auto it = config.begin(); it != config.end(); ++it) {
    auto key = it->first.as<std::string>();
    auto val = it->second.as<std::string>();
    if (key == args.h5_path.tag)
      args.h5_path = val;
    else if (key == args.echo.tag)
      args.echo = val;
    else if (key == args.min_block.tag)
      args.min_block = val;
    else if (key == args.pa_step.tag)
      args.pa_step = val;
    else if (key == args.pa_block.tag)
      args.pa_block = val;
    else if (key == args.w_planes.tag)
      args.w_planes = val;
    else
      invalid_tags.push_front(key);  
  }
  read_config_result_t result;
  if (invalid_tags.empty()) {
#if HAVE_CXX17
    result = args;
#else // !HAVE_CXX17
    result.is_list = false;
    result.args = args;
#endif // HAVE_CXX17
  } else {
#if HAVE_CXX17
    result = invalid_tags;
#else // !HAVE_CXX17
    result.is_list = true;
    result.strings = invalid_tags;
#endif // HAVE_CXX17
  }
  return result;
}

template <typename T>
std::ostringstream&
arg_error(std::ostringstream& oss, const T& arg, const std::string& reason) {
  oss << "'" << arg.tag << "' value '" << arg.value()
      << "': " << reason << std::endl;
  return oss;
}

CXX_OPTIONAL_NAMESPACE::optional<std::string>
load_config(
  const CXX_FILESYSTEM_NAMESPACE::path& path,
  const std::string& config_provenance,
  Args<OPT_STRING_ARGS>& gridder_args) {

  CXX_OPTIONAL_NAMESPACE::optional<std::string> result;
  try {
    auto node = YAML::LoadFile(path);
    auto read_result = read_config(node);
#if HAVE_CXX17
    std::visit(
      hyperion::overloaded {
        [&path, &config_provenance, &result]
        (std::forward_list<std::string>& invalid_tags) {
          std::ostringstream oss;
          oss << "Invalid configuration variable tags in file named by "
              << config_provenance << std::endl
              << " at '" << path << "': " << std::endl
              << invalid_tags.front();
          invalid_tags.pop_front();
          std::for_each(
            invalid_tags.begin(),
            invalid_tags.end(),
            [&oss](auto tg) { oss << ", " << tg; });
          oss << std::endl;
          result = oss.str();
        },
        [&gridder_args](Args<OPT_STRING_ARGS>& args) {
          if (args.h5_path)
            gridder_args.h5_path = args.h5_path.value();
          if (args.echo)
            gridder_args.echo = args.echo.value();
          if (args.min_block)
            gridder_args.min_block = args.min_block.value();
          if (args.pa_step)
            gridder_args.pa_step = args.pa_step.value();
          if (args.pa_block)
            gridder_args.pa_block = args.pa_block.value();
          if (args.w_planes)
            gridder_args.w_planes = args.w_planes.value();
        }
      },
      read_result);
#else // !HAVE_CXX17
    if (read_result.is_list) {
      std::ostringstream oss;
      oss << "Invalid configuration variable tags in file named by "
          << config_provenance << std::endl
          << " at '" << path << "': " << std::endl
          << read_result.strings.front();
      read_result.strings.pop_front();
      std::for_each(
        read_result.strings.begin(),
        read_result.strings.end(),
        [&oss](auto tg) { oss << ", " << tg; });
      oss << std::endl;
      result = oss.str();
    } else {
      if (read_result.args.h5_path)
        gridder_args.h5_path = read_result.args.h5_path.value();
      if (read_result.args.echo)
        gridder_args.echo = read_result.args.echo.value();
      if (read_result.args.min_block)
        gridder_args.min_block = read_result.args.min_block.value();
      if (read_result.args.pa_step)
        gridder_args.pa_step = read_result.args.pa_step.value();
      if (read_result.args.pa_block)
        gridder_args.pa_block = read_result.args.pa_block.value();
      if (read_result.args.w_planes)
        gridder_args.w_planes = read_result.args.w_planes.value();
    }
#endif // HAVE_CXX17
  } catch (const YAML::Exception& e) {
    std::ostringstream oss;
    oss << "Failed to parse configuration file named by "
        << config_provenance << std::endl
        << " at '" << path << "':" << std::endl
        << e.what()
        << std::endl;
    result = oss.str();
  }
  return result;
}

Args<VALUE_ARGS>
hyperion::gridder::as_args(YAML::Node&& node) {

  CXX_FILESYSTEM_NAMESPACE::path h5_path =
    node[ArgsBase::h5_path_tag].as<std::string>();
  CXX_OPTIONAL_NAMESPACE::optional<CXX_FILESYSTEM_NAMESPACE::path> config_path;
  if (node[ArgsBase::config_path_tag])
    config_path = node[ArgsBase::config_path_tag].as<std::string>();
  bool echo = node[ArgsBase::echo_tag].as<bool>();
  size_t min_block = node[ArgsBase::min_block_tag].as<size_t>();
  PARALLACTIC_ANGLE_TYPE pa_step =
    node[ArgsBase::pa_step_tag].as<PARALLACTIC_ANGLE_TYPE>();
  size_t pa_block = node[ArgsBase::pa_block_tag].as<size_t>();
  int w_planes = node[ArgsBase::w_planes_tag].as<int>();
  return
    Args<VALUE_ARGS>(
      h5_path,
      config_path,
      echo,
      min_block,
      pa_step,
      pa_block,
      w_planes);
}

bool
hyperion::gridder::has_help_flag(const Legion::InputArgs& args) {
  for (int i = 0; i < args.argc; ++i) {
    std::string tag = args.argv[i];
    if (std::strcmp(args.argv[i], "--help") == 0)
      return true;
  }
  return false;
}

bool
hyperion::gridder::get_args(
  const Legion::InputArgs& args,
  Args<OPT_STRING_ARGS>& gridder_args) {

  // first look for configuration file given by environment variable
  const char* env_config_pathname = std::getenv("HYPERION_GRIDDER_CONFIG");
  if (env_config_pathname != nullptr) {
    auto errs =
      load_config(
        env_config_pathname,
        "HYPERION_GRIDDER_CONFIG environment variable",
        gridder_args);
    if (errs) {
      std::cerr << errs.value();
      return false;
    }
  }

  // tokenize command line arguments
  std::vector<std::pair<std::string, std::string>> arg_pairs;
  for (int i = 1; i < args.argc;) {
    std::string tag = args.argv[i++];
    if (tag.substr(0, 2) == "--") {
      CXX_OPTIONAL_NAMESPACE::optional<std::string> val;
      tag = tag.substr(2);
      auto eq = tag.find('=');
      if (eq == std::string::npos) {
        if (i < args.argc) {
          val = args.argv[i++];
          if (val.value() == "=") {
            if (i < args.argc)
              val = args.argv[i++];
            else
              val = CXX_OPTIONAL_NAMESPACE::nullopt;
          }
        }
      } else {
        val = tag.substr(eq + 1);
        tag = tag.substr(0, eq);
        if (val.value().size() == 0) {
          if (i < args.argc)
            val = args.argv[i++];
          else
            val = CXX_OPTIONAL_NAMESPACE::nullopt;
        }
      }
      if (val) {
        arg_pairs.emplace_back(tag, val.value());
      } else {
        std::cerr << "No value provided for argument '"
                  << tag << "'" << std::endl;
        return false;
      }
    }
  }

  // apply variables from config file(s) before the rest of the command line
  {
    for (auto& [tag, val] : arg_pairs) {
      std::vector<std::string> matches;
      for (auto& tg : gridder_args.tags())
        if (tg.substr(0, tag.size()) == tag)
          matches.push_back(tg);
      if (std::find(
            matches.begin(),
            matches.end(),
            gridder_args.config_path.tag)
          != matches.end()) {
        if (matches.size() == 1) {
          auto errs =
            load_config(
              val,
              (std::string("command line argument '")
               + gridder_args.config_path.tag + "' value"),
              gridder_args);
          if (errs) {
            std::cerr << errs.value();
            return false;
          }
        } else {
          std::cerr << "Command line argument '"
                    << tag << "' is a prefix of a more than one option"
                    << std::endl;
          return false;
        }
      }
    }
  }
  // apply other variables from the command line
  for (auto& [tag, val] : arg_pairs) {
    std::vector<std::string> matches;
    for (auto& tg : gridder_args.tags())
      if (tg.substr(0, tag.size()) == tag)
        matches.push_back(tg);
    if (matches.size() == 1) {
      auto match = matches.front();
      if (match == gridder_args.h5_path.tag)
        gridder_args.h5_path = val;
      else if (match == gridder_args.pa_step.tag)
        gridder_args.pa_step = val;
      else if (match == gridder_args.pa_block.tag)
        gridder_args.pa_block = val;
      else if (match == gridder_args.w_planes.tag)
        gridder_args.w_planes = val;
      else if (match == gridder_args.min_block.tag)
        gridder_args.min_block = val;
      else if (match == gridder_args.echo.tag)
        gridder_args.echo = val;
      else if (match == gridder_args.config_path.tag)
        gridder_args.config_path = val;
      else
        assert(false);
    } else {
      if (matches.size() == 0)
        std::cerr << "Unrecognized command line argument '"
                  << tag << "'" << std::endl;
      else
        std::cerr << "Command line argument '"
                  << tag << "' is a prefix of a more than one option"
                  << std::endl;
      return false;
    }
  }

  return true;
}

CXX_OPTIONAL_NAMESPACE::optional<std::string>
hyperion::gridder::validate_args(const Args<VALUE_ARGS>& args) {

  std::ostringstream errs;
  if (!CXX_FILESYSTEM_NAMESPACE::is_regular_file(args.h5_path.value()))
    arg_error(errs, args.h5_path, "not a regular file");

  switch (std::fpclassify(args.pa_step.value())) {
  case FP_NORMAL:
    if (std::abs(args.pa_step.value()) > PARALLACTIC_360)
      arg_error(errs, args.pa_step, "not in valid range [-360, 360]");
    break;
  case FP_ZERO:
  case FP_SUBNORMAL:
    arg_error(errs, args.pa_step, "too small");
    break;
  default:
    arg_error(errs, args.pa_step, "invalid");
    break;
  }

  if (args.pa_block.value() == INVALID_MIN_BLOCK_SIZE_VALUE)
    arg_error(
      errs,
      args.pa_block,
      "invalid, value must be at least one");

  if (args.w_planes.value() <= INVALID_W_PROJ_PLANES_VALUE)
    arg_error(
      errs,
      args.w_planes,
      std::string("less than the minimum valid value of ")
      + std::to_string(INVALID_W_PROJ_PLANES_VALUE + 1));
  else if (args.w_planes.value() == AUTO_W_PROJ_PLANES_VALUE)
    arg_error(
      errs,
      args.w_planes,
      "automatic computation of the number of W-projection "
      "planes is unimplemented");

  if (args.min_block.value() == INVALID_MIN_BLOCK_SIZE_VALUE)
    arg_error(
      errs,
      args.min_block,
      "invalid, value must be at least one");

  if (errs.str().size() > 0)
    return errs.str();
  return CXX_OPTIONAL_NAMESPACE::nullopt;
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
