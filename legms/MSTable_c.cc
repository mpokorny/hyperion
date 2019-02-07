#include "MSTable_c.h"
#include "MSTable.h"
#include "c_util.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <mutex>

using namespace legms;

#define TABLE_FUNCTION_DEFNS(T, t)                                      \
const char*                                                             \
t##_table_name() {                                                      \
  return MSTable<MSTables::T>::name;                                    \
}                                                                       \
                                                                        \
const struct legms_axes_t*                                              \
t##_table_element_axes() {                                              \
  static std::once_flag initialized;                                    \
  static std::vector<legms_axes_t> result;                              \
  std::call_once(                                                       \
    initialized,                                                        \
    []() {                                                              \
      std::vector<std::vector<int>> axes;                               \
      std::transform(                                                   \
        MSTable<MSTables::T>::element_axes.begin(),                     \
        MSTable<MSTables::T>::element_axes.end(),                       \
        std::back_inserter(result),                                     \
        [&axes](auto& nm_axs) {                                         \
          auto& [nm, axs] = nm_axs;                                     \
          std::vector<int> iaxs;                                        \
          std::transform(                                               \
            axs.begin(),                                                \
            axs.end(),                                                  \
            std::back_inserter(iaxs),                                   \
            [](auto& ax) { return static_cast<int>(ax); });             \
          axes.push_back(std::move(iaxs));                              \
          return                                                        \
            legms_axes_t {                                              \
            nm.c_str(),                                                 \
              axes.back().data(),                                       \
              static_cast<unsigned>(axes.back().size())};               \
        });                                                             \
    });                                                                 \
  return result.data();                                                 \
}                                                                       \
                                                                        \
unsigned                                                                \
t##_table_num_columns() {                                               \
  return static_cast<unsigned>(MSTable<MSTables::T>::element_axes.size()); \
}                                                                       \
                                                                        \
const char* const*                                                      \
t##_table_axis_names() {                                                \
  typedef MSTable<MSTables::T>::Axes Axes;                              \
  static std::once_flag initialized;                                    \
  static std::vector<const char*> result;                               \
  std::call_once(                                                       \
    initialized,                                                        \
    []() {                                                              \
      auto axis_names = MSTable<MSTables::T>::axis_names();             \
      result.reserve(axis_names.size());                                \
      for (int i = 0; i <= static_cast<int>(Axes::last); ++i) {         \
        result[i] = axis_names[static_cast<Axes>(i)].c_str();           \
      }                                                                 \
    });                                                                 \
  return result.data();                                                 \
}                                                                       \
                                                                        \
unsigned                                                                \
t##_table_num_axes() {                                                  \
  return static_cast<unsigned>(MSTable<MSTables::T>::Axes::last) + 1;   \
}

FOREACH_MS_TABLE_Tt(TABLE_FUNCTION_DEFNS);

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
