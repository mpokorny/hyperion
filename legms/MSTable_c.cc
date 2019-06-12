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
legms_##t##_table_name() {                                              \
  return MSTable<MS_##T>::name;                                         \
}                                                                       \
                                                                        \
const legms_ms_##t##_column_axes_t*                                     \
legms_##t##_table_element_axes() {                                      \
  static std::once_flag initialized;                                    \
  static std::vector<legms_ms_##t##_column_axes_t> result;              \
  std::call_once(                                                       \
    initialized,                                                        \
    []() {                                                              \
      std::transform(                                                   \
        MSTable<MS_##T>::element_axes.begin(),                          \
        MSTable<MS_##T>::element_axes.end(),                            \
        std::back_inserter(result),                                     \
        [](auto& nm_axs) {                                              \
          auto& [nm, axs] = nm_axs;                                     \
          return                                                        \
            legms_ms_##t##_column_axes_t {                              \
            nm.c_str(),                                                 \
              axs.data(),                                               \
              static_cast<unsigned>(axs.size())};                       \
        });                                                             \
    });                                                                 \
  return result.data();                                                 \
}                                                                       \
                                                                        \
unsigned                                                                \
legms_##t##_table_num_columns() {                                       \
  return static_cast<unsigned>(MSTable<MS_##T>::element_axes.size());   \
}                                                                       \
                                                                        \
const char* const*                                                      \
legms_##t##_table_axis_names() {                                        \
  typedef MSTable<MS_##T>::Axes Axes;                                   \
  static std::once_flag initialized;                                    \
  static std::vector<const char*> result;                               \
  std::call_once(                                                       \
    initialized,                                                        \
    []() {                                                              \
      auto axis_names = MSTable<MS_##T>::axis_names();                  \
      result.resize(axis_names.size());                                 \
      for (int i = 0; i <= static_cast<int>(MSTable<MS_##T>::LAST_AXIS); ++i) { \
        result[i] = axis_names[static_cast<Axes>(i)].c_str();           \
      }                                                                 \
    });                                                                 \
  return result.data();                                                 \
}                                                                       \
                                                                        \
unsigned                                                                \
legms_##t##_table_num_axes() {                                          \
  return static_cast<unsigned>(MSTable<MS_##T>::LAST_AXIS) + 1;         \
}

FOREACH_MS_TABLE_Tt(TABLE_FUNCTION_DEFNS);

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
