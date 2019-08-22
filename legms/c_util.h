#ifndef LEGMS_C_UTIL_H_
#define LEGMS_C_UTIL_H_

#include "legms_c.h"

#ifdef __cplusplus
#pragma GCC visibility push(default)
#include <memory>
#include "legion/legion_c_util.h"
#pragma GCC visibility pop

namespace legms {
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

#define FOREACH_MS_TABLE_Tt(__func__)           \
  __func__(MAIN, main)                         \
  __func__(ANTENNA, antenna)                   \
  __func__(DATA_DESCRIPTION, data_description) \
  __func__(DOPPLER, doppler)                   \
  __func__(FEED, feed)                         \
  __func__(FIELD, field)                       \
  __func__(FLAG_CMD, flag_cmd)                 \
  __func__(FREQ_OFFSET, freq_offset)           \
  __func__(HISTORY, history)                   \
  __func__(OBSERVATION, observation)           \
  __func__(POINTING, pointing)                 \
  __func__(POLARIZATION, polarization)         \
  __func__(PROCESSOR, processor)               \
  __func__(SOURCE, source)                     \
  __func__(SPECTRAL_WINDOW, spectral_window)   \
  __func__(STATE, state)                       \
  __func__(SYSCAL, syscal)                     \
  __func__(WEATHER, weather)

#define FOREACH_MS_TABLE_t(__func__)            \
  __func__(main)                               \
  __func__(antenna)                            \
  __func__(data_description)                   \
  __func__(doppler)                            \
  __func__(feed)                               \
  __func__(field)                              \
  __func__(flag_cmd)                           \
  __func__(freq_offset)                        \
  __func__(history)                            \
  __func__(observation)                        \
  __func__(pointing)                           \
  __func__(polarization)                       \
  __func__(processor)                          \
  __func__(source)                             \
  __func__(spectral_window)                    \
  __func__(state)                              \
  __func__(syscal)                             \
  __func__(weather)

#endif // LEGMS_C_UTIL_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
