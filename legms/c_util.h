#ifndef LEGMS_C_UTIL_H_
#define LEGMS_C_UTIL_H_

#include "legms_c.h"

#ifdef __cplusplus
#include <memory>

namespace legms {
namespace CObjectWrapper {

template <typename Wrapper>
struct UniqueWrapped {
  // typedef type_t;
  // typedef std::unique_ptr<type_t> impl_t;
};

template <typename Wrapper>
struct SharedWrapped {
  // typedef type_t;
  // typedef std::shared_ptr<type_t> impl_t;
};

template <typename Wrapped>
struct UniqueWrapper {
  // typedef type_t;
};

template <typename Wrapped>
struct SharedWrapper {
  // typedef type_t
};

template <typename Wrapped>
static typename UniqueWrapper<Wrapped>::type_t
wrap(std::unique_ptr<Wrapped>&& wrapped) {
  typename UniqueWrapper<Wrapped>::type_t result;
  std::unique_ptr<Wrapped>* impl =
    new std::unique_ptr<Wrapped>(std::move(wrapped));
  result.impl = impl;
  return result;
};

template <typename Wrapper>
static typename UniqueWrapped<Wrapper>::type_t*
unwrap(Wrapper wrapper) {
  return
    (static_cast<std::unique_ptr<typename UniqueWrapped<Wrapper>::type_t>*>(
      wrapper.impl))->get();
}

template <
  typename Wrapper,
  typename Impl = typename UniqueWrapped<Wrapper>::impl_t>
static void
destroy(Wrapper& wrapper) {
  Impl* impl = static_cast<Impl *>(wrapper.impl);
  delete impl;
}

template <typename Wrapped>
static typename SharedWrapper<Wrapped>::type_t
wrap(std::shared_ptr<Wrapped>&& wrapped) {
  typename SharedWrapper<Wrapped>::type_t result;
  std::shared_ptr<Wrapped>* impl =
    new std::shared_ptr<Wrapped>(
      std::forward<std::shared_ptr<Wrapped>>(wrapped));
  result.impl = impl;
  return result;
};

template <typename Wrapper>
static std::shared_ptr<typename SharedWrapped<Wrapper>::type_t>&
unwrap(Wrapper wrapper) {
  return
    *(static_cast<std::shared_ptr<typename SharedWrapped<Wrapper>::type_t>*>(
        wrapper.impl));
}

template <
  typename Wrapper,
  typename Impl = typename SharedWrapped<Wrapper>::impl_t>
static void
destroy(Wrapper wrapper) {
  Impl* impl = static_cast<Impl *>(wrapper.impl);
  delete impl;
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
