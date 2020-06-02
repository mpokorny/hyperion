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
#include <hyperion/MSTable.h>
#include <hyperion/c_util.h>

#include <mutex>
#include <type_traits>

using namespace hyperion;

#if __cplusplus < 201703L
#define MST_NAME(T) const constexpr char* MSTable<MS_##T>::name;
HYPERION_FOREACH_MS_TABLE(MST_NAME);
#undef MSTNAME
#endif

template <MSTables T, int N>
inline static std::enable_if_t<(N == 0)>
add_axis_names(std::vector<std::string>& v) {

  typedef typename MSTable<T>::Axes Axes;
  constexpr Axes ax = static_cast<Axes>(N);
  v[N] = MSTableAxis<T, ax>::name;
}

template <MSTables T, int N>
inline static std::enable_if_t<(N > 0)>
add_axis_names(std::vector<std::string>& v) {

  typedef typename MSTable<T>::Axes Axes;
  constexpr Axes ax = static_cast<Axes>(N);
  v[N] = MSTableAxis<T, ax>::name;
  add_axis_names<T, N-1>(v);
}

const std::unordered_map<
  std::string,
  std::vector<MSTable<MS_MAIN>::Axes>>
hyperion::MSTable<MS_MAIN>::element_axes = {
  {"TIME", {}},
  {"TIME_EXTRA_PREC", {}},
  {"ANTENNA1", {}},
  {"ANTENNA2", {}},
  {"ANTENNA3", {}},
  {"FEED1", {}},
  {"FEED2", {}},
  {"FEED3", {}},
  {"DATA_DESC_ID", {}},
  {"PROCESSOR_ID", {}},
  {"PHASE_ID", {}},
  {"FIELD_ID", {}},
  {"INTERVAL", {}},
  {"EXPOSURE", {}},
  {"TIME_CENTROID", {}},
  {"PULSAR_BIN", {}},
  {"PULSAR_GATE_ID", {}},
  {"SCAN_NUMBER", {}},
  {"ARRAY_ID", {}},
  {"OBSERVATION_ID", {}},
  {"STATE_ID", {}},
  {"BASELINE_REF", {}},
  {"UVW", {MAIN_UVW}},
  {"UVW2", {MAIN_UVW}},
  {"DATA", {MAIN_FREQUENCY_CHANNEL, MAIN_CORRELATOR}},
  {"FLOAT_DATA", {MAIN_FREQUENCY_CHANNEL, MAIN_CORRELATOR}},
  {"VIDEO_POINT", {MAIN_FREQUENCY_CHANNEL}},
  {"LAG_DATA", {MAIN_LAG, MAIN_CORRELATOR}},
  {"SIGMA", {MAIN_FREQUENCY_CHANNEL}},
  {"SIGMA_SPECTRUM", {MAIN_FREQUENCY_CHANNEL, MAIN_CORRELATOR}},
  {"WEIGHT", {MAIN_FREQUENCY_CHANNEL}},
  {"WEIGHT_SPECTRUM", {MAIN_FREQUENCY_CHANNEL, MAIN_CORRELATOR}},
  {"FLAG", {MAIN_FREQUENCY_CHANNEL, MAIN_CORRELATOR}},
  {"FLAG_CATEGORY", {MAIN_FLAG_CATEGORY, MAIN_FREQUENCY_CHANNEL,
                     MAIN_CORRELATOR}},
  {"FLAG_ROW", {}}
};

#define AXIS_NAMES(T)                                                   \
  const std::vector<std::string>&                                       \
  hyperion::MSTable<MS_##T>::axis_names() {                                \
    static std::once_flag initialized;                                  \
    static std::vector<std::string> result(T##_last + 1);               \
    std::call_once(initialized, add_axis_names<MS_##T, T##_last>, result); \
    return result;                                                      \
  }

const std::unordered_map<
  std::string,
  std::vector<MSTable<MS_ANTENNA>::Axes>>
hyperion::MSTable<MS_ANTENNA>::element_axes = {
  {"NAME", {}},
  {"STATION", {}},
  {"TYPE", {}},
  {"MOUNT", {}},
  {"POSITION", {ANTENNA_POSITION}},
  {"OFFSET", {ANTENNA_OFFSET}},
  {"DISH_DIAMETER", {}},
  {"ORBIT_ID", {}},
  {"MEAN_ORBIT", {ANTENNA_MEAN_ORBIT}},
  {"PHASED_ARRAY_ID", {}},
  {"FLAG_ROW", {}}
};

const std::unordered_map<
  std::string,
  std::vector<MSTable<MS_DATA_DESCRIPTION>::Axes>>
hyperion::MSTable<MS_DATA_DESCRIPTION>::element_axes = {
  {"SPECTRAL_WINDOW_ID", {}},
  {"POLARIZATION_ID", {}},
  {"LAG_ID", {}},
  {"FLAG_ROW", {}}
};

const std::unordered_map<
  std::string,
  std::vector<MSTable<MS_DOPPLER>::Axes>>
hyperion::MSTable<MS_DOPPLER>::element_axes = {
  {"DOPPLER_ID", {}},
  {"SOURCE_ID", {}},
  {"TRANSITION_ID", {}},
  {"VELDEF", {}}
};

const std::unordered_map<
  std::string,
  std::vector<MSTable<MS_FEED>::Axes>>
hyperion::MSTable<MS_FEED>::element_axes = {
  {"ANTENNA_ID", {}},
  {"FEED_ID", {}},
  {"SPECTRAL_WINDOW_ID", {}},
  {"TIME", {}},
  {"INTERVAL", {}},
  {"NUM_RECEPTORS", {}},
  {"BEAM_ID", {}},
  {"BEAM_OFFSET", {FEED_RECEPTOR, FEED_DIRECTION}},
  {"FOCUS_LENGTH", {}},
  {"PHASED_FEED_ID", {}},
  {"POLARIZATION_TYPE", {FEED_RECEPTOR}},
  {"POL_RESPONSE", {FEED_RECEPTOR1, FEED_RECEPTOR}},
  {"POSITION", {FEED_POSITION}},
  {"RECEPTOR_ANGLE", {FEED_RECEPTOR}}
};

const std::unordered_map<
  std::string,
  std::vector<MSTable<MS_FIELD>::Axes>>
hyperion::MSTable<MS_FIELD>::element_axes = {
  {"NAME", {}},
  {"CODE", {}},
  {"TIME", {}},
  {"NUM_POLY", {}},
  {"DELAY_DIR", {FIELD_POLYNOMIAL, FIELD_DIRECTION}},
  {"PHASE_DIR", {FIELD_POLYNOMIAL, FIELD_DIRECTION}},
  {"REFERENCE_DIR", {FIELD_POLYNOMIAL, FIELD_DIRECTION}},
  {"SOURCE_ID", {}},
  {"EPHEMERIS_ID", {}},
  {"FLAG_ROW", {}}
};

const std::unordered_map<
  std::string,
  std::vector<MSTable<MS_FLAG_CMD>::Axes>>
hyperion::MSTable<MS_FLAG_CMD>::element_axes = {
  {"TIME", {}},
  {"INTERVAL", {}},
  {"TYPE", {}},
  {"REASON", {}},
  {"LEVEL", {}},
  {"SEVERITY", {}},
  {"APPLIED", {}},
  {"COMMAND", {}}
};

const std::unordered_map<
  std::string,
  std::vector<MSTable<MS_FREQ_OFFSET>::Axes>>
hyperion::MSTable<MS_FREQ_OFFSET>::element_axes = {
  {"ANTENNA1", {}},
  {"ANTENNA2", {}},
  {"FEED_ID", {}},
  {"SPECTRAL_WINDOW_ID", {}},
  {"TIME", {}},
  {"INTERVAL", {}},
  {"OFFSET", {}}
};

const std::unordered_map<
  std::string,
  std::vector<MSTable<MS_HISTORY>::Axes>>
hyperion::MSTable<MS_HISTORY>::element_axes = {
  {"TIME", {}},
  {"OBSERVATION_ID", {}},
  {"MESSAGE", {}},
  {"PRIORITY", {}},
  {"ORIGIN", {}},
  {"OBJECT_ID", {}},
  {"APPLICATION", {}},
  {"CLI_COMMAND", {HISTORY_CLI_COMMAND}},
  {"APP_PARAMS", {HISTORY_APP_PARAM}}
};

const std::unordered_map<
  std::string,
  std::vector<MSTable<MS_OBSERVATION>::Axes>>
hyperion::MSTable<MS_OBSERVATION>::element_axes = {
  {"TELESCOPE_NAME", {}},
  {"TIME_RANGE", {OBSERVATION_TIME_RANGE}},
  {"OBSERVER", {}},
  {"LOG", {OBSERVATION_LOG}},
  {"SCHEDULE_TYPE", {}},
  {"SCHEDULE", {OBSERVATION_SCHEDULE}},
  {"PROJECT", {}},
  {"RELEASE_DATE", {}},
  {"FLAG_ROW", {}}
};

const std::unordered_map<
  std::string,
  std::vector<MSTable<MS_POINTING>::Axes>>
hyperion::MSTable<MS_POINTING>::element_axes = {
  {"ANTENNA_ID", {}},
  {"TIME", {}},
  {"INTERVAL", {}},
  {"NAME", {}},
  {"NUM_POLY", {}},
  {"TIME_ORIGIN", {}},
  {"DIRECTION", {POINTING_POLYNOMIAL, POINTING_DIRECTION}},
  {"TARGET", {POINTING_POLYNOMIAL, POINTING_DIRECTION}},
  {"POINTING_OFFSET", {POINTING_POLYNOMIAL, POINTING_DIRECTION}},
  {"SOURCE_OFFSET", {POINTING_POLYNOMIAL, POINTING_DIRECTION}},
  {"ENCODER", {POINTING_DIRECTION}},
  {"POINTING_MODEL_ID", {}},
  {"TRACKING", {}},
  {"ON_SOURCE", {}},
  {"OVER_THE_TOP", {}}
};

const std::unordered_map<
  std::string,
  std::vector<MSTable<MS_POLARIZATION>::Axes>>
hyperion::MSTable<MS_POLARIZATION>::element_axes = {
  {"NUM_CORR", {}},
  {"CORR_TYPE", {POLARIZATION_CORRELATION}},
  {"CORR_PRODUCT", {POLARIZATION_CORRELATION, POLARIZATION_PRODUCT}},
  {"FLAG_ROW", {}}
};

const std::unordered_map<
  std::string,
  std::vector<MSTable<MS_PROCESSOR>::Axes>>
hyperion::MSTable<MS_PROCESSOR>::element_axes = {
  {"TYPE", {}},
  {"SUB_TYPE", {}},
  {"TYPE_ID", {}},
  {"MODE_ID", {}},
  {"PASS_ID", {}},
  {"FLAG_ROW", {}}
};

const std::unordered_map<
  std::string,
  std::vector<MSTable<MS_SOURCE>::Axes>>
hyperion::MSTable<MS_SOURCE>::element_axes = {
  {"SOURCE_ID", {}},
  {"TIME", {}},
  {"INTERVAL", {}},
  {"SPECTRAL_WINDOW_ID", {}},
  {"NUM_LINES", {}},
  {"NAME", {}},
  {"CALIBRATION_GROUP", {}},
  {"CODE", {}},
  {"DIRECTION", {SOURCE_DIRECTION}},
  {"POSITION", {SOURCE_POSITION}},
  {"PROPER_MOTION", {SOURCE_PROPER_MOTION}},
  {"TRANSITION", {SOURCE_LINE}},
  {"REST_FREQUENCY", {SOURCE_LINE}},
  {"SYSVEL", {SOURCE_LINE}},
  {"SOURCE_MODEL", {}},
  {"PULSAR_ID", {}}
};

const std::unordered_map<
  std::string,
  std::vector<MSTable<MS_SPECTRAL_WINDOW>::Axes>>
hyperion::MSTable<MS_SPECTRAL_WINDOW>::element_axes = {
  {"NUM_CHAN", {}},
  {"NAME", {}},
  {"REF_FREQUENCY", {}},
  {"CHAN_FREQ", {SPECTRAL_WINDOW_CHANNEL}},
  {"CHAN_WIDTH", {SPECTRAL_WINDOW_CHANNEL}},
  {"EFFECTIVE_BW", {SPECTRAL_WINDOW_CHANNEL}},
  {"RESOLUTION", {SPECTRAL_WINDOW_CHANNEL}},
  {"TOTAL_BANDWIDTH", {}},
  {"NET_SIDEBAND", {}},
  {"BBC_NO", {}},
  {"BBC_SIDEBAND", {}},
  {"IF_CONV_CHAIN", {}},
  {"RECEIVER_ID", {}},
  {"FREQ_GROUP", {}},
  {"FREQ_GROUP_NAME", {}},
  {"DOPPLER_ID", {}},
  {"ASSOC_SPW_ID", {SPECTRAL_WINDOW_ASSOC_SPW}},
  {"ASSOC_NATURE", {SPECTRAL_WINDOW_ASSOC_SPW}},
  {"FLAG_ROW", {}}
};

const std::unordered_map<
  std::string,
  std::vector<MSTable<MS_STATE>::Axes>>
hyperion::MSTable<MS_STATE>::element_axes = {
  {"SIG", {}},
  {"REF", {}},
  {"CAL", {}},
  {"LOAD", {}},
  {"SUB_SCAN", {}},
  {"OBS_MODE", {}},
  {"FLAG_ROW", {}}
};

const std::unordered_map<
  std::string,
  std::vector<MSTable<MS_SYSCAL>::Axes>>
hyperion::MSTable<MS_SYSCAL>::element_axes = {
  {"ANTENNA_ID", {}},
  {"FEED_ID", {}},
  {"SPECTRAL_WINDOW_ID", {}},
  {"TIME", {}},
  {"INTERVAL", {}},
  {"PHASE_DIFF", {}},
  {"TCAL", {SYSCAL_RECEPTOR}},
  {"TRX", {SYSCAL_RECEPTOR}},
  {"TSKY", {SYSCAL_RECEPTOR}},
  {"TSYS", {SYSCAL_RECEPTOR}},
  {"TANT", {SYSCAL_RECEPTOR}},
  {"TANT_TSYS", {SYSCAL_RECEPTOR}},
  {"TCAL_SPECTRUM", {SYSCAL_CHANNEL, SYSCAL_RECEPTOR}},
  {"TRX_SPECTRUM", {SYSCAL_CHANNEL, SYSCAL_RECEPTOR}},
  {"TSKY_SPECTRUM", {SYSCAL_CHANNEL, SYSCAL_RECEPTOR}},
  {"TSYS_SPECTRUM", {SYSCAL_CHANNEL, SYSCAL_RECEPTOR}},
  {"TANT_SPECTRUM", {SYSCAL_CHANNEL, SYSCAL_RECEPTOR}},
  {"TANT_TSYS_SPECTRUM", {SYSCAL_CHANNEL, SYSCAL_RECEPTOR}},
  {"PHASE_DIFF_FLAG", {}},
  {"TCAL_FLAG", {}},
  {"TRX_FLAG", {}},
  {"TSKY_FLAG", {}},
  {"TSYS_FLAG", {}},
  {"TANT_FLAG", {}},
  {"TANT_TSYS_FLAG", {}}
};

const std::unordered_map<
  std::string,
  std::vector<MSTable<MS_WEATHER>::Axes>>
hyperion::MSTable<MS_WEATHER>::element_axes = {
  {"ANTENNA_ID", {}},
  {"TIME", {}},
  {"INTERVAL", {}},
  {"H2O", {}},
  {"IONOS_ELECTRON", {}},
  {"PRESSURE", {}},
  {"REL_HUMIDITY", {}},
  {"TEMPERATURE", {}},
  {"DEW_POINT", {}},
  {"WIND_DIRECTION", {}},
  {"WIND_SPEED", {}},
  {"H2O_FLAG", {}},
  {"IONOS_ELECTRON_FLAG", {}},
  {"PRESSURE_FLAG", {}},
  {"REL_HUMIDITY_FLAG", {}},
  {"TEMPERATURE_FLAG", {}},
  {"DEW_POINT_FLAG", {}},
  {"WIND_DIRECTION_FLAG", {}},
  {"WIND_SPEED_FLAG", {}}
};

HYPERION_FOREACH_MS_TABLE(AXIS_NAMES);

#ifdef HYPERION_USE_HDF5
template <typename T>
static hid_t
h5_axes_dt()  {
  hid_t result = H5Tenum_create(H5T_NATIVE_UCHAR);
  for (unsigned char a = 0;
       a <= static_cast<unsigned char>(Axes<T>::num_axes - 1);
       ++a) {
    [[maybe_unused]] herr_t err = H5Tenum_insert(result, Axes<T>::names[a].c_str(), &a);
    assert(err >= 0);
  }
  return result;
}

# define MSAXES(T)                                                      \
  const std::vector<std::string>                                        \
  hyperion::Axes<typename MSTable<MS_##T>::Axes>::names = MSTable<MS_##T>::axis_names(); \
  const hid_t                                                           \
  hyperion::Axes<typename MSTable<MS_##T>::Axes>::h5_datatype =                   \
    h5_axes_dt<typename MSTable<MS_##T>::Axes>();
#else
# define MSAXES(T)                                                      \
  const std::vector<std::string>                                        \
  hyperion::Axes<typename MSTable<MS_##T>::Axes>::names = MSTable<MS_##T>::axis_names();
#endif

HYPERION_FOREACH_MS_TABLE(MSAXES);

#undef MSAXES

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
