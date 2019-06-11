#include "MSTable.h"
#include "c_util.h"

#include <mutex>
#include <type_traits>

using namespace legms;

template <MSTables T, int N>
inline std::enable_if_t<(N == 0)>
add_axis_names(
  std::unordered_map<typename MSTable<T>::Axes,
  std::string>& map) {

  typedef typename MSTable<T>::Axes Axes;
  constexpr Axes ax = static_cast<Axes>(N);
  map[ax] = MSTableAxis<T, ax>::name;
}

template <MSTables T, int N>
inline std::enable_if_t<(N > 0)>
add_axis_names(
  std::unordered_map<typename MSTable<T>::Axes,
  std::string>& map) {

  typedef typename MSTable<T>::Axes Axes;
  constexpr Axes ax = static_cast<Axes>(N);
  map[ax] = MSTableAxis<T, ax>::name;
  add_axis_names<T, N-1>(map);
}

template <MSTables T>
inline void
add_axis_names(
  std::unordered_map<typename MSTable<T>::Axes,
  std::string>& map) {

  typedef typename MSTable<T>::Axes Axes;
  add_axis_names<T, static_cast<int>(Axes::last)>(map);
}

const std::unordered_map<
  std::string,
  std::vector<MSTable<MSTables::MAIN>::Axes>>
MSTable<MSTables::MAIN>::element_axes = {
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
  {"UVW", {Axes::UVW}},
  {"UVW2", {Axes::UVW}},
  {"DATA", {Axes::FREQUENCY_CHANNEL, Axes::CORRELATOR}},
  {"FLOAT_DATA", {Axes::FREQUENCY_CHANNEL, Axes::CORRELATOR}},
  {"VIDEO_POINT", {Axes::FREQUENCY_CHANNEL}},
  {"LAG_DATA", {Axes::LAG, Axes::CORRELATOR}},
  {"SIGMA", {Axes::FREQUENCY_CHANNEL}},
  {"SIGMA_SPECTRUM", {Axes::FREQUENCY_CHANNEL, Axes::CORRELATOR}},
  {"WEIGHT", {Axes::FREQUENCY_CHANNEL}},
  {"WEIGHT_SPECTRUM", {Axes::FREQUENCY_CHANNEL, Axes::CORRELATOR}},
  {"FLAG", {Axes::FREQUENCY_CHANNEL, Axes::CORRELATOR}},
  {"FLAG_CATEGORY", {Axes::FLAG_CATEGORY, Axes::FREQUENCY_CHANNEL,
                     Axes::CORRELATOR}},
  {"FLAG_ROW", {}}
};

#define AXIS_NAMES(T)                                                 \
  const std::unordered_map<MSTable<T>::Axes, std::string>&            \
  MSTable<T>::axis_names() {                                          \
    static std::once_flag initialized;                                \
    static std::unordered_map<MSTable<T>::Axes, std::string> result;  \
    std::call_once(initialized, add_axis_names<T>, result);           \
    return result;                                                    \
  }

const std::unordered_map<
  std::string,
  std::vector<MSTable<MSTables::ANTENNA>::Axes>>
MSTable<MSTables::ANTENNA>::element_axes = {
  {"NAME", {}},
  {"STATION", {}},
  {"TYPE", {}},
  {"MOUNT", {}},
  {"POSITION", {Axes::POSITION}},
  {"OFFSET", {Axes::OFFSET}},
  {"DISH_DIAMETER", {}},
  {"ORBIT_ID", {}},
  {"MEAN_ORBIT", {Axes::MEAN_ORBIT}},
  {"PHASED_ARRAY_ID", {}},
  {"FLAG_ROW", {}}
};

const std::unordered_map<
  std::string,
  std::vector<MSTable<MSTables::DATA_DESCRIPTION>::Axes>>
MSTable<MSTables::DATA_DESCRIPTION>::element_axes = {
  {"SPECTRAL_WINDOW_ID", {}},
  {"POLARIZATION_ID", {}},
  {"LAG_ID", {}},
  {"FLAG_ROW", {}}
};

const std::unordered_map<
  std::string,
  std::vector<MSTable<MSTables::DOPPLER>::Axes>>
MSTable<MSTables::DOPPLER>::element_axes = {
  {"DOPPLER_ID", {}},
  {"SOURCE_ID", {}},
  {"TRANSITION_ID", {}},
  {"VELDEF", {}}
};

const std::unordered_map<
  std::string,
  std::vector<MSTable<MSTables::FEED>::Axes>>
MSTable<MSTables::FEED>::element_axes = {
  {"ANTENNA_ID", {}},
  {"FEED_ID", {}},
  {"SPECTRAL_WINDOW_ID", {}},
  {"TIME", {}},
  {"INTERVAL", {}},
  {"NUM_RECEPTORS", {}},
  {"BEAM_ID", {}},
  {"BEAM_OFFSET", {Axes::RECEPTOR, Axes::DIRECTION}},
  {"FOCUS_LENGTH", {}},
  {"PHASED_FEED_ID", {}},
  {"POLARIZATION_TYPE", {Axes::RECEPTOR}},
  {"POL_RESPONSE", {Axes::RECEPTOR1, Axes::RECEPTOR}},
  {"POSITION", {Axes::POSITION}},
  {"RECEPTOR_ANGLE", {Axes::RECEPTOR}}
};

const std::unordered_map<
  std::string,
  std::vector<MSTable<MSTables::FIELD>::Axes>>
MSTable<MSTables::FIELD>::element_axes = {
  {"NAME", {}},
  {"CODE", {}},
  {"TIME", {}},
  {"NUM_POLY", {}},
  {"DELAY_DIR", {Axes::POLYNOMIAL, Axes::DIRECTION}},
  {"PHASE_DIR", {Axes::POLYNOMIAL, Axes::DIRECTION}},
  {"REFERENCE_DIR", {Axes::POLYNOMIAL, Axes::DIRECTION}},
  {"SOURCE_ID", {}},
  {"EPHEMERIS_ID", {}},
  {"FLAG_ROW", {}}
};

const std::unordered_map<
  std::string,
  std::vector<MSTable<MSTables::FLAG_CMD>::Axes>>
MSTable<MSTables::FLAG_CMD>::element_axes = {
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
  std::vector<MSTable<MSTables::FREQ_OFFSET>::Axes>>
MSTable<MSTables::FREQ_OFFSET>::element_axes = {
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
  std::vector<MSTable<MSTables::HISTORY>::Axes>>
MSTable<MSTables::HISTORY>::element_axes = {
  {"TIME", {}},
  {"OBSERVATION_ID", {}},
  {"MESSAGE", {}},
  {"PRIORITY", {}},
  {"ORIGIN", {}},
  {"OBJECT_ID", {}},
  {"APPLICATION", {}},
  {"CLI_COMMAND", {Axes::CLI_COMMAND}},
  {"APP_PARAMS", {Axes::APP_PARAM}}
};

const std::unordered_map<
  std::string,
  std::vector<MSTable<MSTables::OBSERVATION>::Axes>>
MSTable<MSTables::OBSERVATION>::element_axes = {
  {"TELESCOPE_NAME", {}},
  {"TIME_RANGE", {Axes::TIME_RANGE}},
  {"OBSERVER", {}},
  {"LOG", {Axes::LOG}},
  {"SCHEDULE_TYPE", {}},
  {"SCHEDULE", {Axes::SCHEDULE}},
  {"PROJECT", {}},
  {"RELEASE_DATA", {}},
  {"FLAG_ROW", {}}
};

const std::unordered_map<
  std::string,
  std::vector<MSTable<MSTables::POINTING>::Axes>>
MSTable<MSTables::POINTING>::element_axes = {
  {"ANTENNA_ID", {}},
  {"TIME", {}},
  {"INTERVAL", {}},
  {"NAME", {}},
  {"NUM_POLY", {}},
  {"TIME_ORIGIN", {}},
  {"DIRECTION", {Axes::POLYNOMIAL, Axes::DIRECTION}},
  {"TARGET", {Axes::POLYNOMIAL, Axes::DIRECTION}},
  {"POINTING_OFFSET", {Axes::POLYNOMIAL, Axes::DIRECTION}},
  {"SOURCE_OFFSET", {Axes::POLYNOMIAL, Axes::DIRECTION}},
  {"ENCODER", {Axes::DIRECTION}},
  {"POINTING_MODEL_ID", {}},
  {"TRACKING", {}},
  {"ON_SOURCE", {}},
  {"OVER_THE_TOP", {}}
};

const std::unordered_map<
  std::string,
  std::vector<MSTable<MSTables::POLARIZATION>::Axes>>
MSTable<MSTables::POLARIZATION>::element_axes = {
  {"NUM_CORR", {}},
  {"CORR_TYPE", {Axes::CORRELATION}},
  {"CORR_PRODUCT", {Axes::CORRELATION, Axes::PRODUCT}},
  {"FLAG_ROW", {}}
};

const std::unordered_map<
  std::string,
  std::vector<MSTable<MSTables::PROCESSOR>::Axes>>
MSTable<MSTables::PROCESSOR>::element_axes = {
  {"TYPE", {}},
  {"SUB_TYPE", {}},
  {"TYPE_ID", {}},
  {"MODE_ID", {}},
  {"PASS_ID", {}},
  {"FLAG_ROW", {}}
};

const std::unordered_map<
  std::string,
  std::vector<MSTable<MSTables::SOURCE>::Axes>>
MSTable<MSTables::SOURCE>::element_axes = {
  {"SOURCE_ID", {}},
  {"TIME", {}},
  {"INTERVAL", {}},
  {"SPECTRAL_WINDOW_ID", {}},
  {"NUM_LINES", {}},
  {"NAME", {}},
  {"CALIBRATION_GROUP", {}},
  {"CODE", {}},
  {"DIRECTION", {Axes::DIRECTION}},
  {"POSITION", {Axes::POSITION}},
  {"PROPER_MOTION", {Axes::PROPER_MOTION}},
  {"TRANSITION", {Axes::LINE}},
  {"REST_FREQUENCY", {Axes::LINE}},
  {"SYSVEL", {Axes::LINE}},
  {"SOURCE_MODEL", {}},
  {"PULSAR_ID", {}}
};

const std::unordered_map<
  std::string,
  std::vector<MSTable<MSTables::SPECTRAL_WINDOW>::Axes>>
MSTable<MSTables::SPECTRAL_WINDOW>::element_axes = {
  {"NUM_CHAN", {}},
  {"NAME", {}},
  {"REF_FREQUENCY", {}},
  {"CHAN_FREQ", {Axes::CHANNEL}},
  {"CHAN_WIDTH", {Axes::CHANNEL}},
  {"MEAS_FREQ_REF", {}},
  {"EFFECTIVE_BW", {Axes::CHANNEL}},
  {"RESOLUTION", {Axes::CHANNEL}},
  {"TOTAL_BANDWIDTH", {}},
  {"NET_SIDEBAND", {}},
  {"BBC_NO", {}},
  {"BBC_SIDEBAND", {}},
  {"IF_CONV_CHAN", {}},
  {"RECEIVER_ID", {}},
  {"FREQ_GROUP", {}},
  {"FREQ_GROUP_NAME", {}},
  {"DOPPLER_ID", {}},
  {"ASSOC_SPW_ID", {Axes::ASSOC_SPW}},
  {"ASSOC_NATURE", {Axes::ASSOC_SPW}},
  {"FLAG_ROW", {}}
};

const std::unordered_map<
  std::string,
  std::vector<MSTable<MSTables::STATE>::Axes>>
MSTable<MSTables::STATE>::element_axes = {
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
  std::vector<MSTable<MSTables::SYSCAL>::Axes>>
MSTable<MSTables::SYSCAL>::element_axes = {
  {"ANTENNA_ID", {}},
  {"FEED_ID", {}},
  {"SPECTRAL_WINDOW_ID", {}},
  {"TIME", {}},
  {"INTERVAL", {}},
  {"PHASE_DIFF", {}},
  {"TCAL", {Axes::RECEPTOR}},
  {"TRX", {Axes::RECEPTOR}},
  {"TSKY", {Axes::RECEPTOR}},
  {"TSYS", {Axes::RECEPTOR}},
  {"TANT", {Axes::RECEPTOR}},
  {"TANT_TSYS", {Axes::RECEPTOR}},
  {"TCAL_SPECTRUM", {Axes::CHANNEL, Axes::RECEPTOR}},
  {"TRX_SPECTRUM", {Axes::CHANNEL, Axes::RECEPTOR}},
  {"TSKY_SPECTRUM", {Axes::CHANNEL, Axes::RECEPTOR}},
  {"TSYS_SPECTRUM", {Axes::CHANNEL, Axes::RECEPTOR}},
  {"TANT_SPECTRUM", {Axes::CHANNEL, Axes::RECEPTOR}},
  {"TANT_TSYS_SPECTRUM", {Axes::CHANNEL, Axes::RECEPTOR}},
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
  std::vector<MSTable<MSTables::WEATHER>::Axes>>
MSTable<MSTables::WEATHER>::element_axes = {
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

LEGMS_FOREACH_MSTABLE(AXIS_NAMES);

#ifdef USE_CASACORE

std::unique_ptr<Table>
Table::from_ms(
  Legion::Context ctx,
  Legion::Runtime* runtime,
  const std::experimental::filesystem::path& path,
  const std::unordered_set<std::string>& column_selections) {

  std::string table_name = path.filename();

#define FROM_MS_TABLE(N)                                                \
  do {                                                                  \
    if (table_name == MSTable<N>::name)                                 \
      return                                                            \
        legms:: template from_ms<N>(ctx, runtime, path, column_selections); \
  } while (0);

  LEGMS_FOREACH_MSTABLE(FROM_MS_TABLE);

  // try to read as main table
  return
    legms:: template from_ms<MSTables::MAIN>(
      ctx,
      runtime,
      path,
      column_selections);

#undef FROM_MS_TABLE
}

#endif // USE_CASACORE

#if USE_HDF5

#define H5_AXES_DATATYPE(T)                                     \
  template <>                                                   \
  hid_t TableT<typename MSTable<T>::Axes>::m_h5_axes_datatype = \
    legms::h5_axes_datatype<T>();

LEGMS_FOREACH_MSTABLE(H5_AXES_DATATYPE)

#undef H5_AXES_DATATYPE

void
legms::match_h5_axes_datatype(hid_t& id, const char*& uid) {
  if (id < 0)
    return;

#define MATCH_DT(T)                                                 \
  if (H5Tequal(id, TableT<typename MSTable<T>::Axes>::h5_axes())) { \
    id = TableT<typename MSTable<T>::Axes>::h5_axes();              \
    uid = AxesUID<typename MSTable<T>::Axes>::id;                   \
    return;                                                         \
  }

  LEGMS_FOREACH_MSTABLE(MATCH_DT);
#undef MATCH_DT

  id = -1;
}

#endif // USE_HDF5

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
