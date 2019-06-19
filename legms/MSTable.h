#ifndef LEGMS_MS_TABLE_H_
#define LEGMS_MS_TABLE_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "legms.h"
#include "utility.h"

#include "MSTable_c.h"

#if USE_HDF5
# include <hdf5.h>
#endif // USE_HDF5

namespace legms {

typedef ::legms_ms_tables_t MSTables;

#define LEGMS_FOREACH_MSTABLE(FUNC) \
  FUNC(MAIN)              \
  FUNC(ANTENNA) \
  FUNC(DATA_DESCRIPTION) \
  FUNC(DOPPLER) \
  FUNC(FEED) \
  FUNC(FIELD) \
  FUNC(FLAG_CMD) \
  FUNC(FREQ_OFFSET) \
  FUNC(HISTORY) \
  FUNC(OBSERVATION) \
  FUNC(POINTING) \
  FUNC(POLARIZATION) \
  FUNC(PROCESSOR) \
  FUNC(SOURCE) \
  FUNC(SPECTRAL_WINDOW) \
  FUNC(STATE) \
  FUNC(SYSCAL) \
  FUNC(WEATHER)

template <MSTables T>
struct MSTable {
  static const char* name;
  //typedef ... Axes;
  // static const Axes ROW_AXIS;
  // static const Axes LAST_AXIS;
  // static const std::unordered_map<std::string, std::vector<Axes>>
  // element_axes;
  // static const std::unordered_map<Axes, std::string>& axis_names();
};

// defining axis names with the following helper structure should help prevent
// programming errors when the Axes enumeration for a MSTable changes
template <MSTables T, typename MSTable<T>::Axes Axis>
struct MSTableAxis {
  // static const char* name;
};

#define MS_AXIS_NAME(T, A)                      \
  template <>                                   \
  struct MSTableAxis<MS_##T, T##_##A> {         \
    static const constexpr char* name = #A;     \
  }

#define MS_TABLE_AXES_UID(T)                          \
  template <>                                         \
  struct AxesUID<MSTable<MS_##T>::Axes> {        \
    static const constexpr char *id = "legms::" #T;   \
  }

// N.B.: dimension indexes in comments of the Axes members of the MSTable
// specializations in this file correspond to the order of column axes as
// provided in the MS specification document, which is in column-major order; in
// the LogicalRegions, the order is row-major, so while the "in memory" layout
// remains unchanged, indexing within the casacore array-valued elements is
// reversed (this choice was made to maintain a consistent relationship between
// index order and element layout that extends "upward" to row (or higher level)
// indexing)

template <>
struct MSTable<MS_MAIN> {
  static const constexpr char *name = "MAIN";

  typedef ::legms_ms_main_axes_t Axes;

  static const Axes ROW_AXIS = MAIN_ROW;

  static const Axes LAST_AXIS = MAIN_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(MAIN, ROW);
MS_AXIS_NAME(MAIN, UVW);
MS_AXIS_NAME(MAIN, CORRELATOR);
MS_AXIS_NAME(MAIN, FREQUENCY_CHANNEL);
MS_AXIS_NAME(MAIN, LAG);
MS_AXIS_NAME(MAIN, FLAG_CATEGORY);
MS_AXIS_NAME(MAIN, TIME);
MS_AXIS_NAME(MAIN, TIME_EXTRA_PREC);
MS_AXIS_NAME(MAIN, ANTENNA1);
MS_AXIS_NAME(MAIN, ANTENNA2);
MS_AXIS_NAME(MAIN, ANTENNA3);
MS_AXIS_NAME(MAIN, FEED1);
MS_AXIS_NAME(MAIN, FEED2);
MS_AXIS_NAME(MAIN, FEED3);
MS_AXIS_NAME(MAIN, DATA_DESC_ID);
MS_AXIS_NAME(MAIN, PROCESSOR_ID);
MS_AXIS_NAME(MAIN, PHASE_ID);
MS_AXIS_NAME(MAIN, FIELD_ID);
MS_AXIS_NAME(MAIN, SCAN_NUMBER);
MS_AXIS_NAME(MAIN, ARRAY_ID);
MS_AXIS_NAME(MAIN, OBSERVATION_ID);
MS_AXIS_NAME(MAIN, STATE_ID);

MS_TABLE_AXES_UID(MAIN);

template <>
struct MSTable<MS_ANTENNA> {
  static const constexpr char* name = "ANTENNA";

  typedef ::legms_ms_antenna_axes_t Axes;

  static const Axes ROW_AXIS = ANTENNA_ROW;

  static const Axes LAST_AXIS = ANTENNA_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(ANTENNA, ROW);
MS_AXIS_NAME(ANTENNA, POSITION);
MS_AXIS_NAME(ANTENNA, OFFSET);
MS_AXIS_NAME(ANTENNA, MEAN_ORBIT);

MS_TABLE_AXES_UID(ANTENNA);

template <>
struct MSTable<MS_DATA_DESCRIPTION> {
  static const constexpr char* name = "DATA_DESCRIPTION";

  typedef ::legms_ms_data_description_axes_t Axes;

  static const Axes ROW_AXIS = DATA_DESCRIPTION_ROW;

  static const Axes LAST_AXIS = DATA_DESCRIPTION_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(DATA_DESCRIPTION, ROW);

MS_TABLE_AXES_UID(DATA_DESCRIPTION);

template <>
struct MSTable<MS_DOPPLER> {
  static const constexpr char *name = "DOPPLER";

  typedef ::legms_ms_doppler_axes_t Axes;

  static const Axes ROW_AXIS = DOPPLER_ROW;

  static const Axes LAST_AXIS = DOPPLER_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(DOPPLER, ROW);
MS_AXIS_NAME(DOPPLER, DOPPLER_ID);
MS_AXIS_NAME(DOPPLER, SOURCE_ID);
MS_AXIS_NAME(DOPPLER, TRANSITION_ID);

MS_TABLE_AXES_UID(DOPPLER);

template <>
struct MSTable<MS_FEED> {
  static const constexpr char *name = "FEED";

  typedef ::legms_ms_feed_axes_t Axes;

  static const Axes ROW_AXIS = FEED_ROW;

  static const Axes LAST_AXIS = FEED_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(FEED, ROW);
MS_AXIS_NAME(FEED, RECEPTOR);
MS_AXIS_NAME(FEED, RECEPTOR1);
MS_AXIS_NAME(FEED, DIRECTION);
MS_AXIS_NAME(FEED, POSITION);
MS_AXIS_NAME(FEED, ANTENNA_ID);
MS_AXIS_NAME(FEED, FEED_ID);
MS_AXIS_NAME(FEED, SPECTRAL_WINDOW_ID);
MS_AXIS_NAME(FEED, TIME);
MS_AXIS_NAME(FEED, INTERVAL);

MS_TABLE_AXES_UID(FEED);

template <>
struct MSTable<MS_FIELD> {
  static const constexpr char* name = "FIELD";

  typedef ::legms_ms_field_axes_t Axes;

  static const Axes ROW_AXIS = FIELD_ROW;

  static const Axes LAST_AXIS = FIELD_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(FIELD, ROW);
MS_AXIS_NAME(FIELD, POLYNOMIAL);
MS_AXIS_NAME(FIELD, DIRECTION);
MS_AXIS_NAME(FIELD, SOURCE_ID);
MS_AXIS_NAME(FIELD, EPHEMERIS_ID);

MS_TABLE_AXES_UID(FIELD);

template <>
struct MSTable<MS_FLAG_CMD> {
  static const constexpr char *name = "FLAG_CMD";

  typedef ::legms_ms_flag_cmd_axes_t Axes;

  static const Axes ROW_AXIS = FLAG_CMD_ROW;

  static const Axes LAST_AXIS = FLAG_CMD_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(FLAG_CMD, ROW);
MS_AXIS_NAME(FLAG_CMD, TIME);
MS_AXIS_NAME(FLAG_CMD, INTERVAL);

MS_TABLE_AXES_UID(FLAG_CMD);

template <>
struct MSTable<MS_FREQ_OFFSET> {
  static const constexpr char* name = "FREQ_OFFSET";

  typedef ::legms_ms_freq_offset_axes_t Axes;

  static const Axes ROW_AXIS = FREQ_OFFSET_ROW;

  static const Axes LAST_AXIS = FREQ_OFFSET_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(FREQ_OFFSET, ROW);
MS_AXIS_NAME(FREQ_OFFSET, ANTENNA1);
MS_AXIS_NAME(FREQ_OFFSET, ANTENNA2);
MS_AXIS_NAME(FREQ_OFFSET, FEED_ID);
MS_AXIS_NAME(FREQ_OFFSET, SPECTRAL_WINDOW_ID);
MS_AXIS_NAME(FREQ_OFFSET, TIME);
MS_AXIS_NAME(FREQ_OFFSET, INTERVAL);

MS_TABLE_AXES_UID(FREQ_OFFSET);

template <>
struct MSTable<MS_HISTORY> {
  static const constexpr char* name = "HISTORY";

  typedef ::legms_ms_history_axes_t Axes;

  static const Axes ROW_AXIS = HISTORY_ROW;

  static const Axes LAST_AXIS = HISTORY_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(HISTORY, ROW);
MS_AXIS_NAME(HISTORY, CLI_COMMAND);
MS_AXIS_NAME(HISTORY, APP_PARAM);
MS_AXIS_NAME(HISTORY, TIME);
MS_AXIS_NAME(HISTORY, OBSERVATION_ID);

MS_TABLE_AXES_UID(HISTORY);

template <>
struct MSTable<MS_OBSERVATION> {
  static const constexpr char* name = "OBSERVATION";

  typedef ::legms_ms_observation_axes_t Axes;

  static const Axes ROW_AXIS = OBSERVATION_ROW;

  static const Axes LAST_AXIS = OBSERVATION_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(OBSERVATION, ROW);
MS_AXIS_NAME(OBSERVATION, TIME_RANGE);
MS_AXIS_NAME(OBSERVATION, LOG);
MS_AXIS_NAME(OBSERVATION, SCHEDULE);

MS_TABLE_AXES_UID(OBSERVATION);

template <>
struct MSTable<MS_POINTING> {
  static const constexpr char* name = "POINTING";

  typedef ::legms_ms_pointing_axes_t Axes;

  static const Axes ROW_AXIS = POINTING_ROW;

  static const Axes LAST_AXIS = POINTING_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(POINTING, ROW);
MS_AXIS_NAME(POINTING, POLYNOMIAL);
MS_AXIS_NAME(POINTING, DIRECTION);
MS_AXIS_NAME(POINTING, ANTENNA_ID);
MS_AXIS_NAME(POINTING, TIME);
MS_AXIS_NAME(POINTING, INTERVAL);

MS_TABLE_AXES_UID(POINTING);

template <>
struct MSTable<MS_POLARIZATION> {
  static const constexpr char* name = "POLARIZATION";

  typedef ::legms_ms_polarization_axes_t Axes;

  static const Axes ROW_AXIS = POLARIZATION_ROW;

  static const Axes LAST_AXIS = POLARIZATION_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(POLARIZATION, ROW);
MS_AXIS_NAME(POLARIZATION, CORRELATION);
MS_AXIS_NAME(POLARIZATION, PRODUCT);

MS_TABLE_AXES_UID(POLARIZATION);

template <>
struct MSTable<MS_PROCESSOR> {
  static const constexpr char* name = "PROCESSOR";

  typedef ::legms_ms_processor_axes_t Axes;

  static const Axes ROW_AXIS = PROCESSOR_ROW;

  static const Axes LAST_AXIS = PROCESSOR_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(PROCESSOR, ROW);

MS_TABLE_AXES_UID(PROCESSOR);

template <>
struct MSTable<MS_SOURCE> {
  static const constexpr char* name = "SOURCE";

  typedef ::legms_ms_source_axes_t Axes;

  static const Axes ROW_AXIS = SOURCE_ROW;

  static const Axes LAST_AXIS = SOURCE_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(SOURCE, ROW);
MS_AXIS_NAME(SOURCE, DIRECTION);
MS_AXIS_NAME(SOURCE, POSITION);
MS_AXIS_NAME(SOURCE, PROPER_MOTION);
MS_AXIS_NAME(SOURCE, LINE);
MS_AXIS_NAME(SOURCE, SOURCE_ID);
MS_AXIS_NAME(SOURCE, TIME);
MS_AXIS_NAME(SOURCE, INTERVAL);
MS_AXIS_NAME(SOURCE, SPECTRAL_WINDOW_ID);
MS_AXIS_NAME(SOURCE, PULSAR_ID);

MS_TABLE_AXES_UID(SOURCE);

template <>
struct MSTable<MS_SPECTRAL_WINDOW> {
  static const constexpr char* name = "SPECTRAL_WINDOW";

  typedef ::legms_ms_spectral_window_axes_t Axes;

  static const Axes ROW_AXIS = SPECTRAL_WINDOW_ROW;

  static const Axes LAST_AXIS = SPECTRAL_WINDOW_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(SPECTRAL_WINDOW, ROW);
MS_AXIS_NAME(SPECTRAL_WINDOW, CHANNEL);
MS_AXIS_NAME(SPECTRAL_WINDOW, ASSOC_SPW);

MS_TABLE_AXES_UID(SPECTRAL_WINDOW);

template <>
struct MSTable<MS_STATE> {
  static const constexpr char* name = "STATE";

  typedef ::legms_ms_state_axes_t Axes;

  static const Axes ROW_AXIS = STATE_ROW;

  static const Axes LAST_AXIS = STATE_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(STATE, ROW);

MS_TABLE_AXES_UID(STATE);

template <>
struct MSTable<MS_SYSCAL> {
  static const constexpr char* name = "SYSCAL";

  typedef ::legms_ms_syscal_axes_t Axes;

  static const Axes ROW_AXIS = SYSCAL_ROW;

  static const Axes LAST_AXIS = SYSCAL_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(SYSCAL, ROW);
MS_AXIS_NAME(SYSCAL, RECEPTOR);
MS_AXIS_NAME(SYSCAL, CHANNEL);
MS_AXIS_NAME(SYSCAL, ANTENNA_ID);
MS_AXIS_NAME(SYSCAL, FEED_ID);
MS_AXIS_NAME(SYSCAL, SPECTRAL_WINDOW_ID);
MS_AXIS_NAME(SYSCAL, TIME);
MS_AXIS_NAME(SYSCAL, INTERVAL);

MS_TABLE_AXES_UID(SYSCAL);

template <>
struct MSTable<MS_WEATHER> {
  static const constexpr char* name = "WEATHER";

  typedef ::legms_ms_weather_axes_t Axes;

  static const Axes ROW_AXIS = WEATHER_ROW;

  static const Axes LAST_AXIS = WEATHER_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(WEATHER, ROW);
MS_AXIS_NAME(WEATHER, ANTENNA_ID);
MS_AXIS_NAME(WEATHER, TIME);
MS_AXIS_NAME(WEATHER, INTERVAL);

MS_TABLE_AXES_UID(WEATHER);

#undef MS_AXIS_NAME

#undef MS_TABLE_AXES_UID

#define MSAXES(T)                                                       \
  template <>                                                           \
  struct Axes<typename MSTable<MS_##T>::Axes> {                         \
    static const std::unordered_map<typename MSTable<MS_##T>::Axes, std::string> \
    names;                                                              \
  };

LEGMS_FOREACH_MSTABLE(MSAXES);

#undef MSAXES

#if USE_HDF5

template <MSTables T>
hid_t
h5_axes_datatype() {
  hid_t result = H5Tenum_create(H5T_NATIVE_UCHAR);
  typedef typename MSTable<T>::Axes Axes;
  for (auto a = static_cast<unsigned char>(MSTable<T>::ROW_AXIS);
       a <= static_cast<unsigned char>(MSTable<T>::LAST_AXIS);
       ++a) {
    herr_t err =
      H5Tenum_insert(
        result,
        MSTable<T>::axis_names().at(static_cast<Axes>(a)).c_str(),
        &a);
    assert(err >= 0);
  }
  return result;
}

void
match_h5_axes_datatype(hid_t& id, const char*& uid);

#endif // USE_HDF5

} // end namespace legms

#endif // LEGMS_MS_TABLE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
