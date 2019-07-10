#ifndef LEGMS_MS_TABLE_H_
#define LEGMS_MS_TABLE_H_

#include <memory>
#include <string>
#include <vector>

#include "legms.h"
#include "utility.h"

#include "MSTable_c.h"

#ifdef LEGMS_USE_HDF5
# include <hdf5.h>
#endif // LEGMS_USE_HDF5

namespace legms {

typedef ::ms_tables_t MSTables;

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
  // static const std::vector<std::string>& axis_names();
  // static const hid_t h5_axes_datatype;
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

  typedef ::ms_main_axes_t Axes;

  static const unsigned num_axes = MAIN_last + 1;

  static const Axes ROW_AXIS = MAIN_ROW;

  static const Axes LAST_AXIS = MAIN_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::vector<std::string>& axis_names();
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

template <>
struct MSTable<MS_ANTENNA> {
  static const constexpr char* name = "ANTENNA";

  typedef ::ms_antenna_axes_t Axes;

  static const unsigned num_axes = ANTENNA_last + 1;

  static const Axes ROW_AXIS = ANTENNA_ROW;

  static const Axes LAST_AXIS = ANTENNA_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::vector<std::string>& axis_names();
};

MS_AXIS_NAME(ANTENNA, ROW);
MS_AXIS_NAME(ANTENNA, POSITION);
MS_AXIS_NAME(ANTENNA, OFFSET);
MS_AXIS_NAME(ANTENNA, MEAN_ORBIT);

template <>
struct MSTable<MS_DATA_DESCRIPTION> {
  static const constexpr char* name = "DATA_DESCRIPTION";

  typedef ::ms_data_description_axes_t Axes;

  static const unsigned num_axes = DATA_DESCRIPTION_last + 1;

  static const Axes ROW_AXIS = DATA_DESCRIPTION_ROW;

  static const Axes LAST_AXIS = DATA_DESCRIPTION_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::vector<std::string>& axis_names();
};

MS_AXIS_NAME(DATA_DESCRIPTION, ROW);

template <>
struct MSTable<MS_DOPPLER> {
  static const constexpr char *name = "DOPPLER";

  typedef ::ms_doppler_axes_t Axes;

  static const unsigned num_axes = DOPPLER_last + 1;

  static const Axes ROW_AXIS = DOPPLER_ROW;

  static const Axes LAST_AXIS = DOPPLER_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::vector<std::string>& axis_names();
};

MS_AXIS_NAME(DOPPLER, ROW);
MS_AXIS_NAME(DOPPLER, DOPPLER_ID);
MS_AXIS_NAME(DOPPLER, SOURCE_ID);
MS_AXIS_NAME(DOPPLER, TRANSITION_ID);

template <>
struct MSTable<MS_FEED> {
  static const constexpr char *name = "FEED";

  typedef ::ms_feed_axes_t Axes;

  static const unsigned num_axes = FEED_last + 1;

  static const Axes ROW_AXIS = FEED_ROW;

  static const Axes LAST_AXIS = FEED_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::vector<std::string>& axis_names();
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

template <>
struct MSTable<MS_FIELD> {
  static const constexpr char* name = "FIELD";

  typedef ::ms_field_axes_t Axes;

  static const unsigned num_axes = FIELD_last + 1;

  static const Axes ROW_AXIS = FIELD_ROW;

  static const Axes LAST_AXIS = FIELD_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::vector<std::string>& axis_names();
};

MS_AXIS_NAME(FIELD, ROW);
MS_AXIS_NAME(FIELD, POLYNOMIAL);
MS_AXIS_NAME(FIELD, DIRECTION);
MS_AXIS_NAME(FIELD, SOURCE_ID);
MS_AXIS_NAME(FIELD, EPHEMERIS_ID);

template <>
struct MSTable<MS_FLAG_CMD> {
  static const constexpr char *name = "FLAG_CMD";

  typedef ::ms_flag_cmd_axes_t Axes;

  static const unsigned num_axes = FLAG_CMD_last + 1;

  static const Axes ROW_AXIS = FLAG_CMD_ROW;

  static const Axes LAST_AXIS = FLAG_CMD_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::vector<std::string>& axis_names();
};

MS_AXIS_NAME(FLAG_CMD, ROW);
MS_AXIS_NAME(FLAG_CMD, TIME);
MS_AXIS_NAME(FLAG_CMD, INTERVAL);

template <>
struct MSTable<MS_FREQ_OFFSET> {
  static const constexpr char* name = "FREQ_OFFSET";

  typedef ::ms_freq_offset_axes_t Axes;

  static const unsigned num_axes = FREQ_OFFSET_last + 1;

  static const Axes ROW_AXIS = FREQ_OFFSET_ROW;

  static const Axes LAST_AXIS = FREQ_OFFSET_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::vector<std::string>& axis_names();
};

MS_AXIS_NAME(FREQ_OFFSET, ROW);
MS_AXIS_NAME(FREQ_OFFSET, ANTENNA1);
MS_AXIS_NAME(FREQ_OFFSET, ANTENNA2);
MS_AXIS_NAME(FREQ_OFFSET, FEED_ID);
MS_AXIS_NAME(FREQ_OFFSET, SPECTRAL_WINDOW_ID);
MS_AXIS_NAME(FREQ_OFFSET, TIME);
MS_AXIS_NAME(FREQ_OFFSET, INTERVAL);

template <>
struct MSTable<MS_HISTORY> {
  static const constexpr char* name = "HISTORY";

  typedef ::ms_history_axes_t Axes;

  static const unsigned num_axes = HISTORY_last + 1;

  static const Axes ROW_AXIS = HISTORY_ROW;

  static const Axes LAST_AXIS = HISTORY_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::vector<std::string>& axis_names();
};

MS_AXIS_NAME(HISTORY, ROW);
MS_AXIS_NAME(HISTORY, CLI_COMMAND);
MS_AXIS_NAME(HISTORY, APP_PARAM);
MS_AXIS_NAME(HISTORY, TIME);
MS_AXIS_NAME(HISTORY, OBSERVATION_ID);

template <>
struct MSTable<MS_OBSERVATION> {
  static const constexpr char* name = "OBSERVATION";

  typedef ::ms_observation_axes_t Axes;

  static const unsigned num_axes = OBSERVATION_last + 1;

  static const Axes ROW_AXIS = OBSERVATION_ROW;

  static const Axes LAST_AXIS = OBSERVATION_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::vector<std::string>& axis_names();
};

MS_AXIS_NAME(OBSERVATION, ROW);
MS_AXIS_NAME(OBSERVATION, TIME_RANGE);
MS_AXIS_NAME(OBSERVATION, LOG);
MS_AXIS_NAME(OBSERVATION, SCHEDULE);

template <>
struct MSTable<MS_POINTING> {
  static const constexpr char* name = "POINTING";

  typedef ::ms_pointing_axes_t Axes;

  static const unsigned num_axes = POINTING_last + 1;

  static const Axes ROW_AXIS = POINTING_ROW;

  static const Axes LAST_AXIS = POINTING_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::vector<std::string>& axis_names();
};

MS_AXIS_NAME(POINTING, ROW);
MS_AXIS_NAME(POINTING, POLYNOMIAL);
MS_AXIS_NAME(POINTING, DIRECTION);
MS_AXIS_NAME(POINTING, ANTENNA_ID);
MS_AXIS_NAME(POINTING, TIME);
MS_AXIS_NAME(POINTING, INTERVAL);

template <>
struct MSTable<MS_POLARIZATION> {
  static const constexpr char* name = "POLARIZATION";

  typedef ::ms_polarization_axes_t Axes;

  static const unsigned num_axes = POLARIZATION_last + 1;

  static const Axes ROW_AXIS = POLARIZATION_ROW;

  static const Axes LAST_AXIS = POLARIZATION_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::vector<std::string>& axis_names();
};

MS_AXIS_NAME(POLARIZATION, ROW);
MS_AXIS_NAME(POLARIZATION, CORRELATION);
MS_AXIS_NAME(POLARIZATION, PRODUCT);

template <>
struct MSTable<MS_PROCESSOR> {
  static const constexpr char* name = "PROCESSOR";

  typedef ::ms_processor_axes_t Axes;

  static const unsigned num_axes = PROCESSOR_last + 1;

  static const Axes ROW_AXIS = PROCESSOR_ROW;

  static const Axes LAST_AXIS = PROCESSOR_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::vector<std::string>& axis_names();
};

MS_AXIS_NAME(PROCESSOR, ROW);

template <>
struct MSTable<MS_SOURCE> {
  static const constexpr char* name = "SOURCE";

  typedef ::ms_source_axes_t Axes;

  static const unsigned num_axes = SOURCE_last + 1;

  static const Axes ROW_AXIS = SOURCE_ROW;

  static const Axes LAST_AXIS = SOURCE_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::vector<std::string>& axis_names();
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

template <>
struct MSTable<MS_SPECTRAL_WINDOW> {
  static const constexpr char* name = "SPECTRAL_WINDOW";

  typedef ::ms_spectral_window_axes_t Axes;

  static const unsigned num_axes = SPECTRAL_WINDOW_last + 1;

  static const Axes ROW_AXIS = SPECTRAL_WINDOW_ROW;

  static const Axes LAST_AXIS = SPECTRAL_WINDOW_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::vector<std::string>& axis_names();
};

MS_AXIS_NAME(SPECTRAL_WINDOW, ROW);
MS_AXIS_NAME(SPECTRAL_WINDOW, CHANNEL);
MS_AXIS_NAME(SPECTRAL_WINDOW, ASSOC_SPW);

template <>
struct MSTable<MS_STATE> {
  static const constexpr char* name = "STATE";

  typedef ::ms_state_axes_t Axes;

  static const unsigned num_axes = STATE_last + 1;

  static const Axes ROW_AXIS = STATE_ROW;

  static const Axes LAST_AXIS = STATE_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::vector<std::string>& axis_names();
};

MS_AXIS_NAME(STATE, ROW);

template <>
struct MSTable<MS_SYSCAL> {
  static const constexpr char* name = "SYSCAL";

  typedef ::ms_syscal_axes_t Axes;

  static const unsigned num_axes = SYSCAL_last + 1;

  static const Axes ROW_AXIS = SYSCAL_ROW;

  static const Axes LAST_AXIS = SYSCAL_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::vector<std::string>& axis_names();
};

MS_AXIS_NAME(SYSCAL, ROW);
MS_AXIS_NAME(SYSCAL, RECEPTOR);
MS_AXIS_NAME(SYSCAL, CHANNEL);
MS_AXIS_NAME(SYSCAL, ANTENNA_ID);
MS_AXIS_NAME(SYSCAL, FEED_ID);
MS_AXIS_NAME(SYSCAL, SPECTRAL_WINDOW_ID);
MS_AXIS_NAME(SYSCAL, TIME);
MS_AXIS_NAME(SYSCAL, INTERVAL);

template <>
struct MSTable<MS_WEATHER> {
  static const constexpr char* name = "WEATHER";

  typedef ::ms_weather_axes_t Axes;

  static const unsigned num_axes = WEATHER_last + 1;

  static const Axes ROW_AXIS = WEATHER_ROW;

  static const Axes LAST_AXIS = WEATHER_last;

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::vector<std::string>& axis_names();
};

MS_AXIS_NAME(WEATHER, ROW);
MS_AXIS_NAME(WEATHER, ANTENNA_ID);
MS_AXIS_NAME(WEATHER, TIME);
MS_AXIS_NAME(WEATHER, INTERVAL);

#undef MS_AXIS_NAME

#ifdef LEGMS_USE_HDF5
#define MSAXES(T)                                                       \
  template <>                                                           \
  struct Axes<typename MSTable<MS_##T>::Axes> {                         \
    static constexpr const char *uid = "legms::" #T;                    \
    static const std::vector<std::string> names;                        \
    static const unsigned num_axes = MSTable<MS_##T>::num_axes;         \
    static const hid_t h5_datatype;                                     \
  };
#else
#define MSAXES(T)                                                     \
  template <>                                                         \
  struct Axes<typename MSTable<MS_##T>::Axes> {                       \
    static constexpr const char *uid = "legms::" #T;                  \
    static const std::vector<std::string> names;                      \
    static const unsigned num_axes = MSTable<MS_##T>::num_axes;       \
  };
#endif

LEGMS_FOREACH_MSTABLE(MSAXES);

#undef MSAXES

} // end namespace legms

#endif // LEGMS_MS_TABLE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
