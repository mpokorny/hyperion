#ifndef LEGMS_MS_TABLE_C_H_
#define LEGMS_MS_TABLE_C_H_

#include "c_util.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum legms_ms_tables_t {
  MS_MAIN = 0,
  MS_ANTENNA = 1,
  MS_DATA_DESCRIPTION = 2,
  MS_DOPPLER = 3,
  MS_FEED = 4,
  MS_FIELD = 5,
  MS_FLAG_CMD = 6,
  MS_FREQ_OFFSET = 7,
  MS_HISTORY = 8,
  MS_OBSERVATION = 9,
  MS_POINTING = 10,
  MS_POLARIZATION = 11,
  MS_PROCESSOR = 12,
  MS_SOURCE = 13,
  MS_SPECTRAL_WINDOW = 14,
  MS_STATE = 15,
  MS_SYSCAL = 16,
  MS_WEATHER = 17
} legms_mstables_t;

// N.B.: dimension indexes in comments of the Axes members of the MSTable
// specializations in this file correspond to the order of column axes as
// provided in the MS specification document, which is in column-major order; in
// the LogicalRegions, the order is row-major, so while the "in memory" layout
// remains unchanged, indexing within the casacore array-valued elements is
// reversed (this choice was made to maintain a consistent relationship between
// index order and element layout that extends "upward" to row (or higher level)
// indexing)

typedef enum legms_ms_main_axes_t {
  MAIN_ROW = 0,
  // UVW(0)
  MAIN_UVW,
  // DATA(0), FLOAT_DATA(0), VIDEO_POINT(0), LAG_DATA(0), SIGMA(0),
  // SIGMA_SPECTRUM(0), WEIGHT_SPECTRUM(0), FLAG(0), FLAG_CATEGORY(0)
  MAIN_CORRELATOR,
  // DATA(1), FLOAT_DATA(1), SIGMA_SPECTRUM(1), WEIGHT_SPECTRUM(1), FLAG(1),
  // FLAG_CATEGORY(1)
  MAIN_FREQUENCY_CHANNEL,
  // LAG_DATA(1)
  MAIN_LAG,
  // FLAG_CATEGORY(2)
  MAIN_FLAG_CATEGORY,

  // key column axes
  MAIN_TIME,
  MAIN_TIME_EXTRA_PREC,
  MAIN_ANTENNA1,
  MAIN_ANTENNA2,
  MAIN_ANTENNA3,
  MAIN_FEED1,
  MAIN_FEED2,
  MAIN_FEED3,
  MAIN_DATA_DESC_ID,
  MAIN_PROCESSOR_ID,
  MAIN_PHASE_ID,
  MAIN_FIELD_ID,

  // additional index column axes
  MAIN_SCAN_NUMBER,
  MAIN_ARRAY_ID,
  MAIN_OBSERVATION_ID,
  MAIN_STATE_ID,

  MAIN_last = MAIN_STATE_ID
} legms_ms_main_axes_t;

typedef enum legms_ms_antenna_axes_t {
  ANTENNA_ROW = 0,
  // POSITION(0)
  ANTENNA_POSITION,
  // OFFSET(0)
  ANTENNA_OFFSET,
  // MEAN_ORBIT(0)
  ANTENNA_MEAN_ORBIT,

  ANTENNA_last = ANTENNA_MEAN_ORBIT
} legms_ms_antenna_axes_t;

typedef enum legms_ms_data_description_axes_t {
  DATA_DESCRIPTION_ROW = 0,

  DATA_DESCRIPTION_last = DATA_DESCRIPTION_ROW
} legms_ms_data_description_axes_t;

typedef enum legms_ms_doppler_axes_t  {
  DOPPLER_ROW = 0,

  // key column axes
  DOPPLER_DOPPLER_ID,
  DOPPLER_SOURCE_ID,

  // additional index column axes
  DOPPLER_TRANSITION_ID,

  DOPPLER_last = DOPPLER_TRANSITION_ID
} legms_ms_doppler_axes_t;

typedef enum legms_ms_feed_axes_t {
  FEED_ROW = 0,
  // BEAM_OFFSET(1), POLARIZATION_TYPE(0), POL_RESPONSE(0), RECEPTOR_ANGLE(0)
  FEED_RECEPTOR,
  // POL_RESPONSE(1)
  FEED_RECEPTOR1,
  // BEAM_OFFSET(0)
  FEED_DIRECTION,
  // POSITION(0)
  FEED_POSITION,

  // key column axes
  FEED_ANTENNA_ID,
  FEED_FEED_ID,
  FEED_SPECTRAL_WINDOW_ID,
  FEED_TIME,
  FEED_INTERVAL,

  FEED_last = FEED_INTERVAL
} legms_ms_feed_axes_t;

typedef enum legms_ms_field_axes_t {
  FIELD_ROW = 0,
  // DELAY_DIR(1), PHASE_DIR(1), REFERENCE_DIR(1)
  FIELD_POLYNOMIAL,
  // DELAY_DIR(0), PHASE_DIR(0), REFERENCE_DIR(0)
  FIELD_DIRECTION,

  // additional index column axes
  FIELD_SOURCE_ID,
  FIELD_EPHEMERIS_ID,

  FIELD_last = FIELD_EPHEMERIS_ID
} legms_ms_field_axes_t;

typedef enum legms_ms_flag_cmd_axes_t {
  FLAG_CMD_ROW = 0,

  // key column axes
  FLAG_CMD_TIME,
  FLAG_CMD_INTERVAL,

  FLAG_CMD_last = FLAG_CMD_INTERVAL
} legms_ms_flag_cmd_axes_t;

typedef enum legms_ms_freq_offset_axes_t {
  FREQ_OFFSET_ROW = 0,

  // key column axes
  FREQ_OFFSET_ANTENNA1,
  FREQ_OFFSET_ANTENNA2,
  FREQ_OFFSET_FEED_ID,
  FREQ_OFFSET_SPECTRAL_WINDOW_ID,
  FREQ_OFFSET_TIME,
  FREQ_OFFSET_INTERVAL,

  FREQ_OFFSET_last = FREQ_OFFSET_INTERVAL
} legms_ms_freq_offset_axes_t;

typedef enum legms_ms_history_axes_t {
  HISTORY_ROW = 0,
  // CLI_COMMAND(0)
  HISTORY_CLI_COMMAND,
  // APP_PARAMS(0)
  HISTORY_APP_PARAM,

  // key column axes
  HISTORY_TIME,
  HISTORY_OBSERVATION_ID,

  HISTORY_last = HISTORY_OBSERVATION_ID
} legms_ms_history_axes_t;

typedef enum legms_ms_observation_axes_t {
  OBSERVATION_ROW = 0,
  // TIME_RANGE(0)
  OBSERVATION_TIME_RANGE,
  // LOG(0)
  OBSERVATION_LOG,
  // SCHEDULE(0)
  OBSERVATION_SCHEDULE,

  OBSERVATION_last = OBSERVATION_SCHEDULE
} legms_ms_observation_axes_t;

typedef enum legms_ms_pointing_axes_t {
  POINTING_ROW = 0,
  // DIRECTION(1), TARGET(1), POINTING_OFFSET(1), SOURCE_OFFSET(1)
  POINTING_POLYNOMIAL,
  // DIRECTION(0), TARGET(0), POINTING_OFFSET(0), SOURCE_OFFSET(0), ENCODER(0)
  POINTING_DIRECTION,

  // key column axes
  POINTING_ANTENNA_ID,
  POINTING_TIME,
  POINTING_INTERVAL,

  POINTING_last = POINTING_INTERVAL
} legms_ms_pointing_axes_t;

typedef enum legms_ms_polarization_axes_t {
  POLARIZATION_ROW = 0,
  // CORR_TYPE(0), CORR_PRODUCT(1)
  POLARIZATION_CORRELATION,
  // CORR_PRODUCT(0)
  POLARIZATION_PRODUCT,

  POLARIZATION_last = POLARIZATION_PRODUCT
} legms_ms_polarization_axes_t;

typedef enum legms_ms_processor_axes_t {
  PROCESSOR_ROW = 0,

  PROCESSOR_last = PROCESSOR_ROW
} legms_ms_processor_axes_t;

typedef enum legms_ms_source_axes_t {
  SOURCE_ROW = 0,
  // DIRECTION(0)
  SOURCE_DIRECTION,
  // POSITION(0)
  SOURCE_POSITION,
  // PROPER_MOTION(0)
  SOURCE_PROPER_MOTION,
  // TRANSITION(0), REST_FREQUENCY(0), SYSVEL(0)
  SOURCE_LINE,

  // key column axes
  SOURCE_SOURCE_ID,
  SOURCE_TIME,
  SOURCE_INTERVAL,
  SOURCE_SPECTRAL_WINDOW_ID,

  // additional column index axes
  SOURCE_PULSAR_ID,

  SOURCE_last = SOURCE_PULSAR_ID
} legms_ms_source_axes_t;

typedef enum legms_ms_spectral_window_axes_t {
  SPECTRAL_WINDOW_ROW = 0,
  // CHAN_FREQ(0), CHAN_WIDTH(0), EFFECTIVE_BW(0), RESOLUTION(0)
  SPECTRAL_WINDOW_CHANNEL,
  // ASSOC_SPW_ID(0), ASSOC_NATURE(0)
  SPECTRAL_WINDOW_ASSOC_SPW,

  SPECTRAL_WINDOW_last = SPECTRAL_WINDOW_ASSOC_SPW
} legms_ms_spectral_window_axes_t;

typedef enum legms_ms_state_axes_t {
  STATE_ROW = 0,

  STATE_last = STATE_ROW
} legms_ms_state_axes_t;

typedef enum legms_ms_syscal_axes_t {
  SYSCAL_ROW = 0,
  // TCAL(0), TRX(0), TSKY(0), TSYS(0), TANT(0), TANT_TSYS(0), TCAL_SPECTRUM(0),
  // TRX_SPECTRUM(0), TSKY_SPECTRUM(0), TSYS_SPECTRUM(0), TANT_SPECTRUM(0),
  // TANT_TSYS_SPECTRUM(0)
  SYSCAL_RECEPTOR,
  // TCAL_SPECTRUM(1), TRX_SPECTRUM(1), TSKY_SPECTRUM(1), TSYS_SPECTRUM(1),
  // TANT_SPECTRUM(1), TANT_TSYS_SPECTRUM(1)
  SYSCAL_CHANNEL,

  // key column axes
  SYSCAL_ANTENNA_ID,
  SYSCAL_FEED_ID,
  SYSCAL_SPECTRAL_WINDOW_ID,
  SYSCAL_TIME,
  SYSCAL_INTERVAL,

  SYSCAL_last = SYSCAL_INTERVAL
} legms_ms_syscal_axes_t;

typedef enum legms_ms_weather_axes_t {
  WEATHER_ROW = 0,

  // key column axes
  WEATHER_ANTENNA_ID,
  WEATHER_TIME,
  WEATHER_INTERVAL,

  WEATHER_last = WEATHER_INTERVAL
} legms_ms_weather_axes_t;

#define TABLE_FUNCTION_DECLS(t)                 \
  typedef struct legms_ms_##t##_column_axes_t { \
    const char* column;                         \
    const legms_ms_##t##_axes_t* axes;          \
    unsigned num_axes;                          \
  } legms_ms_##t##_column_axes_t;               \
                                                \
  const char*                                   \
  legms_##t##_table_name();                     \
                                                \
  const legms_ms_##t##_column_axes_t*           \
  legms_##t##_table_element_axes();             \
                                                \
  unsigned                                      \
  legms_##t##_table_num_columns();              \
                                                \
  const char* const*                            \
  legms_##t##_table_axis_names();               \
                                                \
  unsigned                                      \
  legms_##t##_table_num_axes();

FOREACH_MS_TABLE_t(TABLE_FUNCTION_DECLS);

#if 0
// In case the Terra compiler doesn't expand the above macro...
typedef struct legms_ms_main_column_axes_t {
  const char* column;
  const legms_ms_main_axes_t* axes;
  unsigned num_axes;
} legms_ms_main_column_axes_t;
const char* legms_main_table_name();
const legms_ms_main_column_axes_t* legms_main_table_element_axes();
unsigned legms_main_table_num_columns();
const char* const* legms_main_table_axis_names();
unsigned legms_main_table_num_axes();

typedef struct legms_ms_antenna_column_axes_t {
  const char* column;
  const legms_ms_antenna_axes_t* axes;
  unsigned num_axes;
} legms_ms_antenna_column_axes_t;
const char* legms_antenna_table_name();
const legms_ms_antenna_column_axes_t* legms_antenna_table_element_axes();
unsigned legms_antenna_table_num_columns();
const char* const* legms_antenna_table_axis_names();
unsigned legms_antenna_table_num_axes();

typedef struct legms_ms_data_description_column_axes_t {
  const char* column;
  const legms_ms_data_description_axes_t* axes;
  unsigned num_axes;
} legms_ms_data_description_column_axes_t;
const char* legms_data_description_table_name();
const legms_ms_data_description_column_axes_t* legms_data_description_table_element_axes();
unsigned legms_data_description_table_num_columns();
const char* const* legms_data_description_table_axis_names();
unsigned legms_data_description_table_num_axes();

typedef struct legms_ms_doppler_column_axes_t {
  const char* column;
  const legms_ms_doppler_axes_t* axes;
  unsigned num_axes;
} legms_ms_doppler_column_axes_t;
const char* legms_doppler_table_name();
const legms_ms_doppler_column_axes_t* legms_doppler_table_element_axes();
unsigned legms_doppler_table_num_columns();
const char* const* legms_doppler_table_axis_names();
unsigned legms_doppler_table_num_axes();

typedef struct legms_ms_feed_column_axes_t {
  const char* column;
  const legms_ms_feed_axes_t* axes;
  unsigned num_axes;
} legms_ms_feed_column_axes_t;
const char* legms_feed_table_name();
const legms_ms_feed_column_axes_t* legms_feed_table_element_axes();
unsigned legms_feed_table_num_columns();
const char* const* legms_feed_table_axis_names();
unsigned legms_feed_table_num_axes();

typedef struct legms_ms_field_column_axes_t {
  const char* column;
  const legms_ms_field_axes_t* axes;
  unsigned num_axes;
} legms_ms_field_column_axes_t;
const char* legms_field_table_name();
const legms_ms_field_column_axes_t* legms_field_table_element_axes();
unsigned legms_field_table_num_columns();
const char* const* legms_field_table_axis_names();
unsigned legms_field_table_num_axes();

typedef struct legms_ms_flag_cmd_column_axes_t {
  const char* column;
  const legms_ms_flag_cmd_axes_t* axes;
  unsigned num_axes;
} legms_ms_flag_cmd_column_axes_t;
const char* legms_flag_cmd_table_name();
const legms_ms_flag_cmd_column_axes_t* legms_flag_cmd_table_element_axes();
unsigned legms_flag_cmd_table_num_columns();
const char* const* legms_flag_cmd_table_axis_names();
unsigned legms_flag_cmd_table_num_axes();

typedef struct legms_ms_freq_offset_column_axes_t {
  const char* column;
  const legms_ms_freq_offset_axes_t* axes;
  unsigned num_axes;
} legms_ms_freq_offset_column_axes_t;
const char* legms_freq_offset_table_name();
const legms_ms_freq_offset_column_axes_t* legms_freq_offset_table_element_axes();
unsigned legms_freq_offset_table_num_columns();
const char* const* legms_freq_offset_table_axis_names();
unsigned legms_freq_offset_table_num_axes();

typedef struct legms_ms_history_column_axes_t {
  const char* column;
  const legms_ms_history_axes_t* axes;
  unsigned num_axes;
} legms_ms_history_column_axes_t;
const char* legms_history_table_name();
const legms_ms_history_column_axes_t* legms_history_table_element_axes();
unsigned legms_history_table_num_columns();
const char* const* legms_history_table_axis_names();
unsigned legms_history_table_num_axes();

typedef struct legms_ms_observation_column_axes_t {
  const char* column;
  const legms_ms_observation_axes_t* axes;
  unsigned num_axes;
} legms_ms_observation_column_axes_t;
const char* legms_observation_table_name();
const legms_ms_observation_column_axes_t* legms_observation_table_element_axes();
unsigned legms_observation_table_num_columns();
const char* const* legms_observation_table_axis_names();
unsigned legms_observation_table_num_axes();

typedef struct legms_ms_pointing_column_axes_t {
  const char* column;
  const legms_ms_pointing_axes_t* axes;
  unsigned num_axes;
} legms_ms_pointing_column_axes_t;
const char* legms_pointing_table_name();
const legms_ms_pointing_column_axes_t* legms_pointing_table_element_axes();
unsigned legms_pointing_table_num_columns();
const char* const* legms_pointing_table_axis_names();
unsigned legms_pointing_table_num_axes();

typedef struct legms_ms_polarization_column_axes_t {
  const char* column;
  const legms_ms_polarization_axes_t* axes;
  unsigned num_axes;
} legms_ms_polarization_column_axes_t;
const char* legms_polarization_table_name();
const legms_ms_polarization_column_axes_t* legms_polarization_table_element_axes();
unsigned legms_polarization_table_num_columns();
const char* const* legms_polarization_table_axis_names();
unsigned legms_polarization_table_num_axes();

typedef struct legms_ms_processor_column_axes_t {
  const char* column;
  const legms_ms_processor_axes_t* axes;
  unsigned num_axes;
} legms_ms_processor_column_axes_t;
const char* legms_processor_table_name();
const legms_ms_processor_column_axes_t* legms_processor_table_element_axes();
unsigned legms_processor_table_num_columns();
const char* const* legms_processor_table_axis_names();
unsigned legms_processor_table_num_axes();

typedef struct legms_ms_source_column_axes_t {
  const char* column;
  const legms_ms_source_axes_t* axes;
  unsigned num_axes;
} legms_ms_source_column_axes_t;
const char* legms_source_table_name();
const legms_ms_source_column_axes_t* legms_source_table_element_axes();
unsigned legms_source_table_num_columns();
const char* const* legms_source_table_axis_names();
unsigned legms_source_table_num_axes();

typedef struct legms_ms_spectral_window_column_axes_t {
  const char* column;
  const legms_ms_spectral_window_axes_t* axes;
  unsigned num_axes;
} legms_ms_spectral_window_column_axes_t;
const char* legms_spectral_window_table_name();
const legms_ms_spectral_window_column_axes_t* legms_spectral_window_table_element_axes();
unsigned legms_spectral_window_table_num_columns();
const char* const* legms_spectral_window_table_axis_names();
unsigned legms_spectral_window_table_num_axes();

typedef struct legms_ms_state_column_axes_t {
  const char* column;
  const legms_ms_state_axes_t* axes;
  unsigned num_axes;
} legms_ms_state_column_axes_t;
const char* legms_state_table_name();
const legms_ms_state_column_axes_t* legms_state_table_element_axes();
unsigned legms_state_table_num_columns();
const char* const* legms_state_table_axis_names();
unsigned legms_state_table_num_axes();

typedef struct legms_ms_syscal_column_axes_t {
  const char* column;
  const legms_ms_syscal_axes_t* axes;
  unsigned num_axes;
} legms_ms_syscal_column_axes_t;
const char* legms_syscal_table_name();
const legms_ms_syscal_column_axes_t* legms_syscal_table_element_axes();
unsigned legms_syscal_table_num_columns();
const char* const* legms_syscal_table_axis_names();
unsigned legms_syscal_table_num_axes();

typedef struct legms_ms_weather_column_axes_t {
  const char* column;
  const legms_ms_weather_axes_t* axes;
  unsigned num_axes;
} legms_ms_weather_column_axes_t;
const char* legms_weather_table_name();
const legms_ms_weather_column_axes_t* legms_weather_table_element_axes();
unsigned legms_weather_table_num_columns();
const char* const* legms_weather_table_axis_names();
unsigned legms_weather_table_num_axes();
#endif


#ifdef __cplusplus
}
#endif

#endif // LEGMS_MS_TABLE_C_H_

// Local Variables:
// mode: c
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
