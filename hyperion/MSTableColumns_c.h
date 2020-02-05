/*
 * Copyright 2019 Associated Universities, Inc. Washington DC, USA.
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
#ifndef HYPERION_MS_TABLE_COLUMNS_C_H_
#define HYPERION_MS_TABLE_COLUMNS_C_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ms_main_col_t {
  MS_MAIN_COL_TIME,
  MS_MAIN_COL_TIME_EXTRA_PREC,
  MS_MAIN_COL_ANTENNA1,
  MS_MAIN_COL_ANTENNA2,
  MS_MAIN_COL_ANTENNA3,
  MS_MAIN_COL_FEED1,
  MS_MAIN_COL_FEED2,
  MS_MAIN_COL_FEED3,
  MS_MAIN_COL_DATA_DESC_ID,
  MS_MAIN_COL_PROCESSOR_ID,
  MS_MAIN_COL_PHASE_ID,
  MS_MAIN_COL_FIELD_ID,
  MS_MAIN_COL_INTERVAL,
  MS_MAIN_COL_EXPOSURE,
  MS_MAIN_COL_TIME_CENTROID,
  MS_MAIN_COL_PULSAR_BIN,
  MS_MAIN_COL_PULSAR_GATE_ID,
  MS_MAIN_COL_SCAN_NUMBER,
  MS_MAIN_COL_ARRAY_ID,
  MS_MAIN_COL_OBSERVATION_ID,
  MS_MAIN_COL_STATE_ID,
  MS_MAIN_COL_BASELINE_REF,
  MS_MAIN_COL_UVW,
  MS_MAIN_COL_UVW2,
  MS_MAIN_COL_DATA,
  MS_MAIN_COL_FLOAT_DATA,
  MS_MAIN_COL_VIDEO_POINT,
  MS_MAIN_COL_LAG_DATA,
  MS_MAIN_COL_SIGMA,
  MS_MAIN_COL_SIGMA_SPECTRUM,
  MS_MAIN_COL_WEIGHT,
  MS_MAIN_COL_WEIGHT_SPECTRUM,
  MS_MAIN_COL_FLAG,
  MS_MAIN_COL_FLAG_CATEGORY,
  MS_MAIN_COL_FLAG_ROW,
  MS_MAIN_NUM_COLS
} ms_main_col_t;

#define MS_MAIN_COL_FID_BASE 100

#define MS_MAIN_COLUMN_NAMES {                  \
    "TIME",                                     \
      "TIME_EXTRA_PREC",                        \
      "ANTENNA1",                               \
      "ANTENNA2",                               \
      "ANTENNA3",                               \
      "FEED1",                                  \
      "FEED2",                                  \
      "FEED3",                                  \
      "DATA_DESC_ID",                           \
      "PROCESSOR_ID",                           \
      "PHASE_ID",                               \
      "FIELD_ID",                               \
      "INTERVAL",                               \
      "EXPOSURE",                               \
      "TIME_CENTROID",                          \
      "PULSAR_BIN",                             \
      "PULSAR_GATE_ID",                         \
      "SCAN_NUMBER",                            \
      "ARRAY_ID",                               \
      "OBSERVATION_ID",                         \
      "STATE_ID",                               \
      "BASELINE_REF",                           \
      "UVW",                                    \
      "UVW2",                                   \
      "DATA",                                   \
      "FLOAT_DATA",                             \
      "VIDEO_POINT",                            \
      "LAG_DATA",                               \
      "SIGMA",                                  \
      "SIGMA_SPECTRUM",                         \
      "WEIGHT",                                 \
      "WEIGHT_SPECTRUM",                        \
      "FLAG",                                   \
      "FLAG_CATEGORY",                          \
      "FLAG_ROW"                                \
      }

#define MS_MAIN_COLUMN_ELEMENT_RANKS {          \
    0,                                          \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      1,                                        \
      1,                                        \
      2,                                        \
      2,                                        \
      1,                                        \
      2,                                        \
      1,                                        \
      2,                                        \
      1,                                        \
      2,                                        \
      2,                                        \
      3,                                        \
      0                                         \
      }

#define MS_MAIN_COLUMN_UNITS { \
  {MS_MAIN_COL_TIME, "s"}, \
  {MS_MAIN_COL_TIME_EXTRA_PREC, "s"}, \
  {MS_MAIN_COL_INTERVAL, "s"}, \
  {MS_MAIN_COL_EXPOSURE, "s"}, \
  {MS_MAIN_COL_TIME_CENTROID, "s"}, \
  {MS_MAIN_COL_UVW, "m"},\
  {MS_MAIN_COL_UVW2, "m"} \
}

#define MS_MAIN_COLUMN_MEASURE_NAMES {\
  {MS_MAIN_COL_TIME, "TIME_MEASURE_EPOCH"},\
  {MS_MAIN_COL_TIME_CENTROID, "TIME_CENTROID_MEASURE_EPOCH"},\
  {MS_MAIN_COL_UVW, "UVW_MEASURE_UVW"},\
  {MS_MAIN_COL_UVW2, "UVW2_MEASURE_UVW"}\
}

typedef enum ms_antenna_col_t {
  MS_ANTENNA_COL_NAME,
  MS_ANTENNA_COL_STATION,
  MS_ANTENNA_COL_TYPE,
  MS_ANTENNA_COL_MOUNT,
  MS_ANTENNA_COL_POSITION,
  MS_ANTENNA_COL_OFFSET,
  MS_ANTENNA_COL_DISH_DIAMETER,
  MS_ANTENNA_COL_ORBIT_ID,
  MS_ANTENNA_COL_MEAN_ORBIT,
  MS_ANTENNA_COL_PHASED_ARRAY_ID,
  MS_ANTENNA_COL_FLAG_ROW,
  MS_ANTENNA_NUM_COLS
} ms_antenna_col_t;

#define MS_ANTENNA_COL_FID_BASE 200

#define MS_ANTENNA_COLUMN_NAMES {               \
    "NAME",                                     \
      "STATION",                                \
      "TYPE",                                   \
      "MOUNT",                                  \
      "POSITION",                               \
      "OFFSET",                                 \
      "DISH_DIAMETER",                          \
      "ORBIT_ID",                               \
      "MEAN_ORBIT",                             \
      "PHASED_ARRAY_ID",                        \
      "FLAG_ROW"                                \
      }

#define MS_ANTENNA_COLUMN_ELEMENT_RANKS {       \
    0,                                          \
      0,                                        \
      0,                                        \
      0,                                        \
      1,                                        \
      1,                                        \
      0,                                        \
      0,                                        \
      1,                                        \
      0,                                        \
      0                                         \
      }

#define MS_ANTENNA_COLUMN_UNITS { \
  {MS_ANTENNA_COL_POSITION, "m"}, \
  {MS_ANTENNA_COL_OFFSET, "m"}, \
  {MS_ANTENNA_COL_DISH_DIAMETER, "m"} \
}

#define MS_ANTENNA_COLUMN_MEASURE_NAMES {\
  {MS_ANTENNA_COL_POSITION, "POSITION_MEASURE_POSITION"},\
  {MS_ANTENNA_COL_OFFSET, "OFFSET_MEASURE_POSITION"}\
}

typedef enum ms_data_description_col_t {
  MS_DATA_DESCRIPTION_COL_SPECTRAL_WINDOW_ID,
  MS_DATA_DESCRIPTION_COL_POLARIZATION_ID,
  MS_DATA_DESCRIPTION_COL_LAG_ID,
  MS_DATA_DESCRIPTION_COL_FLAG_ROW,
  MS_DATA_DESCRIPTION_NUM_COLS
} ms_data_description_col_t;

#define MS_DATA_DESCRIPTION_COL_FID_BASE 300

#define MS_DATA_DESCRIPTION_COLUMN_NAMES {      \
    "SPECTRAL_WINDOW_ID",                       \
      "POLARIZATION_ID",                        \
      "LAG_ID",                                 \
      "FLAG_ROW"                                \
      }

#define MS_DATA_DESCRIPTION_COLUMN_ELEMENT_RANKS {  \
    0,                                          \
      0,                                        \
      0,                                        \
      0                                         \
      }

#define MS_DATA_DESCRIPTION_COLUMN_UNITS {}

#define MS_DATA_DESCRIPTION_COLUMN_MEASURE_NAMES {}

typedef enum ms_doppler_col_t  {
  MS_DOPPLER_COL_DOPPLER_ID,
  MS_DOPPLER_COL_SOURCE_ID,
  MS_DOPPLER_COL_TRANSITION_ID,
  MS_DOPPLER_COL_VELDEF,
  MS_DOPPLER_NUM_COLS
} ms_doppler_col_t;

#define MS_DOPPLER_COL_FID_BASE 400

#define MS_DOPPLER_COLUMN_NAMES {               \
    "SPECTRAL_WINDOW_ID",                       \
      "POLARIZATION_ID",                        \
      "LAG_ID",                                 \
      "FLAG_ROW"                                \
      }

#define MS_DOPPLER_COLUMN_ELEMENT_RANKS {       \
    0,                                          \
      0,                                        \
      0,                                        \
      0                                         \
      }

#define MS_DOPPLER_COLUMN_UNITS {\
  {MS_DOPPLER_COL_VELDEF, "m/s"}\
}

#define MS_DOPPLER_COLUMN_MEASURE_NAMES {\
  {MS_DOPPLER_COL_VELDEF, "VELDEF_MEASURE_DOPPLER"}\
}

typedef enum ms_feed_col_t {
  MS_FEED_COL_ANTENNA_ID,
  MS_FEED_COL_FEED_ID,
  MS_FEED_COL_SPECTRAL_WINDOW_ID,
  MS_FEED_COL_TIME,
  MS_FEED_COL_INTERVAL,
  MS_FEED_COL_NUM_RECEPTORS,
  MS_FEED_COL_BEAM_ID,
  MS_FEED_COL_BEAM_OFFSET,
  MS_FEED_COL_FOCUS_LENGTH,
  MS_FEED_COL_PHASED_FEED_ID,
  MS_FEED_COL_POLARIZATION_TYPE,
  MS_FEED_COL_POL_RESPONSE,
  MS_FEED_COL_POSITION,
  MS_FEED_COL_RECEPTOR_ANGLE,
  MS_FEED_NUM_COLS
} ms_feed_col_t;

#define MS_FEED_COL_FID_BASE 500

#define MS_FEED_COLUMN_NAMES {                  \
    "ANTENNA_ID",                               \
      "FEED_ID",                                \
      "SPECTRAL_WINDOW_ID",                     \
      "TIME",                                   \
      "INTERVAL",                               \
      "NUM_RECEPTORS",                          \
      "BEAM_ID",                                \
      "BEAM_OFFSET",                            \
      "FOCUS_LENGTH",                           \
      "PHASED_FEED_ID",                         \
      "POLARIZATION_TYPE",                      \
      "POL_RESPONSE",                           \
      "POSITION",                               \
      "RECEPTOR_ANGLE"                          \
      }

#define MS_FEED_COLUMN_ELEMENT_RANKS {          \
    0,                                          \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      2,                                        \
      0,                                        \
      0,                                        \
      1,                                        \
      2,                                        \
      1,                                        \
      1                                         \
      }

#define MS_FEED_COLUMN_UNITS {\
  {MS_FEED_COL_TIME, "s"},\
  {MS_FEED_COL_INTERVAL, "s"},\
  {MS_FEED_COL_BEAM_OFFSET, "rad"},\
  {MS_FEED_COL_FOCUS_LENGTH, "m"},\
  {MS_FEED_COL_POSITION, "m"},\
  {MS_FEED_COL_RECEPTOR_ANGLE, "rad"}\
}

#define MS_FEED_COLUMN_MEASURE_NAMES {\
  {MS_FEED_COL_TIME, "TIME_MEASURE_EPOCH"},\
  {MS_FEED_COL_BEAM_OFFSET, "BEAM_OFFSET_MEASURE_DIRECTION"},\
  {MS_FEED_COL_POSITION, "POSITION_MEASURE_POSITION"}\
}

typedef enum ms_field_col_t {
  MS_FIELD_COL_NAME,
  MS_FIELD_COL_CODE,
  MS_FIELD_COL_TIME,
  MS_FIELD_COL_NUM_POLY,
  MS_FIELD_COL_DELAY_DIR,
  MS_FIELD_COL_PHASE_DIR,
  MS_FIELD_COL_REFERENCE_DIR,
  MS_FIELD_COL_SOURCE_ID,
  MS_FIELD_COL_EPHEMERIS_ID,
  MS_FIELD_COL_FLAG_ROW,
  MS_FIELD_NUM_COLS
} ms_field_col_t;

#define MS_FIELD_COL_FID_BASE 600

#define MS_FIELD_COLUMN_NAMES {                 \
    "NAME",                                     \
      "CODE",                                   \
      "TIME",                                   \
      "NUM_POLY",                               \
      "DELAY_DIR",                              \
      "PHASE_DIR",                              \
      "REFERENCE_DIR",                          \
      "SOURCE_ID",                              \
      "EPHEMERIS_ID",                           \
      "FLAG_ROW"                                \
      }

#define MS_FIELD_COLUMN_ELEMENT_RANKS {         \
    0,                                          \
      0,                                        \
      0,                                        \
      0,                                        \
      2,                                        \
      2,                                        \
      2,                                        \
      0,                                        \
      0,                                        \
      0                                         \
      }

#define MS_FIELD_COLUMN_UNITS {\
  {MS_FIELD_COL_TIME, "s"},\
  {MS_FIELD_COL_DELAY_DIR, "rad"},\
  {MS_FIELD_COL_PHASE_DIR, "rad"},\
  {MS_FIELD_COL_REFERENCE_DIR, "rad"}\
}

#define MS_FIELD_COLUMN_MEASURE_NAMES {\
  {MS_FIELD_COL_TIME, "TIME_MEASURE_EPOCH"},\
  {MS_FIELD_COL_DELAY_DIR, "DELAY_DIR_MEASURE_DIRECTION"},\
  {MS_FIELD_COL_PHASE_DIR, "PHASE_DIR_MEASURE_DIRECTION"},\
  {MS_FIELD_COL_REFERENCE_DIR, "REFERENCE_DIR_MEASURE_DIRECTION"}\
}

typedef enum ms_flag_cmd_col_t {
  MS_FLAG_CMD_COL_TIME,
  MS_FLAG_CMD_COL_INTERVAL,
  MS_FLAG_CMD_COL_TYPE,
  MS_FLAG_CMD_COL_REASON,
  MS_FLAG_CMD_COL_LEVEL,
  MS_FLAG_CMD_COL_SEVERITY,
  MS_FLAG_CMD_COL_APPLIED,
  MS_FLAG_CMD_COL_COMMAND,
  MS_FLAG_CMD_NUM_COLS
} ms_flag_cmd_col_t;

#define MS_FLAG_CMD_COL_FID_BASE 700

#define MS_FLAG_CMD_COLUMN_NAMES {              \
    "TIME",                                     \
      "INTERVAL",                               \
      "TYPE",                                   \
      "REASON",                                 \
      "LEVEL",                                  \
      "SEVERITY",                               \
      "APPLIED",                                \
      "COMMAND"                                 \
      }

#define MS_FLAG_CMD_COLUMN_ELEMENT_RANKS {      \
    0,                                          \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0                                         \
      }

#define MS_FLAG_CMD_COLUMN_UNITS {\
  {MS_FLAG_CMD_COL_TIME, "s"},\
  {MS_FLAG_CMD_COL_INTERVAL, "s"}\
}

#define MS_FLAG_CMD_COLUMN_MEASURE_NAMES {\
  {MS_FLAG_CMD_COL_TIME, "TIME_MEASURE_EPOCH"}\
}

typedef enum ms_freq_offset_col_t {
  MS_FREQ_OFFSET_COL_ANTENNA1,
  MS_FREQ_OFFSET_COL_ANTENNA2,
  MS_FREQ_OFFSET_COL_FEED_ID,
  MS_FREQ_OFFSET_COL_SPECTRAL_WINDOW_ID,
  MS_FREQ_OFFSET_COL_TIME,
  MS_FREQ_OFFSET_COL_INTERVAL,
  MS_FREQ_OFFSET_COL_OFFSET,
  MS_FREQ_OFFSET_NUM_COLS
} ms_freq_offset_col_t;

#define MS_FREQ_OFFSET_COL_FID_BASE 800

#define MS_FREQ_OFFSET_COLUMN_NAMES {           \
    "ANTENNA1",                                 \
      "ANTENNA2",                               \
      "FEED_ID",                                \
      "SPECTRAL_WINDOW_ID",                     \
      "TIME",                                   \
      "INTERVAL",                               \
      "OFFSET"                                  \
      }

#define MS_FREQ_OFFSET_COLUMN_ELEMENT_RANKS {   \
    0,                                          \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0                                         \
      }

#define MS_FREQ_OFFSET_COLUMN_UNITS {\
  {MS_FREQ_OFFSET_COL_TIME, "s"},\
  {MS_FREQ_OFFSET_COL_INTERVAL, "s"},\
  {MS_FREQ_OFFSET_COL_OFFSET, "Hz"}\
}

#define MS_FREQ_OFFSET_COLUMN_MEASURE_NAMES {\
  {MS_FREQ_OFFSET_COL_TIME, "TIME_MEASURE_EPOCH"}\
}

typedef enum ms_history_col_t {
  MS_HISTORY_COL_TIME,
  MS_HISTORY_COL_OBSERVATION_ID,
  MS_HISTORY_COL_MESSAGE,
  MS_HISTORY_COL_PRIORITY,
  MS_HISTORY_COL_ORIGIN,
  MS_HISTORY_COL_OBJECT_ID,
  MS_HISTORY_COL_APPLICATION,
  MS_HISTORY_COL_CLI_COMMAND,
  MS_HISTORY_COL_APP_PARAMS,
  MS_HISTORY_NUM_COLS
} ms_history_col_t;

#define MS_HISTORY_COL_FID_BASE 900

#define MS_HISTORY_COLUMN_NAMES {               \
    "TIME",                                     \
      "OBSERVATION_ID",                         \
      "MESSAGE",                                \
      "PRIORITY",                               \
      "ORIGIN",                                 \
      "OBJECT_ID",                              \
      "APPLICATION",                            \
      "CLI_COMMAND",                            \
      "APP_PARAMS"                              \
      }

#define MS_HISTORY_COLUMN_ELEMENT_RANKS {       \
    0,                                          \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      1,                                        \
      1                                         \
      }

#define MS_HISTORY_COLUMN_UNITS {\
  {MS_HISTORY_COL_TIME, "s"}\
}

#define MS_HISTORY_COLUMN_MEASURE_NAMES {\
  {MS_HISTORY_COL_TIME, "TIME_MEASURE_EPOCH"}\
}

typedef enum ms_observation_col_t {
  MS_OBSERVATION_COL_TELESCOPE_NAME,
  MS_OBSERVATION_COL_TIME_RANGE,
  MS_OBSERVATION_COL_OBSERVER,
  MS_OBSERVATION_COL_LOG,
  MS_OBSERVATION_COL_SCHEDULE_TYPE,
  MS_OBSERVATION_COL_SCHEDULE,
  MS_OBSERVATION_COL_PROJECT,
  MS_OBSERVATION_COL_RELEASE_DATE,
  MS_OBSERVATION_COL_FLAG_ROW,
  MS_OBSERVATION_NUM_COLS
} ms_observation_col_t;

#define MS_OBSERVATION_COL_FID_BASE 1000

#define MS_OBSERVATION_COLUMN_NAMES {           \
    "TELESCOPE_NAME",                           \
      "TIME_RANGE",                             \
      "OBSERVER",                               \
      "LOG",                                    \
      "SCHEDULE_TYPE",                          \
      "SCHEDULE",                               \
      "PROJECT",                                \
      "RELEASE_DATE",                           \
      "FLAG_ROW"                                \
      }

#define MS_OBSERVATION_COLUMN_ELEMENT_RANKS {   \
    0,                                          \
      1,                                        \
      0,                                        \
      1,                                        \
      0,                                        \
      1,                                        \
      0,                                        \
      0,                                        \
      0                                         \
      }

#define MS_OBSERVATION_COLUMN_UNITS {\
  {MS_OBSERVATION_COL_TIME_RANGE, "s"},\
  {MS_OBSERVATION_COL_RELEASE_DATE, "s"}\
}

#define MS_OBSERVATION_COLUMN_MEASURE_NAMES {\
  {MS_OBSERVATION_COL_TIME_RANGE, "TIME_RANGE_MEASURE_EPOCH"},\
  {MS_OBSERVATION_COL_RELEASE_DATE, "RELEASE_DATE_MEASURE_EPOCH"}\
}

typedef enum ms_pointing_col_t {
  MS_POINTING_COL_ANTENNA_ID,
  MS_POINTING_COL_TIME,
  MS_POINTING_COL_INTERVAL,
  MS_POINTING_COL_NAME,
  MS_POINTING_COL_NUM_POLY,
  MS_POINTING_COL_TIME_ORIGIN,
  MS_POINTING_COL_DIRECTION,
  MS_POINTING_COL_TARGET,
  MS_POINTING_COL_POINTING_OFFSET,
  MS_POINTING_COL_SOURCE_OFFSET,
  MS_POINTING_COL_ENCODER,
  MS_POINTING_COL_POINTING_MODEL_ID,
  MS_POINTING_COL_TRACKING,
  MS_POINTING_COL_ON_SOURCE,
  MS_POINTING_COL_OVER_THE_TOP,
  MS_POINTING_NUM_COLS
} ms_pointing_col_t;

#define MS_POINTING_COL_FID_BASE 1100

#define MS_POINTING_COLUMN_NAMES {              \
    "ANTENNA_ID",                               \
      "TIME",                                   \
      "INTERVAL",                               \
      "NAME",                                   \
      "NUM_POLY",                               \
      "TIME_ORIGIN",                            \
      "DIRECTION",                              \
      "TARGET",                                 \
      "POINTING_OFFSET",                        \
      "SOURCE_OFFSET",                          \
      "ENCODER",                                \
      "POINTING_MODEL_ID",                      \
      "TRACKING",                               \
      "ON_SOURCE",                              \
      "OVER_THE_TOP"                            \
      }

#define MS_POINTING_COLUMN_ELEMENT_RANKS {      \
    0,                                          \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      2,                                        \
      2,                                        \
      2,                                        \
      2,                                        \
      1,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0                                         \
      }

#define MS_POINTING_COLUMN_UNITS {\
  {MS_POINTING_COL_TIME, "s"},\
  {MS_POINTING_COL_INTERVAL, "s"},\
  {MS_POINTING_COL_TIME_ORIGIN, "s"},\
  {MS_POINTING_COL_DIRECTION, "rad"},\
  {MS_POINTING_COL_TARGET, "rad"},\
  {MS_POINTING_COL_POINTING_OFFSET, "rad"},\
  {MS_POINTING_COL_SOURCE_OFFSET, "rad"},\
  {MS_POINTING_COL_ENCODER, "rad"}\
}

#define MS_POINTING_COLUMN_MEASURE_NAMES {\
  {MS_POINTING_COL_TIME, "TIME_MEASURE_EPOCH"},\
  {MS_POINTING_COL_TIME_ORIGIN, "TIME_ORIGIN_MEASURE_EPOCH"},\
  {MS_POINTING_COL_DIRECTION, "DIRECTION_MEASURE_DIRECTION"},\
  {MS_POINTING_COL_TARGET, "TARGET_MEASURE_DIRECTION"},\
  {MS_POINTING_COL_POINTING_OFFSET, "POINTING_OFFSET_MEASURE_DIRECTION"},\
  {MS_POINTING_COL_SOURCE_OFFSET, "SOURCE_OFFSET_MEASURE_DIRECTION"},\
  {MS_POINTING_COL_ENCODER, "ENCODER_MEASURE_DIRECTION"}\
}

typedef enum ms_polarization_col_t {
  MS_POLARIZATION_COL_NUM_CORR,
  MS_POLARIZATION_COL_CORR_TYPE,
  MS_POLARIZATION_COL_CORR_PRODUCT,
  MS_POLARIZATION_COL_FLAG_ROW,
  MS_POLARIZATION_NUM_COLS
} ms_polarization_col_t;

#define MS_POLARIZATION_COL_FID_BASE 1200

#define MS_POLARIZATION_COLUMN_NAMES {          \
    "NUM_CORR",                                 \
      "CORR_TYPE",                              \
      "CORR_PRODUCT",                           \
      "FLAG_ROW"                                \
      }

#define MS_POLARIZATION_COLUMN_ELEMENT_RANKS {  \
    0,                                          \
      1,                                        \
      2,                                        \
      0                                         \
      }

#define MS_POLARIZATION_COLUMN_UNITS {}

#define MS_POLARIZATION_COLUMN_MEASURE_NAMES {}

typedef enum ms_processor_col_t {
  MS_PROCESSOR_COL_TYPE,
  MS_PROCESSOR_COL_SUB_TYPE,
  MS_PROCESSOR_COL_TYPE_ID,
  MS_PROCESSOR_COL_MODE_ID,
  MS_PROCESSOR_COL_PASS_ID,
  MS_PROCESSOR_COL_FLAG_ROW,
  MS_PROCESSOR_NUM_COLS
} ms_processor_col_t;

#define MS_PROCESSOR_COL_FID_BASE 1300

#define MS_PROCESSOR_COLUMN_NAMES {             \
    "TYPE",                                     \
      "SUB_TYPE",                               \
      "TYPE_ID",                                \
      "MODE_ID",                                \
      "PASS_ID",                                \
      "FLAG_ROW"                                \
      }

#define MS_PROCESSOR_COLUMN_ELEMENT_RANKS {     \
    0,                                          \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0                                         \
      }

#define MS_PROCESSOR_COLUMN_UNITS {}

#define MS_PROCESSOR_COLUMN_MEASURE_NAMES {}

typedef enum ms_source_col_t {
  MS_SOURCE_COL_SOURCE_ID,
  MS_SOURCE_COL_TIME,
  MS_SOURCE_COL_INTERVAL,
  MS_SOURCE_COL_SPECTRAL_WINDOW_ID,
  MS_SOURCE_COL_NUM_LINES,
  MS_SOURCE_COL_NAME,
  MS_SOURCE_COL_CALIBRATION_GROUP,
  MS_SOURCE_COL_CODE,
  MS_SOURCE_COL_DIRECTION,
  MS_SOURCE_COL_POSITION,
  MS_SOURCE_COL_PROPER_MOTION,
  MS_SOURCE_COL_TRANSITION,
  MS_SOURCE_COL_REST_FREQUENCY,
  MS_SOURCE_COL_SYSVEL,
  MS_SOURCE_COL_SOURCE_MODEL,
  MS_SOURCE_COL_PULSAR_ID,
  MS_SOURCE_NUM_COLS
} ms_source_col_t;

#define MS_SOURCE_COL_FID_BASE 1400

#define MS_SOURCE_COLUMN_NAMES {                \
    "SOURCE_ID",                                \
      "TIME",                                   \
      "INTERVAL",                               \
      "SPECTRAL_WINDOW_ID",                     \
      "NUM_LINES",                              \
      "NAME",                                   \
      "CALIBRATION_GROUP",                      \
      "CODE",                                   \
      "DIRECTION",                              \
      "POSITION",                               \
      "PROPER_MOTION",                          \
      "TRANSITION",                             \
      "REST_FREQUENCY",                         \
      "SYSVEL",                                 \
      "SOURCE_MODEL",                           \
      "PULSAR_ID",                              \
      }

#define MS_SOURCE_COLUMN_ELEMENT_RANKS {        \
    0,                                          \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      1,                                        \
      1,                                        \
      1,                                        \
      1,                                        \
      1,                                        \
      1,                                        \
      0,                                        \
      0                                         \
      }

#define MS_SOURCE_COLUMN_UNITS {\
  {MS_SOURCE_COL_TIME, "s"} ,\
  {MS_SOURCE_COL_INTERVAL, "s"},\
  {MS_SOURCE_COL_DIRECTION, "rad"},\
  {MS_SOURCE_COL_POSITION, "m"},\
  {MS_SOURCE_COL_PROPER_MOTION, "rad/s"},\
  {MS_SOURCE_COL_REST_FREQUENCY, "Hz"},\
  {MS_SOURCE_COL_SYSVEL, "m/s"}\
}

#define MS_SOURCE_COLUMN_MEASURE_NAMES {\
  {MS_SOURCE_COL_TIME, "TIME_MEASURE_EPOCH"},\
  {MS_SOURCE_COL_DIRECTION, "DIRECTION_MEASURE_DIRECTION"},\
  {MS_SOURCE_COL_POSITION, "POSITION_MEASURE_POSITION"},\
  {MS_SOURCE_COL_REST_FREQUENCY, "REST_FREQUENCY_MEASURE_FREQUENCY"},\
  {MS_SOURCE_COL_SYSVEL, "SYSVEL_MEASURE_RADIAL_VELOCITY"}\
}

typedef enum ms_spectral_window_col_t {
  MS_SPECTRAL_WINDOW_COL_NUM_CHAN,
  MS_SPECTRAL_WINDOW_COL_NAME,
  MS_SPECTRAL_WINDOW_COL_REF_FREQUENCY,
  MS_SPECTRAL_WINDOW_COL_CHAN_FREQ,
  MS_SPECTRAL_WINDOW_COL_CHAN_WIDTH,
  MS_SPECTRAL_WINDOW_COL_MEAS_FREQ_REF,
  MS_SPECTRAL_WINDOW_COL_EFFECTIVE_BW,
  MS_SPECTRAL_WINDOW_COL_RESOLUTION,
  MS_SPECTRAL_WINDOW_COL_TOTAL_BANDWIDTH,
  MS_SPECTRAL_WINDOW_COL_NET_SIDEBAND,
  MS_SPECTRAL_WINDOW_COL_BBC_NO,
  MS_SPECTRAL_WINDOW_COL_BBC_SIDEBAND,
  MS_SPECTRAL_WINDOW_COL_IF_CONV_CHAIN,
  MS_SPECTRAL_WINDOW_COL_RECEIVER_ID,
  MS_SPECTRAL_WINDOW_COL_FREQ_GROUP,
  MS_SPECTRAL_WINDOW_COL_FREQ_GROUP_NAME,
  MS_SPECTRAL_WINDOW_COL_DOPPLER_ID,
  MS_SPECTRAL_WINDOW_COL_ASSOC_SPW_ID,
  MS_SPECTRAL_WINDOW_COL_ASSOC_NATURE,
  MS_SPECTRAL_WINDOW_COL_FLAG_ROW,
  MS_SPECTRAL_WINDOW_NUM_COLS
} ms_spectral_window_col_t;

#define MS_SPECTRAL_WINDOW_COL_FID_BASE 1500

#define MS_SPECTRAL_WINDOW_COLUMN_NAMES {       \
    "NUM_CHAN",                                 \
      "NAME",                                   \
      "REF_FREQUENCY",                          \
      "CHAN_FREQ",                              \
      "CHAN_WIDTH",                             \
      "MEAS_FREQ_REF",                          \
      "EFFECTIVE_BW",                           \
      "RESOLUTION",                             \
      "TOTAL_BANDWIDTH",                        \
      "NET_SIDEBAND",                           \
      "BBC_NO",                                 \
      "BBC_SIDEBAND",                           \
      "IF_CONV_CHAIN",                          \
      "RECEIVER_ID",                            \
      "FREQ_GROUP",                             \
      "FREQ_GROUP_NAME",                        \
      "DOPPLER_ID",                             \
      "ASSOC_SPW_ID",                           \
      "ASSOC_NATURE",                           \
      "FLAG_ROW"                                \
      }

#define MS_SPECTRAL_WINDOW_COLUMN_ELEMENT_RANKS { \
    0,                                          \
      0,                                        \
      0,                                        \
      1,                                        \
      1,                                        \
      0,                                        \
      1,                                        \
      1,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      1,                                        \
      1,                                        \
      0                                         \
      }

#define MS_SPECTRAL_WINDOW_COLUMN_UNITS {\
  {MS_SPECTRAL_WINDOW_COL_REF_FREQUENCY, "Hz"},\
  {MS_SPECTRAL_WINDOW_COL_CHAN_FREQ, "Hz"},\
  {MS_SPECTRAL_WINDOW_COL_CHAN_WIDTH, "Hz"},\
  {MS_SPECTRAL_WINDOW_COL_EFFECTIVE_BW, "Hz"},\
  {MS_SPECTRAL_WINDOW_COL_RESOLUTION, "Hz"},\
  {MS_SPECTRAL_WINDOW_COL_TOTAL_BANDWIDTH, "Hz"}\
}

#define MS_SPECTRAL_WINDOW_COLUMN_MEASURE_NAMES {\
  {MS_SPECTRAL_WINDOW_COL_REF_FREQUENCY, "REF_FREQUENCY_MEASURE_FREQUENCY"},\
  {MS_SPECTRAL_WINDOW_COL_CHAN_FREQ, "CHAN_FREQ_MEASURE_FREQUENCY"}\
}

typedef enum ms_state_col_t {
  MS_STATE_COL_SIG,
  MS_STATE_COL_REF,
  MS_STATE_COL_CAL,
  MS_STATE_COL_LOAD,
  MS_STATE_COL_SUB_SCAN,
  MS_STATE_COL_OBS_MODE,
  MS_STATE_COL_FLAG_ROW,
  MS_STATE_NUM_COLS
} ms_state_col_t;

#define MS_STATE_COL_FID_BASE 1600

#define MS_STATE_COLUMN_NAMES {                 \
    "SIG",                                      \
      "REF",                                    \
      "CAL",                                    \
      "LOAD",                                   \
      "SUB_SCAN",                               \
      "OBS_MODE",                               \
      "FLAG_ROW"                                \
      }

#define MS_STATE_COLUMN_ELEMENT_RANKS {         \
    0,                                          \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0                                         \
      }

#define MS_STATE_COLUMN_UNITS {\
  {MS_STATE_COL_CAL, "K"},\
  {MS_STATE_COL_LOAD, "K"}\
}

#define MS_STATE_COLUMN_MEASURE_NAMES {}

typedef enum ms_syscal_col_t {
  MS_SYSCAL_COL_ANTENNA_ID,
  MS_SYSCAL_COL_FEED_ID,
  MS_SYSCAL_COL_SPECTRAL_WINDOW_ID,
  MS_SYSCAL_COL_TIME,
  MS_SYSCAL_COL_INTERVAL,
  MS_SYSCAL_COL_PHASE_DIFF,
  MS_SYSCAL_COL_TCAL,
  MS_SYSCAL_COL_TRX,
  MS_SYSCAL_COL_TSKY,
  MS_SYSCAL_COL_TSYS,
  MS_SYSCAL_COL_TANT,
  MS_SYSCAL_COL_TANT_TSYS,
  MS_SYSCAL_COL_TCAL_SPECTRUM,
  MS_SYSCAL_COL_TRX_SPECTRUM,
  MS_SYSCAL_COL_TSKY_SPECTRUM,
  MS_SYSCAL_COL_TSYS_SPECTRUM,
  MS_SYSCAL_COL_TANT_SPECTRUM,
  MS_SYSCAL_COL_TANT_TSYS_SPECTRUM,
  MS_SYSCAL_COL_PHASE_DIFF_FLAG,
  MS_SYSCAL_COL_TCAL_FLAG,
  MS_SYSCAL_COL_TRX_FLAG,
  MS_SYSCAL_COL_TSKY_FLAG,
  MS_SYSCAL_COL_TSYS_FLAG,
  MS_SYSCAL_COL_TANT_FLAG,
  MS_SYSCAL_COL_TANT_TSYS_FLAG,
  MS_SYSCAL_NUM_COLS
} ms_syscal_col_t;

#define MS_SYSCAL_COL_FID_BASE 1700

#define MS_SYSCAL_COLUMN_NAMES {                \
    "ANTENNA_ID",                               \
      "FEED_ID",                                \
      "SPECTRAL_WINDOW_ID",                     \
      "TIME",                                   \
      "INTERVAL",                               \
      "PHASE_DIFF",                             \
      "TCAL",                                   \
      "TRX",                                    \
      "TSKY",                                   \
      "TSYS",                                   \
      "TANT",                                   \
      "TANT_TSYS",                              \
      "TCAL_SPECTRUM",                          \
      "TRX_SPECTRUM",                           \
      "TSKY_SPECTRUM",                          \
      "TSYS_SPECTRUM",                          \
      "TANT_SPECTRUM",                          \
      "TANT_TSYS_SPECTRUM",                     \
      "PHASE_DIFF_FLAG",                        \
      "TCAL_FLAG",                              \
      "TRX_FLAG",                               \
      "TSKY_FLAG",                              \
      "TSYS_FLAG",                              \
      "TANT_FLAG",                              \
      "TANT_TSYS_FLAG"                          \
      }

#define MS_SYSCAL_COLUMN_ELEMENT_RANKS {        \
    0,                                          \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      1,                                        \
      1,                                        \
      1,                                        \
      1,                                        \
      1,                                        \
      1,                                        \
      2,                                        \
      2,                                        \
      2,                                        \
      2,                                        \
      2,                                        \
      2,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0                                         \
      }

#define MS_SYSCAL_COLUMN_UNITS {\
  {MS_SYSCAL_COL_TIME, "s"},\
  {MS_SYSCAL_COL_INTERVAL, "s"},\
  {MS_SYSCAL_COL_PHASE_DIFF, "rad"},\
  {MS_SYSCAL_COL_TCAL, "K"},\
  {MS_SYSCAL_COL_TRX, "K"},\
  {MS_SYSCAL_COL_TSKY, "K"},\
  {MS_SYSCAL_COL_TSYS, "K"},\
  {MS_SYSCAL_COL_TANT, "K"},\
  {MS_SYSCAL_COL_TCAL_SPECTRUM, "K"},\
  {MS_SYSCAL_COL_TRX_SPECTRUM, "K"},\
  {MS_SYSCAL_COL_TSKY_SPECTRUM, "K"},  \
  {MS_SYSCAL_COL_TSYS_SPECTRUM, "K"}, \
  {MS_SYSCAL_COL_TANT_SPECTRUM, "K"} \
}

#define MS_SYSCAL_COLUMN_MEASURE_NAMES {\
  {MS_SYSCAL_COL_TIME, "TIME_MEASURE_EPOCH"}\
}

typedef enum ms_weather_col_t {
  MS_WEATHER_COL_ANTENNA_ID,
  MS_WEATHER_COL_TIME,
  MS_WEATHER_COL_INTERVAL,
  MS_WEATHER_COL_H2O,
  MS_WEATHER_COL_IONOS_ELECTRON,
  MS_WEATHER_COL_PRESSURE,
  MS_WEATHER_COL_REL_HUMIDITY,
  MS_WEATHER_COL_TEMPERATURE,
  MS_WEATHER_COL_DEW_POINT,
  MS_WEATHER_COL_WIND_DIRECTION,
  MS_WEATHER_COL_WIND_SPEED,
  MS_WEATHER_COL_H2O_FLAG,
  MS_WEATHER_COL_IONOS_ELECTRON_FLAG,
  MS_WEATHER_COL_PRESSURE_FLAG,
  MS_WEATHER_COL_REL_HUMIDITY_FLAG,
  MS_WEATHER_COL_TEMPERATURE_FLAG,
  MS_WEATHER_COL_DEW_POINT_FLAG,
  MS_WEATHER_COL_WIND_DIRECTION_FLAG,
  MS_WEATHER_COL_WIND_SPEED_FLAG,
  MS_WEATHER_NUM_COLS
} ms_weather_col_t;

#define MS_WEATHER_COL_FID_BASE 1800

#define MS_WEATHER_COLUMN_NAMES {               \
    "ANTENNA_ID",                               \
      "TIME",                                   \
      "INTERVAL",                               \
      "H2O",                                    \
      "IONOS_ELECTRON",                         \
      "PRESSURE",                               \
      "REL_HUMIDITY",                           \
      "TEMPERATURE",                            \
      "DEW_POINT",                              \
      "WIND_DIRECTION",                         \
      "WIND_SPEED",                             \
      "H2O_FLAG",                               \
      "IONOS_ELECTRON_FLAG",                    \
      "PRESSURE_FLAG",                          \
      "REL_HUMIDITY_FLAG",                      \
      "TEMPERATURE_FLAG",                       \
      "DEW_POINT_FLAG",                         \
      "WIND_DIRECTION_FLAG",                    \
      "WIND_SPEED_FLAG"                         \
      }

#define MS_WEATHER_COLUMN_ELEMENT_RANKS {       \
    0,                                          \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0,                                        \
      0                                         \
      }

#define MS_WEATHER_COLUMN_UNITS {\
  {MS_WEATHER_COL_TIME, "s"},\
  {MS_WEATHER_COL_INTERVAL, "s"},\
  {MS_WEATHER_COL_H2O, "m-2"},\
  {MS_WEATHER_COL_IONOS_ELECTRON, "m-2"},\
  {MS_WEATHER_COL_PRESSURE, "hPa"},\
  {MS_WEATHER_COL_TEMPERATURE, "K"},\
  {MS_WEATHER_COL_DEW_POINT, "K"},\
  {MS_WEATHER_COL_WIND_DIRECTION, "rad"},\
  {MS_WEATHER_COL_WIND_SPEED, "m/s"}\
}

#define MS_WEATHER_COLUMN_MEASURE_NAMES {\
  {MS_WEATHER_COL_TIME, "TIME_MEASURE_EPOCH"}\
}

#ifdef __cplusplus
}
#endif

#endif // HYPERION_MS_TABLE_COLUMNS_C_H_

// Local Variables:
// mode: c
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
