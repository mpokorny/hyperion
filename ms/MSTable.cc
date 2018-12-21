#include "MSTable.h"

using namespace legms::ms;

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
    {"UVW", {Axes::uvw}},
    {"UVW2", {Axes::uvw}},
    {"DATA", {Axes::frequency_channel, Axes::correlator}},
    {"FLOAT_DATA", {Axes::frequency_channel, Axes::correlator}},
    {"VIDEO_POINT", {Axes::frequency_channel}},
    {"LAG_DATA", {Axes::lag, Axes::correlator}},
    {"SIGMA", {Axes::frequency_channel}},
    {"SIGMA_SPECTRUM", {Axes::frequency_channel, Axes::correlator}},
    {"WEIGHT", {Axes::frequency_channel}},
    {"WEIGHT_SPECTRUM", {Axes::frequency_channel, Axes::correlator}},
    {"FLAG", {Axes::frequency_channel, Axes::correlator}},
    {"FLAG_CATEGORY", {Axes::flag_category, Axes::frequency_channel,
                       Axes::correlator}},
    {"FLAG_ROW", {}}
  };

const std::unordered_map<
    std::string,
    std::vector<MSTable<MSTables::ANTENNA>::Axes>>
MSTable<MSTables::ANTENNA>::element_axes = {
    {"NAME", {}},
    {"STATION", {}},
    {"TYPE", {}},
    {"MOUNT", {}},
    {"POSITION", {Axes::position}},
    {"OFFSET", {Axes::offset}},
    {"DISH_DIAMETER", {}},
    {"ORBIT_ID", {}},
    {"MEAN_ORBIT", {Axes::mean_orbit}},
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
    {"BEAM_OFFSET", {Axes::receptor, Axes::direction}},
    {"FOCUS_LENGTH", {}},
    {"PHASED_FEED_ID", {}},
    {"POLARIZATION_TYPE", {Axes::receptor}},
    {"POL_RESPONSE", {Axes::receptor1, Axes::receptor}},
    {"POSITION", {Axes::position}},
    {"RECEPTOR_ANGLE", {Axes::receptor}}
};

const std::unordered_map<
    std::string,
    std::vector<MSTable<MSTables::FIELD>::Axes>>
MSTable<MSTables::FIELD>::element_axes = {
    {"NAME", {}},
    {"CODE", {}},
    {"TIME", {}},
    {"NUM_POLY", {}},
    {"DELAY_DIR", {Axes::polynomial, Axes::direction}},
    {"PHASE_DIR", {Axes::polynomial, Axes::direction}},
    {"REFERENCE_DIR", {Axes::polynomial, Axes::direction}},
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
    {"CLI_COMMAND", {Axes::cli_command}},
    {"APP_PARAMS", {Axes::app_param}}
};

const std::unordered_map<
    std::string,
    std::vector<MSTable<MSTables::OBSERVATION>::Axes>>
MSTable<MSTables::OBSERVATION>::element_axes = {
    {"TELESCOPE_NAME", {}},
    {"TIME_RANGE", {Axes::time_range}},
    {"OBSERVER", {}},
    {"LOG", {Axes::log}},
    {"SCHEDULE_TYPE", {}},
    {"SCHEDULE", {Axes::schedule}},
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
    {"DIRECTION", {Axes::polynomial, Axes::direction}},
    {"TARGET", {Axes::polynomial, Axes::direction}},
    {"POINTING_OFFSET", {Axes::polynomial, Axes::direction}},
    {"SOURCE_OFFSET", {Axes::polynomial, Axes::direction}},
    {"ENCODER", {Axes::direction}},
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
    {"CORR_TYPE", {Axes::correlation}},
    {"CORR_PRODUCT", {Axes::correlation, Axes::product}},
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
    {"DIRECTION", {Axes::direction}},
    {"POSITION", {Axes::position}},
    {"PROPER_MOTION", {Axes::proper_motion}},
    {"TRANSITION", {Axes::line}},
    {"REST_FREQUENCY", {Axes::line}},
    {"SYSVEL", {Axes::line}},
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
    {"CHAN_FREQ", {Axes::channel}},
    {"CHAN_WIDTH", {Axes::channel}},
    {"MEAS_FREQ_REF", {}},
    {"EFFECTIVE_BW", {Axes::channel}},
    {"RESOLUTION", {Axes::channel}},
    {"TOTAL_BANDWIDTH", {}},
    {"NET_SIDEBAND", {}},
    {"BBC_NO", {}},
    {"BBC_SIDEBAND", {}},
    {"IF_CONV_CHAN", {}},
    {"RECEIVER_ID", {}},
    {"FREQ_GROUP", {}},
    {"FREQ_GROUP_NAME", {}},
    {"DOPPLER_ID", {}},
    {"ASSOC_SPW_ID", {Axes::assoc_spw}},
    {"ASSOC_NATURE", {Axes::assoc_spw}},
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
    {"TCAL", {Axes::receptor}},
    {"TRX", {Axes::receptor}},
    {"TSKY", {Axes::receptor}},
    {"TSYS", {Axes::receptor}},
    {"TANT", {Axes::receptor}},
    {"TANT_TSYS", {Axes::receptor}},
    {"TCAL_SPECTRUM", {Axes::channel, Axes::receptor}},
    {"TRX_SPECTRUM", {Axes::channel, Axes::receptor}},
    {"TSKY_SPECTRUM", {Axes::channel, Axes::receptor}},
    {"TSYS_SPECTRUM", {Axes::channel, Axes::receptor}},
    {"TANT_SPECTRUM", {Axes::channel, Axes::receptor}},
    {"TANT_TSYS_SPECTRUM", {Axes::channel, Axes::receptor}},
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


