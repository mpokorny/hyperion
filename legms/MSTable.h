#ifndef LEGMS_MS_TABLE_H_
#define LEGMS_MS_TABLE_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace legms {

enum struct MSTables {
  MAIN,
  ANTENNA,
  DATA_DESCRIPTION,
  DOPPLER,
  FEED,
  FIELD,
  FLAG_CMD,
  FREQ_OFFSET,
  HISTORY,
  OBSERVATION,
  POINTING,
  POLARIZATION,
  PROCESSOR,
  SOURCE,
  SPECTRAL_WINDOW,
  STATE,
  SYSCAL,
  WEATHER
};

template <MSTables T>
struct MSTable {
  // static const char* name;
  // enum struct Axes;
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

#define MS_AXIS_NAME(T, A)                                          \
  template <>                                                       \
  struct MSTableAxis<MSTables::T, MSTable<MSTables::T>::Axes::A> {  \
    static const constexpr char* name = #A;                         \
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
struct MSTable<MSTables::MAIN> {
  static const constexpr char *name = "MAIN";

  enum struct Axes {
    index = -1,

    ROW = 0,
    // UVW(0)
    UVW,
    // DATA(0), FLOAT_DATA(0), VIDEO_POINT(0), LAG_DATA(0), SIGMA(0),
    // SIGMA_SPECTRUM(0), WEIGHT_SPECTRUM(0), FLAG(0), FLAG_CATEGORY(0)
    CORRELATOR,
    // DATA(1), FLOAT_DATA(1), SIGMA_SPECTRUM(1), WEIGHT_SPECTRUM(1), FLAG(1),
    // FLAG_CATEGORY(1)
    FREQUENCY_CHANNEL,
    // LAG_DATA(1)
    LAG,
    // FLAG_CATEGORY(2)
    FLAG_CATEGORY,

    // key column axes
    TIME,
    TIME_EXTRA_PREC,
    ANTENNA1,
    ANTENNA2,
    ANTENNA3,
    FEED1,
    FEED2,
    FEED3,
    DATA_DESC_ID,
    PROCESSOR_ID,
    PHASE_ID,
    FIELD_ID,

    // additional index column axes
    SCAN_NUMBER,
    ARRAY_ID,
    OBSERVATION_ID,
    STATE_ID,

    last = STATE_ID
  };

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

template <>
struct MSTable<MSTables::ANTENNA> {
  static const constexpr char* name = "ANTENNA";

  enum struct Axes {
    ROW = 0,
    // POSITION(0)
    POSITION,
    // OFFSET(0)
    OFFSET,
    // MEAN_ORBIT(0)
    MEAN_ORBIT,

    last = MEAN_ORBIT
  };

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(ANTENNA, ROW);
MS_AXIS_NAME(ANTENNA, POSITION);
MS_AXIS_NAME(ANTENNA, OFFSET);
MS_AXIS_NAME(ANTENNA, MEAN_ORBIT);

template <>
struct MSTable<MSTables::DATA_DESCRIPTION> {
  static const constexpr char* name = "DATA_DESCRIPTION";

  enum struct Axes {
    ROW,

    last = ROW
  };

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(DATA_DESCRIPTION, ROW);

template <>
struct MSTable<MSTables::DOPPLER> {
  static const constexpr char *name = "DOPPLER";

  enum struct Axes {
    index = -1,

    ROW = 0,

    // key column axes
    DOPPLER_ID,
    SOURCE_ID,

    // additional index column axes
    TRANSITION_ID,

    last = TRANSITION_ID
  };

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(DOPPLER, ROW);
MS_AXIS_NAME(DOPPLER, DOPPLER_ID);
MS_AXIS_NAME(DOPPLER, SOURCE_ID);
MS_AXIS_NAME(DOPPLER, TRANSITION_ID);

template <>
struct MSTable<MSTables::FEED> {
  static const constexpr char *name = "FEED";

  enum struct Axes {
    index = -1,

    ROW = 0,
    // BEAM_OFFSET(1), POLARIZATION_TYPE(0), POL_RESPONSE(0), RECEPTOR_ANGLE(0)
    RECEPTOR,
    // POL_RESPONSE(1)
    RECEPTOR1,
    // BEAM_OFFSET(0)
    DIRECTION,
    // POSITION(0)
    POSITION,

    // key column axes
    ANTENNA_ID,
    FEED_ID,
    SPECTRAL_WINDOW_ID,
    TIME,
    INTERVAL,

    last = INTERVAL
  };

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

template <>
struct MSTable<MSTables::FIELD> {
  static const constexpr char* name = "FIELD";

  enum struct Axes {
    index = -1,

    ROW = 0,
    // DELAY_DIR(1), PHASE_DIR(1), REFERENCE_DIR(1)
    POLYNOMIAL,
    // DELAY_DIR(0), PHASE_DIR(0), REFERENCE_DIR(0)
    DIRECTION,

    // additional index column axes
    SOURCE_ID,
    EPHEMERIS_ID,

    last = EPHEMERIS_ID
  };

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

template <>
struct MSTable<MSTables::FLAG_CMD> {
  static const constexpr char *name = "FLAG_CMD";

  enum struct Axes {
    index = -1,

    ROW = 0,

    // key column axes
    TIME,
    INTERVAL,

    last = INTERVAL
  };

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(FLAG_CMD, ROW);
MS_AXIS_NAME(FLAG_CMD, TIME);
MS_AXIS_NAME(FLAG_CMD, INTERVAL);

template <>
struct MSTable<MSTables::FREQ_OFFSET> {
  static const constexpr char* name = "FREQ_OFFSET";

  enum struct Axes {
    index = -1,

    ROW = 0,

    // key column axes
    ANTENNA1,
    ANTENNA2,
    FEED_ID,
    SPECTRAL_WINDOW_ID,
    TIME,
    INTERVAL,

    last = INTERVAL
  };

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

template <>
struct MSTable<MSTables::HISTORY> {
  static const constexpr char* name = "HISTORY";

  enum struct Axes {
    index = -1,

    ROW = 0,
    // CLI_COMMAND(0)
    CLI_COMMAND,
    // APP_PARAMS(0)
    APP_PARAM,

    // key column axes
    TIME,
    OBSERVATION_ID,

    last = OBSERVATION_ID
  };

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

template <>
struct MSTable<MSTables::OBSERVATION> {
  static const constexpr char* name = "OBSERVATION";

  enum struct Axes {
    ROW,
    // TIME_RANGE(0)
    TIME_RANGE,
    // LOG(0)
    LOG,
    // SCHEDULE(0)
    SCHEDULE,

    last = SCHEDULE
  };

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(OBSERVATION, ROW);
MS_AXIS_NAME(OBSERVATION, TIME_RANGE);
MS_AXIS_NAME(OBSERVATION, LOG);
MS_AXIS_NAME(OBSERVATION, SCHEDULE);

template <>
struct MSTable<MSTables::POINTING> {
  static const constexpr char* name = "POINTING";

  enum struct Axes {
    index = -1,

    ROW = 0,
    // DIRECTION(1), TARGET(1), POINTING_OFFSET(1), SOURCE_OFFSET(1)
    POLYNOMIAL,
    // DIRECTION(0), TARGET(0), POINTING_OFFSET(0), SOURCE_OFFSET(0), ENCODER(0)
    DIRECTION,

    // key column axes
    ANTENNA_ID,
    TIME,
    INTERVAL,

    last = INTERVAL
  };

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

template <>
struct MSTable<MSTables::POLARIZATION> {
  static const constexpr char* name = "POLARIZATION";

  enum struct Axes {
    ROW,
    // CORR_TYPE(0), CORR_PRODUCT(1)
    CORRELATION,
    // CORR_PRODUCT(0)
    PRODUCT,

    last = PRODUCT
  };

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(POLARIZATION, ROW);
MS_AXIS_NAME(POLARIZATION, CORRELATION);
MS_AXIS_NAME(POLARIZATION, PRODUCT);

template <>
struct MSTable<MSTables::PROCESSOR> {
  static const constexpr char* name = "PROCESSOR";

  enum struct Axes {
    ROW,

    last = ROW
  };

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(PROCESSOR, ROW);


template <>
struct MSTable<MSTables::SOURCE> {
  static const constexpr char* name = "SOURCE";

  enum struct Axes {
    index = -1,

    ROW = 0,
    // DIRECTION(0)
    DIRECTION,
    // POSITION(0)
    POSITION,
    // PROPER_MOTION(0)
    PROPER_MOTION,
    // TRANSITION(0), REST_FREQUENCY(0), SYSVEL(0)
    LINE,

    // key column axes
    SOURCE_ID,
    TIME,
    INTERVAL,
    SPECTRAL_WINDOW_ID,

    // additional column index axes
    PULSAR_ID,

    last = PULSAR_ID
  };

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

template <>
struct MSTable<MSTables::SPECTRAL_WINDOW> {
  static const constexpr char* name = "SPECTRAL_WINDOW";

  enum struct Axes {
    ROW,
    // CHAN_FREQ(0), CHAN_WIDTH(0), EFFECTIVE_BW(0), RESOLUTION(0)
    CHANNEL,
    // ASSOC_SPW_ID(0), ASSOC_NATURE(0)
    ASSOC_SPW,

    last = ASSOC_SPW
  };

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(SPECTRAL_WINDOW, ROW);
MS_AXIS_NAME(SPECTRAL_WINDOW, CHANNEL);
MS_AXIS_NAME(SPECTRAL_WINDOW, ASSOC_SPW);

template <>
struct MSTable<MSTables::STATE> {
  static const constexpr char* name = "STATE";

  enum struct Axes {
    ROW,

    last = ROW
  };

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(STATE, ROW);

template <>
struct MSTable<MSTables::SYSCAL> {
  static const constexpr char* name = "SYSCAL";

  enum struct Axes {
    index = -1,

    ROW = 0,
    // TCAL(0), TRX(0), TSKY(0), TSYS(0), TANT(0), TANT_TSYS(0), TCAL_SPECTRUM(0),
    // TRX_SPECTRUM(0), TSKY_SPECTRUM(0), TSYS_SPECTRUM(0), TANT_SPECTRUM(0),
    // TANT_TSYS_SPECTRUM(0)
    RECEPTOR,
    // TCAL_SPECTRUM(1), TRX_SPECTRUM(1), TSKY_SPECTRUM(1), TSYS_SPECTRUM(1),
    // TANT_SPECTRUM(1), TANT_TSYS_SPECTRUM(1)
    CHANNEL,

    // key column axes
    ANTENNA_ID,
    FEED_ID,
    SPECTRAL_WINDOW_ID,
    TIME,
    INTERVAL,

    last = INTERVAL
  };

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

template <>
struct MSTable<MSTables::WEATHER> {
  static const constexpr char* name = "WEATHER";

  enum struct Axes {
    index = -1,

    ROW,

    // key column axes
    ANTENNA_ID,
    TIME,
    INTERVAL,

    last = INTERVAL
  };

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(WEATHER, ROW);
MS_AXIS_NAME(WEATHER, ANTENNA_ID);
MS_AXIS_NAME(WEATHER, TIME);
MS_AXIS_NAME(WEATHER, INTERVAL);

#undef MS_AXIS_NAME

} // end namespace legms

#endif // LEGMS_MS_TABLE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
