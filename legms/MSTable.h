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
    row,
    // UVW(0)
    uvw,
    // DATA(0), FLOAT_DATA(0), VIDEO_POINT(0), LAG_DATA(0), SIGMA(0),
    // SIGMA_SPECTRUM(0), WEIGHT_SPECTRUM(0), FLAG(0), FLAG_CATEGORY(0)
    correlator,
    // DATA(1), FLOAT_DATA(1), SIGMA_SPECTRUM(1), WEIGHT_SPECTRUM(1), FLAG(1),
    // FLAG_CATEGORY(1)
    frequency_channel,
    // LAG_DATA(1)
    lag,
    // FLAG_CATEGORY(2)
    flag_category,

    last = flag_category
  };

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(MAIN, row);
MS_AXIS_NAME(MAIN, uvw);
MS_AXIS_NAME(MAIN, correlator);
MS_AXIS_NAME(MAIN, frequency_channel);
MS_AXIS_NAME(MAIN, lag);
MS_AXIS_NAME(MAIN, flag_category);

template <>
struct MSTable<MSTables::ANTENNA> {
  static const constexpr char* name = "ANTENNA";

  enum struct Axes {
    row,
    // POSITION(0)
    position,
    // OFFSET(0)
    offset,
    // MEAN_ORBIT(0)
    mean_orbit,

    last = mean_orbit
  };

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(ANTENNA, row);
MS_AXIS_NAME(ANTENNA, position);
MS_AXIS_NAME(ANTENNA, offset);
MS_AXIS_NAME(ANTENNA, mean_orbit);

template <>
struct MSTable<MSTables::DATA_DESCRIPTION> {
  static const constexpr char* name = "DATA_DESCRIPTION";

  enum struct Axes {
    row,

    last = row
  };

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(DATA_DESCRIPTION, row);

template <>
struct MSTable<MSTables::DOPPLER> {
  static const constexpr char *name = "DOPPLER";

  enum struct Axes {
    row,

    last = row
  };

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(DOPPLER, row);

template <>
struct MSTable<MSTables::FEED> {
  static const constexpr char *name = "FEED";

  enum struct Axes {
    row,
    // BEAM_OFFSET(1), POLARIZATION_TYPE(0), POL_RESPONSE(0), RECEPTOR_ANGLE(0)
    receptor,
    // POL_RESPONSE(1)
    receptor1,
    // BEAM_OFFSET(0)
    direction,
    // POSITION(0)
    position,

    last = position
  };

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(FEED, row);
MS_AXIS_NAME(FEED, receptor);
MS_AXIS_NAME(FEED, receptor1);
MS_AXIS_NAME(FEED, direction);
MS_AXIS_NAME(FEED, position);

template <>
struct MSTable<MSTables::FIELD> {
  static const constexpr char* name = "FIELD";

  enum struct Axes {
    row,
    // DELAY_DIR(1), PHASE_DIR(1), REFERENCE_DIR(1)
    polynomial,
    // DELAY_DIR(0), PHASE_DIR(0), REFERENCE_DIR(0)
    direction,

    last = direction
  };

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();  
};

MS_AXIS_NAME(FIELD, row);
MS_AXIS_NAME(FIELD, polynomial);
MS_AXIS_NAME(FIELD, direction);

template <>
struct MSTable<MSTables::FLAG_CMD> {
  static const constexpr char *name = "FLAG_CMD";

  enum struct Axes {
    row,

    last = row
  };

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(FLAG_CMD, row);

template <>
struct MSTable<MSTables::FREQ_OFFSET> {
  static const constexpr char* name = "FREQ_OFFSET";

  enum struct Axes {
    row,

    last = row
  };

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(FREQ_OFFSET, row);

template <>
struct MSTable<MSTables::HISTORY> {
  static const constexpr char* name = "HISTORY";

  enum struct Axes {
    row,
    // CLI_COMMAND(0)
    cli_command,
    // APP_PARAMS(0)
    app_param,

    last = app_param
  };

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;
  
  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(HISTORY, row);
MS_AXIS_NAME(HISTORY, cli_command);
MS_AXIS_NAME(HISTORY, app_param);

template <>
struct MSTable<MSTables::OBSERVATION> {
  static const constexpr char* name = "OBSERVATION";

  enum struct Axes {
    row,
    // TIME_RANGE(0)
    time_range,
    // LOG(0)
    log,
    // SCHEDULE(0)
    schedule,

    last = schedule
  };

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(OBSERVATION, row);
MS_AXIS_NAME(OBSERVATION, time_range);
MS_AXIS_NAME(OBSERVATION, log);
MS_AXIS_NAME(OBSERVATION, schedule);

template <>
struct MSTable<MSTables::POINTING> {
  static const constexpr char* name = "POINTING";

  enum struct Axes {
    row,
    // DIRECTION(1), TARGET(1), POINTING_OFFSET(1), SOURCE_OFFSET(1)
    polynomial,
    // DIRECTION(0), TARGET(0), POINTING_OFFSET(0), SOURCE_OFFSET(0), ENCODER(0)
    direction,

    last = direction
  };

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(POINTING, row);
MS_AXIS_NAME(POINTING, polynomial);
MS_AXIS_NAME(POINTING, direction);

template <>
struct MSTable<MSTables::POLARIZATION> {
  static const constexpr char* name = "POLARIZATION";

  enum struct Axes {
    row,
    // CORR_TYPE(0), CORR_PRODUCT(1)
    correlation,
    // CORR_PRODUCT(0)
    product,

    last = product
  };

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(POLARIZATION, row);
MS_AXIS_NAME(POLARIZATION, correlation);
MS_AXIS_NAME(POLARIZATION, product);

template <>
struct MSTable<MSTables::PROCESSOR> {
  static const constexpr char* name = "PROCESSOR";

  enum struct Axes {
    row,

    last = row
  };

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(PROCESSOR, row);


template <>
struct MSTable<MSTables::SOURCE> {
  static const constexpr char* name = "SOURCE";

  enum struct Axes {
    row,
    // DIRECTION(0)
    direction,
    // POSITION(0)
    position,
    // PROPER_MOTION(0)
    proper_motion,
    // TRANSITION(0), REST_FREQUENCY(0), SYSVEL(0)
    line,

    last = line
  };

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(SOURCE, row);
MS_AXIS_NAME(SOURCE, direction);
MS_AXIS_NAME(SOURCE, position);
MS_AXIS_NAME(SOURCE, proper_motion);
MS_AXIS_NAME(SOURCE, line);

template <>
struct MSTable<MSTables::SPECTRAL_WINDOW> {
  static const constexpr char* name = "SPECTRAL_WINDOW";

  enum struct Axes {
    row,
    // CHAN_FREQ(0), CHAN_WIDTH(0), EFFECTIVE_BW(0), RESOLUTION(0)
    channel,
    // ASSOC_SPW_ID(0), ASSOC_NATURE(0)
    assoc_spw,

    last = assoc_spw
  };

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(SPECTRAL_WINDOW, row);
MS_AXIS_NAME(SPECTRAL_WINDOW, channel);
MS_AXIS_NAME(SPECTRAL_WINDOW, assoc_spw);

template <>
struct MSTable<MSTables::STATE> {
  static const constexpr char* name = "STATE";

  enum struct Axes {
    row,

    last = row
  };

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(STATE, row);

template <>
struct MSTable<MSTables::SYSCAL> {
  static const constexpr char* name = "SYSCAL";

  enum struct Axes {
    row,
    // TCAL(0), TRX(0), TSKY(0), TSYS(0), TANT(0), TANT_TSYS(0), TCAL_SPECTRUM(0),
    // TRX_SPECTRUM(0), TSKY_SPECTRUM(0), TSYS_SPECTRUM(0), TANT_SPECTRUM(0),
    // TANT_TSYS_SPECTRUM(0)
    receptor,
    // TCAL_SPECTRUM(1), TRX_SPECTRUM(1), TSKY_SPECTRUM(1), TSYS_SPECTRUM(1),
    // TANT_SPECTRUM(1), TANT_TSYS_SPECTRUM(1)
    channel,

    last = channel
  };

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(SYSCAL, row);
MS_AXIS_NAME(SYSCAL, receptor);
MS_AXIS_NAME(SYSCAL, channel);

template <>
struct MSTable<MSTables::WEATHER> {
  static const constexpr char* name = "WEATHER";

  enum struct Axes {
    row,

    last = row
  };

  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;

  static const std::unordered_map<Axes, std::string>&
  axis_names();
};

MS_AXIS_NAME(WEATHER, row);

#undef MS_AXIS_NAME

} // end namespace legms

#endif // LEGMS_MS_TABLE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
