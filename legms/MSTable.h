#ifndef LEGMS_MS_TABLE_H_
#define LEGMS_MS_TABLE_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "legms.h"
#include "utility.h"
#include "Table.h"
#include "TableBuilder.h"

#if USE_HDF5
# include <hdf5.h>
#endif // USE_HDF5

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

#define LEGMS_FOREACH_MSTABLE(FUNC) \
  FUNC(MSTables::MAIN)              \
  FUNC(MSTables::ANTENNA) \
  FUNC(MSTables::DATA_DESCRIPTION) \
  FUNC(MSTables::DOPPLER) \
  FUNC(MSTables::FEED) \
  FUNC(MSTables::FIELD) \
  FUNC(MSTables::FLAG_CMD) \
  FUNC(MSTables::FREQ_OFFSET) \
  FUNC(MSTables::HISTORY) \
  FUNC(MSTables::OBSERVATION) \
  FUNC(MSTables::POINTING) \
  FUNC(MSTables::POLARIZATION) \
  FUNC(MSTables::PROCESSOR) \
  FUNC(MSTables::SOURCE) \
  FUNC(MSTables::SPECTRAL_WINDOW) \
  FUNC(MSTables::STATE) \
  FUNC(MSTables::SYSCAL) \
  FUNC(MSTables::WEATHER)

template <MSTables T>
struct MSTable {
  static const char* name;
  enum struct Axes;
  static const std::unordered_map<std::string, std::vector<Axes>>
  element_axes;
  static const std::unordered_map<Axes, std::string>& axis_names();
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

#define MS_TABLE_AXES_UID(T)                          \
  template <>                                         \
  struct AxesUID<MSTable<MSTables::T>::Axes> {        \
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

MS_TABLE_AXES_UID(MAIN);

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

MS_TABLE_AXES_UID(ANTENNA);

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

MS_TABLE_AXES_UID(DATA_DESCRIPTION);

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

MS_TABLE_AXES_UID(DOPPLER);

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

MS_TABLE_AXES_UID(FEED);

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

MS_TABLE_AXES_UID(FIELD);

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

MS_TABLE_AXES_UID(FLAG_CMD);

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

MS_TABLE_AXES_UID(FREQ_OFFSET);

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

MS_TABLE_AXES_UID(HISTORY);

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

MS_TABLE_AXES_UID(OBSERVATION);

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

MS_TABLE_AXES_UID(POINTING);

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

MS_TABLE_AXES_UID(POLARIZATION);

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

MS_TABLE_AXES_UID(PROCESSOR);

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

MS_TABLE_AXES_UID(SOURCE);

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

MS_TABLE_AXES_UID(SPECTRAL_WINDOW);

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

MS_TABLE_AXES_UID(STATE);

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

MS_TABLE_AXES_UID(SYSCAL);

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

MS_TABLE_AXES_UID(WEATHER);

#undef MS_AXIS_NAME

#undef MS_TABLE_AXES_UID

#if USE_CASACORE

struct TableBuilder {

  template <MSTables T>
  static TableBuilderT<typename MSTable<T>::Axes>
  from_ms(
    const std::experimental::filesystem::path& path,
    const std::unordered_set<std::string>& column_selections) {

    return
      TableBuilderT<typename MSTable<T>::Axes>::from_casacore_table(
        ((path.filename() == MSTable<T>::name)
         ? path
         : (path / MSTable<T>::name)),
        column_selections,
        MSTable<T>::element_axes);
  }
};

template <MSTables T>
static std::unique_ptr<TableT<typename MSTable<T>::Axes>>
from_ms(
  Legion::Context ctx,
  Legion::Runtime* runtime,
  const std::experimental::filesystem::path& path,
  const std::unordered_set<std::string>& column_selections) {

  typedef typename MSTable<T>::Axes D;
  auto builder = TableBuilder::from_ms<T>(path, column_selections);
  return
    std::make_unique<TableT<typename MSTable<T>::Axes>>(
      ctx,
      runtime,
      builder.name(),
      std::vector<int>{static_cast<int>(D::ROW)},
      builder.column_generators(),
      builder.keywords());
}

#endif // USE_CASACORE

#if USE_HDF5

template <MSTables T>
hid_t
h5_axes_datatype() {
  hid_t result = H5Tenum_create(H5T_NATIVE_UCHAR);
  typedef typename MSTable<T>::Axes Axes;
  for (auto a = static_cast<unsigned char>(Axes::ROW);
       a <= static_cast<unsigned char>(Axes::last);
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

template <MSTables T>
static std::optional<typename MSTable<T>::Axes>
column_is_axis(
  const std::string& colname,
  const std::vector<typename MSTable<T>::Axes>& axes) {
  auto axis_names = MSTable<T>::axis_names();
  auto colax =
    find(
      axes.begin(),
      axes.end(),
      [&axis_names, &colname](auto& ax) {
        return colname == axis_names.at(ax);
      });
  return ((colax != axes.end()) ? *colax : std::nullopt);
}

template <MSTables T>
static std::optional<Legion::Future/*TableGenArgs*/>
reindexed(
  const TableT<typename MSTable<T>::Axes>* table,
  const std::vector<typename MSTable<T>::Axes>& axes,
  bool allow_rows = true) {

  typedef typename MSTable<T>::Axes D;

  // 'allow_rows' is intended to support the case where the reindexing may not
  // result in a single value in a column per aggregate index, necessitating the
  // maintenance of a row index. A value of 'true' for this argument is always
  // safe, but may result in a degenerate axis when an aggregate index always
  // identifies a single value in a column. If the value is 'false' and a
  // non-degenerate axis is required by the reindexing, this method will return
  // an empty value. TODO: remove degenerate axes after the fact, and do that
  // automatically in this method, which would allow us to remove the
  // 'allow_rows' argument.

  // can only reindex along an axis if table has a column with the associated
  // name
  //
  // TODO: add support for index columns that already exist in the table
  if ((table->index_axes().size() > 1)
      || (table->index_axes().back() != static_cast<int>(D::ROW)))
    return std::nullopt;

  // for every column in table, determine which axes need indexing
  std::unordered_map<std::string, std::vector<D>> col_reindex_axes;
  std::transform(
    table->column_names().begin(),
    table->column_names().end(),
    std::inserter(col_reindex_axes, col_reindex_axes.end()),
    [table, &axes](auto& nm) {
      std::vector<D> ax;
      auto col_axes = table->columnT(nm)->axesT();
      // skip the column if it does not have a "row" axis
      if (col_axes.back() == D::ROW) {
        // if column is a reindexing axis, reindexing depends only on itself
        auto myaxis = column_is_axis(nm, axes);
        if (myaxis) {
          ax.push_back(myaxis.value());
        } else {
          // select those axes in "axes" that are not already an axis of the
          // column
          std::for_each(
            axes.begin(),
            axes.end(),
            [&col_axes, &ax](auto& d) {
              if (find(col_axes.begin(), col_axes.end(), d) == col_axes.end())
                ax.push_back(d);
            });
        }
      }
      return std::pair(nm, std::move(ax));
    });

  // index associated columns; the Future in "index_cols" below contains a
  // ColumnGenArgs of a LogicalRegion with two fields: at Column::value_fid, the
  // column values (sorted in ascending order); and at Column::value_fid +
  // IndexColumnTask::rows_fid, a sorted vector of DomainPoints in the original
  // column.
  std::unordered_map<D, Legion::Future> index_cols;
  std::for_each(
    col_reindex_axes.begin(),
    col_reindex_axes.end(),
    [table, &index_cols](auto& nm_ds) {
      const std::vector<D>& ds = std::get<1>(nm_ds);
      std::for_each(
        ds.begin(),
        ds.end(),
        [table, &index_cols](auto& d) {
          if (index_cols.count(d) == 0) {
            auto col = table->columnT(D::axis_names().at(d));
            IndexColumnTask task(col, static_cast<int>(d));
            index_cols[d] = task.dispatch(table->context(), table->runtime());
          }
        });
    });

  // do reindexing of columns
  std::vector<Legion::Future> reindexed;
  std::transform(
    col_reindex_axes.begin(),
    col_reindex_axes.end(),
    std::back_inserter(reindexed),
    [table, &index_cols, &allow_rows](auto& nm_ds) {
      auto& [nm, ds] = nm_ds;
      // if this column is an index column, we've already launched a task to
      // create its logical region, so we can use that
      if (ds.size() == 1 && index_cols.count(ds[0]) > 0)
        return index_cols.at(ds[0]);

      // create reindexing task launcher
      // TODO: start intermediary task dependent on Futures of index columns
      std::vector<std::shared_ptr<Column>> ixcols;
      std::vector<int> index_axes;
      for (auto d : ds) {
        ixcols.push_back(
          index_cols.at(d)
          .template get_result<ColumnGenArgs>()
          .operator()<T>(table->context(), table->runtime()));
        index_axes.push_back(static_cast<int>(d));
      }
      auto col = table->columnT(nm);
      auto col_axes = col->axesT();
      auto row_axis_offset =
        std::distance(
          col_axes.begin(),
          find(col_axes.begin(), col_axes.end(), D::ROW));
      ReindexColumnTask task(
        col,
        row_axis_offset,
        ixcols,
        index_axes,
        allow_rows);
      return task.dispatch(table->context(), table->runtime());
    });

  // launch task that creates the reindexed table
  std::vector<int> index_axes;
  std::transform(
    axes.begin(),
    axes.end(),
    std::back_inserter(index_axes),
    [](auto& d) {
      return static_cast<int>(d);
    });
  if (allow_rows)
    index_axes.push_back(static_cast<int>(D::ROW));
  ReindexedTableTask
    task(
      table->name(),
      table->axes_uid(),
      index_axes,
      table->keywords_region(),
      reindexed);
  return task.dispatch(table->context(), table->runtime());
}

} // end namespace legms

#endif // LEGMS_MS_TABLE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
