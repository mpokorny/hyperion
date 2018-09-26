#ifndef LEGMS_MS_SPECTRAL_WINDOW_TABLE_H_
#define LEGMS_MS_SPECTRAL_WINDOW_TABLE_H_

#include <experimental/filesystem>

#include <casacore/casa/aipstype.h>
#include <casacore/casa/BasicSL/String.h>
#include <casacore/tables/Tables.h>

#include "IndexTree.h"
#include "TableBuilder.h"
#include "Table.h"

namespace legms {
namespace ms {

class SpectralWindowTable
  : public Table {
public:

  SpectralWindowTable(const std::experimental::filesystem::path& path)
    : Table(builder(path))
    , m_path(path / table_name) {
  }

  const std::experimental::filesystem::path&
  path() const {
    return m_path;
  }

private:

  static TableBuilder
  builder(const std::experimental::filesystem::path& path) {

    casacore::Table table(
      casacore::String(path / table_name),
      casacore::TableLock::NoLocking);
    auto nrow = table.nrow();
    auto tdesc = table.tableDesc();

    TableBuilder result(table_name, IndexTreeL(1));

    // required columns
    result.add_scalar_column<casacore::Int>("NUM_CHAN");
    result.add_scalar_column<casacore::String>("NAME");
    result.add_scalar_column<casacore::Double>("REF_FREQUENCY");
    auto num_chan = [](const std::any& arg) -> std::array<size_t, 1> {
      return { static_cast<size_t>(std::any_cast<casacore::Int>(arg)) };
    };
    result.add_array_column<1, casacore::Double>("CHAN_FREQ", num_chan);
    result.add_array_column<1, casacore::Double>("CHAN_WIDTH", num_chan);
    result.add_scalar_column<casacore::Int>("MEAS_FREQ_REF");
    result.add_array_column<1, casacore::Double>("EFFECTIVE_BW", num_chan);
    result.add_array_column<1, casacore::Double>("RESOLUTION", num_chan);
    result.add_scalar_column<casacore::Double>("TOTAL_BANDWIDTH");
    result.add_scalar_column<casacore::Int>("NET_SIDEBAND");
    result.add_scalar_column<casacore::Int>("IF_CONV_CHAIN");
    result.add_scalar_column<casacore::Int>("FREQ_GROUP");
    result.add_scalar_column<casacore::String>("FREQ_GROUP_NAME");
    result.add_scalar_column<casacore::Bool>("FLAG_ROW");

    // optional columns
    if (tdesc.isColumn("BBC_NO"))
      result.add_scalar_column<casacore::Int>("BBC_NO");
    if (tdesc.isColumn("BBC_SIDEBAND"))
      result.add_scalar_column<casacore::Int>("BBC_SIDEBAND");
    if (tdesc.isColumn("RECEIVER_ID"))
      result.add_scalar_column<casacore::Int>("RECEIVER_ID");
    if (tdesc.isColumn("DOPPLER_ID"))
      result.add_scalar_column<casacore::Int>("DOPPLER_ID");
    if (tdesc.isColumn("ASSOC_SPW_ID"))
      result.add_scalar_column<std::vector<casacore::Int>>("ASSOC_SPW_ID");
    if (tdesc.isColumn("ASSOC_NATURE"))
      result.add_scalar_column<std::vector<casacore::String>>("ASSOC_NATURE");

    casacore::ScalarColumn<casacore::Int> col(table, "NUM_CHAN");
    std::unordered_map<std::string, std::any> args;
    for (casacore::uInt i = 0; i < nrow; ++i) {
      auto nc = col.get(i);
      args["CHAN_FREQ"] = nc;
      args["CHAN_WIDTH"] = nc;
      args["EFFECTIVE_BW"] = nc;
      args["RESOLUTION"] = nc;
      result.add_row(args);
    }
    return result;
  }

  constexpr static const char* table_name = "SPECTRAL_WINDOW";

  std::experimental::filesystem::path m_path;
};

}
}
#endif // LEGMS_MS_SPECTRAL_WINDOW_TABLE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
