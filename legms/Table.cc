#include <algorithm>
#include "Table.h"

using namespace legms;
using namespace legms::ms;
using namespace std;

std::unique_ptr<ColumnPartition>
Table::row_block_partition(std::optional<size_t> block_length) const {
  auto nr = num_rows();
  auto bl = block_length.value_or(nr);
  std::vector<std::vector<Column::row_number_t>> rowp((nr + bl - 1) / bl);
  for (Column::row_number_t i = 0; i < nr; ++i)
    rowp[i / bl].push_back(i);
  return row_partition(rowp, false, true);
}

optional<Legion::coord_t>
Table::find_color(
  const vector<vector<Column::row_number_t>>& rowp,
  Column::row_number_t rn,
  bool sorted_selections) {

  optional<Legion::coord_t> result;
  if (sorted_selections) {
    for (size_t i = 0; !result && i < rowp.size(); ++i) {
      auto rns = rowp[i];
      if (binary_search(rns.begin(), rns.end(), rn))
        result = i;
    }
  } else {
    for (size_t i = 0; !result && i < rowp.size(); ++i) {
      auto rns = rowp[i];
      if (find(rns.begin(), rns.end(), rn) != rns.end())
        result = i;
    }
  }
  return result;
}

std::unique_ptr<Table>
Table::from_ms(
  Legion::Context ctx,
  Legion::Runtime* runtime,
  const std::experimental::filesystem::path& path,
  const std::unordered_set<std::string>& column_selections) {

  std::string table_name = path.filename();

#define FROM_MS_TABLE(N)                                \
  do {                                                  \
    if (table_name == MSTable<MSTables::N>::name)       \
      return legms::ms:: template from_ms<MSTables::N>( \
        ctx, runtime, path, column_selections);         \
  } while (0)

  FROM_MS_TABLE(MAIN);
  FROM_MS_TABLE(ANTENNA);
  FROM_MS_TABLE(DATA_DESCRIPTION);
  FROM_MS_TABLE(DOPPLER);
  FROM_MS_TABLE(FEED);
  FROM_MS_TABLE(FIELD);
  FROM_MS_TABLE(FLAG_CMD);
  FROM_MS_TABLE(FREQ_OFFSET);
  FROM_MS_TABLE(HISTORY);
  FROM_MS_TABLE(OBSERVATION);
  FROM_MS_TABLE(POINTING);
  FROM_MS_TABLE(POLARIZATION);
  FROM_MS_TABLE(PROCESSOR);
  FROM_MS_TABLE(SOURCE);
  FROM_MS_TABLE(SPECTRAL_WINDOW);
  FROM_MS_TABLE(STATE);
  FROM_MS_TABLE(SYSCAL);
  FROM_MS_TABLE(WEATHER);
  // try to read as main table
  return
    legms::ms:: template from_ms<MSTables::MAIN>(
      ctx,
      runtime,
      path,
      column_selections);

#undef FROM_MS_TABLE
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
