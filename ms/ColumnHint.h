#ifndef LEGMS_MS_COLUMN_HINT_H_
#define LEGMS_MS_COLUMN_HINT_H_

#include <array>

namespace legms {
namespace ms {

struct ColumnHint {
  static const unsigned MAX_RANK = 16;
  unsigned ms_value_rank;
  unsigned index_rank;
  std::array<unsigned, MAX_RANK> index_permutations;
};

} // end namespace ms
} // end namespace legms

#endif // LEGMS_MS_COLUMN_HINT_H_


// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
