#include "Measures.h"

#ifdef LEGMS_USE_CASACORE

#define MCLASS_NAME(M) \
  const std::string MClassT<M>::name = MClassT<M>::type::showMe();
FOREACH_MCLASS(MCLASS_NAME)
#undef MCLASS_NAME

#endif // LEGMS_USE_CASACORE

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
