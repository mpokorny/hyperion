#ifndef LEGMS_LEGMS_H_
#define LEGMS_LEGMS_H_

#pragma GCC visibility push(default)
#include "legion.h"
#pragma GCC visibility pop

#include "legms_config.h"

#define LEGMS_API __attribute__((visibility("default")))
#define LEGMS_LOCAL __attribute__((visibility("hidden")))

#endif // LEGMS_LEGMS_H_

// Local Variables:
// mode: c
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
