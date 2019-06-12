#include "utility_c.h"
#include "utility.h"
#include "legion/legion_c_util.h"

void
legms_preregister_all() {
  legms::preregister_all();
}

void
legms_register_tasks(legion_runtime_t runtime) {
  legms::register_tasks(Legion::CObjectWrapper::unwrap(runtime));
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
