#include <legms/utility_c.h>
#include <legms/utility.h>

#pragma GCC visibility push(default)
#include <legion/legion_c_util.h>
#pragma GCC visibility pop

void
preregister_all() {
  legms::preregister_all();
}

void
register_tasks(legion_context_t context, legion_runtime_t runtime) {
  legms::register_tasks(
    Legion::CObjectWrapper::unwrap(context)->context(),
    Legion::CObjectWrapper::unwrap(runtime));
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
