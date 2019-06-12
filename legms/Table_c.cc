#include "c_util.h"
#include "Table_c.h"
#include "Table.h"

#include "legion/legion_c_util.h"

#include <cstdlib>
#include <cstring>
#include <memory>

using namespace legms;
using namespace legms::CObjectWrapper;

const char*
legms_table_name(legms_table_t table) {
  return unwrap(table)->name().c_str();
}

int
legms_table_is_empty(legms_table_t table) {
  return unwrap(table)->is_empty();
}

unsigned
legms_table_num_columns(legms_table_t table) {
  return unwrap(table)->column_names().size();
}

void
legms_table_column_names(legms_table_t table, char** names) {
  auto cols = unwrap(table)->column_names();
  std::accumulate(
    cols.begin(),
    cols.end(),
    0u,
    [names](unsigned i, const auto& nm) {
      names[i] =
        static_cast<char*>(std::malloc((nm.size() + 1) * sizeof(char)));
      std::strcpy(names[i], nm.c_str());
      return i + 1;
    });
}

int
legms_table_has_column(legms_table_t table, const char* name) {
  return unwrap(table)->has_column(name);
}

legms_column_t
legms_table_column(legms_table_t table, const char* name) {
  return wrap(unwrap(table)->column(name));
}

const char *
legms_table_min_rank_column_name(legms_table_t table) {
  return unwrap(table)->min_rank_column_name().c_str();
}

const char *
legms_table_max_rank_column_name(legms_table_t table) {
  return unwrap(table)->max_rank_column_name().c_str();
}

const char*
legms_table_axes_uid(legms_table_t table) {
  return unwrap(table)->axes_uid();
}

const int*
legms_table_index_axes(legms_table_t table) {
  return unwrap(table)->index_axes().data();
}

void
legms_table_destroy(legms_table_t table) {
  destroy(table);
}

#ifdef USE_CASACORE
legms_table_t
legms_table_from_ms(
  legion_context_t context,
  legion_runtime_t runtime,
  const char* path,
  const char** column_selections) {

  std::unordered_set<std::string> cs;
  while (*column_selections != NULL) {
    cs.insert(*column_selections);
    ++column_selections;
  }

  return
    wrap(
      Table::from_ms(
        Legion::CObjectWrapper::unwrap(context)->context(),
        Legion::CObjectWrapper::unwrap(runtime),
        path,
        cs));
}
#endif // USE_CASACORE

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
