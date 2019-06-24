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
table_name(table_t table) {
  return unwrap(table)->name().c_str();
}

int
table_is_empty(table_t table) {
  return unwrap(table)->is_empty();
}

unsigned
table_num_columns(table_t table) {
  return unwrap(table)->column_names().size();
}

void
table_column_names(table_t table, char** names) {
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
table_has_column(table_t table, const char* name) {
  return unwrap(table)->has_column(name);
}

column_t
table_column(table_t table, const char* name) {
  return wrap(unwrap(table)->column(name));
}

const char *
table_min_rank_column_name(table_t table) {
  auto cn = unwrap(table)->min_rank_column_name();
  return cn ? cn.value().c_str() : NULL;
}

const char *
table_max_rank_column_name(table_t table) {
  auto cn = unwrap(table)->max_rank_column_name();
  return cn ? cn.value().c_str() : NULL;
}

const char*
table_axes_uid(table_t table) {
  return unwrap(table)->axes_uid().c_str();
}

unsigned
table_num_index_axes(table_t table) {
  return unwrap(table)->index_axes().size();
}

const int*
table_index_axes(table_t table) {
  return unwrap(table)->index_axes().data();
}

void
table_destroy(table_t table) {
  destroy(table);
}

table_t
table_reindexed(
  table_t table,
  const int* axes,
  unsigned num_axes,
  int allow_rows) {

  std::vector<int> as(num_axes);
  for (size_t i = 0; i < num_axes; ++i)
    as[i] = axes[i];
  return
    wrap(
      unwrap(table)
      ->reindexed(as, allow_rows)
      .get_result<TableGenArgs>()
      .operator()(unwrap(table)->context(), unwrap(table)->runtime()));
}

#ifdef USE_CASACORE
table_t
table_from_ms(
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
