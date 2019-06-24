#include "c_util.h"
#include "Table_c.h"
#include "Table.h"
#ifdef USE_HDF5
# include "legms_hdf5.h"
#endif

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

#ifdef USE_HDF5
static char**
struset2strv(const std::unordered_set<std::string>& us) {
  char** result = (char**)std::calloc(us.size() + 1, sizeof(char*));
  if (result)
    std::accumulate(
      us.begin(),
      us.end(),
      0,
      [result](const size_t& i, const std::string& p) {
        result[i] = (char*)std::malloc(p.size() + 1 * sizeof(char));
        std::strcpy(result[i], p.c_str());
        return i + 1;
      });
  return result;
}

char **
tables_in_h5(const char* path) {
  auto tblpaths = hdf5::get_table_paths(path);
  return struset2strv(tblpaths);
}

char **
columns_in_h5(const char* path, const char* table_path) {
  auto colnames = hdf5::get_column_names(path, table_path);
  return struset2strv(colnames);
}

table_t
table_from_h5(
  legion_context_t context,
  legion_runtime_t runtime,
  const char* path,
  const char* table_path,
  // NULL-terminated array of string pointers
  const char** column_selections) {

  std::unordered_set<std::string> colnames;
  while (*column_selections != NULL) {
    colnames.insert(*column_selections);
    ++column_selections;
  }
  Legion::Context ctx = Legion::CObjectWrapper::unwrap(context)->context();
  Legion::Runtime* rt = Legion::CObjectWrapper::unwrap(runtime);
  auto tbgen = hdf5::init_table(ctx, rt, path, table_path, colnames);
  return
    wrap(
      tbgen
      ? tbgen.value().operator()(ctx, rt)
      : std::make_unique<Table>(ctx, rt, table_path, "", std::vector<int>()));
}
#endif

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
