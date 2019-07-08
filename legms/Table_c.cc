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

static std::unique_ptr<Table>
to_table(
  legion_context_t context,
  legion_runtime_t runtime,
  const table_t* table) {

  return
    TableGenArgs(*table)(
      Legion::CObjectWrapper::unwrap(context)->context(),
      Legion::CObjectWrapper::unwrap(runtime));
}

int
table_has_column(const table_t* table, const char* name) {
  int nomatch = 1;
  for (unsigned i = 0; (nomatch != 0) && (i < table->num_columns); ++i)
    nomatch =
      std::strncmp(
        table->columns[i].name,
        name,
        sizeof(table->columns[i].name));
  return nomatch ? 0 : 1;
}

const column_t*
table_column(const table_t* table, const char* name) {
  for (unsigned i = 0; i < table->num_columns; ++i)
    if (std::strncmp(
          table->columns[i].name,
          name,
          sizeof(table->columns[i].name) == 0))
      return &table->columns[i];
  return NULL;
}

table_t
table_reindexed(
  legion_context_t context,
  legion_runtime_t runtime,
  const table_t* table,
  unsigned num_axes,
  const int* axes,
  int allow_rows) {

  std::vector<int> as(num_axes);
  for (size_t i = 0; i < num_axes; ++i)
    as[i] = axes[i];
  return
    to_table(context, runtime, table)
    ->reindexed(as, allow_rows).get_result<TableGenArgs>().to_table_t();
}

void
table_partition_by_value(
  legion_context_t context,
  legion_runtime_t runtime,
  const table_t* table,
  unsigned num_axes,
  const int* axes,
  /* length of col_names and col_partitions arrays must equal value of
   * table_num_columns() */
  char** col_names,
  legion_logical_partition_t* col_partitions) {

  std::vector<int> as(num_axes);
  for (size_t i = 0; i < num_axes; ++i)
    as[i] = axes[i];

  auto t = to_table(context, runtime, table);
  auto fps = t->partition_by_value(t->context(), t->runtime(), as);
  // NB: the following blocks
  std::accumulate(
    fps.begin(),
    fps.end(),
    0u,
    [&t, col_names, col_partitions](unsigned i, auto& n_f) {
      auto& [n, f] = n_f;
      col_names[i] = static_cast<char*>(std::malloc(n.size() + 1));
      std::strcpy(col_names[i], n.c_str());
      auto p = f.template get_result<Legion::IndexPartition>();
      auto lp =
        t->runtime()->get_logical_partition(
          t->context(),
          t->column(n)->logical_region(),
          p);
      col_partitions[i] = Legion::CObjectWrapper::wrap(lp);
      return i + 1;
    });
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
    Table::from_ms(
      Legion::CObjectWrapper::unwrap(context)->context(),
      Legion::CObjectWrapper::unwrap(runtime),
      path,
      cs)
    ->generator_args().to_table_t();
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
  unsigned num_column_selections,
  const char** column_selections) {

  std::unordered_set<std::string> colnames;
  for (unsigned i = 0; i < num_column_selections; ++i)
    colnames.insert(*(column_selections + i));
  Legion::Context ctx = Legion::CObjectWrapper::unwrap(context)->context();
  Legion::Runtime* rt = Legion::CObjectWrapper::unwrap(runtime);
  auto tbgen = hdf5::init_table(ctx, rt, path, table_path, colnames);
  return tbgen.value_or(TableGenArgs()).to_table_t();
}

void
table_keyword_paths(
  legion_context_t context,
  legion_runtime_t runtime,
  const table_t* table,
  char** keywords,
  char** paths) {

  auto kwps = hdf5::get_table_keyword_paths(*to_table(context, runtime, table));
  std::accumulate(
    kwps.begin(),
    kwps.end(),
    0u,
    [keywords, paths](unsigned i, auto& kwp) {
      auto& [kw, pth] = kwp;
      char *k = (char*)std::malloc((kw.size() + 1) * sizeof(char));
      std::strcpy(k, kw.c_str());
      char *p = (char*)std::malloc((pth.size() + 1) * sizeof(char));
      std::strcpy(p, pth.c_str());
      keywords[i] = k;
      paths[i] = p;
      return i + 1;
    });
}

void
table_column_value_path(
  legion_context_t context,
  legion_runtime_t runtime,
  const table_t* table,
  const char* colname,
  char** path) {

  auto pth =
    hdf5::get_table_column_value_path(
      *to_table(context, runtime, table),
      colname);
  *path = (char*)std::malloc((pth.size() + 1) * sizeof(char));
  std::strcpy(*path, pth.c_str());
}

void
table_column_keyword_paths(
  legion_context_t context,
  legion_runtime_t runtime,
  const table_t* table,
  const char* colname,
  char** keywords,
  char** paths) {
  auto pths =
    hdf5::get_table_column_keyword_paths(
      *to_table(context, runtime, table),
      colname);
  std::accumulate(
    pths.begin(),
    pths.end(),
    0u,
    [keywords, paths](unsigned i, auto& kwp) {
      auto& [kw, pth] = kwp;
      char *k = (char*)std::malloc((kw.size() + 1) * sizeof(char));
      std::strcpy(k, kw.c_str());
      char *p = (char*)std::malloc((pth.size() + 1) * sizeof(char));
      std::strcpy(p, pth.c_str());
      keywords[i] = k;
      paths[i] = p;
      return i + 1;
    });
}

#endif

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
