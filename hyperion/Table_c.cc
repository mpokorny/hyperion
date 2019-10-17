/*
 * Copyright 2019 Associated Universities, Inc. Washington DC, USA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <hyperion/c_util.h>
#include <hyperion/Table_c.h>
#include <hyperion/Table.h>
#ifdef HYPERION_USE_HDF5
# include <hyperion/hdf5.h>
#endif

#pragma GCC visibility push(default)
#include <legion/legion_c_util.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#pragma GCC visibility pop

using namespace hyperion;
using namespace hyperion::CObjectWrapper;

const Legion::FieldID metadata_fs[2] =
  {Table::METADATA_NAME_FID, Table::METADATA_AXES_UID_FID};
const Legion::FieldID axes_fs[1] = {Table::AXES_FID};

/* metadata field types: [char*, char*] */
const legion_field_id_t*
table_metadata_fs() {
  return metadata_fs;
}

/* axes field types: [int] */
const legion_field_id_t*
table_axes_fs() {
  return axes_fs;
}

int
table_is_empty(legion_context_t ctx, legion_runtime_t rt, table_t tab) {
  return
    unwrap(tab)
    .is_empty(
      Legion::CObjectWrapper::unwrap(ctx)->context(),
      Legion::CObjectWrapper::unwrap(rt));
}

HYPERION_API char**
table_column_names(
  legion_context_t ctx,
  legion_runtime_t rt,
  table_t tab) {

  std::vector<std::string> names =
    unwrap(tab)
    .column_names(
      Legion::CObjectWrapper::unwrap(ctx)->context(),
      Legion::CObjectWrapper::unwrap(rt));
  char **result = static_cast<char**>(std::calloc(names.size(), sizeof(char*)));
  char **nms = result;
  for (auto& name : names) {
    char* nm = static_cast<char*>(std::malloc(name.size() + 1));
    std::strcpy(nm, name.c_str());
    *nms++ = nm;
  }
  *nms++ = NULL;
  return result;
}

HYPERION_API column_t
table_column(
  legion_context_t ctx,
  legion_runtime_t rt,
  table_t tab,
  const char* name) {
  return
    wrap(
      unwrap(tab)
      .column(
        Legion::CObjectWrapper::unwrap(ctx)->context(),
        Legion::CObjectWrapper::unwrap(rt),
        name));
}

HYPERION_API void
table_destroy(
  legion_context_t ctx,
  legion_runtime_t rt,
  table_t tab,
  int destroy_columns) {
  unwrap(tab)
    .destroy(
      Legion::CObjectWrapper::unwrap(ctx)->context(),
      Legion::CObjectWrapper::unwrap(rt),
      destroy_columns);
}

// static std::unique_ptr<Table>
// to_table(
//   legion_context_t context,
//   legion_runtime_t runtime,
//   const table_t* table) {

//   return
//     TableGenArgs(*table)(
//       Legion::CObjectWrapper::unwrap(context)->context(),
//       Legion::CObjectWrapper::unwrap(runtime));
// }

// int
// table_has_column(const table_t* table, const char* name) {
//   int nomatch = 1;
//   for (unsigned i = 0; (nomatch != 0) && (i < table->num_columns); ++i)
//     nomatch =
//       std::strncmp(
//         table->columns[i].name,
//         name,
//         sizeof(table->columns[i].name));
//   return nomatch ? 0 : 1;
// }

// const column_t*
// table_column(const table_t* table, const char* name) {
//   for (unsigned i = 0; i < table->num_columns; ++i) {
//     if (std::strncmp(
//           table->columns[i].name,
//           name,
//           sizeof(table->columns[i].name)) == 0)
//       return &table->columns[i];
//   }
//   return NULL;
// }

// table_t
// table_reindexed(
//   legion_context_t context,
//   legion_runtime_t runtime,
//   const table_t* table,
//   unsigned num_axes,
//   const int* axes,
//   int allow_rows) {

//   std::vector<int> as(num_axes);
//   for (size_t i = 0; i < num_axes; ++i)
//     as[i] = axes[i];
//   return
//     to_table(context, runtime, table)
//     ->reindexed(as, allow_rows).get_result<TableGenArgs>().to_table_t();
// }

// void
// table_partition_by_value(
//   legion_context_t context,
//   legion_runtime_t runtime,
//   const table_t* table,
//   unsigned num_axes,
//   const int* axes,
//   char** col_names,
//   legion_logical_partition_t* col_partitions) {

//   std::vector<int> as(num_axes);
//   for (size_t i = 0; i < num_axes; ++i)
//     as[i] = axes[i];

//   auto t = to_table(context, runtime, table);
//   auto fps = t->partition_by_value(t->context(), t->runtime(), as);
//   // NB: the following blocks
//   std::accumulate(
//     fps.begin(),
//     fps.end(),
//     0u,
//     [&t, col_names, col_partitions](unsigned i, auto& n_f) {
//       auto& [n, f] = n_f;
//       col_names[i] = static_cast<char*>(std::malloc(n.size() + 1));
//       std::strcpy(col_names[i], n.c_str());
//       auto p = f.template get_result<Legion::IndexPartition>();
//       auto lp =
//         t->runtime()->get_logical_partition(
//           t->context(),
//           t->column(n)->logical_region(),
//           p);
//       col_partitions[i] = Legion::CObjectWrapper::wrap(lp);
//       return i + 1;
//     });
// }

// #ifdef HYPERION_USE_CASACORE
// table_t
// table_from_ms(
//   legion_context_t context,
//   legion_runtime_t runtime,
//   const char* path,
//   const char** column_selections) {

//   std::unordered_set<std::string> cs;
//   while (*column_selections != NULL) {
//     cs.insert(*column_selections);
//     ++column_selections;
//   }

//   return
//     Table::from_ms(
//       Legion::CObjectWrapper::unwrap(context)->context(),
//       Legion::CObjectWrapper::unwrap(runtime),
//       path,
//       cs)
//     ->generator_args().to_table_t();
// }
// #endif // HYPERION_USE_CASACORE

// #ifdef HYPERION_USE_HDF5
// static char**
// struset2strv(const std::unordered_set<std::string>& us) {
//   char** result = (char**)std::calloc(us.size() + 1, sizeof(char*));
//   if (result)
//     std::accumulate(
//       us.begin(),
//       us.end(),
//       0,
//       [result](const size_t& i, const std::string& p) {
//         result[i] = (char*)std::malloc(p.size() + 1 * sizeof(char));
//         std::strcpy(result[i], p.c_str());
//         return i + 1;
//       });
//   return result;
// }

// char **
// tables_in_h5(const char* path) {
//   auto tblpaths = hdf5::get_table_paths(path);
//   return struset2strv(tblpaths);
// }

// char **
// columns_in_h5(const char* path, const char* table_path) {
//   auto colnames = hdf5::get_column_names(path, table_path);
//   return struset2strv(colnames);
// }

// table_t
// table_from_h5(
//   legion_context_t context,
//   legion_runtime_t runtime,
//   const char* path,
//   const char* table_path,
//   unsigned num_column_selections,
//   const char** column_selections) {

//   std::unordered_set<std::string> colnames;
//   for (unsigned i = 0; i < num_column_selections; ++i)
//     colnames.insert(*(column_selections + i));
//   Legion::Context ctx = Legion::CObjectWrapper::unwrap(context)->context();
//   Legion::Runtime* rt = Legion::CObjectWrapper::unwrap(runtime);
//   auto tbgen = hdf5::init_table(ctx, rt, path, table_path, colnames);
//   return tbgen.value_or(TableGenArgs()).to_table_t();
// }

// void
// table_keyword_paths(
//   legion_context_t context,
//   legion_runtime_t runtime,
//   const table_t* table,
//   char** keywords,
//   char** paths) {

//   auto kwps = hdf5::get_table_keyword_paths(*to_table(context, runtime, table));
//   std::accumulate(
//     kwps.begin(),
//     kwps.end(),
//     0u,
//     [keywords, paths](unsigned i, auto& kwp) {
//       auto& [kw, pth] = kwp;
//       char *k = (char*)std::malloc((kw.size() + 1) * sizeof(char));
//       std::strcpy(k, kw.c_str());
//       char *p = (char*)std::malloc((pth.size() + 1) * sizeof(char));
//       std::strcpy(p, pth.c_str());
//       keywords[i] = k;
//       paths[i] = p;
//       return i + 1;
//     });
// }

// void
// table_column_value_path(
//   legion_context_t context,
//   legion_runtime_t runtime,
//   const table_t* table,
//   const char* colname,
//   char** path) {

//   auto pth =
//     hdf5::get_table_column_value_path(
//       *to_table(context, runtime, table),
//       colname);
//   *path = (char*)std::malloc((pth.size() + 1) * sizeof(char));
//   std::strcpy(*path, pth.c_str());
// }

// void
// table_column_keyword_paths(
//   legion_context_t context,
//   legion_runtime_t runtime,
//   const table_t* table,
//   const char* colname,
//   char** keywords,
//   char** paths) {
//   auto pths =
//     hdf5::get_table_column_keyword_paths(
//       *to_table(context, runtime, table),
//       colname);
//   std::accumulate(
//     pths.begin(),
//     pths.end(),
//     0u,
//     [keywords, paths](unsigned i, auto& kwp) {
//       auto& [kw, pth] = kwp;
//       char *k = (char*)std::malloc((kw.size() + 1) * sizeof(char));
//       std::strcpy(k, kw.c_str());
//       char *p = (char*)std::malloc((pth.size() + 1) * sizeof(char));
//       std::strcpy(p, pth.c_str());
//       keywords[i] = k;
//       paths[i] = p;
//       return i + 1;
//     });
// }

// #endif

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
