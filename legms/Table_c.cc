#include "Table_c.h"
#include "c_util.h"

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

char**
table_column_names(table_t table) {
  auto names = unwrap(table)->column_names();
  char** result = static_cast<char**>(std::calloc(names.size(), sizeof(char*)));
  std::accumulate(
    names.begin(),
    names.end(),
    0,
    [result](auto& i, auto& nm) {
      result[i] = static_cast<char*>(std::calloc(nm.size(), sizeof(char)));
      std::strcpy(result[i], nm.c_str());
      return i + 1;
    });
  return result;
}

column_t
table_column(table_t table, const char* name) {
  return wrap(unwrap(table)->column(name));
}

const char *
table_min_rank_column_name(table_t table) {
  return unwrap(table)->min_rank_column_name().c_str();
}

const char *
table_max_rank_column_name(table_t table) {
  return unwrap(table)->max_rank_column_name().c_str();
}

#if 0
column_row_number_t
table_num_rows(table_t table) {
  return unwrap(table)->num_rows();
}

column_partition_t
table_row_block_partition(table_t table, size_t block_length) {
  return wrap(unwrap(table)->row_block_partition(block_length));
}

column_partition_t
table_all_rows_partition(table_t table) {
  return wrap(unwrap(table)->row_block_partition(std::nullopt));
}

column_partition_t
table_row_partition(
  table_t table,
  column_row_number_t** rowp,
  int include_unselected,
  int sorted_selections) {

  std::vector<std::vector<Column::row_number_t>> rpv;
  while (*rowp != NULL) {
    column_row_number_t* rp = *rowp;
    auto len = rp[0];
    ++rp;
    {
      std::vector<Column::row_number_t> v;
      std::transform(
        rp,
        rp + len,
        std::back_inserter(v),
        [](auto& n) { return n; });
      rpv.push_back(std::move(v));
    }
    ++rowp;
  }

  return
    wrap(
      unwrap(table)->row_partition(rpv, include_unselected, sorted_selections));
}
#endif

void
table_destroy(table_t table) {
  destroy(table);
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
