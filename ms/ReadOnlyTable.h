#ifndef LEGMS_MS_READ_ONLY_TABLE_H_
#define LEGMS_MS_READ_ONLY_TABLE_H_

#include <experimental/filesystem>

#include <casacore/casa/aipstype.h>
#include <casacore/casa/BasicSL/String.h>
#include <casacore/tables/Tables.h>

#include "legion.h"
#include "Table.h"

namespace legms {
namespace ms {

template <typename T>
class ReadOnlyTable
  : public Table {
public:

  ReadOnlyTable(const std::experimental::filesystem::path& path)
    : Table(T::builder(path / T::table_name))
    , m_path(path / T::table_name) {
  }

  const std::experimental::filesystem::path&
  path() const {
    return m_path;
  }

private:

  std::experimental::filesystem::path m_path;
};

class ROTable
  : public Table {
public:

  ROTable(
    const std::experimental::filesystem::path& path)
    : Table(builder(path))
    , m_ms_path(path.parent_path())
    , m_table_name(path.filename()) {
  }

  const std::experimental::filesystem::path&
  ms_path() const {
    return m_ms_path;
  }

  const std::string&
  table_name() const {
    return m_table_name;
  }

  std::experimental::filesystem::path
  table_path() const {
    return m_ms_path / m_table_name;
  }

private:

  static TableBuilder
  builder(
    const std::experimental::filesystem::path& path);

  struct SizeArgs {
    std::shared_ptr<casacore::TableColumn> tcol;
    unsigned row;
    casacore::IPosition shape;
  };

  template <int DIM>
  static std::array<size_t, DIM>
  size(const std::any& args) {
    std::array<size_t, DIM> result;
    auto sa = std::any_cast<SizeArgs>(args);
    const casacore::IPosition& shape =
      (sa.tcol ? sa.tcol->shape(sa.row) : sa.shape);
    assert(shape.size() == DIM);
    shape.copy(result.begin());
    return result;
  }

  std::experimental::filesystem::path m_ms_path;
  std::string m_table_name;
};

}
}
#endif // LEGMS_MS_READ_ONLY_TABLE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
