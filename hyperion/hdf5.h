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
#ifndef HYPERION_HDF_HDF5_H_
#define HYPERION_HDF_HDF5_H_

#include <hyperion/utility.h>
#include <hyperion/IndexTree.h>
#include <hyperion/Keywords.h>
#include <hyperion/Table.h>

#ifdef HYPERION_USE_CASACORE
# include <hyperion/MeasRef.h>
#endif

#pragma GCC visibility push(default)
# include <cassert>
# include <cstdint>
# include <cstring>
# include <exception>
# include CXX_FILESYSTEM_HEADER
# include <optional>
# include <string>
# include <unordered_set>
# include <vector>

# include <hdf5.h>
#pragma GCC visibility pop

namespace hyperion {

class Table;
class Column;
class Keywords;
class ColumnSpace;
class PhysicalTable;
class PhysicalColumn;

namespace hdf5 {

#define HYPERION_NAMESPACE "hyperion"
#define HYPERION_NAME_SEP ":"
#define HYPERION_NAMESPACE_PREFIX HYPERION_NAMESPACE HYPERION_NAME_SEP
#define HYPERION_ATTRIBUTE_DT HYPERION_NAMESPACE_PREFIX "dt"
#define HYPERION_ATTRIBUTE_DT_PREFIX HYPERION_ATTRIBUTE_DT HYPERION_NAME_SEP
#define HYPERION_ATTRIBUTE_SID HYPERION_NAMESPACE_PREFIX "sid"
#define HYPERION_ATTRIBUTE_SID_PREFIX HYPERION_ATTRIBUTE_SID HYPERION_NAME_SEP
#define HYPERION_ATTRIBUTE_DS HYPERION_NAMESPACE_PREFIX "ds"
#define HYPERION_ATTRIBUTE_DS_PREFIX HYPERION_ATTRIBUTE_DS HYPERION_NAME_SEP
#define HYPERION_ATTRIBUTE_FID HYPERION_NAMESPACE_PREFIX "fid"
#define HYPERION_COLUMN_DS HYPERION_NAMESPACE_PREFIX "col"
#define HYPERION_COLUMN_SPACE_GROUP HYPERION_NAMESPACE_PREFIX "colsp"
#define HYPERION_COLUMN_SPACE_GROUP_PREFIX HYPERION_COLUMN_SPACE_GROUP HYPERION_NAME_SEP
#define HYPERION_COLUMN_SPACE_AXES HYPERION_NAMESPACE_PREFIX "axes"
#define HYPERION_COLUMN_SPACE_FLAG HYPERION_NAMESPACE_PREFIX "iflag"
#define HYPERION_COLUMN_SPACE_INDEX_TREE HYPERION_NAMESPACE_PREFIX "indexes"
#ifdef HYPERION_USE_CASACORE
#define HYPERION_MEASURE_GROUP HYPERION_NAMESPACE_PREFIX "measure"
#define HYPERION_MEAS_REF_MCLASS_DS HYPERION_NAMESPACE_PREFIX "mclass"
#define HYPERION_MEAS_REF_RTYPE_DS HYPERION_NAMESPACE_PREFIX "rtype"
#define HYPERION_MEAS_REF_NVAL_DS HYPERION_NAMESPACE_PREFIX "nval"
#define HYPERION_MEAS_REF_VALUES_DS HYPERION_NAMESPACE_PREFIX "values"
#define HYPERION_MEAS_REF_INDEX_DS HYPERION_NAMESPACE_PREFIX "index"
#endif // HYPERION_USE_CASACORE

#define HYPERION_LARGE_TREE_MIN (64 * (1 << 10))

#define CHECK_H5(F) ({                          \
      auto _rc = F;                             \
      assert(_rc >= 0);                         \
      _rc;                                      \
    })

// TODO: it might be nice to support use of types of IndexSpace descriptions
// other than IndexTree...this might require some sort of type registration
// interface, the descriptions would have to support a
// serialization/deserialization interface, and the type would have to be
// recorded in another HDF5 attribute

// FIXME: HDF5 call error handling

template <typename SERDEZ>
void
write_index_tree_to_attr(
  hid_t grp_id,
  const std::string& attr_name,
  const IndexTreeL& spec) {

  std::string attr_ds_name =
    std::string(HYPERION_ATTRIBUTE_DS_PREFIX) + attr_name;

  // remove current attribute value
  htri_t attr_exists = CHECK_H5(H5Aexists(grp_id, attr_name.c_str()));
  if (attr_exists > 0) {
    CHECK_H5(H5Adelete(grp_id, attr_name.c_str()));
    htri_t attr_ds_exists =
      CHECK_H5(H5Lexists(grp_id, attr_ds_name.c_str(), H5P_DEFAULT));
    if (attr_ds_exists > 0)
      CHECK_H5(H5Ldelete(grp_id, attr_ds_name.c_str(), H5P_DEFAULT));
  }

  auto size = SERDEZ::serialized_size(spec);
  std::vector<char> buf(size);
  SERDEZ::serialize(spec, buf.data());
  hsize_t value_dims = size;
  hid_t value_space_id = H5Screate_simple(1, &value_dims, NULL);
  if (size < HYPERION_LARGE_TREE_MIN) {
    // small serialized size: save byte string as an attribute
    hid_t attr_id =
      CHECK_H5(
        H5Acreate(
          grp_id,
          attr_name.c_str(),
          H5T_NATIVE_UINT8,
          value_space_id,
          H5P_DEFAULT, H5P_DEFAULT));
    CHECK_H5(H5Awrite(attr_id, H5T_NATIVE_UINT8, buf.data()));
    CHECK_H5(H5Aclose(attr_id));
  } else {
    // large serialized size: create a new dataset containing byte string, and
    // save reference to that dataset as attribute
    hid_t attr_ds =
      CHECK_H5(
        H5Dcreate(
          grp_id,
          attr_ds_name.c_str(),
          H5T_NATIVE_UINT8,
          value_space_id,
          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    CHECK_H5(
      H5Dwrite(
        attr_ds,
        H5T_NATIVE_UINT8,
        H5S_ALL,
        H5S_ALL,
        H5P_DEFAULT,
        buf.data()));
    CHECK_H5(H5Dclose(attr_ds));

    hid_t ref_space_id = CHECK_H5(H5Screate(H5S_SCALAR));
    hid_t attr_type = H5T_STD_REF_OBJ;
    hid_t attr_id =
      CHECK_H5(
        H5Acreate(
          grp_id,
          attr_name.c_str(),
          attr_type,
          ref_space_id,
          H5P_DEFAULT, H5P_DEFAULT));
    hobj_ref_t attr_ref;
    CHECK_H5(
      H5Rcreate(&attr_ref, grp_id, attr_ds_name.c_str(), H5R_OBJECT, -1));
    CHECK_H5(H5Awrite(attr_id, H5T_STD_REF_OBJ, &attr_ref));
    CHECK_H5(H5Sclose(ref_space_id));
    CHECK_H5(H5Aclose(attr_id));
  }
  H5Sclose(value_space_id);

  // write serdez id
  {
    std::string md_name =
      std::string(HYPERION_ATTRIBUTE_SID_PREFIX) + attr_name;
    hid_t md_space_id = H5Screate(H5S_SCALAR);
    hid_t md_attr_dt =
      hyperion::H5DatatypeManager::datatype<HYPERION_TYPE_STRING>();
    hid_t md_attr_id =
      CHECK_H5(
        H5Acreate(
          grp_id,
          md_name.c_str(),
          md_attr_dt,
          md_space_id,
          H5P_DEFAULT, H5P_DEFAULT));
    hyperion::string attr;
    attr = SERDEZ::id;
    CHECK_H5(H5Awrite(md_attr_id, md_attr_dt, attr.val));
    CHECK_H5(H5Aclose(md_attr_id));
    CHECK_H5(H5Sclose(md_space_id));
  }
}

HYPERION_API std::optional<std::string>
read_index_tree_attr_metadata(hid_t grp_id, const std::string& attr_name);

template <typename SERDEZ>
std::optional<IndexTreeL>
read_index_tree_from_attr(
  hid_t grp_id,
  const std::string& attr_name) {

  std::optional<IndexTreeL> result;

  auto metadata = read_index_tree_attr_metadata(grp_id, attr_name);
  if (!metadata || metadata.value() != SERDEZ::id)
    return result;

  if (!H5Aexists(grp_id, attr_name.c_str()))
    return result;

  hid_t attr_id = CHECK_H5(H5Aopen(grp_id, attr_name.c_str(), H5P_DEFAULT));

  hid_t attr_type = CHECK_H5(H5Aget_type(attr_id));
  if (H5Tequal(attr_type, H5T_NATIVE_UINT8) > 0) {
    // serialized value was written into attribute
    hid_t attr_ds = CHECK_H5(H5Aget_space(attr_id));
    hssize_t attr_sz = CHECK_H5(H5Sget_simple_extent_npoints(attr_ds));
    CHECK_H5(H5Sclose(attr_ds));
    std::vector<char> buf(static_cast<size_t>(attr_sz));
    CHECK_H5(H5Aread(attr_id, H5T_NATIVE_UINT8, buf.data()));
    IndexTreeL tree;
    SERDEZ::deserialize(tree, buf.data());
    result = tree;
  } else if (H5Tequal(attr_type, H5T_STD_REF_OBJ) > 0) {
    // serialized value is in a dataset referenced by attribute
    hobj_ref_t attr_ref;
    CHECK_H5(H5Aread(attr_id, H5T_STD_REF_OBJ, &attr_ref));
    hid_t attr_ds =
      CHECK_H5(H5Rdereference2(grp_id, H5P_DEFAULT, H5R_OBJECT, &attr_ref));
    hid_t attr_sp = CHECK_H5(H5Dget_space(attr_ds));
    hssize_t attr_sz = CHECK_H5(H5Sget_simple_extent_npoints(attr_sp));
    std::vector<char> buf(static_cast<size_t>(attr_sz));
    CHECK_H5(
      H5Dread(
        attr_ds,
        H5T_NATIVE_UINT8,
        H5S_ALL,
        H5S_ALL,
        H5P_DEFAULT,
        buf.data()));
    CHECK_H5(H5Dclose(attr_ds));
    CHECK_H5(H5Sclose(attr_sp));
    IndexTreeL tree;
    SERDEZ::deserialize(tree, buf.data());
    result = tree;
  }
  CHECK_H5(H5Tclose(attr_type));
  CHECK_H5(H5Aclose(attr_id));
  return result;
}

HYPERION_API void
write_keywords(
  Legion::Runtime *rt,
  hid_t loc_id,
  const Keywords::pair<Legion::PhysicalRegion>& kw_prs);

HYPERION_API void
write_keywords(
  Legion::Context ctx,
  Legion::Runtime *rt,
  hid_t loc_id,
  const Keywords& keywords);

#ifdef HYPERION_USE_CASACORE
HYPERION_API void
write_measure(
  Legion::Runtime* rt,
  hid_t mr_id,
  const MeasRef::DataRegions& mr_drs);

HYPERION_API void
write_measure(
  Legion::Context ctx,
  Legion::Runtime* rt,
  hid_t mr_id,
  const MeasRef& mr);
#endif

HYPERION_API void
write_column(
  Legion::Runtime* rt,
  hid_t col_grp_id,
  const std::string& csp_name,
  const PhysicalColumn& column);

HYPERION_API void
write_column(
  Legion::Context ctx,
  Legion::Runtime* rt,
  hid_t col_grp_id,
  const std::string& csp_name,
  const Column& column);

HYPERION_API void
write_columnspace(
  Legion::Runtime* rt,
  hid_t csp_grp_id,
  const Legion::PhysicalRegion& csp_md,
  const Legion::IndexSpace& csp_is,
  hid_t table_axes_dt);

HYPERION_API void
write_columnspace(
  Legion::Context ctx,
  Legion::Runtime* rt,
  hid_t csp_grp_id,
  const ColumnSpace& csp,
  hid_t table_axes_dt);

HYPERION_API void
write_table(
  Legion::Runtime* rt,
  hid_t table_grp_id,
  const PhysicalTable& table);

HYPERION_API void
write_table(
  Legion::Context ctx,
  Legion::Runtime* rt,
  hid_t table_grp_id,
  const Table& table,
  const std::unordered_set<std::string>& columns);

HYPERION_API void
write_table(
  Legion::Context ctx,
  Legion::Runtime* rt,
  hid_t table_grp_id,
  const Table& table);

HYPERION_API hyperion::Keywords::kw_desc_t
init_keywords(hid_t loc_id);

HYPERION_API hyperion::ColumnSpace
init_columnspace(
  Legion::Context ctx,
  Legion::Runtime* rt,
  hid_t table_grp_id,
  hid_t table_axes_dt,
  const std::string& csp_name);

HYPERION_API
  std::optional<
    std::tuple<
      Table::fields_t,
      std::unordered_map<std::string, std::string>>>
table_fields(
  Legion::Context ctx,
  Legion::Runtime* rt,
  hid_t loc_id,
  const std::string& table_name);

HYPERION_API std::tuple<
  hyperion::Table,
  std::unordered_map<std::string, std::string>>
init_table(
  Legion::Context ctx,
  Legion::Runtime* rt,
  hid_t loc_id,
  const std::string& table_name);

HYPERION_API std::unordered_map<
  std::string,
    std::tuple<
      Table::fields_t,
      std::unordered_map<std::string, std::string>>>
all_table_fields(Legion::Context ctx, Legion::Runtime* rt, hid_t loc_id);

HYPERION_API std::unordered_map<std::string, std::string>
get_table_column_paths(
  hid_t file_id,
  const std::string& table_path,
  const std::unordered_set<std::string>& columns);

HYPERION_API std::unordered_map<std::string, std::string>
get_table_column_paths(
  const CXX_FILESYSTEM_NAMESPACE::path& file_path,
  const std::string& table_path,
  const std::unordered_set<std::string>& columns);

HYPERION_API Legion::PhysicalRegion
attach_keywords(
  Legion::Context context,
  Legion::Runtime* runtime,
  const CXX_FILESYSTEM_NAMESPACE::path& file_path,
  const std::string& keywords_path,
  const Keywords& keywords,
  bool read_only = true);

HYPERION_API std::optional<Legion::PhysicalRegion>
attach_table_columns(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const CXX_FILESYSTEM_NAMESPACE::path& file_path,
  const std::string& root_path,
  const Table& table,
  const std::unordered_set<std::string>& columns,
  const std::unordered_map<std::string, std::string>& column_paths,
  bool read_only,
  bool mapped);

HYPERION_API std::map<
  Legion::PhysicalRegion,
  std::unordered_map<std::string, Column>>
attach_all_table_columns(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const CXX_FILESYSTEM_NAMESPACE::path& file_path,
  const std::string& root_path,
  const Table& table,
  const std::unordered_set<std::string>& exclude,
  const std::unordered_map<std::string, std::string>& column_paths,
  bool read_only,
  bool mapped);

HYPERION_API std::map<
  Legion::PhysicalRegion,
  std::unordered_map<std::string, Column>>
attach_some_table_columns(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const CXX_FILESYSTEM_NAMESPACE::path& file_path,
  const std::string& root_path,
  const Table& table,
  const std::unordered_set<std::string>& include,
  const std::unordered_map<std::string, std::string>& column_paths,
  bool read_only,
  bool mapped);

struct HYPERION_API binary_index_tree_serdez {

  static const constexpr char* id = "hyperion::hdf5::binary_index_tree_serdez";

  static size_t
  serialized_size(const IndexTreeL& tree) {
    return tree.serialized_size();
  }

  static size_t
  serialize(const IndexTreeL& tree, void *buffer) {
    return tree.serialize(static_cast<char*>(buffer));
  }

  static size_t
  deserialize(IndexTreeL& tree, const void* buffer) {
    tree = IndexTreeL::deserialize(static_cast<const char*>(buffer));
    return tree.serialized_size();
  }
};

struct HYPERION_API string_index_tree_serdez {

  static const constexpr char* id = "hyperion::hdf5::string_index_tree_serdez";

  static size_t
  serialized_size(const IndexTreeL& tree) {
    return tree.show().size() + 1;
  }

  static size_t
  serialize(const IndexTreeL& tree, void *buffer) {
    auto tr = tree.show();
    std::memcpy(static_cast<char*>(buffer), tr.c_str(), tr.size() + 1);
    return tr.size() + 1;
  }

  static size_t
  deserialize(IndexTreeL&, const void*) {
    // TODO
    assert(false);
  }
};

} // end namespace hdf5
} // end namespace hyperion

#endif // HYPERION_HDF_HDF5_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
