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

#ifdef HYPERION_USE_CASACORE
# include <hyperion/MeasRefContainer.h>
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
#define HYPERION_COLUMN_DS HYPERION_NAMESPACE_PREFIX "col"
#ifdef HYPERION_USE_CASACORE
#define HYPERION_MEASURES_GROUP HYPERION_NAMESPACE_PREFIX "measures"
#define HYPERION_MEAS_REF_MCLASS_DS HYPERION_NAMESPACE_PREFIX "mclass"
#define HYPERION_MEAS_REF_RTYPE_DS HYPERION_NAMESPACE_PREFIX "rtype"
#define HYPERION_MEAS_REF_NVAL_DS HYPERION_NAMESPACE_PREFIX "nval"
#define HYPERION_MEAS_REF_VALUES_DS HYPERION_NAMESPACE_PREFIX "values"
#define HYPERION_MEAS_REF_LINK_PREFIX HYPERION_NAMESPACE_PREFIX "mr" HYPERION_NAME_SEP
#endif // HYPERION_USE_CASACORE

#define HYPERION_LARGE_TREE_MIN (64 * (1 << 10))

// TODO: it might be nice to support use of types of IndexSpace descriptions
// other than IndexTree...this might require some sort of type registration
// interface, the descriptions would have to support a
// serialization/deserialization interface, and the type would have to be
// recorded in another HDF5 attribute

// FIXME: HDF5 call error handling

template <typename SERDEZ>
void
write_index_tree_to_attr(
  const IndexTreeL& spec,
  hid_t parent_id,
  const std::string& obj_name,
  const std::string& attr_name) {

  // remove current attribute value
  std::string hyperion_attr_name =
    std::string(HYPERION_NAMESPACE_PREFIX) + attr_name;
  std::string attr_ds_name =
    std::string(HYPERION_ATTRIBUTE_DS_PREFIX)
    + obj_name
    + std::string(HYPERION_NAME_SEP)
    + attr_name;

  if (H5Aexists_by_name(
        parent_id,
        obj_name.c_str(),
        hyperion_attr_name.c_str(),
        H5P_DEFAULT)) {
    H5Adelete_by_name(
      parent_id,
      obj_name.c_str(),
      hyperion_attr_name.c_str(),
      H5P_DEFAULT);
    if (H5Lexists(parent_id, attr_ds_name.c_str(), H5P_DEFAULT) > 0)
      H5Ldelete(parent_id, attr_ds_name.c_str(), H5P_DEFAULT);
  }

  auto size = SERDEZ::serialized_size(spec);
  std::vector<char> buf(size);
  SERDEZ::serialize(spec, buf.data());
  hsize_t value_dims = size;
  hid_t value_space_id = H5Screate_simple(1, &value_dims, NULL);
  if (size < HYPERION_LARGE_TREE_MIN) {
    // small serialized size: save byte string as an attribute
    hid_t attr_id =
      H5Acreate_by_name(
        parent_id,
        obj_name.c_str(),
        hyperion_attr_name.c_str(),
        H5T_NATIVE_UINT8,
        value_space_id,
        H5P_DEFAULT,
        H5P_DEFAULT,
        H5P_DEFAULT);
    assert(attr_id >= 0);
    herr_t rc = H5Awrite(attr_id, H5T_NATIVE_UINT8, buf.data());
    assert (rc >= 0);
  } else {
    // large serialized size: create a new dataset containing byte string, and
    // save reference to that dataset as attribute
    hid_t attr_ds =
      H5Dcreate(
        parent_id,
        attr_ds_name.c_str(),
        H5T_NATIVE_UINT8,
        value_space_id,
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    herr_t rc =
      H5Dwrite(
        attr_ds,
        H5T_NATIVE_UINT8,
        H5S_ALL,
        H5S_ALL,
        H5P_DEFAULT,
        buf.data());
    assert(rc >= 0);

    hid_t ref_space_id = H5Screate(H5S_SCALAR);
    hid_t attr_type = H5T_STD_REF_OBJ;
    hid_t attr_id =
      H5Acreate_by_name(
        parent_id,
        obj_name.c_str(),
        hyperion_attr_name.c_str(),
        attr_type,
        ref_space_id,
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(attr_id >= 0);
    hobj_ref_t attr_ref;
    rc = H5Rcreate(&attr_ref, parent_id, attr_ds_name.c_str(), H5R_OBJECT, -1);
    assert (rc >= 0);
    rc = H5Awrite(attr_id, H5T_STD_REF_OBJ, &attr_ref);
    assert (rc >= 0);
    H5Sclose(ref_space_id);
  }
  H5Sclose(value_space_id);

  // write serdez id
  {
    std::string md_name = std::string(HYPERION_ATTRIBUTE_SID_PREFIX) + attr_name;
    hid_t md_space_id = H5Screate(H5S_SCALAR);
    hid_t md_attr_dt =
      hyperion::H5DatatypeManager::datatype<ValueType<std::string>::DataType>();
    hid_t md_attr_id =
      H5Acreate_by_name(
        parent_id,
        obj_name.c_str(),
        md_name.c_str(),
        md_attr_dt,
        md_space_id,
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(md_attr_id >= 0);
    char attr[HYPERION_MAX_STRING_SIZE];
    std::strncpy(attr, SERDEZ::id, sizeof(attr));
    attr[sizeof(attr) - 1] = '\0';
    herr_t rc = H5Awrite(md_attr_id, md_attr_dt, attr);
    assert(rc >= 0);
  }
}

HYPERION_API std::optional<std::string>
read_index_tree_attr_metadata(hid_t loc_id, const std::string& attr_name);

template <typename SERDEZ>
std::optional<IndexTreeL>
read_index_tree_from_attr(
  hid_t loc_id,
  const std::string& attr_name) {

  std::optional<IndexTreeL> result;

  auto metadata = read_index_tree_attr_metadata(loc_id, attr_name);
  if (!metadata || metadata.value() != SERDEZ::id)
    return result;

  std::string hyperion_attr_name =
    std::string(HYPERION_NAMESPACE_PREFIX) + attr_name;
  if (!H5Aexists(loc_id, hyperion_attr_name.c_str()))
    return result;

  hid_t attr_id = H5Aopen(loc_id, hyperion_attr_name.c_str(), H5P_DEFAULT);

  if (attr_id < 0)
    return result;

  hid_t attr_type = H5Aget_type(attr_id);
  if (H5Tequal(attr_type, H5T_NATIVE_UINT8) > 0) {
    // serialized value was written into attribute
    hid_t attr_ds = H5Aget_space(attr_id);
    assert(attr_ds >= 0);
    hssize_t attr_sz = H5Sget_simple_extent_npoints(attr_ds);
    assert(attr_sz >= 0);
    H5Sclose(attr_ds);
    std::vector<char> buf(static_cast<size_t>(attr_sz));
    herr_t rc = H5Aread(attr_id, H5T_NATIVE_UINT8, buf.data());
    assert(rc >= 0);
    IndexTreeL tree;
    SERDEZ::deserialize(tree, buf.data());
    result = tree;

  } else if (H5Tequal(attr_type, H5T_STD_REF_OBJ) > 0) {
    // serialized value is in a dataset referenced by attribute
    hobj_ref_t attr_ref;
    herr_t rc = H5Aread(attr_id, H5T_STD_REF_OBJ, &attr_ref);
    assert(rc >= 0);
    hid_t attr_ds = H5Rdereference2(loc_id, H5P_DEFAULT, H5R_OBJECT, &attr_ref);
    assert(attr_ds >= 0);
    hid_t attr_sp = H5Dget_space(attr_ds);
    assert(attr_ds >= 0);
    hssize_t attr_sz = H5Sget_simple_extent_npoints(attr_sp);
    assert(attr_sz >= 0);
    std::vector<char> buf(static_cast<size_t>(attr_sz));
    rc =
      H5Dread(
        attr_ds,
        H5T_NATIVE_UINT8,
        H5S_ALL,
        H5S_ALL,
        H5P_DEFAULT,
        buf.data());
    assert(rc >= 0);
    H5Dclose(attr_ds);
    H5Sclose(attr_sp);
    IndexTreeL tree;
    SERDEZ::deserialize(tree, buf.data());
    result = tree;
  }
  H5Tclose(attr_type);
  return result;
}

HYPERION_API void
write_keywords(
  Legion::Context ctx,
  Legion::Runtime *rt,
  hid_t loc_id,
  const Keywords& keywords,
  bool with_data = true);

#ifdef HYPERION_USE_CASACORE
HYPERION_API void
write_measures(
  Legion::Context ctx,
  Legion::Runtime* rt,
  hid_t loc_id,
  const std::string& component_path,
  const MeasRefContainer& meas_refs);
#endif

HYPERION_API void
write_column(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const CXX_FILESYSTEM_NAMESPACE::path& path,
  hid_t table_id,
  const std::string& table_name,
  const Column& column,
  hid_t table_axes_dt,
  bool with_data = true);

HYPERION_API void
write_table(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const CXX_FILESYSTEM_NAMESPACE::path& path,
  hid_t loc_id,
  const Table& table,
  const std::unordered_set<std::string>& excluded_columns = {},
  bool with_data = true);

HYPERION_API hyperion::Keywords::kw_desc_t
init_keywords(
  Legion::Context context,
  Legion::Runtime* runtime,
  hid_t loc_id);

HYPERION_API hyperion::Column
init_column(
  Legion::Context context,
  Legion::Runtime* runtime,
  const std::string& column_name,
  const std::string& axes_uid,
  hid_t loc_id,
  hid_t axes_dt,
#ifdef HYPERION_USE_CASACORE
  const MeasRefContainer& table_meas_ref,
#endif
  const std::string& name_prefix = "");

HYPERION_API hyperion::Table
init_table(
  Legion::Context context,
  Legion::Runtime* runtime,
  const CXX_FILESYSTEM_NAMESPACE::path& file_path,
  const std::string& table_path,
  const std::unordered_set<std::string>& column_names,
#ifdef HYPERION_USE_CASACORE
  const MeasRefContainer& ms_meas_ref,
#endif
  unsigned flags = H5F_ACC_RDONLY);

HYPERION_API hyperion::Table
init_table(
  Legion::Context context,
  Legion::Runtime* runtime,
  const std::string& table_name,
  hid_t loc_id,
  const std::unordered_set<std::string>& column_names,
#ifdef HYPERION_USE_CASACORE
  const MeasRefContainer& ms_meas_ref,
#endif
  const std::string& name_prefix = "");

HYPERION_API std::unordered_set<std::string>
get_table_paths(const CXX_FILESYSTEM_NAMESPACE::path& file_path);

HYPERION_API std::unordered_set<std::string>
get_column_names(
  const CXX_FILESYSTEM_NAMESPACE::path& file_path,
  const std::string& table_path);

HYPERION_API std::unordered_map<std::string, std::string>
get_table_keyword_paths(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const Table& table);

HYPERION_API std::string
get_table_column_value_path(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const Table& table,
  const std::string& colname);

HYPERION_API std::unordered_map<std::string, std::string>
get_table_column_keyword_paths(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const Table& table,
  const std::string& colname);

HYPERION_API Legion::PhysicalRegion
attach_keywords(
  Legion::Context context,
  Legion::Runtime* runtime,
  const CXX_FILESYSTEM_NAMESPACE::path& file_path,
  const std::string& keywords_path,
  const Keywords& keywords,
  bool read_only = true);

HYPERION_API Legion::PhysicalRegion
attach_column_values(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const CXX_FILESYSTEM_NAMESPACE::path& file_path,
  const std::string& table_root,
  const Column& column,
  bool mapped = true,
  bool read_only = true);

HYPERION_API Legion::PhysicalRegion
attach_column_keywords(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const CXX_FILESYSTEM_NAMESPACE::path& file_path,
  const std::string& table_root,
  const Column& column,
  bool read_only = true);

HYPERION_API Legion::PhysicalRegion
attach_table_keywords(
  Legion::Context context,
  Legion::Runtime* runtime,
  const CXX_FILESYSTEM_NAMESPACE::path& file_path,
  const std::string& root_path,
  const Table& table,
  bool read_only = true);

HYPERION_API void
release_table_column_values(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const Table& table);

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
