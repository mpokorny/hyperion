#ifndef LEGMS_HDF_HDF5_H_
#define LEGMS_HDF_HDF5_H_

#pragma GCC visibility push(default)
#include <cassert>
#include <cstdint>
#include <cstring>
#include <exception>
#if GCC_VERSION >= 90000
# include <filesystem>
namespace fs = std::filesystem;
#else
# include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>


#pragma GCC visibility pop

#include "utility.h"
#include "IndexTree.h"
#include "Column.h"
#include "Table.h"

#pragma GCC visibility push(default)
#include <hdf5.h>
#pragma GCC visibility pop

namespace legms {
namespace hdf5 {

#define LEGMS_NAMESPACE "legms"
#define LEGMS_NAME_SEP "::"
#define LEGMS_NAMESPACE_PREFIX LEGMS_NAMESPACE LEGMS_NAME_SEP
#define LEGMS_ATTRIBUTE_DT LEGMS_NAMESPACE_PREFIX "dt"
#define LEGMS_ATTRIBUTE_DT_PREFIX LEGMS_ATTRIBUTE_DT LEGMS_NAME_SEP
#define LEGMS_ATTRIBUTE_SID LEGMS_NAMESPACE_PREFIX "sid"
#define LEGMS_ATTRIBUTE_SID_PREFIX LEGMS_ATTRIBUTE_SID LEGMS_NAME_SEP
#define LEGMS_ATTRIBUTE_DS LEGMS_NAMESPACE_PREFIX "ds"
#define LEGMS_ATTRIBUTE_DS_PREFIX LEGMS_ATTRIBUTE_DS LEGMS_NAME_SEP
#define LEGMS_COLUMN_DS LEGMS_NAMESPACE_PREFIX "col"

#define LEGMS_LARGE_TREE_MIN (64 * (1 << 10))

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
  const std::string& attr_name,
  hid_t link_creation_pl = H5P_DEFAULT,
  hid_t link_access_pl = H5P_DEFAULT,
  hid_t dataset_creation_pl = H5P_DEFAULT,
  hid_t dataset_access_pl = H5P_DEFAULT,
  hid_t xfer_pl = H5P_DEFAULT) {

  // remove current attribute value
  std::string legms_attr_name =
    std::string(LEGMS_NAMESPACE_PREFIX) + attr_name;
  std::string attr_ds_name =
    std::string(LEGMS_ATTRIBUTE_DS_PREFIX)
    + obj_name
    + std::string(LEGMS_NAME_SEP)
    + attr_name;

  if (H5Aexists_by_name(
        parent_id,
        obj_name.c_str(),
        legms_attr_name.c_str(),
        link_access_pl)) {
    H5Adelete_by_name(
      parent_id,
      obj_name.c_str(),
      legms_attr_name.c_str(),
      link_access_pl);
    if (H5Lexists(parent_id, attr_ds_name.c_str(), link_access_pl) > 0)
      H5Ldelete(parent_id, attr_ds_name.c_str(), link_access_pl);
  }

  auto size = SERDEZ::serialized_size(spec);
  std::vector<char> buf(size);
  SERDEZ::serialize(spec, buf.data());
  hsize_t value_dims = size;
  hid_t value_space_id = H5Screate_simple(1, &value_dims, NULL);
  if (size < LEGMS_LARGE_TREE_MIN) {
    // small serialized size: save byte string as an attribute
    hid_t attr_id =
      H5Acreate_by_name(
        parent_id,
        obj_name.c_str(),
        legms_attr_name.c_str(),
        H5T_NATIVE_UINT8,
        value_space_id,
        H5P_DEFAULT,
        H5P_DEFAULT,
        link_access_pl);
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
        link_creation_pl,
        dataset_creation_pl,
        dataset_access_pl);
    herr_t rc =
      H5Dwrite(
        attr_ds,
        H5T_NATIVE_UINT8,
        H5S_ALL,
        H5S_ALL,
        xfer_pl,
        buf.data());
    assert(rc >= 0);

    hid_t ref_space_id = H5Screate(H5S_SCALAR);
    hid_t attr_type = H5T_STD_REF_OBJ;
    hid_t attr_id =
      H5Acreate_by_name(
        parent_id,
        obj_name.c_str(),
        legms_attr_name.c_str(),
        attr_type,
        ref_space_id,
        H5P_DEFAULT,
        H5P_DEFAULT,
        link_access_pl);
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
    std::string md_name = std::string(LEGMS_ATTRIBUTE_SID_PREFIX) + attr_name;
    hid_t md_space_id = H5Screate(H5S_SCALAR);
    hid_t md_attr_dt =
      legms::H5DatatypeManager::datatype<ValueType<std::string>::DataType>();
    hid_t md_attr_id =
      H5Acreate_by_name(
        parent_id,
        obj_name.c_str(),
        md_name.c_str(),
        md_attr_dt,
        md_space_id,
        H5P_DEFAULT,
        H5P_DEFAULT,
        link_access_pl);
    assert(md_attr_id >= 0);
    char attr[LEGMS_MAX_STRING_SIZE];
    std::strncpy(attr, SERDEZ::id, sizeof(attr));
    attr[sizeof(attr) - 1] = '\0';
    herr_t rc = H5Awrite(md_attr_id, md_attr_dt, attr);
    assert(rc >= 0);
  }
}

LEGMS_API std::optional<std::string>
read_index_tree_attr_metadata(
  hid_t loc_id,
  const std::string& attr_name,
  hid_t access_pl = H5P_DEFAULT);

template <typename SERDEZ>
std::optional<IndexTreeL>
read_index_tree_from_attr(
  hid_t loc_id,
  const std::string& attr_name,
  hid_t attr_access_pl = H5P_DEFAULT,
  hid_t xfer_pl = H5P_DEFAULT) {

  std::optional<IndexTreeL> result;

  auto metadata = read_index_tree_attr_metadata(loc_id, attr_name);
  if (!metadata || metadata.value() != SERDEZ::id)
    return result;

  std::string legms_attr_name =
    std::string(LEGMS_NAMESPACE_PREFIX) + attr_name;
  if (!H5Aexists(loc_id, legms_attr_name.c_str()))
    return result;

  hid_t attr_id = H5Aopen(loc_id, legms_attr_name.c_str(), attr_access_pl);

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
    hid_t attr_ds = H5Rdereference(loc_id, H5P_DEFAULT, H5R_OBJECT, &attr_ref);
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
        xfer_pl,
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

LEGMS_API void
write_keywords(
  Legion::Context ctx,
  Legion::Runtime *rt,
  hid_t loc_id,
  const Keywords& keywords,
  bool with_data = true,
  hid_t link_creation_pl = H5P_DEFAULT,
  hid_t link_access_pl = H5P_DEFAULT,
  hid_t dataset_creation_pl = H5P_DEFAULT,
  hid_t dataset_access_pl = H5P_DEFAULT,
  hid_t xfer_pl = H5P_DEFAULT);

LEGMS_API void
write_column(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const fs::path& path,
  hid_t table_id,
  const std::string& table_name,
  const Column& column,
  hid_t table_axes_dt,
  bool with_data = true,
  hid_t link_creation_pl = H5P_DEFAULT,
  hid_t link_access_pl = H5P_DEFAULT,
  hid_t group_creation_pl = H5P_DEFAULT,
  hid_t group_access_pl = H5P_DEFAULT,
  hid_t dataset_creation_pl = H5P_DEFAULT,
  hid_t dataset_access_pl = H5P_DEFAULT,
  hid_t attr_creation_pl = H5P_DEFAULT,
  hid_t attr_access_pl = H5P_DEFAULT,
  hid_t xfer_pl = H5P_DEFAULT);

LEGMS_API void
write_table(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const fs::path& path,
  hid_t loc_id,
  const Table& table,
  const std::unordered_set<std::string>& excluded_columns = {},
  bool with_data = true,
  hid_t link_creation_pl = H5P_DEFAULT,
  hid_t link_access_pl = H5P_DEFAULT,
  hid_t group_creation_pl = H5P_DEFAULT,
  hid_t group_access_pl = H5P_DEFAULT,
  hid_t type_creation_pl = H5P_DEFAULT,
  hid_t type_access_pl = H5P_DEFAULT,
  hid_t dataset_creation_pl = H5P_DEFAULT,
  hid_t dataset_access_pl = H5P_DEFAULT,
  hid_t attr_creation_pl = H5P_DEFAULT,
  hid_t attr_access_pl = H5P_DEFAULT,
  hid_t xfer_pl = H5P_DEFAULT);

LEGMS_API legms::Keywords::kw_desc_t
init_keywords(
  Legion::Context context,
  Legion::Runtime* runtime,
  hid_t loc_id,
  hid_t attr_access_pl = H5P_DEFAULT,
  hid_t link_access_pl = H5P_DEFAULT);

LEGMS_API legms::Column
init_column(
  Legion::Context context,
  Legion::Runtime* runtime,
  const std::string& column_name,
  const std::string& axes_uid,
  hid_t loc_id,
  hid_t axes_dt,
  hid_t attr_access_pl = H5P_DEFAULT,
  hid_t link_access_pl = H5P_DEFAULT,
  hid_t xfer_pl = H5P_DEFAULT);

LEGMS_API legms::Table
init_table(
  Legion::Context context,
  Legion::Runtime* runtime,
  const fs::path& file_path,
  const std::string& table_path,
  const std::unordered_set<std::string>& column_names,
  unsigned flags = H5F_ACC_RDONLY,
  hid_t file_access_pl = H5P_DEFAULT,
  hid_t table_access_pl = H5P_DEFAULT,
  hid_t type_access_pl = H5P_DEFAULT,
  hid_t attr_access_pl = H5P_DEFAULT,
  hid_t link_access_pl = H5P_DEFAULT,
  hid_t xfer_pl = H5P_DEFAULT);

LEGMS_API legms::Table
init_table(
  Legion::Context context,
  Legion::Runtime* runtime,
  const std::string& table_name,
  hid_t loc_id,
  const std::unordered_set<std::string>& column_names,
  hid_t type_access_pl = H5P_DEFAULT,
  hid_t attr_access_pl = H5P_DEFAULT,
  hid_t link_access_pl = H5P_DEFAULT,
  hid_t xfer_pl = H5P_DEFAULT);

LEGMS_API std::unordered_set<std::string>
get_table_paths(const fs::path& file_path);

LEGMS_API std::unordered_set<std::string>
get_column_names(
  const fs::path& file_path,
  const std::string& table_path);

LEGMS_API std::unordered_map<std::string, std::string>
get_table_keyword_paths(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const Table& table);

LEGMS_API std::string
get_table_column_value_path(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const Table& table,
  const std::string& colname);

LEGMS_API std::unordered_map<std::string, std::string>
get_table_column_keyword_paths(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const Table& table,
  const std::string& colname);

LEGMS_API std::optional<Legion::PhysicalRegion>
attach_keywords(
  Legion::Context context,
  Legion::Runtime* runtime,
  const fs::path& file_path,
  const std::string& keywords_path,
  const Keywords& keywords,
  bool read_only = true);

// returns value/keywords region pairs by column
LEGMS_API std::unordered_map<
  std::string,
  std::tuple<
    std::optional<Legion::PhysicalRegion>,
    std::optional<Legion::PhysicalRegion>>>
attach_table_columns(
  Legion::Context context,
  Legion::Runtime* runtime,
  const std::experimental::filesystem::path& file_path,
  const std::string& root_path,
  const Table& table,
  bool mapped = true,
  bool read_only = true);

LEGMS_API std::optional<Legion::PhysicalRegion>
attach_table_keywords(
  Legion::Context context,
  Legion::Runtime* runtime,
  const fs::path& file_path,
  const std::string& root_path,
  const Table& table,
  bool read_only = true);

struct LEGMS_API binary_index_tree_serdez {

  static const constexpr char* id = "legms::hdf5::binary_index_tree_serdez";

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

struct LEGMS_API string_index_tree_serdez {

  static const constexpr char* id = "legms::hdf5::string_index_tree_serdez";

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
} // end namespace legms

#endif // LEGMS_HDF_HDF5_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
