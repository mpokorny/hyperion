#ifndef LEGMS_HDF_HDF5_H_
#define LEGMS_HDF_HDF5_H_

#include <cassert>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include <experimental/filesystem>

#include <hdf5.h>

#include "IndexTree.h"
#include "Column.h"
#include "Table.h"

namespace legms {
namespace hdf5 {

const size_t large_tree_min = (64 * (1 << 10));
#define LEGMS_ATTRIBUTE_NAME_PREFIX "legms::"
const char* table_index_axes_attr_name =
  LEGMS_ATTRIBUTE_NAME_PREFIX "index_axes";
const char* table_axes_uid_attr_name =
  LEGMS_ATTRIBUTE_NAME_PREFIX "axes_uid";
const char* column_axes_attr_name =
  LEGMS_ATTRIBUTE_NAME_PREFIX "axes";

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
  hid_t loc_id,
  const std::string& obj_name,
  const std::string& attr_name) {

  // remove current attribute value
  std::string legms_attr_name =
    std::string(LEGMS_ATTRIBUTE_NAME_PREFIX) + attr_name;
  std::string attr_ds_name =
    std::string(LEGMS_ATTRIBUTE_NAME_PREFIX) + obj_name + "-" + attr_name;

  if (H5Aexists_by_name(
        loc_id,
        obj_name.c_str(),
        legms_attr_name.c_str(),
        H5P_DEFAULT)) {
    H5Adelete_by_name(
      loc_id,
      obj_name.c_str(),
      legms_attr_name.c_str(),
      H5P_DEFAULT);
    if (H5Lexists(loc_id, attr_ds_name.c_str(), H5P_DEFAULT) > 0)
      H5Ldelete(loc_id, attr_ds_name.c_str(), H5P_DEFAULT);
  }

  auto size = SERDEZ::serialized_size(spec);
  std::vector<char> buf(size);
  SERDEZ::serialize(spec, buf.data());
  hsize_t value_dims = size;
  hid_t value_space_id = H5Screate_simple(1, &value_dims, NULL);
  if (size < large_tree_min) {
    // small serialized size: save byte string as an attribute
    hid_t attr_id =
      H5Acreate_by_name(
        loc_id,
        obj_name.c_str(),
        legms_attr_name.c_str(),
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
        loc_id,
        attr_ds_name.c_str(),
        H5T_NATIVE_UINT8,
        value_space_id,
        H5P_DEFAULT,
        H5P_DEFAULT,
        H5P_DEFAULT);
    herr_t rc =
      H5Dwrite(
        attr_ds,
        H5T_NATIVE_UINT8,
        H5S_ALL,
        H5S_ALL,
        H5P_DEFAULT,
        buf.data());
    assert(rc >= 0);

    hsize_t ref_dims = 1;
    hid_t ref_space_id = H5Screate_simple(1, &ref_dims, NULL);
    hid_t attr_type = H5T_STD_REF_OBJ;
    hid_t attr_id =
      H5Acreate_by_name(
        loc_id,
        obj_name.c_str(),
        legms_attr_name.c_str(),
        attr_type,
        ref_space_id,
        H5P_DEFAULT,
        H5P_DEFAULT,
        H5P_DEFAULT);
    assert(attr_id >= 0);
    hobj_ref_t attr_ref;
    rc = H5Rcreate(&attr_ref, loc_id, attr_ds_name.c_str(), H5R_OBJECT, -1);
    assert (rc >= 0);
    rc = H5Awrite(attr_id, H5T_STD_REF_OBJ, &attr_ref);
    assert (rc >= 0);
    H5Sclose(ref_space_id);
  }
  H5Sclose(value_space_id);

  // write serdez id
  {
    std::string md_name = legms_attr_name + "-sid";
    hid_t md_space_id = H5Screate(H5S_SCALAR);
    hid_t md_attr_dt =
      legms::H5DatatypeManager::datatype<ValueType<casacore::String>::DataType>();
    hid_t md_attr_id =
      H5Acreate_by_name(
        loc_id,
        obj_name.c_str(),
        md_name.c_str(),
        md_attr_dt,
        md_space_id,
        H5P_DEFAULT,
        H5P_DEFAULT,
        H5P_DEFAULT);
    assert(md_attr_id >= 0);
    char attr[LEGMS_H5_STRING_SIZE];
    std::strncpy(attr, SERDEZ::id, sizeof(attr));
    attr[sizeof(attr) - 1] = '\0';
    herr_t rc = H5Awrite(md_attr_id, md_attr_dt, attr);
    assert(rc >= 0);
  }
}

std::optional<std::string>
read_index_tree_attr_metadata(
  hid_t loc_id,
  const std::string& obj_name,
  const std::string& attr_name);

template <typename SERDEZ>
std::optional<IndexTreeL>
read_index_tree_from_attr(
  hid_t loc_id,
  const std::string& obj_name,
  const std::string& attr_name) {

  std::optional<IndexTreeL> result;

  auto metadata = read_index_tree_attr_metadata(loc_id, obj_name, attr_name);
  if (!metadata || std::strcmp(metadata.value().c_str(), SERDEZ::id) != 0)
    return result;

  std::string legms_attr_name =
    std::string(LEGMS_ATTRIBUTE_NAME_PREFIX) + attr_name;
  if (!H5Aexists_by_name(
        loc_id,
        obj_name.c_str(),
        legms_attr_name.c_str(),
        H5P_DEFAULT))
    return result;

  hid_t attr_id =
    H5Aopen_by_name(
      loc_id,
      obj_name.c_str(),
      legms_attr_name.c_str(),
      H5P_DEFAULT,
      H5P_DEFAULT);

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

void
write_keywords(
  hid_t loc_id,
  const WithKeywords* with_keywords,
  bool with_data = true);

void
write_column(
  const std::experimental::filesystem::path& path,
  hid_t table_id,
  const std::string& table_name,
  const Column* column,
  bool with_data = true,
  hid_t creation_pl = H5P_DEFAULT,
  hid_t access_pl = H5P_DEFAULT,
  hid_t transfer_pl = H5P_DEFAULT);

void
write_table(
  const std::experimental::filesystem::path& path,
  hid_t loc_id,
  const Table* table,
  const std::unordered_set<std::string>& excluded_columns = {},
  bool with_data = true,
  hid_t link_creation_pl = H5P_DEFAULT,
  hid_t link_access_pl = H5P_DEFAULT,
  hid_t group_creation_pl = H5P_DEFAULT,
  hid_t group_access_pl = H5P_DEFAULT);

struct binary_index_tree_serdez {

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

struct string_index_tree_serdez {

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
  deserialize(IndexTreeL& tree, const void* buffer) {
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
