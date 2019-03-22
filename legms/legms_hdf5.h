#ifndef LEGMS_HDF_HDF5_H_
#define LEGMS_HDF_HDF5_H_

#include <cassert>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <hdf5.h>

#include "IndexTree.h"

namespace legms {
namespace hdf5 {

const size_t large_tree_min = (64 * (1 << 10));
const char* table_index_axes_attr_name = "index_axes";
const size_t table_index_axes_attr_max_length = 160;

// TODO: it might be nice to support use of types of IndexSpace descriptions
// other than IndexTree...this might require some sort of type registration
// interface, the descriptions would have to support a
// serialization/deserialization interface, and the type would have to be
// recorded in another HDF5 attribute

// FIXME: HDF5 call error handling

template <typename SERDEZ>
void
write_index_tree_to_attr(
  const IndexTree<typename SERDEZ::coord_t>& spec,
  hid_t loc_id,
  const std::string& obj_name,
  const std::string& attr_name) {

  // remove current attribute value
  std::string attr_ds_name = obj_name + "-" + attr_name;

  if (H5Aexists_by_name(
        loc_id,
        obj_name.c_str(),
        attr_name.c_str(),
        H5P_DEFAULT)) {
    H5Adelete_by_name(
      loc_id,
      obj_name.c_str(),
      attr_name.c_str(),
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
        attr_name.c_str(),
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
        attr_name.c_str(),
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
    std::string md_name = attr_name + "-metadata";
    hsize_t md_dims = 1;
    hid_t md_space_id = H5Screate_simple(1, &md_dims, NULL);
    hid_t md_attr_id =
      H5Acreate_by_name(
        loc_id,
        obj_name.c_str(),
        md_name.c_str(),
        H5T_NATIVE_UINT8,
        md_space_id,
        H5P_DEFAULT,
        H5P_DEFAULT,
        H5P_DEFAULT);
    assert(md_attr_id >= 0);
    uint8_t metadata = SERDEZ::id;
    herr_t rc = H5Awrite(md_attr_id, H5T_NATIVE_UINT8, &metadata);
    assert(rc >= 0);
  }
}

std::optional<uint8_t>
read_index_tree_attr_metadata(
  hid_t loc_id,
  const std::string& obj_name,
  const std::string& attr_name);

template <typename SERDEZ>
std::optional<IndexTree<typename SERDEZ::coord_t>>
read_index_tree_from_attr(
  hid_t loc_id,
  const std::string& obj_name,
  const std::string& attr_name) {

  std::optional<IndexTree<typename SERDEZ::coord_t>> result;

  auto metadata = read_index_tree_attr_metadata(loc_id, obj_name, attr_name);
  if (!metadata || metadata.value() != SERDEZ::id)
    return result;

  if (!H5Aexists_by_name(
        loc_id,
        obj_name.c_str(),
        attr_name.c_str(),
        H5P_DEFAULT))
    return result;

  hid_t attr_id =
    H5Aopen_by_name(
      loc_id,
      obj_name.c_str(),
      attr_name.c_str(),
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
    IndexTree<typename SERDEZ::coord_t> tree;
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
    IndexTree<typename SERDEZ::coord_t> tree;
    SERDEZ::deserialize(tree, buf.data());
    result = tree;
  }
  H5Tclose(attr_type);
  return result;
}

void
write_column(
  hid_t table_id,
  const std::shared_ptr<Column>& column,
  hid_t creation_pl = H5P_DEFAULT,
  hid_t access_pl = H5P_DEFAULT,
  hid_t transfer_pl = H5P_DEFAULT);

void
write_keywords(hid_t loc_id, Legion::LogicalRegion& keywords);

void
write_table(hid_t loc_id, const std::shared_ptr<Table>& table);

template <typename COORD_T = Legion::coord_t>
struct binary_index_tree_serdez {

  typedef COORD_T coord_t;
  static const constexpr uint8_t id = 10 + sizeof(COORD_T);

  static size_t
  serialized_size(const IndexTree<COORD_T>& tree) {
    return tree.serialized_size();
  }

  static size_t
  serialize(const IndexTree<COORD_T>& tree, void *buffer) {
    return tree.serialize(static_cast<char*>(buffer));
  }

  static size_t
  deserialize(IndexTree<COORD_T>& tree, const void* buffer) {
    tree = IndexTree<COORD_T>::deserialize(static_cast<const char*>(buffer));
    return tree.serialized_size();
  }
};

// template <typename COORD_T = Legion::coord_t>
// struct string_index_tree_serdez {

//   typedef COORD_T coord_t;
//   static const constexpr uint8_t id = 20 + sizeof(COORD_T);

//   static size_t
//   serialized_size(const IndexTree<COORD_T>& tree) {
//     return tree.show().size() + 1;
//   }

//   static size_t
//   serialize(const IndexTree<COORD_T>& tree, void *buffer) {
//     auto tr = tree.show();
//     std::memcpy(static_cast<char*>(buffer), tr.c_str(), tr.size() + 1);
//     return return tr.size() + 1;
//   }

//   static size_t
//   deserialize(IndexTree<COORD_T>& tree, const void* buffer) {
//     // TODO
//   }
// };

} // end namespace hdf5
} // end namespace legms

#endif // LEGMS_HDF_HDF5_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
