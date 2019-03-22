#include "legms_hdf5.h"

#ifdef USE_HDF

using namespace legms::hdf5;

std::optional<uint8_t>
legms::hdf5::read_index_tree_attr_metadata(
  hid_t loc_id,
  const std::string& obj_name,
  const std::string& attr_name) {

  std::optional<uint8_t> result;

  std::string md_id_name = attr_name + "-metadata";
  if (H5Aexists_by_name(
        loc_id,
        obj_name.c_str(),
        md_id_name.c_str(),
        H5P_DEFAULT)) {

    hid_t attr_id =
      H5Aopen_by_name(
        loc_id,
        obj_name.c_str(),
        md_id_name.c_str(),
        H5P_DEFAULT,
        H5P_DEFAULT);

    if (attr_id >= 0) {
      hid_t attr_type = H5Aget_type(attr_id);

      if (H5Tequal(attr_type, H5T_NATIVE_UINT8) > 0) {
        hid_t attr_ds = H5Aget_space(attr_id);
        assert(attr_ds >= 0);
        hssize_t attr_sz = H5Sget_simple_extent_npoints(attr_ds);
        assert(attr_sz == 1);
        H5Sclose(attr_ds);
        uint8_t metadata;
        herr_t rc = H5Aread(attr_id, H5T_NATIVE_UINT8, &metadata);
        assert(rc >= 0);
        result = metadata;
      }
    }
  }
  return result;
}

#endif

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
