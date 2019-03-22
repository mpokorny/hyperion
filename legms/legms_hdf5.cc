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

void
legms::hdf5::write_column(
  hid_t table_id,
  const std::string& table_name,
  const std::shared_ptr<Column>& column,
  hid_t creation_pl = H5P_DEFAULT,
  hid_t access_pl = H5P_DEFAULT,
  hid_t transfer_pl = H5P_DEFAULT) {

  herr_t err;
  h5tri_t ds_exists =
    H5Lexists(table_id, column->name().c_str((), H5P_DEFAULT));
  if (ds_exists > 0) {
    err = H5Ldelete(table_id, column->name().c_str(), H5P_DEFAULT);
    assert(err >= 0);
  } else {
    assert(ds_exists == 0);
  }
  
  std::string column_ds_name = table_name + "/" + column.name();
  std::map<FieldID, const char*>
    field_map{{Column::value_fid, column_ds_name.c_str()}};
  LogicalRegion values_lr =
    runtime->create_logical_region(
      context,
      column->index_space(),
      column->logical_region().get_field_space());
  AttachLauncher attach(EXTERNAL_HDF5_FILE, values_lr, values_lr);
  attach.attach_hdf5(FILENAME, field_map, LEGION_FILE_CREATE);
  PhysicalRegion values_pr = runtime->attach_external_resource(context, attach);
  AcquireLauncher acquire(values_lr, values_lr, values_pr);
  acquire.add_field(Column::value_fid);
  runtime->issue_acquire(context, acquire);
  RegionRequirement src(
    column->logical_region(),
    READ_ONLY,
    EXCLUSIVE,
    column->logical_region());
  src.add_field(Column::value_fid);
  RegionRequirement dst(
    values_lr,
    WRITE_DISCARD,
    EXCLUSIVE,
    values_lr);
  CopyLauncher copy(src, dst);
  runtime->issue_copy_operation(context, copy);
  ReleaseLauncher release(values_lr, values_lr, values_pr);
  release.add_field(Column::value_fid);
  runtime->issue_release(context, acquire);
  runtime->detach_external_resource(context, values_pr);
}

void
write_keywords(hid_t loc_id, Legion::LogicalRegion& keywords) {
  // FIXME: implement this
}

void
legms::hdf5::write_table(
  hid_t loc_id,
  const std::shared_ptr<Table>& table) {

  // open or create the group for the table
  hid_t table_id;
  {
    htri_t rc = H5Lexists(loc_id, table->name().c_str(), H5P_DEFAULT);
    if (rc == 0) {
      table_id =
        H5Gcreate(
          file_id,
          table->name().c_str(),
          H5P_DEFAULT,
          H5P_DEFAULT,
          H5P_DEFAULT);
    } else {
      assert(rc > 0);
      table_id = H5Gopen(loc_id, table->name().c_str(), H5P_DEFAULT);
    }
    assert(table_id >= 0);
  }

  try {
    {
      // TODO: change to writing table->index_axes() values
      hid_t index_axes_type = H5Tcopy(H5T_C_S1);
      herr_t err =
        H5Tset_size(index_axes_type, table_index_axes_attr_max_length);
      assert(err >= 0);

      try {
        hid_t index_axes_id;
        htri_t rc = H5Aexists(table_id, table_index_axes_attr_name);

        try {
          if (rc == 0) {
            index_axes_id =
              H5Acreate(
                table_id,
                table_index_axes_attr_name,
                index_axes_type,
                sp_id,
                H5P_DEFAULT,
                H5P_DEFAULT);
          } else {
            assert(rc > 0);
            index_axes_id =
              H5Aopen(table_id, table_index_axes_attr_name, H5P_DEFAULT);
          }
          assert(index_axes_id >= 0);

          const char* sep = "";
          std::ostringstream oss;
          std::for_each(
            table->row_axes.begin(),
            table->row_axes.end(),
            [&oss, &sep](auto& i) {
              oss << *sep << std::to_string(i);
              sep = ",";
            });
          std::string index_axes = oss.str();
          // FIXME: return errors from this function
          assert(index_axes.size() < table_index_axes_attr_max_length);
          char index_axes_buf[table_index_axes_attr_max_length];
          std::memcpy(index_axes_str, index_axes.c_str(), index_axes.size() + 1);
          err = H5Awrite(index_axes_id, index_axes_type, index_axes_buf);
          assert(err >= 0);
        } catch (...) {
          H5Aclose(index_axes_id);
          throw;
        }
        err = H5Aclose(index_axes_id);
        assert(err >= 0);
      } catch (...) {
        H5Tclose(index_axes_type);
        throw;
      }
      err = H5Tclose(index_axes_type);
      assert(err >= 0);
    }

    std::for_each(
      table->column_names().begin(),
      table->column_names().end(),
      [&table, table_id](auto& nm) {
        write_column(table_id, table->column(nm));
      });

    write_keywords(table_id, table);
    
  } catch (...) {
    H5Gclose(table_id);
    throw;
  }
  H5Gclose(table_id);
}

#endif

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
