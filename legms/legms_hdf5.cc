#include "legms_hdf5.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <optional>
#include <sstream>

using namespace legms::hdf5;
using namespace std;

optional<std::string>
legms::hdf5::read_index_tree_attr_metadata(
  hid_t loc_id,
  const string& obj_name,
  const string& attr_name) {

  optional<string> result;

  string md_id_name = string(LEGMS_INDEX_TREE_SID_PREFIX) + attr_name;
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

      hid_t attr_dt =
        legms::H5DatatypeManager::datatype<
          ValueType<casacore::String>::DataType>();
      if (H5Tequal(attr_type, attr_dt) > 0) {
        char metadata[LEGMS_MAX_STRING_SIZE];
        herr_t rc = H5Aread(attr_id, attr_dt, metadata);
        assert(rc >= 0);
        result = std::string(metadata);
      }
    }
  }
  return result;
}

template <casacore::DataType DT>
using KW =
  Legion::FieldAccessor<
  READ_ONLY,
  typename legms::DataType<DT>::ValueType,
  1,
  Legion::coord_t,
  Legion::AffineAccessor<
    typename legms::DataType<DT>::ValueType,
    1,
    Legion::coord_t>,
  false>;

void
init_datatype_attr(hid_t loc_id, const char* name, casacore::DataType dt) {

  std::string attr_name =
    std::string(
      (std::strlen(name) > 0) ? LEGMS_ATTRIBUTE_DT_PREFIX : LEGMS_ATTRIBUTE_DT)
    + name;

  htri_t rc = H5Aexists(loc_id, attr_name.c_str());
  if (rc > 0) {
    herr_t err = H5Adelete(loc_id, attr_name.c_str());
    assert(err >= 0);
  }

  hid_t ds = H5Screate(H5S_SCALAR);
  assert(ds >= 0);
  hid_t did = legms::H5DatatypeManager::datatypes()[
    legms::H5DatatypeManager::CASACORE_DATATYPE_H5T];
  hid_t attr_id =
    H5Acreate(
      loc_id,
      attr_name.c_str(),
      did,
      ds,
      H5P_DEFAULT,
      H5P_DEFAULT);
  assert(attr_id >= 0);
  unsigned char udt = dt;
  // herr_t err =
  //   H5Tconvert(H5T_NATIVE_UCHAR, attr_dt, 1, &udt, NULL, H5P_DEFAULT);
  // assert(err >= 0);
  herr_t err = H5Awrite(attr_id, did, &udt);
  assert(err >= 0);
  err = H5Sclose(ds);
  assert(err >= 0);
  err = H5Aclose(attr_id);
  assert(err >= 0);
}

hid_t
init_kw_attr(
  hid_t loc_id,
  const char *attr_name,
  hid_t type_id,
  casacore::DataType dt) {

  {
    htri_t rc = H5Aexists(loc_id, attr_name);
    if (rc > 0) {
      herr_t err = H5Adelete(loc_id, attr_name);
      assert(err >= 0);
    }
  }
  init_datatype_attr(loc_id, attr_name, dt);
  hid_t result;
  {
    hid_t attr_ds = H5Screate(H5S_SCALAR);
    assert(attr_ds >= 0);
    result =
      H5Acreate(loc_id, attr_name, type_id, attr_ds, H5P_DEFAULT, H5P_DEFAULT);
    assert(result >= 0);
    herr_t err = H5Sclose(attr_ds);
    assert(err >= 0);
  }
  return result;
}

template <casacore::DataType DT>
void
write_kw_attr(
  hid_t loc_id,
  const char *attr_name,
  std::optional<Legion::PhysicalRegion>& region,
  Legion::FieldID fid) {

  hid_t dt = legms::H5DatatypeManager::datatype<DT>();
  hid_t attr = init_kw_attr(loc_id, attr_name, dt, DT);
  if (region) {
    const KW<DT> kw(region.value(), fid);
    herr_t err = H5Awrite(attr, dt, kw.ptr(0));
    assert(err >= 0);
  }
  herr_t err = H5Aclose(attr);
  assert(err >= 0);
}

template <>
void
write_kw_attr<casacore::TpString> (
  hid_t loc_id,
  const char *attr_name,
  std::optional<Legion::PhysicalRegion>& region,
  Legion::FieldID fid) {

  hid_t dt = legms::H5DatatypeManager::datatype<casacore::TpString>();
  hid_t attr = init_kw_attr(loc_id, attr_name, dt, casacore::TpString);
  if (region) {
    const KW<casacore::TpString> kw(region.value(), fid);
    const string& val = kw[0];
    char buf[LEGMS_MAX_STRING_SIZE];
    assert(val.size() < sizeof(buf));
    strncpy(buf, val.c_str(), sizeof(buf));
    buf[sizeof(buf) - 1] = '\0';
    herr_t err = H5Awrite(attr, dt, buf);
    assert(err >= 0);
  }
  herr_t err = H5Aclose(attr);
  assert(err >= 0);
}

void
legms::hdf5::write_keywords(
  hid_t loc_id,
  const WithKeywords* with_keywords,
  bool with_data) {

  if (with_keywords->keywords_region() == Legion::LogicalRegion::NO_REGION)
    return;

  Legion::Runtime* runtime = with_keywords->runtime();
  Legion::Context context = with_keywords->context();

  std::optional<Legion::PhysicalRegion> pr;
  if (with_data) {
    Legion::RegionRequirement req(
      with_keywords->keywords_region(),
      READ_ONLY,
      EXCLUSIVE,
      with_keywords->keywords_region());
    std::vector<Legion::FieldID> fids(with_keywords->num_keywords());
    std::iota(fids.begin(), fids.end(), 0);
    req.add_fields(fids);
    pr = runtime->map_region(context, req);
  }

  auto kws = with_keywords->keywords();
  for (size_t i = 0; i < kws.size(); ++i) {
    auto [nm, dt] = kws[i];
    assert(nm.substr(0, sizeof(LEGMS_ATTRIBUTE_NAME_PREFIX) - 1)
           != LEGMS_ATTRIBUTE_NAME_PREFIX);

#define WRITE_KW(DT)                                  \
    case (DT): {                                      \
      write_kw_attr<DT>(loc_id, nm.c_str(), pr, i);   \
      break;                                          \
    }

    switch (dt) {
      FOREACH_DATATYPE(WRITE_KW)
    default:
        assert(false);
    }

#undef WRITE_KW
  }
}

void
legms::hdf5::write_column(
  const experimental::filesystem::path& path,
  hid_t table_id,
  const string& table_name,
  const Column* column,
  bool with_data,
  hid_t creation_pl,
  hid_t access_pl,
  hid_t transfer_pl) {


  Legion::Runtime* runtime = column->runtime();
  Legion::Context context = column->context();

  // delete column dataset if it exists
  htri_t ds_exists =
    H5Lexists(table_id, column->name().c_str(), H5P_DEFAULT);
  if (ds_exists > 0) {
    herr_t err = H5Ldelete(table_id, column->name().c_str(), H5P_DEFAULT);
    assert(err >= 0);
  } else {
    assert(ds_exists == 0);
  }

  // create column dataset
  hid_t col_id;
  {
    int rank = column->index_space().get_dim();
    hsize_t dims[rank];

#define DIMS(N) \
    case N: {\
      Legion::Rect<N> rect =                                            \
        runtime->get_index_space_domain(context, column->index_space()) \
        .bounds<N, Legion::coord_t>();                                  \
      for (size_t i = 0; i < N; ++i)                                    \
        dims[i] = rect.hi[i] + 1;                                       \
      break;                                                            \
    }

    switch (rank) {
      LEGMS_FOREACH_N(DIMS)
    default:
      assert(false);
      break;
    }
#undef DIMS

    hid_t ds = H5Screate_simple(rank, dims, NULL);
    assert(ds >= 0);

    hid_t dt;

#define DT(T) case T: { dt = legms::H5DatatypeManager::datatype<T>(); break; }

    switch (column->datatype()) {
      FOREACH_DATATYPE(DT)
    default:
      assert(false);
      break;
    }
#undef DT

    col_id =
      H5Dcreate(
        table_id,
        column->name().c_str(),
        dt,
        ds,
        H5P_DEFAULT, // TODO
        H5P_DEFAULT, // TODO
        H5P_DEFAULT); // TODO
    assert(col_id >= 0);
    herr_t err = H5Sclose(ds);
    assert(err >= 0);
  }

  // write column value datatype
  init_datatype_attr(col_id, "", column->datatype());

  // write axes attribute to column
  {
    hid_t axes_dt =
      legms::H5DatatypeManager::datatype<ValueType<int>::DataType>();

    htri_t rc = H5Aexists(table_id, column_axes_attr_name);
    if (rc > 0) {
      herr_t err = H5Adelete(table_id, column_axes_attr_name);
      assert(err >= 0);
    }

    hsize_t dims = column->axes().size();
    hid_t axes_ds = H5Screate_simple(1, &dims, NULL);
    assert(axes_ds >= 0);

    try {
      hid_t axes_id =
        H5Acreate(
          col_id,
          column_axes_attr_name,
          axes_dt,
          axes_ds,
          H5P_DEFAULT,
          H5P_DEFAULT);
      try {
        assert(axes_id >= 0);
        herr_t err = H5Awrite(axes_id, axes_dt, column->axes().data());
        assert(err >= 0);
      } catch (...) {
        herr_t err = H5Aclose(axes_id);
        assert(err >= 0);
        throw;
      }
      herr_t err = H5Aclose(axes_id);
      assert(err >= 0);
    } catch (...) {
      herr_t err = H5Sclose(axes_ds);
      assert(err >= 0);
      throw;
    }
    herr_t err = H5Sclose(axes_ds);
    assert(err >= 0);
  }

  // write data to dataset
  if (with_data) {
    string column_ds_name = string("/") + table_name + "/" + column->name();
    map<Legion::FieldID, const char*>
      field_map{{Column::value_fid, column_ds_name.c_str()}};
    Legion::LogicalRegion values_lr =
      runtime->create_logical_region(
        context,
        column->index_space(),
        column->logical_region().get_field_space());
    Legion::AttachLauncher attach(EXTERNAL_HDF5_FILE, values_lr, values_lr);
    attach.attach_hdf5(path.c_str(), field_map, LEGION_FILE_READ_WRITE);
    Legion::PhysicalRegion values_pr =
      runtime->attach_external_resource(context, attach);
    Legion::RegionRequirement src(
      column->logical_region(),
      READ_ONLY,
      EXCLUSIVE,
      column->logical_region());
    src.add_field(Column::value_fid);
    Legion::RegionRequirement dst(
      values_lr,
      WRITE_ONLY,
      EXCLUSIVE,
      values_lr);
    dst.add_field(Column::value_fid);
    Legion::CopyLauncher copy;
    copy.add_copy_requirements(src, dst);
    runtime->issue_copy_operation(context, copy);
    runtime->detach_external_resource(context, values_pr);
  }
  write_keywords(col_id, column, with_data);
  herr_t rc = H5Dclose(col_id);
  assert(rc >= 0);

  write_index_tree_to_attr<binary_index_tree_serdez>(
    column->index_tree(),
    table_id,
    column->name(),
    "index_tree");
}

void
legms::hdf5::write_table(
  const experimental::filesystem::path& path,
  hid_t loc_id,
  const Table* table,
  const std::unordered_set<std::string>& excluded_columns,
  bool with_data,
  hid_t link_creation_pl,
  hid_t link_access_pl,
  hid_t group_creation_pl,
  hid_t group_access_pl) {

  // open or create the group for the table
  hid_t table_id;
  {
    htri_t rc = H5Lexists(loc_id, table->name().c_str(), link_access_pl);
    if (rc == 0) {
      table_id =
        H5Gcreate(
          loc_id,
          table->name().c_str(),
          link_creation_pl,
          group_creation_pl,
          group_access_pl);
    } else {
      assert(rc > 0);
      table_id = H5Gopen(loc_id, table->name().c_str(), group_access_pl);
    }
    assert(table_id >= 0);
  }

  // write axes uid attribute to table
  {
    hid_t axes_uid_dt =
      legms::H5DatatypeManager::datatype<
        ValueType<casacore::String>::DataType>();

    htri_t rc = H5Aexists(table_id, table_axes_uid_attr_name);
    assert(rc >= 0);

    hid_t axes_uid_id;
    if (rc == 0) {
      hid_t axes_uid_ds = H5Screate(H5S_SCALAR);
      assert(axes_uid_ds >= 0);
      axes_uid_id =
        H5Acreate(
          table_id,
          table_axes_uid_attr_name,
          axes_uid_dt,
          axes_uid_ds,
          H5P_DEFAULT,
          H5P_DEFAULT);
      H5Sclose(axes_uid_ds);
    } else {
      axes_uid_id = H5Aopen(table_id, table_axes_uid_attr_name, H5P_DEFAULT);
    }
    assert(axes_uid_id >= 0);
    try {
      char buff[LEGMS_MAX_STRING_SIZE];
      assert(std::strlen(table->axes_uid()) < sizeof(buff));
      strncpy(buff, table->axes_uid(), sizeof(buff));
      buff[sizeof(buff) - 1] = '\0';
      herr_t err = H5Awrite(axes_uid_id, axes_uid_dt, buff);
      assert(err >= 0);
    } catch (...) {
      herr_t err = H5Aclose(axes_uid_id);
      assert(err >= 0);
      throw;
    }
    herr_t err = H5Aclose(axes_uid_id);
    assert(err >= 0);
  }

  // write index axes attribute to table
  try {
    hid_t index_axes_dt =
      legms::H5DatatypeManager::datatype<ValueType<int>::DataType>();

    htri_t rc = H5Aexists(table_id, table_index_axes_attr_name);
    if (rc > 0) {
      herr_t err = H5Adelete(table_id, table_index_axes_attr_name);
      assert(err >= 0);
    }

    hsize_t dims = table->index_axes().size();
    hid_t index_axes_ds = H5Screate_simple(1, &dims, NULL);
    assert(index_axes_ds >= 0);

    try {
      hid_t index_axes_id =
        H5Acreate(
          table_id,
          table_index_axes_attr_name,
          index_axes_dt,
          index_axes_ds,
          H5P_DEFAULT,
          H5P_DEFAULT);
      try {
        assert(index_axes_id >= 0);
        herr_t err =
          H5Awrite(index_axes_id, index_axes_dt, table->index_axes().data());
        assert(err >= 0);
      } catch (...) {
        herr_t err = H5Aclose(index_axes_id);
        assert(err >= 0);
        throw;
      }
      herr_t err = H5Aclose(index_axes_id);
      assert(err >= 0);
    } catch (...) {
      herr_t err = H5Sclose(index_axes_ds);
      assert(err >= 0);
      throw;
    }
    herr_t err = H5Sclose(index_axes_ds);
    assert(err >= 0);

    for_each(
      table->column_names().begin(),
      table->column_names().end(),
      [&path, &table, &excluded_columns, &with_data, table_id](auto& nm) {
        if (excluded_columns.count(nm) == 0)
          write_column(
            path,
            table_id,
            table->name(),
            table->column(nm).get(),
            with_data);
      });

    write_keywords(table_id, table, with_data);

  } catch (...) {
    herr_t err = H5Gclose(table_id);
    assert(err >= 0);
    throw;
  }
  herr_t err = H5Gclose(table_id);
  assert(err >= 0);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
