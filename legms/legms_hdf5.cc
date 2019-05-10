#include "legms_hdf5.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <optional>
#include <sstream>

#include "tree_index_space.h"

using namespace legms::hdf5;
using namespace std;
using namespace Legion;

optional<string>
legms::hdf5::read_index_tree_attr_metadata(
  hid_t loc_id,
  const string& attr_name,
  hid_t access_pl) {

  optional<string> result;

  string md_id_name = string(LEGMS_ATTRIBUTE_SID_PREFIX) + attr_name;
  if (H5Aexists(loc_id, md_id_name.c_str())) {

    hid_t attr_id = H5Aopen(loc_id, md_id_name.c_str(), access_pl);

    if (attr_id >= 0) {
      hid_t attr_type = H5Aget_type(attr_id);

      hid_t attr_dt =
        H5DatatypeManager::datatype<ValueType<casacore::String>::DataType>();
      if (H5Tequal(attr_type, attr_dt) > 0) {
        char metadata[LEGMS_MAX_STRING_SIZE];
        herr_t rc = H5Aread(attr_id, attr_dt, metadata);
        assert(rc >= 0);
        result = string(metadata);
      }
    }
  }
  return result;
}

template <casacore::DataType DT>
using KW =
  FieldAccessor<
  READ_ONLY,
  typename legms::DataType<DT>::ValueType,
  1,
  coord_t,
  AffineAccessor<
    typename legms::DataType<DT>::ValueType,
    1,
    coord_t>,
  false>;

static void
init_datatype_attr(
  hid_t loc_id,
  casacore::DataType dt,
  hid_t creation_pl = H5P_DEFAULT,
  hid_t access_pl = H5P_DEFAULT);

static void
init_datatype_attr(
  hid_t loc_id,
  casacore::DataType dt,
  hid_t creation_pl,
  hid_t access_pl) {

  htri_t rc = H5Aexists(loc_id, LEGMS_ATTRIBUTE_DT);
  if (rc > 0) {
    herr_t err = H5Adelete(loc_id, LEGMS_ATTRIBUTE_DT);
    assert(err >= 0);
  }

  hid_t ds = H5Screate(H5S_SCALAR);
  assert(ds >= 0);
  hid_t did = legms::H5DatatypeManager::datatypes()[
    legms::H5DatatypeManager::CASACORE_DATATYPE_H5T];
  hid_t attr_id =
    H5Acreate(
      loc_id,
      LEGMS_ATTRIBUTE_DT,
      did,
      ds,
      creation_pl,
      access_pl);
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

static hid_t
init_kw(
  hid_t loc_id,
  const char *attr_name,
  hid_t type_id,
  casacore::DataType dt,
  hid_t link_creation_pl = H5P_DEFAULT,
  hid_t link_access_pl = H5P_DEFAULT,
  hid_t dataset_creation_pl = H5P_DEFAULT,
  hid_t dataset_access_pl = H5P_DEFAULT);

static hid_t
init_kw(
  hid_t loc_id,
  const char *attr_name,
  hid_t type_id,
  casacore::DataType dt,
  hid_t link_creation_pl,
  hid_t link_access_pl,
  hid_t dataset_creation_pl,
  hid_t dataset_access_pl) {

  {
    htri_t rc = H5Lexists(loc_id, attr_name, link_access_pl);
    assert(rc >= 0);
    if (rc > 0) {
      herr_t err = H5Ldelete(loc_id, attr_name, link_access_pl);
      assert(err >= 0);
    }
  }
  hid_t result;
  {
    hid_t attr_ds = H5Screate(H5S_SCALAR);
    assert(attr_ds >= 0);
    result =
      H5Dcreate(
        loc_id,
        attr_name,
        type_id,
        attr_ds,
        link_creation_pl,
        dataset_creation_pl,
        dataset_access_pl);
    assert(result >= 0);
    herr_t err = H5Sclose(attr_ds);
    assert(err >= 0);
  }
  init_datatype_attr(result, dt);
  return result;
}

template <casacore::DataType DT>
static void
write_kw(
  hid_t loc_id,
  const char *attr_name,
  optional<PhysicalRegion>& region,
  FieldID fid,
  hid_t link_creation_pl = H5P_DEFAULT,
  hid_t link_access_pl = H5P_DEFAULT,
  hid_t dataset_creation_pl = H5P_DEFAULT,
  hid_t dataset_access_pl = H5P_DEFAULT,
  hid_t xfer_pl = H5P_DEFAULT);

template <casacore::DataType DT>
static void
write_kw(
  hid_t loc_id,
  const char *attr_name,
  optional<PhysicalRegion>& region,
  FieldID fid,
  hid_t link_creation_pl,
  hid_t link_access_pl,
  hid_t dataset_creation_pl,
  hid_t dataset_access_pl,
  hid_t xfer_pl) {

  hid_t dt = legms::H5DatatypeManager::datatype<DT>();
  hid_t attr_id =
    init_kw(
      loc_id,
      attr_name,
      dt,
      DT,
      link_creation_pl,
      link_access_pl,
      dataset_creation_pl,
      dataset_access_pl);
  if (region) {
    const KW<DT> kw(region.value(), fid);
    herr_t err = H5Dwrite(attr_id, dt, H5S_ALL, H5S_ALL, xfer_pl, kw.ptr(0));
    assert(err >= 0);
  }
  herr_t err = H5Dclose(attr_id);
  assert(err >= 0);
}

template <>
void
write_kw<casacore::TpString> (
  hid_t loc_id,
  const char *attr_name,
  optional<PhysicalRegion>& region,
  FieldID fid,
  hid_t link_creation_pl,
  hid_t link_access_pl,
  hid_t dataset_creation_pl,
  hid_t dataset_access_pl,
  hid_t xfer_pl) {

  hid_t dt = legms::H5DatatypeManager::datatype<casacore::TpString>();
  hid_t attr_id =
    init_kw(
      loc_id,
      attr_name,
      dt,
      casacore::TpString,
      link_creation_pl,
      link_access_pl,
      dataset_creation_pl,
      dataset_access_pl);
  if (region) {
    const KW<casacore::TpString> kw(region.value(), fid);
    const string& val = kw[0];
    char buf[LEGMS_MAX_STRING_SIZE];
    assert(val.size() < sizeof(buf));
    strncpy(buf, val.c_str(), sizeof(buf));
    buf[sizeof(buf) - 1] = '\0';
    herr_t err = H5Dwrite(attr_id, dt, H5S_ALL, H5S_ALL, xfer_pl, buf);
    assert(err >= 0);
  }
  herr_t err = H5Dclose(attr_id);
  assert(err >= 0);
}

void
legms::hdf5::write_keywords(
  hid_t loc_id,
  const WithKeywords* with_keywords,
  bool with_data,
  hid_t link_creation_pl,
  hid_t link_access_pl,
  hid_t dataset_creation_pl,
  hid_t dataset_access_pl,
  hid_t xfer_pl) {

  if (with_keywords->keywords_region() == LogicalRegion::NO_REGION)
    return;

  Runtime* runtime = with_keywords->runtime();
  Context context = with_keywords->context();

  optional<PhysicalRegion> pr;
  if (with_data) {
    RegionRequirement req(
      with_keywords->keywords_region(),
      READ_ONLY,
      EXCLUSIVE,
      with_keywords->keywords_region());
    vector<FieldID> fids(with_keywords->num_keywords());
    iota(fids.begin(), fids.end(), 0);
    req.add_fields(fids);
    pr = runtime->map_region(context, req);
  }

  auto kws = with_keywords->keywords();
  for (size_t i = 0; i < kws.size(); ++i) {
    auto [nm, dt] = kws[i];
    assert(nm.substr(0, sizeof(LEGMS_NAMESPACE_PREFIX) - 1)
           != LEGMS_NAMESPACE_PREFIX);

#define WRITE_KW(DT)                            \
    case (DT): {                                \
      write_kw<DT>(                             \
        loc_id,                                 \
        nm.c_str(),                             \
        pr,                                     \
        i,                                      \
        link_creation_pl,                       \
        link_access_pl,                         \
        dataset_creation_pl,                    \
        dataset_access_pl,                      \
        xfer_pl);                               \
      break;                                    \
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
  hid_t link_creation_pl,
  hid_t link_access_pl,
  hid_t group_creation_pl,
  hid_t group_access_pl,
  hid_t dataset_creation_pl,
  hid_t dataset_access_pl,
  hid_t attr_creation_pl,
  hid_t attr_access_pl,
  hid_t xfer_pl) {

  Runtime* runtime = column->runtime();
  Context context = column->context();

  // delete column dataset if it exists
  htri_t ds_exists =
    H5Lexists(table_id, column->name().c_str(), link_access_pl);
  if (ds_exists > 0) {
    herr_t err = H5Ldelete(table_id, column->name().c_str(), link_access_pl);
    assert(err >= 0);
  } else {
    assert(ds_exists == 0);
  }

  // create column group
  hid_t col_group_id =
    H5Gcreate(
      table_id,
      column->name().c_str(),
      link_creation_pl,
      group_creation_pl,
      group_access_pl);
  assert(col_group_id >= 0);

  // create column dataset
  hid_t col_id;
  {
    int rank = column->index_space().get_dim();
    hsize_t dims[rank];

#define DIMS(N) \
    case N: {\
      Rect<N> rect =                                            \
        runtime->get_index_space_domain(context, column->index_space()) \
        .bounds<N, coord_t>();                                  \
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

#define DT(T) case T: { dt = H5DatatypeManager::datatype<T>(); break; }

    switch (column->datatype()) {
      FOREACH_DATATYPE(DT)
    default:
      assert(false);
      break;
    }
#undef DT

    col_id =
      H5Dcreate(
        col_group_id,
        LEGMS_COLUMN_DS,
        dt,
        ds,
        link_creation_pl,
        dataset_creation_pl,
        dataset_access_pl);
    assert(col_id >= 0);
    herr_t err = H5Sclose(ds);
    assert(err >= 0);

    // write column value datatype
    init_datatype_attr(col_id, column->datatype());

    err = H5Dclose(col_id);
    assert(err >= 0);
  }

  // write axes attribute to column
  {
    hid_t axes_dt = H5DatatypeManager::datatype<ValueType<int>::DataType>();

    htri_t rc = H5Aexists(col_group_id, column_axes_attr_name);
    if (rc > 0) {
      herr_t err = H5Adelete(col_group_id, column_axes_attr_name);
      assert(err >= 0);
    }

    hsize_t dims = column->axes().size();
    hid_t axes_ds = H5Screate_simple(1, &dims, NULL);
    assert(axes_ds >= 0);

    try {
      hid_t axes_id =
        H5Acreate(
          col_group_id,
          column_axes_attr_name,
          axes_dt,
          axes_ds,
          attr_creation_pl,
          attr_access_pl);
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
    string column_ds_name =
      string("/") + table_name + "/" + column->name() + "/" + LEGMS_COLUMN_DS;
    map<FieldID, const char*>
      field_map{{Column::value_fid, column_ds_name.c_str()}};
    LogicalRegion values_lr =
      runtime->create_logical_region(
        context,
        column->index_space(),
        column->logical_region().get_field_space());
    AttachLauncher attach(EXTERNAL_HDF5_FILE, values_lr, values_lr);
    attach.attach_hdf5(path.c_str(), field_map, LEGION_FILE_READ_WRITE);
    PhysicalRegion values_pr =
      runtime->attach_external_resource(context, attach);
    RegionRequirement src(
      column->logical_region(),
      READ_ONLY,
      EXCLUSIVE,
      column->logical_region());
    src.add_field(Column::value_fid);
    RegionRequirement dst(
      values_lr,
      WRITE_ONLY,
      EXCLUSIVE,
      values_lr);
    dst.add_field(Column::value_fid);
    CopyLauncher copy;
    copy.add_copy_requirements(src, dst);
    runtime->issue_copy_operation(context, copy);
    runtime->detach_external_resource(context, values_pr);
  }
  write_keywords(col_group_id, column, with_data);
  herr_t err = H5Gclose(col_group_id);
  assert(err >= 0);

  write_index_tree_to_attr<binary_index_tree_serdez>(
    column->index_tree(),
    table_id,
    column->name(),
    "index_tree",
    link_creation_pl,
    link_access_pl,
    dataset_creation_pl,
    dataset_access_pl,
    xfer_pl);
}

void
legms::hdf5::write_table(
  const experimental::filesystem::path& path,
  hid_t loc_id,
  const Table* table,
  const unordered_set<string>& excluded_columns,
  bool with_data,
  hid_t link_creation_pl,
  hid_t link_access_pl,
  hid_t group_creation_pl,
  hid_t group_access_pl,
  hid_t dataset_creation_pl,
  hid_t dataset_access_pl,
  hid_t attr_creation_pl,
  hid_t attr_access_pl,
  hid_t xfer_pl) {

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
      H5DatatypeManager::datatype<ValueType<casacore::String>::DataType>();

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
          attr_creation_pl,
          attr_access_pl);
      H5Sclose(axes_uid_ds);
    } else {
      axes_uid_id = H5Aopen(table_id, table_axes_uid_attr_name, attr_access_pl);
    }
    assert(axes_uid_id >= 0);
    try {
      char buff[LEGMS_MAX_STRING_SIZE];
      assert(strlen(table->axes_uid()) < sizeof(buff));
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
      H5DatatypeManager::datatype<ValueType<int>::DataType>();

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
          attr_creation_pl,
          attr_access_pl);
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
      [&](auto& nm) {
        if (excluded_columns.count(nm) == 0)
          write_column(
            path,
            table_id,
            table->name(),
            table->column(nm).get(),
            with_data,
            link_creation_pl,
            link_access_pl,
            group_creation_pl,
            group_access_pl,
            dataset_creation_pl,
            dataset_access_pl,
            attr_creation_pl,
            attr_access_pl,
            xfer_pl);
      });

    write_keywords(
      table_id,
      table,
      with_data,
      link_creation_pl,
      link_access_pl,
      dataset_creation_pl,
      dataset_access_pl,
      xfer_pl);

  } catch (...) {
    herr_t err = H5Gclose(table_id);
    assert(err >= 0);
    throw;
  }
  herr_t err = H5Gclose(table_id);
  assert(err >= 0);
}

static bool
starts_with(const char* str, const char* pref) {
  bool result = true;
  while (result && *str != '\0' && *pref != '\0') {
    result = *str == *pref;
    ++str; ++pref;
  }
  return result && *pref == '\0';
}

// static bool
// ends_with(const char* str, const char* suff) {
//   bool result = true;
//   const char* estr = strchr(str, '\0');
//   const char* esuff = strchr(suff, '\0');
//   do {
//     --estr; --esuff;
//     result = *estr == *esuff;
//   } while (result && estr != str && esuff != suff);
//   return result && esuff == suff;
// }

static herr_t
acc_kw_names(
  hid_t loc_id,
  const char* name,
  const H5L_info_t*,
  void* ctx) {

  vector<string>* acc = static_cast<vector<string>*>(ctx);
  if (!starts_with(name, LEGMS_NAMESPACE_PREFIX)) {
    H5O_info_t infobuf;
    herr_t err = H5Oget_info_by_name(loc_id, name, &infobuf, H5P_DEFAULT);
    assert(err >= 0);
    if (infobuf.type == H5O_TYPE_DATASET)
      acc->push_back(name);
  }
  return 0;
}

tuple<LogicalRegion, vector<casacore::DataType>>
legms::hdf5::init_keywords(
  hid_t loc_id,
  Runtime* runtime,
  Context context,
  hid_t attr_access_pl,
  hid_t link_access_pl) {

  vector<string> kw_names;
  hsize_t n = 0;
  herr_t err =
    H5Literate(
      loc_id,
      H5_INDEX_NAME,
      H5_ITER_INC,
      &n,
      acc_kw_names,
      &kw_names);
  assert(err >= 0);

  if (kw_names.size() == 0)
    return make_tuple(LogicalRegion::NO_REGION, vector<casacore::DataType>());

  WithKeywords::kw_desc_t kws;
  hid_t did =
    legms::H5DatatypeManager::datatypes()[
      legms::H5DatatypeManager::CASACORE_DATATYPE_H5T];
  transform(
    kw_names.begin(),
    kw_names.end(),
    back_inserter(kws),
    [&](auto& nm) {
      hid_t dt_id =
        H5Aopen_by_name(
          loc_id,
          nm.c_str(),
          LEGMS_ATTRIBUTE_DT,
          attr_access_pl,
          link_access_pl);
      assert(dt_id >= 0);

      unsigned char dt;
      herr_t err = H5Aread(dt_id, did, &dt);
      assert(err >= 0);
      err = H5Aclose(dt_id);
      assert(err >= 0);
      return make_tuple(nm, static_cast<casacore::DataType>(dt));
    });

  IndexSpaceT<1> is =
    runtime->create_index_space(context, Rect<1>(0, 0));
  FieldSpace fs = runtime->create_field_space(context);
  FieldAllocator fa = runtime->create_field_allocator(context, fs);
  vector<casacore::DataType> dts;
  for (size_t i = 0; i < kws.size(); ++i) {
    auto [nm, dt] = kws[i];
    add_field(dt, fa, i);
    runtime->attach_name(fs, i, nm.c_str());
    dts.push_back(dt);
  }
  LogicalRegion region =
    runtime->create_logical_region(context, is, fs);
  runtime->destroy_field_space(context, fs);
  runtime->destroy_index_space(context, is);
  return make_tuple(region, move(dts));
}

optional<legms::ColumnGenArgs>
legms::hdf5::init_column(
  hid_t loc_id,
  Runtime* runtime,
  Context context,
  hid_t attr_access_pl,
  hid_t link_access_pl,
  hid_t xfer_pl) {

  optional<ColumnGenArgs> result;

  casacore::DataType datatype = ValueType<int>::DataType;
  hid_t datatype_id = -1;
  vector<int> axes;
  hid_t axes_id = -1;
  hid_t axes_id_ds = -1;
  LogicalRegion values = LogicalRegion::NO_REGION;
  LogicalRegion keywords = LogicalRegion::NO_REGION;
  vector<casacore::DataType> keyword_datatypes;

  htri_t rc = H5Lexists(loc_id, LEGMS_COLUMN_DS, link_access_pl);
  if (rc > 0) {
    H5O_info_t infobuf;
    H5Oget_info_by_name(loc_id, LEGMS_COLUMN_DS, &infobuf, H5P_DEFAULT);
    if (infobuf.type == H5O_TYPE_DATASET) {
      {
        htri_t axes_exists = H5Aexists(loc_id, column_axes_attr_name);
        assert(axes_exists >= 0);
        if (axes_exists == 0)
          goto return_nothing;
        axes_id = H5Aopen(loc_id, column_axes_attr_name, attr_access_pl);
        assert(axes_id >= 0);
        axes_id_ds = H5Aget_space(axes_id);
        assert(axes_id_ds >= 0);
        int ndims = H5Sget_simple_extent_ndims(axes_id_ds);
        if (ndims != 1)
          goto return_nothing;
        axes.resize(H5Sget_simple_extent_npoints(axes_id_ds));
        hid_t axes_dt = H5DatatypeManager::datatype<ValueType<int>::DataType>();
        herr_t err = H5Aread(axes_id, axes_dt, axes.data());
        assert(err >= 0);
      }
      {
        optional<string> sid =
          read_index_tree_attr_metadata(loc_id, "index_tree");
        if (!sid || sid.value() != "legms::hdf5::binary_index_tree_serdez")
          goto return_nothing;
        optional<IndexTreeL> ixtree =
          read_index_tree_from_attr<binary_index_tree_serdez>(
            loc_id,
            "index_tree",
            attr_access_pl,
            xfer_pl);
        assert(ixtree);
        IndexSpace is = tree_index_space(ixtree.value(), context, runtime);
        FieldSpace fs = runtime->create_field_space(context);
        FieldAllocator fa = runtime->create_field_allocator(context, fs);
        add_field(datatype, fa, Column::value_fid);
        values = runtime->create_logical_region(context, is, fs);
        runtime->destroy_field_space(context, fs);
        runtime->destroy_index_space(context, is);
      }
      {
        string datatype_name(LEGMS_ATTRIBUTE_DT);
        htri_t datatype_exists =
          H5Aexists_by_name(
            loc_id,
            LEGMS_COLUMN_DS,
            datatype_name.c_str(),
            link_access_pl);
        assert(datatype_exists >= 0);
        hid_t did =
          H5DatatypeManager::datatypes()[
            H5DatatypeManager::CASACORE_DATATYPE_H5T];
        unsigned char dt;
        if (datatype_exists == 0)
          goto return_nothing;
        datatype_id =
          H5Aopen_by_name(
            loc_id,
            LEGMS_COLUMN_DS,
            datatype_name.c_str(),
            attr_access_pl,
            link_access_pl);
        assert(datatype_id >= 0);
        herr_t err = H5Aread(datatype_id, did, &dt);
        assert(err >= 0);
        datatype = static_cast<casacore::DataType>(dt);
      }
    }
  }
  tie(keywords, keyword_datatypes) =
    init_keywords(loc_id, runtime, context, attr_access_pl, link_access_pl);

  result =
    ColumnGenArgs{"", "", datatype, axes, values, keywords, keyword_datatypes};

return_nothing:
  if (datatype_id >= 0) {
    herr_t err = H5Aclose(datatype_id);
    assert(err >= 0);
  }
  if (axes_id_ds >= 0) {
    herr_t err = H5Sclose(axes_id_ds);
    assert(err >= 0);
  }
  if (axes_id >= 0) {
    herr_t err = H5Aclose(axes_id);
    assert(err >= 0);
  }
  return result;
}

struct acc_col_genargs_ctx {
  vector<legms::ColumnGenArgs>* acc;
  string axes_uid;
  hid_t attr_access_pl;
  hid_t link_access_pl;
  hid_t xfer_pl;
  Runtime* runtime;
  Context context;
};

static herr_t
acc_col_genargs(
  hid_t table_id,
  const char* name,
  const H5L_info_t*,
  void* ctx) {

  struct acc_col_genargs_ctx *args =
    static_cast<struct acc_col_genargs_ctx*>(ctx);
  htri_t rc = H5Lexists(table_id, name, H5P_DEFAULT);
  if (rc > 0) {
    H5O_info_t infobuf;
    H5Oget_info_by_name(table_id, name, &infobuf, H5P_DEFAULT);
    if (infobuf.type == H5O_TYPE_GROUP) {
      hid_t col_group_id = H5Gopen(table_id, name, H5P_DEFAULT);
      assert(col_group_id >= 0);
      auto cga =
        init_column(
          col_group_id,
          args->runtime,
          args->context,
          args->attr_access_pl,
          args->link_access_pl,
          args->xfer_pl);
      if (cga) {
        legms::ColumnGenArgs& a = cga.value();
        a.name = name;
        a.axes_uid = args->axes_uid;
        args->acc->push_back(move(a));
      }
    }
  }
  return 0;
}

optional<legms::TableGenArgs>
legms::hdf5::init_table(
  hid_t loc_id,
  Runtime* runtime,
  Context context,
  hid_t attr_access_pl,
  hid_t link_access_pl,
  hid_t xfer_pl) {

  optional<TableGenArgs> result;

  vector<int> index_axes;
  hid_t index_axes_id = -1;
  hid_t index_axes_id_ds = -1;
  string axes_uid;
  vector<ColumnGenArgs> col_genargs;
  LogicalRegion keywords = LogicalRegion::NO_REGION;
  vector<casacore::DataType> keyword_datatypes;
  {
    htri_t index_axes_exists = H5Aexists(loc_id, table_index_axes_attr_name);
    assert(index_axes_exists >= 0);
    if (index_axes_exists == 0)
      goto return_nothing;
    index_axes_id =
      H5Aopen(loc_id, table_index_axes_attr_name, attr_access_pl);
    assert(index_axes_id >= 0);
    index_axes_id_ds = H5Aget_space(index_axes_id);
    assert(index_axes_id_ds >= 0);
    int ndims = H5Sget_simple_extent_ndims(index_axes_id_ds);
    if (ndims != 1)
      goto return_nothing;
    index_axes.resize(H5Sget_simple_extent_npoints(index_axes_id_ds));
    hid_t index_axes_dt =
      H5DatatypeManager::datatype<ValueType<int>::DataType>();
    herr_t err = H5Aread(index_axes_id, index_axes_dt, index_axes.data());
    assert(err >= 0);
  }
  {
    htri_t axes_uid_exists = H5Aexists(loc_id, table_axes_uid_attr_name);
    assert(axes_uid_exists >= 0);
    if (axes_uid_exists == 0)
      goto return_nothing;
    hid_t did =
      H5DatatypeManager::datatypes()[H5DatatypeManager::CASACORE_STRING_H5T];
    char str[LEGMS_MAX_STRING_SIZE];
    hid_t axes_uid_id =
      H5Aopen(loc_id, table_axes_uid_attr_name, attr_access_pl);
    assert(axes_uid_id >= 0);
    herr_t err = H5Aread(axes_uid_id, did, str);
    assert(err >= 0);
    axes_uid = str;
    err = H5Aclose(axes_uid_id);
    assert(err >= 0);
  }
  {
    struct acc_col_genargs_ctx ctx{
      &col_genargs, axes_uid, attr_access_pl, link_access_pl, xfer_pl,
      runtime, context};
    hsize_t position = 0;
    herr_t err =
      H5Literate(
        loc_id,
        H5_INDEX_NAME,
        H5_ITER_NATIVE,
        &position,
        acc_col_genargs,
        &ctx);
    assert(err >= 0);
  }

  tie(keywords, keyword_datatypes) =
    init_keywords(loc_id, runtime, context, attr_access_pl, link_access_pl);

  result =
    TableGenArgs{
    "",
    axes_uid,
    index_axes,
    col_genargs,
    keywords,
    keyword_datatypes};

return_nothing:
  if (index_axes_id_ds >= 0) {
    herr_t err = H5Sclose(index_axes_id_ds);
    assert(err >= 0);
  }
  if (index_axes_id >= 0) {
    herr_t err = H5Aclose(index_axes_id);
    assert(err >= 0);
  }
  return result;
}

std::optional<legms::TableGenArgs>
legms::hdf5::init_table(
  const experimental::filesystem::path& file_path,
  const string& table_path,
  Runtime* runtime,
  Context context,
  unsigned flags,
  hid_t file_access_pl,
  hid_t table_access_pl,
  hid_t attr_access_pl,
  hid_t link_access_pl,
  hid_t xfer_pl) {

  optional<TableGenArgs> result;

  hid_t fid = H5Fopen(file_path.c_str(), flags, file_access_pl);
  if (fid >= 0) {
    try {
      hid_t table_loc = H5Gopen(fid, table_path.c_str(), table_access_pl);
      if (table_loc >= 0) {
        try {
          result =
            init_table(
              table_loc,
              runtime,
              context,
              attr_access_pl,
              link_access_pl,
              xfer_pl);
          result.value().name = table_path.substr(table_path.rfind('/') + 1);
        } catch (...) {
          herr_t err = H5Gclose(table_loc);
          assert(err >= 0);
          throw;
        }
        herr_t err = H5Gclose(table_loc);
        assert(err >= 0);
      }
    } catch (...) {
      herr_t err = H5Fclose(fid);
      assert(err >= 0);
      throw;
    }
    herr_t err = H5Fclose(fid);
    assert(err >= 0);
  }
  return result;
}

optional<PhysicalRegion>
legms::hdf5::attach_keywords(
  const experimental::filesystem::path& file_path,
  const string& with_keywords_path,
  const WithKeywords* with_keywords,
  Runtime* runtime,
  Context context,
  bool read_only) {

  optional<PhysicalRegion> result;
  auto kws = with_keywords->keywords_region();
  if (kws != LogicalRegion::NO_REGION) {
    vector<string> field_paths(with_keywords->num_keywords());
    map<FieldID, const char*> fields;
    for (size_t i = 0; i < with_keywords->num_keywords(); ++i) {
      field_paths[i] =
        with_keywords_path + "/" + get<0>(with_keywords->keywords()[i]);
      fields[i] = field_paths[i].c_str();
    }
    AttachLauncher kws_attach(EXTERNAL_HDF5_FILE, kws, kws);
    kws_attach.attach_hdf5(
      file_path.c_str(),
      fields,
      read_only ? LEGION_FILE_READ_ONLY : LEGION_FILE_READ_WRITE);
    result = runtime->attach_external_resource(context, kws_attach);
  }
  return result;
}

vector<
  tuple<
    optional<PhysicalRegion>,
    optional<PhysicalRegion>>>
legms::hdf5::attach_table_columns(
  const experimental::filesystem::path& file_path,
  const string& root_path,
  const Table* table,
  const std::vector<std::string>& columns,
  Runtime* runtime,
  Context context,
  bool read_only) {

  vector<
    tuple<
      optional<PhysicalRegion>,
      optional<PhysicalRegion>>> result;
  string table_root = root_path;
  if (table_root.back() != '/')
    table_root.push_back('/');
  table_root += table->name() + "/";
  transform(
    columns.begin(),
    columns.end(),
    back_inserter(result),
    [&file_path,&table_root, &table, &runtime, &context, &read_only](auto& nm) {
      auto c = table->column(nm);
      tuple<optional<PhysicalRegion>, optional<PhysicalRegion>> regions;
      if (c) {
        auto col = c->logical_region();
        if (col != LogicalRegion::NO_REGION) {
          AttachLauncher col_attach(EXTERNAL_HDF5_FILE, col, col);
          string col_path = table_root + c->name() + "/" + LEGMS_COLUMN_DS;
          map<FieldID, const char*>
            fields{{Column::value_fid, col_path.c_str()}};
          col_attach.attach_hdf5(
            file_path.c_str(),
            fields,
            read_only ? LEGION_FILE_READ_ONLY : LEGION_FILE_READ_WRITE);
          get<0>(regions) =
            runtime->attach_external_resource(context, col_attach);
          get<1>(regions) =
            attach_keywords(
              file_path,
              col_path,
              c.get(),
              runtime,
              context,
              read_only);
        }
      }
      return regions;
    });
  return result;
}

optional<PhysicalRegion>
legms::hdf5::attach_table_keywords(
  const experimental::filesystem::path& file_path,
  const string& root_path,
  const Table* table,
  Runtime* runtime,
  Context context,
  bool read_only) {

  string table_root = root_path;
  if (table_root.back() != '/')
    table_root.push_back('/');
  table_root += table->name();
  return
    attach_keywords(file_path, table_root, table, runtime, context, read_only);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
