#include "legms_hdf5.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <optional>
#include <sstream>

#include "tree_index_space.h"
#include "MSTable.h"

using namespace legms::hdf5;
using namespace legms;
using namespace Legion;

const char* table_index_axes_attr_name =
  LEGMS_NAMESPACE_PREFIX "index_axes";
const char* table_axes_dt_name =
  LEGMS_NAMESPACE_PREFIX "table_axes";
const char* column_axes_attr_name =
  LEGMS_NAMESPACE_PREFIX "axes";

std::optional<std::string>
legms::hdf5::read_index_tree_attr_metadata(
  hid_t loc_id,
  const std::string& attr_name,
  hid_t access_pl) {

  std::optional<std::string> result;

  std::string md_id_name = std::string(LEGMS_ATTRIBUTE_SID_PREFIX) + attr_name;
  if (H5Aexists(loc_id, md_id_name.c_str())) {

    hid_t attr_id = H5Aopen(loc_id, md_id_name.c_str(), access_pl);

    if (attr_id >= 0) {
      hid_t attr_type = H5Aget_type(attr_id);

      hid_t attr_dt =
        H5DatatypeManager::datatype<ValueType<std::string>::DataType>();
      if (H5Tequal(attr_type, attr_dt) > 0) {
        string attr;
        herr_t rc = H5Aread(attr_id, attr_dt, attr.val);
        assert(rc >= 0);
        result = attr.val;
      }
    }
  }
  return result;
}

template <legms::TypeTag DT>
using KW =
  FieldAccessor<
  READ_ONLY,
  typename DataType<DT>::ValueType,
  1,
  coord_t,
  AffineAccessor<
    typename DataType<DT>::ValueType,
    1,
    coord_t>,
  false>;

static void
init_datatype_attr(
  hid_t loc_id,
  legms::TypeTag dt,
  hid_t creation_pl = H5P_DEFAULT,
  hid_t access_pl = H5P_DEFAULT);

static void
init_datatype_attr(
  hid_t loc_id,
  legms::TypeTag dt,
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
    legms::H5DatatypeManager::DATATYPE_H5T];
  hid_t attr_id =
    H5Acreate(
      loc_id,
      LEGMS_ATTRIBUTE_DT,
      did,
      ds,
      creation_pl,
      access_pl);
  assert(attr_id >= 0);
  herr_t err = H5Awrite(attr_id, did, &dt);
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
  legms::TypeTag dt,
  hid_t link_creation_pl = H5P_DEFAULT,
  hid_t link_access_pl = H5P_DEFAULT,
  hid_t dataset_creation_pl = H5P_DEFAULT,
  hid_t dataset_access_pl = H5P_DEFAULT);

static hid_t
init_kw(
  hid_t loc_id,
  const char *attr_name,
  hid_t type_id,
  legms::TypeTag dt,
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

template <legms::TypeTag DT>
static void
write_kw(
  hid_t loc_id,
  const char *attr_name,
  std::optional<PhysicalRegion>& region,
  FieldID fid,
  hid_t link_creation_pl = H5P_DEFAULT,
  hid_t link_access_pl = H5P_DEFAULT,
  hid_t dataset_creation_pl = H5P_DEFAULT,
  hid_t dataset_access_pl = H5P_DEFAULT,
  hid_t xfer_pl = H5P_DEFAULT);

template <legms::TypeTag DT>
static void
write_kw(
  hid_t loc_id,
  const char *attr_name,
  std::optional<PhysicalRegion>& region,
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
write_kw<LEGMS_TYPE_STRING> (
  hid_t loc_id,
  const char *attr_name,
  std::optional<PhysicalRegion>& region,
  FieldID fid,
  hid_t link_creation_pl,
  hid_t link_access_pl,
  hid_t dataset_creation_pl,
  hid_t dataset_access_pl,
  hid_t xfer_pl) {

  hid_t dt = legms::H5DatatypeManager::datatype<LEGMS_TYPE_STRING>();
  hid_t attr_id =
    init_kw(
      loc_id,
      attr_name,
      dt,
      LEGMS_TYPE_STRING,
      link_creation_pl,
      link_access_pl,
      dataset_creation_pl,
      dataset_access_pl);
  if (region) {
    const KW<LEGMS_TYPE_STRING> kw(region.value(), fid);
    const legms::string& kwval = kw[0];
    legms::string buf;
    strncpy(buf.val, kwval.val, sizeof(buf));
    buf.val[sizeof(buf.val) - 1] = '\0';
    herr_t err = H5Dwrite(attr_id, dt, H5S_ALL, H5S_ALL, xfer_pl, buf.val);
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

  std::optional<PhysicalRegion> pr;
  if (with_data) {
    RegionRequirement req(
      with_keywords->keywords_region(),
      READ_ONLY,
      EXCLUSIVE,
      with_keywords->keywords_region());
    std::vector<FieldID> fids(with_keywords->num_keywords());
    std::iota(fids.begin(), fids.end(), 0);
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
      LEGMS_FOREACH_DATATYPE(WRITE_KW)
    default:
        assert(false);
    }

#undef WRITE_KW
  }
}

void
legms::hdf5::write_column(
  const std::experimental::filesystem::path& path,
  hid_t table_id,
  const std::string& table_name,
  const Column* column,
  hid_t table_axes_dt,
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
    hsize_t dims[column->rank()];

#define DIMS(N) \
    case N: {\
      Rect<N> rect =                                            \
        runtime->get_index_space_domain(context, column->index_space()) \
        .bounds<N, coord_t>();                                  \
      for (size_t i = 0; i < N; ++i)                                    \
        dims[i] = rect.hi[i] + 1;                                       \
      break;                                                            \
    }

    switch (column->rank()) {
      LEGMS_FOREACH_N(DIMS)
    default:
      assert(false);
      break;
    }
#undef DIMS

    hid_t ds = H5Screate_simple(column->rank(), dims, NULL);
    assert(ds >= 0);

    hid_t dt;

#define DT(T) case T: { dt = H5DatatypeManager::datatype<T>(); break; }

    switch (column->datatype()) {
      LEGMS_FOREACH_DATATYPE(DT)
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
          table_axes_dt,
          axes_ds,
          attr_creation_pl,
          attr_access_pl);
      assert(axes_id >= 0);
      try {
        auto axes = column->axes();
        std::vector<unsigned char> ax;
        ax.reserve(axes.size());
        std::copy(axes.begin(), axes.end(), std::back_inserter(ax));
        assert(axes_id >= 0);
        herr_t err = H5Awrite(axes_id, table_axes_dt, ax.data());
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
    // FIXME: the value of column_ds_name is only correct when the table group
    // occurs at the HDF5 root...must add some way to pass in the path to the
    // table HDF5 group
    std::string column_ds_name =
      std::string("/") + table_name + "/" + column->name()
      + "/" + LEGMS_COLUMN_DS;
    std::map<FieldID, const char*>
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
  const std::experimental::filesystem::path& path,
  hid_t loc_id,
  const Table* table,
  const std::unordered_set<std::string>& excluded_columns,
  bool with_data,
  hid_t link_creation_pl,
  hid_t link_access_pl,
  hid_t group_creation_pl,
  hid_t group_access_pl,
  hid_t type_creation_pl,
  hid_t type_access_pl,
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

  // write axes datatype to table
  auto axes = AxesRegistrar::axes(table->axes_uid());
  assert(axes);
  hid_t table_axes_dt = axes.value().h5_datatype;
  {
    herr_t err =
      H5Tcommit(
        table_id,
        table_axes_dt_name,
        table_axes_dt,
        link_creation_pl,
        type_creation_pl,
        type_access_pl);
    assert(err >= 0);
  }

  // write index axes attribute to table
  try {
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
          table_axes_dt,
          index_axes_ds,
          attr_creation_pl,
          attr_access_pl);
      assert(index_axes_id >= 0);
      try {
        auto axes = table->index_axes();
        std::vector<unsigned char> ax;
        ax.reserve(axes.size());
        std::copy(axes.begin(), axes.end(), std::back_inserter(ax));
        herr_t err = H5Awrite(index_axes_id, table_axes_dt, ax.data());
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
        if (excluded_columns.count(nm) == 0
            && table->column(nm)->index_space() != IndexSpace::NO_SPACE)
          write_column(
            path,
            table_id,
            table->name(),
            table->column(nm).get(),
            table_axes_dt,
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

  std::vector<std::string>* acc = static_cast<std::vector<std::string>*>(ctx);
  if (!starts_with(name, LEGMS_NAMESPACE_PREFIX)) {
    H5O_info_t infobuf;
    herr_t err = H5Oget_info_by_name(loc_id, name, &infobuf, H5P_DEFAULT);
    assert(err >= 0);
    if (infobuf.type == H5O_TYPE_DATASET)
      acc->push_back(name);
  }
  return 0;
}

static legms::TypeTag
read_dt_value(hid_t dt_id) {
  legms::TypeTag dt;
  // enumeration datatypes are converted by libhdf5 based on symbol names, which
  // ensures interoperability for legms HDF5 files written with one enumeration
  // definition and read with a different enumeration definition (for example,
  // in two legms codes built with and without USE_CASACORE)
  herr_t err = H5Aread(dt_id, H5T_NATIVE_INT, &dt);
  assert(err >= 0);
  return dt;
}

std::tuple<LogicalRegion, std::vector<legms::TypeTag>>
legms::hdf5::init_keywords(
  Context context,
  Runtime* runtime,
  hid_t loc_id,
  hid_t attr_access_pl,
  hid_t link_access_pl) {

  std::vector<std::string> kw_names;
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
    return
      std::make_tuple(LogicalRegion::NO_REGION, std::vector<legms::TypeTag>());

  WithKeywords::kw_desc_t kws =
    legms::map(
      kw_names,
      [&](const auto& nm) {
        hid_t dt_id =
          H5Aopen_by_name(
            loc_id,
            nm.c_str(),
            LEGMS_ATTRIBUTE_DT,
            attr_access_pl,
            link_access_pl);
        assert(dt_id >= 0);
        legms::TypeTag dt = read_dt_value(dt_id);
        err = H5Aclose(dt_id);
        assert(err >= 0);
        return std::make_tuple(nm, dt);
      });

  IndexSpaceT<1> is =
    runtime->create_index_space(context, Rect<1>(0, 0));
  FieldSpace fs = runtime->create_field_space(context);
  FieldAllocator fa = runtime->create_field_allocator(context, fs);
  std::vector<legms::TypeTag> dts;
  for (size_t i = 0; i < kws.size(); ++i) {
    auto [nm, dt] = kws[i];
    add_field(dt, fa, i);
    runtime->attach_name(fs, i, nm.c_str());
    dts.push_back(dt);
  }
  LogicalRegion region = runtime->create_logical_region(context, is, fs);
  // TODO: remove?
  //runtime->destroy_field_space(context, fs);
  //runtime->destroy_index_space(context, is);
  return make_tuple(region, move(dts));
}

std::optional<legms::ColumnGenArgs>
legms::hdf5::init_column(
  Context context,
  Runtime* runtime,
  hid_t loc_id,
  hid_t axes_dt,
  hid_t attr_access_pl,
  hid_t link_access_pl,
  hid_t xfer_pl) {

  std::optional<ColumnGenArgs> result;

  legms::TypeTag datatype = ValueType<int>::DataType;
  hid_t datatype_id = -1;
  std::vector<int> axes;
  hid_t axes_id = -1;
  hid_t axes_id_ds = -1;
  LogicalRegion values = LogicalRegion::NO_REGION;
  LogicalRegion keywords = LogicalRegion::NO_REGION;
  std::vector<legms::TypeTag> keyword_datatypes;

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
        std::vector<unsigned char> ax(H5Sget_simple_extent_npoints(axes_id_ds));
        herr_t err = H5Aread(axes_id, axes_dt, ax.data());
        assert(err >= 0);
        axes.reserve(ax.size());
        std::copy(ax.begin(), ax.end(), std::back_inserter(axes));
      }
      {
        std::string datatype_name(LEGMS_ATTRIBUTE_DT);
        htri_t datatype_exists =
          H5Aexists_by_name(
            loc_id,
            LEGMS_COLUMN_DS,
            datatype_name.c_str(),
            link_access_pl);
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
        datatype = read_dt_value(datatype_id);
      }
      {
        std::optional<std::string> sid =
          read_index_tree_attr_metadata(loc_id, "index_tree");
        if (!sid
            || (sid.value() != "legms::hdf5::binary_index_tree_serdez"))
          goto return_nothing;
        std::optional<IndexTreeL> ixtree =
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
        // TODO: remove?
        // runtime->destroy_field_space(context, fs);
        // runtime->destroy_index_space(context, is);
      }
    }
  }
  tie(keywords, keyword_datatypes) =
    init_keywords(context, runtime, loc_id, attr_access_pl, link_access_pl);

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
  const std::unordered_set<std::string>* column_names;
  std::vector<legms::ColumnGenArgs>* acc;
  std::string axes_uid;
  hid_t axes_dt;
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
    if (infobuf.type == H5O_TYPE_GROUP
        && args->column_names->count(name) > 0) {
      hid_t col_group_id = H5Gopen(table_id, name, H5P_DEFAULT);
      assert(col_group_id >= 0);
      auto cga =
        init_column(
          args->context,
          args->runtime,
          col_group_id,
          args->axes_dt,
          args->attr_access_pl,
          args->link_access_pl,
          args->xfer_pl);
      if (cga) {
        legms::ColumnGenArgs& a = cga.value();
        a.name = name;
        a.axes_uid = args->axes_uid;
        args->acc->push_back(std::move(a));
      }
    }
  }
  return 0;
}

std::optional<legms::TableGenArgs>
legms::hdf5::init_table(
  Context context,
  Runtime* runtime,
  hid_t loc_id,
  const std::unordered_set<std::string>& column_names,
  hid_t type_access_pl,
  hid_t attr_access_pl,
  hid_t link_access_pl,
  hid_t xfer_pl) {

  std::optional<TableGenArgs> result;

  if (column_names.size() == 0)
    return result;

  std::vector<int> index_axes;
  hid_t index_axes_id = -1;
  hid_t index_axes_id_ds = -1;
  std::string axes_uid;
  std::vector<ColumnGenArgs> col_genargs;
  LogicalRegion keywords = LogicalRegion::NO_REGION;
  std::vector<legms::TypeTag> keyword_datatypes;
  hid_t axes_dt;
  {
    hid_t dt = H5Topen(loc_id, table_axes_dt_name, type_access_pl);
    auto uid = AxesRegistrar::match_axes_datatype(dt);
    if (!uid)
      goto return_nothing;
    axes_uid = uid.value();
    axes_dt = AxesRegistrar::axes(axes_uid).value().h5_datatype;
    herr_t err = H5Tclose(dt);
    assert(err >= 0);
  }
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
    std::vector<unsigned char>
      ax(H5Sget_simple_extent_npoints(index_axes_id_ds));
    herr_t err = H5Aread(index_axes_id, axes_dt, ax.data());
    assert(err >= 0);
    index_axes.reserve(ax.size());
    std::copy(ax.begin(), ax.end(), std::back_inserter(index_axes));
  }
  {
    struct acc_col_genargs_ctx ctx{
      &column_names, &col_genargs, axes_uid, axes_dt,
      attr_access_pl, link_access_pl, xfer_pl,
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
    init_keywords(context, runtime, loc_id, attr_access_pl, link_access_pl);

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
  Context context,
  Runtime* runtime,
  const std::experimental::filesystem::path& file_path,
  const std::string& table_path,
  const std::unordered_set<std::string>& column_names,
  unsigned flags,
  hid_t file_access_pl,
  hid_t table_access_pl,
  hid_t type_access_pl,
  hid_t attr_access_pl,
  hid_t link_access_pl,
  hid_t xfer_pl) {

  std::optional<TableGenArgs> result;

  hid_t fid = H5Fopen(file_path.c_str(), flags, file_access_pl);
  if (fid >= 0) {
    try {
      hid_t table_loc = H5Gopen(fid, table_path.c_str(), table_access_pl);
      if (table_loc >= 0) {
        try {
          result =
            init_table(
              context,
              runtime,
              table_loc,
              column_names,
              type_access_pl,
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

herr_t
acc_table_paths(hid_t loc_id, const char* name, const H5L_info_t*, void* ctx) {

  std::unordered_set<std::string>* tblpaths =
    (std::unordered_set<std::string>*)(ctx);
  H5O_info_t infobuf;
  herr_t err = H5Oget_info_by_name(loc_id, name, &infobuf, H5P_DEFAULT);
  assert(err >= 0);
  if (infobuf.type == H5O_TYPE_GROUP)
    tblpaths->insert(std::string("/") + name);
  return 0;
}

std::unordered_set<std::string>
legms::hdf5::get_table_paths(
  const std::experimental::filesystem::path& file_path) {

  std::unordered_set<std::string> result;
  hid_t fid = H5Fopen(file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (fid >= 0) {
    herr_t err =
      H5Literate(
        fid,
        H5_INDEX_NAME,
        H5_ITER_NATIVE,
        NULL,
        acc_table_paths,
        &result);
    assert(err >= 0);
    err = H5Fclose(fid);
    assert(err >= 0);
  }
  return result;
}

herr_t
acc_column_names(hid_t loc_id, const char* name, const H5L_info_t*, void* ctx) {
  std::unordered_set<std::string>* colnames =
    (std::unordered_set<std::string>*)(ctx);
  H5O_info_t infobuf;
  herr_t err = H5Oget_info_by_name(loc_id, name, &infobuf, H5P_DEFAULT);
  assert(err >= 0);
  if (infobuf.type == H5O_TYPE_GROUP) {
    hid_t gid = H5Gopen(loc_id, name, H5P_DEFAULT);
    assert(gid >= 0);
    htri_t has_col_ds = H5Oexists_by_name(gid, LEGMS_COLUMN_DS, H5P_DEFAULT);
    assert(has_col_ds >= 0);
    if (has_col_ds > 0) {
      herr_t err =
        H5Oget_info_by_name(gid, LEGMS_COLUMN_DS, &infobuf, H5P_DEFAULT);
      assert(err >= 0);
      if (infobuf.type == H5O_TYPE_DATASET)
        colnames->insert(name);
    }
  }
  return 0;
}

std::unordered_set<std::string>
legms::hdf5::get_column_names(
  const std::experimental::filesystem::path& file_path,
  const std::string& table_path) {

  std::unordered_set<std::string> result;
  hid_t fid = H5Fopen(file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (fid >= 0) {
    hid_t tid = H5Gopen(fid, table_path.c_str(), H5P_DEFAULT);
    if (tid >= 0) {
      herr_t err =
        H5Literate(
          tid,
          H5_INDEX_NAME,
          H5_ITER_NATIVE,
          NULL,
          acc_column_names,
          &result);
      assert(err >= 0);
      err = H5Gclose(tid);
      assert(err >= 0);
    }
    herr_t err = H5Fclose(fid);
    assert(err >= 0);
  }
  return result;
}

std::unordered_map<std::string, std::string>
legms::hdf5::get_table_keyword_paths(const Table& table) {
  std::unordered_map<std::string, std::string> result;
  std::transform(
    table.keywords().begin(),
    table.keywords().end(),
    std::inserter(result, result.end()),
    [tn=std::string("/") + table.name() + "/"](auto& kwd) {
      return std::make_pair(std::get<0>(kwd), tn + std::get<0>(kwd));
    });
  return result;
}

std::string
legms::hdf5::get_table_column_value_path(
  const Table& table,
  const std::string& colname) {

  return
    std::string("/") + table.name() + "/" + colname + "/" + LEGMS_COLUMN_DS;
}

std::unordered_map<std::string, std::string>
legms::hdf5::get_table_column_keyword_paths(
  const Table& table,
  const std::string& colname) {

  auto col = table.column(colname);
  auto prefix = std::string("/") + table.name() + "/" + col->name() + "/";
  std::unordered_map<std::string, std::string> result;
  std::transform(
    col->keywords().begin(),
    col->keywords().end(),
    std::inserter(result, result.end()),
    [&prefix](auto& kwd) {
      return std::make_pair(std::get<0>(kwd), prefix + std::get<0>(kwd));
    });
  return result;
}

std::optional<PhysicalRegion>
legms::hdf5::attach_keywords(
  Context context,
  Runtime* runtime,
  const std::experimental::filesystem::path& file_path,
  const std::string& with_keywords_path,
  const WithKeywords* with_keywords,
  bool read_only) {

  std::optional<PhysicalRegion> result;
  auto kws = with_keywords->keywords_region();
  if (kws != LogicalRegion::NO_REGION) {
    std::vector<std::string> field_paths(with_keywords->num_keywords());
    std::map<FieldID, const char*> fields;
    for (size_t i = 0; i < with_keywords->num_keywords(); ++i) {
      field_paths[i] =
        with_keywords_path + "/" + std::get<0>(with_keywords->keywords()[i]);
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

std::unordered_map<
  std::string,
  std::tuple<
    std::optional<PhysicalRegion>,
    std::optional<PhysicalRegion>>>
legms::hdf5::attach_table_columns(
  Context context,
  Runtime* runtime,
  const std::experimental::filesystem::path& file_path,
  const std::string& root_path,
  const Table* table,
  bool read_only) {

  std::unordered_map<
    std::string,
    std::tuple<
      std::optional<PhysicalRegion>,
      std::optional<PhysicalRegion>>> result;
  std::string table_root = root_path;
  if (table_root.back() != '/')
    table_root.push_back('/');
  table_root += table->name() + "/";
  std::transform(
    table->column_names().begin(),
    table->column_names().end(),
    std::inserter(result, result.end()),
    [&file_path,&table_root, &table, &runtime, &context, &read_only](auto& nm) {
      auto c = table->column(nm);
      std::tuple<std::optional<PhysicalRegion>, std::optional<PhysicalRegion>>
        regions;
      auto col = c->logical_region();
      if (col != LogicalRegion::NO_REGION) {
        AttachLauncher col_attach(EXTERNAL_HDF5_FILE, col, col);
        std::string col_path = table_root + c->name() + "/" + LEGMS_COLUMN_DS;
        std::map<FieldID, const char*>
          fields{{Column::value_fid, col_path.c_str()}};
        col_attach.attach_hdf5(
          file_path.c_str(),
          fields,
          read_only ? LEGION_FILE_READ_ONLY : LEGION_FILE_READ_WRITE);
        std::get<0>(regions) =
          runtime->attach_external_resource(context, col_attach);
        std::get<1>(regions) =
          attach_keywords(
            context,
            runtime,
            file_path,
            col_path,
            c.get(),
            read_only);
      }
      return std::make_pair(nm, regions);
    });
  return result;
}

std::optional<PhysicalRegion>
legms::hdf5::attach_table_keywords(
  Context context,
  Runtime* runtime,
  const std::experimental::filesystem::path& file_path,
  const std::string& root_path,
  const Table* table,
  bool read_only) {

  std::string table_root = root_path;
  if (table_root.back() != '/')
    table_root.push_back('/');
  table_root += table->name();
  return
    attach_keywords(context, runtime, file_path, table_root, table, read_only);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
