#include "legms_hdf5.h"

#pragma GCC visibility push(default)
#include <algorithm>
#include <cstring>
#include <numeric>
#include <optional>
#include <sstream>
#pragma GCC visibility pop

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
  Legion::Context ctx,
  Legion::Runtime *rt,
  hid_t loc_id,
  const Keywords& keywords,
  bool with_data,
  hid_t link_creation_pl,
  hid_t link_access_pl,
  hid_t dataset_creation_pl,
  hid_t dataset_access_pl,
  hid_t xfer_pl) {

  if (keywords.values_lr == LogicalRegion::NO_REGION)
    return;

  std::optional<PhysicalRegion> pr;
  std::vector<std::string> keys = keywords.keys(rt);
  std::vector<FieldID> fids(keys.size());
  std::iota(fids.begin(), fids.end(), 0);
  if (with_data) {
    RegionRequirement
      req(keywords.values_lr, READ_ONLY, EXCLUSIVE, keywords.values_lr);
    req.add_fields(fids);
    pr = rt->map_region(ctx, req);
  }

  auto value_types = keywords.value_types(ctx, rt, fids);
  for (size_t i = 0; i < keys.size(); ++i) {
    assert(keys[i].substr(0, sizeof(LEGMS_NAMESPACE_PREFIX) - 1)
           != LEGMS_NAMESPACE_PREFIX);
    switch (value_types[i]) {
#define WRITE_KW(DT)                            \
      case (DT): {                              \
        write_kw<DT>(                           \
          loc_id,                               \
          keys[i].c_str(),                      \
          pr,                                   \
          i,                                    \
          link_creation_pl,                     \
          link_access_pl,                       \
          dataset_creation_pl,                  \
          dataset_access_pl,                    \
          xfer_pl);                             \
        break;                                  \
      }
      LEGMS_FOREACH_DATATYPE(WRITE_KW)
#undef WRITE_KW
    default:
        assert(false);
    }
  }
}

void
legms::hdf5::write_column(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const std::experimental::filesystem::path& path,
  hid_t table_id,
  const std::string& table_name,
  const Column& column,
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

  // delete column dataset if it exists
  auto colname = column.name(ctx, rt);
  auto datatype = column.datatype(ctx, rt);

  htri_t ds_exists =
    H5Lexists(table_id, colname.c_str(), link_access_pl);
  if (ds_exists > 0) {
    herr_t err = H5Ldelete(table_id, colname.c_str(), link_access_pl);
    assert(err >= 0);
  } else {
    assert(ds_exists == 0);
  }

  // create column group
  hid_t col_group_id =
    H5Gcreate(
      table_id,
      colname.c_str(),
      link_creation_pl,
      group_creation_pl,
      group_access_pl);
  assert(col_group_id >= 0);

  // create column dataset
  hid_t col_id;
  {
    unsigned rank = column.rank(rt);
    hsize_t dims[rank];
    switch (rank) {
#define DIMS(N)                                 \
      case N: {                                 \
        Rect<N> rect =                          \
          rt->get_index_space_domain(           \
            ctx,                                \
            column.values_lr.get_index_space()) \
          .bounds<N, coord_t>();                \
        for (size_t i = 0; i < N; ++i)          \
          dims[i] = rect.hi[i] + 1;             \
        break;                                  \
      }
      LEGMS_FOREACH_N(DIMS)
#undef DIMS
    default:
      assert(false);
      break;
    }

    hid_t ds = H5Screate_simple(rank, dims, NULL);
    assert(ds >= 0);

    hid_t dt;


    switch (datatype) {
#define DT(T) \
      case T: dt = H5DatatypeManager::datatype<T>(); break;
      LEGMS_FOREACH_DATATYPE(DT)
#undef DT
    default:
      assert(false);
      break;
    }

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
    init_datatype_attr(col_id, datatype);

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

    auto axes = column.axes(ctx, rt);
    hsize_t dims = axes.size();
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
      std::string("/") + table_name + "/" + colname + "/" + LEGMS_COLUMN_DS;
    std::map<FieldID, const char*>
      field_map{{Column::VALUE_FID, column_ds_name.c_str()}};
    LogicalRegion values_lr =
      rt->create_logical_region(
        ctx,
        column.values_lr.get_index_space(),
        column.values_lr.get_field_space());
    AttachLauncher attach(EXTERNAL_HDF5_FILE, values_lr, values_lr);
    attach.attach_hdf5(path.c_str(), field_map, LEGION_FILE_READ_WRITE);
    PhysicalRegion values_pr = rt->attach_external_resource(ctx, attach);
    RegionRequirement src(column.values_lr, READ_ONLY, EXCLUSIVE, column.values_lr);
    src.add_field(Column::VALUE_FID);
    RegionRequirement dst(values_lr, WRITE_ONLY, EXCLUSIVE, values_lr);
    dst.add_field(Column::VALUE_FID);
    CopyLauncher copy;
    copy.add_copy_requirements(src, dst);
    rt->issue_copy_operation(ctx, copy);
    rt->detach_external_resource(ctx, values_pr);
    rt->destroy_logical_region(ctx, values_lr);
  }
  write_keywords(ctx, rt, col_group_id, column.keywords, with_data);
  herr_t err = H5Gclose(col_group_id);
  assert(err >= 0);

  write_index_tree_to_attr<binary_index_tree_serdez>(
    column.index_tree(rt),
    table_id,
    colname,
    "index_tree",
    link_creation_pl,
    link_access_pl,
    dataset_creation_pl,
    dataset_access_pl,
    xfer_pl);
}

void
legms::hdf5::write_table(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const std::experimental::filesystem::path& path,
  hid_t loc_id,
  const Table& table,
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
  auto tabname = table.name(ctx, rt);
  hid_t table_id;
  {
    htri_t rc = H5Lexists(loc_id, tabname.c_str(), link_access_pl);
    if (rc == 0) {
      table_id =
        H5Gcreate(
          loc_id,
          tabname.c_str(),
          link_creation_pl,
          group_creation_pl,
          group_access_pl);
    } else {
      assert(rc > 0);
      table_id = H5Gopen(loc_id, tabname.c_str(), group_access_pl);
    }
    assert(table_id >= 0);
  }

  // write axes datatype to table
  auto axes = AxesRegistrar::axes(table.axes_uid(ctx, rt));
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

    auto index_axes = table.index_axes(ctx, rt);
    hsize_t dims = index_axes.size();
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
        std::vector<unsigned char> ax;
        ax.reserve(index_axes.size());
        std::copy(index_axes.begin(), index_axes.end(), std::back_inserter(ax));
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

    {
      RegionRequirement
        req(table.columns_lr, READ_ONLY, EXCLUSIVE, table.columns_lr);
      req.add_field(Table::COLUMNS_FID);
      auto columns = rt->map_region(ctx, req);
      auto colnames = Table::column_names(ctx, rt, columns);
      for (auto& nm : colnames) {
        auto col = table.column(ctx, rt, columns, nm);
        if (excluded_columns.count(nm) == 0 && !col.is_empty())
          write_column(
            ctx,
            rt,
            path,
            table_id,
            tabname,
            col,
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
      }
      rt->unmap_region(ctx, columns);
    }

    write_keywords(
      ctx,
      rt,
      table_id,
      table.keywords,
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
  // in two legms codes built with and without LEGMS_USE_CASACORE)
  herr_t err = H5Aread(dt_id, H5T_NATIVE_INT, &dt);
  assert(err >= 0);
  return dt;
}

legms::Keywords::kw_desc_t
legms::hdf5::init_keywords(
  Context ctx,
  Runtime* rt,
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
    return {};

  return
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
}

Column
legms::hdf5::init_column(
  Context ctx,
  Runtime* rt,
  const std::string& column_name,
  const std::string& axes_uid,
  hid_t loc_id,
  hid_t axes_dt,
  hid_t attr_access_pl,
  hid_t link_access_pl,
  hid_t xfer_pl) {

  Column result;

  legms::TypeTag datatype = ValueType<int>::DataType;
  hid_t datatype_id = -1;
  std::vector<int> axes;
  hid_t axes_id = -1;
  hid_t axes_id_ds = -1;

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
      auto keywords =
        init_keywords(ctx, rt, loc_id, attr_access_pl, link_access_pl);
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
        result =
          Column::create(
            ctx,
            rt,
            column_name,
            axes_uid,
            axes,
            datatype,
            ixtree.value(),
            keywords);
      }
    }
  }

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

struct acc_col_ctx {
  const std::unordered_set<std::string>* column_names;
  std::vector<legms::Column>* acc;
  std::string axes_uid;
  hid_t axes_dt;
  hid_t attr_access_pl;
  hid_t link_access_pl;
  hid_t xfer_pl;
  Runtime* runtime;
  Context context;
};

static herr_t
acc_col(
  hid_t table_id,
  const char* name,
  const H5L_info_t*,
  void* ctx) {

  struct acc_col_ctx *args =
    static_cast<struct acc_col_ctx*>(ctx);
  htri_t rc = H5Lexists(table_id, name, H5P_DEFAULT);
  if (rc > 0) {
    H5O_info_t infobuf;
    H5Oget_info_by_name(table_id, name, &infobuf, H5P_DEFAULT);
    if (infobuf.type == H5O_TYPE_GROUP
        && args->column_names->count(name) > 0) {
      hid_t col_group_id = H5Gopen(table_id, name, H5P_DEFAULT);
      assert(col_group_id >= 0);
      auto col =
        init_column(
          args->context,
          args->runtime,
          name,
          args->axes_uid,
          col_group_id,
          args->axes_dt,
          args->attr_access_pl,
          args->link_access_pl,
          args->xfer_pl);
      args->acc->push_back(std::move(col));
    }
  }
  return 0;
}

Table
legms::hdf5::init_table(
  Context ctx,
  Runtime* rt,
  const std::string& table_name,
  hid_t loc_id,
  const std::unordered_set<std::string>& column_names,
  hid_t type_access_pl,
  hid_t attr_access_pl,
  hid_t link_access_pl,
  hid_t xfer_pl) {

  Table result;

  if (column_names.size() == 0)
    return result;

  std::vector<int> index_axes;
  hid_t index_axes_id = -1;
  hid_t index_axes_id_ds = -1;
  std::string axes_uid;
  std::vector<Column> cols;
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
    struct acc_col_ctx acc_col_ctx{
      &column_names, &cols, axes_uid, axes_dt,
      attr_access_pl, link_access_pl, xfer_pl,
      rt, ctx};
    hsize_t position = 0;
    herr_t err =
      H5Literate(
        loc_id,
        H5_INDEX_NAME,
        H5_ITER_NATIVE,
        &position,
        acc_col,
        &acc_col_ctx);
    assert(err >= 0);
  }
  {
    auto keywords =
      init_keywords(ctx, rt, loc_id, attr_access_pl, link_access_pl);

    result =
      Table::create(
        ctx,
        rt,
        table_name,
        axes_uid,
        index_axes,
        cols,
        keywords);
  }
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

Table
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

  Table result;

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
              table_path.substr(table_path.rfind('/') + 1),
              table_loc,
              column_names,
              type_access_pl,
              attr_access_pl,
              link_access_pl,
              xfer_pl);
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

static herr_t
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

static herr_t
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
legms::hdf5::get_table_keyword_paths(
  Context ctx,
  Runtime* rt,
  const Table& table) {

  std::string tn = std::string("/") + table.name(ctx, rt) + "/";
  std::unordered_map<std::string, std::string> result;
  for (auto& k : table.keywords.keys(rt))
    result[k] = tn + k;
  return result;
}

std::string
legms::hdf5::get_table_column_value_path(
  Context ctx,
  Runtime* rt,
  const Table& table,
  const std::string& colname) {

  return
    std::string("/") + table.name(ctx, rt) + "/"
    + colname + "/" + LEGMS_COLUMN_DS;
}

std::unordered_map<std::string, std::string>
legms::hdf5::get_table_column_keyword_paths(
  Context ctx,
  Runtime* rt,
  const Table& table,
  const std::string& colname) {

  auto col = table.column(ctx, rt, colname);
  auto prefix =
    std::string("/") + table.name(ctx, rt) + "/" + col.name(ctx, rt) + "/";
  std::unordered_map<std::string, std::string> result;
  for (auto& k : col.keywords.keys(rt))
    result[k] = prefix + k;
  return result;
}

std::optional<PhysicalRegion>
legms::hdf5::attach_keywords(
  Context ctx,
  Runtime* rt,
  const std::experimental::filesystem::path& file_path,
  const std::string& keywords_path,
  const Keywords& keywords,
  bool read_only) {

  std::optional<PhysicalRegion> result;
  auto kws = keywords.values_lr;
  if (kws != LogicalRegion::NO_REGION) {
    std::vector<std::string> keys = keywords.keys(rt);
    std::vector<std::string> field_paths(keys.size());
    std::map<FieldID, const char*> fields;
    for (size_t i = 0; i < keys.size(); ++i) {
      field_paths[i] = keywords_path + "/" + keys[i];
      fields[i] = field_paths[i].c_str();
    }
    AttachLauncher kws_attach(EXTERNAL_HDF5_FILE, kws, kws);
    kws_attach.attach_hdf5(
      file_path.c_str(),
      fields,
      read_only ? LEGION_FILE_READ_ONLY : LEGION_FILE_READ_WRITE);
    result = rt->attach_external_resource(ctx, kws_attach);
  }
  return result;
}

std::unordered_map<
  std::string,
  std::tuple<
    std::optional<PhysicalRegion>,
    std::optional<PhysicalRegion>>>
legms::hdf5::attach_table_columns(
  Context ctx,
  Runtime* rt,
  const std::experimental::filesystem::path& file_path,
  const std::string& root_path,
  const Table& table,
  bool mapped,
  bool read_only) {

  std::unordered_map<
    std::string,
    std::tuple<
      std::optional<PhysicalRegion>,
      std::optional<PhysicalRegion>>> result;
  std::string table_root = root_path;
  if (table_root.back() != '/')
    table_root.push_back('/');
  table_root += table.name(ctx, rt) + "/";

  {
    RegionRequirement
      req(table.columns_lr, READ_ONLY, EXCLUSIVE, table.columns_lr);
    req.add_field(Table::COLUMNS_FID);
    auto columns = rt->map_region(ctx, req);
    auto colnames = Table::column_names(ctx, rt, columns);
    for (auto& colname : colnames) {
      auto col = Table::column(ctx, rt, columns, colname);
      std::tuple<std::optional<PhysicalRegion>, std::optional<PhysicalRegion>>
        regions;
      if (!col.is_empty()) {
        AttachLauncher col_attach(EXTERNAL_HDF5_FILE, col.values_lr, col.values_lr);
        col_attach.mapped = mapped;
        std::string col_path = table_root + colname + "/" + LEGMS_COLUMN_DS;
        std::map<FieldID, const char*>
          fields{{Column::VALUE_FID, col_path.c_str()}};
        col_attach.attach_hdf5(
          file_path.c_str(),
          fields,
          read_only ? LEGION_FILE_READ_ONLY : LEGION_FILE_READ_WRITE);
        std::get<0>(regions) =
          rt->attach_external_resource(ctx, col_attach);
        std::get<1>(regions) =
          attach_keywords(ctx, rt, file_path, col_path, col.keywords, read_only);
      }
      result[colname] = std::move(regions);
    }
    rt->unmap_region(ctx, columns);
  }
  return result;
}

std::optional<PhysicalRegion>
legms::hdf5::attach_table_keywords(
  Context ctx,
  Runtime* rt,
  const std::experimental::filesystem::path& file_path,
  const std::string& root_path,
  const Table& table,
  bool read_only) {

  std::string table_root = root_path;
  if (table_root.back() != '/')
    table_root.push_back('/');
  table_root += table.name(ctx, rt);
  return
    attach_keywords(
      ctx,
      rt,
      file_path,
      table_root,
      table.keywords,
      read_only);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
