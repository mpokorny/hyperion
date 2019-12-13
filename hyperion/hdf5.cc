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
#include <hyperion/hdf5.h>
#include <hyperion/tree_index_space.h>
#include <hyperion/MSTable.h>
#include <hyperion/Table.h>
#include <hyperion/Column.h>

#pragma GCC visibility push(default)
# include <algorithm>
# include <cstring>
# include <numeric>
# include <optional>
# include <sstream>
#pragma GCC visibility pop

using namespace hyperion::hdf5;
using namespace hyperion;
using namespace Legion;

const char* table_index_axes_attr_name =
  HYPERION_NAMESPACE_PREFIX "index_axes";
const char* table_axes_dt_name =
  HYPERION_NAMESPACE_PREFIX "table_axes";
const char* column_axes_attr_name =
  HYPERION_NAMESPACE_PREFIX "axes";
const char* column_refcol_attr_name =
  HYPERION_NAMESPACE_PREFIX "refcol";

#define CHECK_H5(F) do {                        \
    herr_t err = F;                             \
    assert(err >= 0);                           \
  } while (0)

template <
  typename OPEN,
  typename F,
  typename CLOSE,
  std::enable_if_t<
    !std::is_void_v<
      std::invoke_result_t<F, std::invoke_result_t<OPEN>>>,
    int> = 0>
std::invoke_result_t<F, std::invoke_result_t<OPEN>>
using_resource(OPEN open, F f, CLOSE close) {
  auto r = open();
  std::invoke_result_t<F, std::invoke_result_t<OPEN>> result;
  try {
    result = f(r);
  } catch (...) {
    close(r);
    throw;
  }
  close(r);
  return result;
}

template <
  typename OPEN,
  typename F,
  typename CLOSE,
  std::enable_if_t<
    std::is_void_v<
      std::invoke_result_t<F, std::invoke_result_t<OPEN>>>,
    int> = 0>
void
using_resource(OPEN open, F f, CLOSE close) {
  auto r = open();
  try {
    f(r);
  } catch (...) {
    close(r);
    throw;
  }
  close(r);
}

std::optional<std::string>
hyperion::hdf5::read_index_tree_attr_metadata(
  hid_t loc_id,
  const std::string& attr_name) {

  std::optional<std::string> result;

  std::string md_id_name =
    std::string(HYPERION_ATTRIBUTE_SID_PREFIX) + attr_name;
  if (H5Aexists(loc_id, md_id_name.c_str())) {

    hid_t attr_id = H5Aopen(loc_id, md_id_name.c_str(), H5P_DEFAULT);

    if (attr_id >= 0) {
      hid_t attr_type = H5Aget_type(attr_id);

      // FIXME: shouldn't I be using hyperion::string?
      hid_t attr_dt =
        H5DatatypeManager::datatype<ValueType<std::string>::DataType>();
      if (H5Tequal(attr_type, attr_dt) > 0) {
        string attr;
        CHECK_H5(H5Aread(attr_id, attr_dt, attr.val));
        result = attr.val;
      }
      CHECK_H5(H5Aclose(attr_id));
    }
  }
  return result;
}

template <hyperion::TypeTag DT>
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
  hyperion::TypeTag dt);

static void
init_datatype_attr(
  hid_t loc_id,
  hyperion::TypeTag dt) {

  htri_t rc = H5Aexists(loc_id, HYPERION_ATTRIBUTE_DT);
  if (rc > 0)
    CHECK_H5(H5Adelete(loc_id, HYPERION_ATTRIBUTE_DT));

  hid_t ds = H5Screate(H5S_SCALAR);
  assert(ds >= 0);
  hid_t did = hyperion::H5DatatypeManager::datatypes()[
    hyperion::H5DatatypeManager::DATATYPE_H5T];
  hid_t attr_id =
    H5Acreate(
      loc_id,
      HYPERION_ATTRIBUTE_DT,
      did,
      ds,
      H5P_DEFAULT,
      H5P_DEFAULT);
  assert(attr_id >= 0);
  CHECK_H5(H5Awrite(attr_id, did, &dt));
  CHECK_H5(H5Sclose(ds));
  CHECK_H5(H5Aclose(attr_id));
}

static hid_t
init_kw(
  hid_t loc_id,
  const char *attr_name,
  hid_t type_id,
  hyperion::TypeTag dt);

static hid_t
init_kw(
  hid_t loc_id,
  const char *attr_name,
  hid_t type_id,
  hyperion::TypeTag dt) {

  {
    htri_t rc = H5Lexists(loc_id, attr_name, H5P_DEFAULT);
    assert(rc >= 0);
    if (rc > 0)
      CHECK_H5(H5Ldelete(loc_id, attr_name, H5P_DEFAULT));
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
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(result >= 0);
    CHECK_H5(H5Sclose(attr_ds));
  }
  init_datatype_attr(result, dt);
  return result;
}

template <hyperion::TypeTag DT>
static void
write_kw(
  hid_t loc_id,
  const char *attr_name,
  std::optional<PhysicalRegion>& region,
  FieldID fid);

template <hyperion::TypeTag DT>
static void
write_kw(
  hid_t loc_id,
  const char *attr_name,
  std::optional<PhysicalRegion>& region,
  FieldID fid) {

  hid_t dt = hyperion::H5DatatypeManager::datatype<DT>();
  hid_t attr_id = init_kw(loc_id, attr_name, dt, DT);
  if (region) {
    const KW<DT> kw(region.value(), fid);
    CHECK_H5(H5Dwrite(attr_id, dt, H5S_ALL, H5S_ALL, H5P_DEFAULT, kw.ptr(0)));
  }
  CHECK_H5(H5Dclose(attr_id));
}

template <>
void
write_kw<HYPERION_TYPE_STRING> (
  hid_t loc_id,
  const char *attr_name,
  std::optional<PhysicalRegion>& region,
  FieldID fid) {

  hid_t dt = hyperion::H5DatatypeManager::datatype<HYPERION_TYPE_STRING>();
  hid_t attr_id =
    init_kw(loc_id, attr_name, dt, HYPERION_TYPE_STRING);
  if (region) {
    const KW<HYPERION_TYPE_STRING> kw(region.value(), fid);
    const hyperion::string& kwval = kw[0];
    hyperion::string buf;
    fstrcpy(buf.val, kwval.val);
    CHECK_H5(H5Dwrite(attr_id, dt, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.val));
  }
  CHECK_H5(H5Dclose(attr_id));
}

void
hyperion::hdf5::write_keywords(
  Context ctx,
  Runtime *rt,
  hid_t loc_id,
  const Keywords& keywords,
  bool with_data) {

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
    assert(keys[i].substr(0, sizeof(HYPERION_NAMESPACE_PREFIX) - 1)
           != HYPERION_NAMESPACE_PREFIX);
    switch (value_types[i].value()) {
#define WRITE_KW(DT)                                  \
      case (DT):                                      \
        write_kw<DT>(loc_id, keys[i].c_str(), pr, i); \
        break;
      HYPERION_FOREACH_DATATYPE(WRITE_KW)
#undef WRITE_KW
    default:
        assert(false);
    }
  }
}

#ifdef HYPERION_USE_CASACORE
template <int D, typename A, typename T>
std::vector<T>
copy_mr_region(
  Context ctx,
  Runtime *rt,
  LogicalRegion lr,
  FieldID fid) {

  // copy values into buff...lr index space may be sparse
  RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
  req.add_field(fid);
  auto pr = rt->map_region(ctx, req);
  const A acc(pr, fid);
  Domain dom = rt->get_index_space_domain(lr.get_index_space());
  Rect<D,coord_t> rect = dom.bounds<D,coord_t>();
  size_t sz = 1;
  for (size_t i = 0; i < D; ++i)
    sz *= rect.hi[i] + 1;
  std::vector<T> result(sz);
  auto t = result.begin();
  for (PointInRectIterator<D> pir(rect, false); pir(); pir++) {
    if (dom.contains(*pir))
      *t = acc[*pir];
    ++t;
  }
  rt->unmap_region(ctx, pr);
  return result;
}

template <int D, typename A, typename T>
static void
write_mr_region(
  Context ctx,
  Runtime *rt,
  hid_t ds,
  hid_t dt,
  LogicalRegion lr,
  FieldID fid) {

  std::vector<T> buff = copy_mr_region<D, A, T>(ctx, rt, lr, fid);
  CHECK_H5(H5Dwrite(ds, dt, H5S_ALL, H5S_ALL, H5P_DEFAULT, buff.data()));
}

void
hyperion::hdf5::write_measure(
  Context ctx,
  Runtime* rt,
  hid_t loc_id,
  const char* name,
  const MeasRef& mr) {

  hid_t mr_id = H5Gcreate(loc_id, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  assert(mr_id > 0);

  if (!mr.is_empty()) {
    std::vector<hsize_t> dims, dims1;
    hid_t sp, sp1 = -1;
    switch (mr.metadata_lr.get_index_space().get_dim()) {
#define SP(D)                                                           \
      case D:                                                           \
      {                                                                 \
        Rect<D> bounds =                                                \
          rt->get_index_space_domain(mr.metadata_lr.get_index_space()); \
        dims.resize(D);                                                 \
        for (size_t i = 0; i < D; ++i)                                  \
          dims[i] = bounds.hi[i] + 1;                                   \
        sp = H5Screate_simple(D, dims.data(), NULL);                    \
        assert(sp >= 0);                                                \
      }                                                                 \
                                                                        \
      if (mr.values_lr != LogicalRegion::NO_REGION) {                   \
        Rect<D+1> bounds =                                              \
          rt->get_index_space_domain(mr.values_lr.get_index_space());   \
        dims1.resize(D + 1);                                            \
        for (size_t i = 0; i < D + 1; ++i)                              \
          dims1[i] = bounds.hi[i] + 1;                                  \
        sp1 = H5Screate_simple(D + 1, dims1.data(), NULL);              \
        assert(sp1 >= 0);                                               \
      }                                                                 \
      break;
      HYPERION_FOREACH_N_LESS_MAX(SP)
#undef SP
    default:
        assert(false);
      break;
    }
    {
      // Write the datasets for the MeasRef values directly, without going through
      // the Legion HDF5 interface, as the dataset sizes are small. Not worrying
      // too much about efficiency for this, in any case.
      {
        hid_t ds =
          H5Dcreate(
            mr_id,
            HYPERION_MEAS_REF_MCLASS_DS,
            H5DatatypeManager::datatypes()[H5DatatypeManager::MEASURE_CLASS_H5T],
            sp,
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        assert(ds >= 0);

        switch (dims.size()) {
#define W_MCLASS(D)                                                     \
          case D:                                                       \
            write_mr_region<                                            \
              D, \
              MeasRef::MeasureClassAccessor<READ_ONLY, D>, \
              MeasRef::MEASURE_CLASS_TYPE>( \
                ctx, \
                rt, \
                ds, \
                H5DatatypeManager::datatypes()[ \
                  H5DatatypeManager::MEASURE_CLASS_H5T], \
                mr.metadata_lr, \
                MeasRef::MEASURE_CLASS_FID); \
            break;
          HYPERION_FOREACH_N_LESS_MAX(W_MCLASS);
#undef W_MCLASS
        default:
          assert(false);
          break;
        }
        CHECK_H5(H5Dclose(ds));
      }
      {
        hid_t ds =
          H5Dcreate(
            mr_id,
            HYPERION_MEAS_REF_RTYPE_DS,
            H5DatatypeManager::datatype<
            ValueType<MeasRef::REF_TYPE_TYPE>::DataType>(),
            sp,
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        assert(ds >= 0);

        switch (dims.size()) {
#define W_RTYPE(D)                                                      \
          case D:                                                       \
            write_mr_region<                                            \
              D, \
              MeasRef::RefTypeAccessor<READ_ONLY, D>, \
              MeasRef::REF_TYPE_TYPE>( \
                ctx, \
                rt, \
                ds, \
                H5DatatypeManager::datatype< \
                  ValueType<MeasRef::REF_TYPE_TYPE>::DataType>(), \
                mr.metadata_lr, \
                MeasRef::REF_TYPE_FID); \
            break;
          HYPERION_FOREACH_N_LESS_MAX(W_RTYPE);
#undef W_RTYPE
        default:
          assert(false);
          break;
        }
        CHECK_H5(H5Dclose(ds));
      }
      {
        hid_t ds =
          H5Dcreate(
            mr_id,
            HYPERION_MEAS_REF_NVAL_DS,
            H5DatatypeManager::datatype<
            ValueType<MeasRef::NUM_VALUES_TYPE>::DataType>(),
            sp,
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        assert(ds >= 0);

        switch (dims.size()) {
#define W_NVAL(D)                                                       \
          case D:                                                       \
            write_mr_region<                                            \
              D, \
              MeasRef::NumValuesAccessor<READ_ONLY, D>, \
              MeasRef::NUM_VALUES_TYPE>( \
                ctx, \
                rt, \
                ds, \
                H5DatatypeManager::datatype< \
                  ValueType<MeasRef::NUM_VALUES_TYPE>::DataType>(), \
                mr.metadata_lr, \
                MeasRef::NUM_VALUES_FID); \
            break;
          HYPERION_FOREACH_N_LESS_MAX(W_NVAL);
#undef W_NVAL
        default:
          assert(false);
          break;
        }
        CHECK_H5(H5Dclose(ds));
      }
    }
    if (dims1.size() > 0) {
      hid_t ds =
        H5Dcreate(
          mr_id,
          HYPERION_MEAS_REF_VALUES_DS,
          H5DatatypeManager::datatype<ValueType<MeasRef::VALUE_TYPE>::DataType>(),
          sp1,
          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      assert(ds >= 0);

      switch (dims1.size()) {
#define W_VALUES(D)                                                     \
        case D + 1:                                                     \
          write_mr_region<                                              \
            D + 1, \
            MeasRef::ValueAccessor<READ_ONLY, D + 1>, \
            MeasRef::VALUE_TYPE>( \
              ctx, \
              rt, \
              ds, \
              H5DatatypeManager::datatype< \
                ValueType<MeasRef::VALUE_TYPE>::DataType>(), \
              mr.values_lr, \
              0); \
          break;
        HYPERION_FOREACH_N_LESS_MAX(W_VALUES);
#undef W_VALUES
      default:
        assert(false);
        break;
      }
      CHECK_H5(H5Dclose(ds));
    }
    // write the index array, if it exists
    if (mr.index_lr != LogicalRegion::NO_REGION) {
      hid_t udt =
        H5DatatypeManager::datatype<
          ValueType<MeasRef::M_CODE_TYPE>::DataType>();
      hid_t ds =
        H5Dcreate(
          mr_id,
          HYPERION_MEAS_REF_INDEX_DS,
          udt,
          sp1,
          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      assert(ds >= 0);
      write_mr_region<
        1,
        MeasRef::MCodeAccessor<READ_ONLY>,
        MeasRef::M_CODE_TYPE>(
          ctx,
          rt,
          ds,
          udt,
          mr.index_lr,
          MeasRef::M_CODE_FID);
      CHECK_H5(H5Dclose(ds));
    }
  }
  IndexTreeL metadata_tree =
    index_space_as_tree(rt, mr.metadata_lr.get_index_space());
  write_index_tree_to_attr<binary_index_tree_serdez>(
    metadata_tree,
    loc_id,
    name,
    "metadata_index_tree");
  if (mr.values_lr != LogicalRegion::NO_REGION) {
    IndexTreeL value_tree =
      index_space_as_tree(rt, mr.values_lr.get_index_space());
    write_index_tree_to_attr<binary_index_tree_serdez>(
      value_tree,
      loc_id,
      name,
      "value_index_tree");
  }
  CHECK_H5(H5Gclose(mr_id));
}
#endif //HYPERION_USE_CASACORE

void
hyperion::hdf5::write_column(
  Context ctx,
  Runtime* rt,
  const CXX_FILESYSTEM_NAMESPACE::path& path,
  hid_t table_id,
  const std::string& table_name,
  const Column& column,
  hid_t table_axes_dt,
  bool with_data) {

  // delete column dataset if it exists
  auto colname = column.name(ctx, rt);
  auto datatype = column.datatype(ctx, rt);

  // FIXME: the value of column_path is only correct when the table group
  // occurs at the HDF5 root...must add some way to pass in the path to the
  // table HDF5 group
  std::string column_path =
    std::string("/") + table_name + "/" + colname;

  htri_t ds_exists =
    H5Lexists(table_id, colname.c_str(), H5P_DEFAULT);
  if (ds_exists > 0)
    CHECK_H5(H5Ldelete(table_id, colname.c_str(), H5P_DEFAULT));
  else
    assert(ds_exists == 0);

  // create column group
  hid_t col_group_id =
    H5Gcreate(table_id, colname.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
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
      HYPERION_FOREACH_N(DIMS)
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
      HYPERION_FOREACH_DATATYPE(DT)
#undef DT
    default:
      assert(false);
      break;
    }

    col_id =
      H5Dcreate(
        col_group_id,
        HYPERION_COLUMN_DS,
        dt,
        ds,
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(col_id >= 0);
    CHECK_H5(H5Sclose(ds));

    // write column value datatype
    init_datatype_attr(col_id, datatype);

    CHECK_H5(H5Dclose(col_id));
  }

  // write axes attribute to column
  {
    htri_t rc = H5Aexists(col_group_id, column_axes_attr_name);
    if (rc > 0)
      CHECK_H5(H5Adelete(col_group_id, column_axes_attr_name));

    auto axes = column.axes(ctx, rt);    
    hsize_t dims = axes.size();
    hid_t axes_ds = H5Screate_simple(1, &dims, NULL);
    assert(axes_ds >= 0);
    hid_t axes_id =
      H5Acreate(
        col_group_id,
        column_axes_attr_name,
        table_axes_dt,
        axes_ds,
        H5P_DEFAULT, H5P_DEFAULT);
    assert(axes_id >= 0);
    std::vector<unsigned char> ax;
    ax.reserve(axes.size());
    std::copy(axes.begin(), axes.end(), std::back_inserter(ax));
    CHECK_H5(H5Awrite(axes_id, table_axes_dt, ax.data()));
    CHECK_H5(H5Aclose(axes_id));
    CHECK_H5(H5Sclose(axes_ds));
  }

  // write measure reference column name to attribute
  {
    htri_t rc = H5Aexists(col_group_id, column_refcol_attr_name);
    if (rc > 0)
      CHECK_H5(H5Adelete(col_group_id, column_refcol_attr_name));

    auto refcol = column.ref_column(ctx, rt);
    if (refcol) {
      hsize_t dims = 1;
      hid_t refcol_ds = H5Screate_simple(1, &dims, NULL);
      assert(refcol_ds >= 0);
      const hid_t sdt = H5DatatypeManager::datatype<HYPERION_TYPE_STRING>();
      hid_t refcol_id =
        H5Acreate(
          col_group_id,
          column_refcol_attr_name,
          sdt,
          refcol_ds,
          H5P_DEFAULT, H5P_DEFAULT);
      assert(refcol_id >= 0);
      string s = refcol.value();
      CHECK_H5(H5Awrite(refcol_id, sdt, s.val));
      CHECK_H5(H5Aclose(refcol_id));
      CHECK_H5(H5Sclose(refcol_ds));
    }
  }

  // write data to dataset
  if (with_data) {
    std::string column_ds_name = column_path + "/" + HYPERION_COLUMN_DS;
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
    RegionRequirement
      src(column.values_lr, READ_ONLY, EXCLUSIVE, column.values_lr);
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

#ifdef HYPERION_USE_CASACORE
  if (!column.meas_ref.is_empty()){
    {
      htri_t rc =
        H5Lexists(col_group_id, HYPERION_MEASURES_GROUP, H5P_DEFAULT);
      if (rc > 0)
        CHECK_H5(H5Ldelete(col_group_id, HYPERION_MEASURES_GROUP, H5P_DEFAULT));
    }
    hid_t measures_id =
      H5Gcreate(
        col_group_id,
        HYPERION_MEASURES_GROUP,
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(measures_id >= 0);
    write_measure(
      ctx,
      rt,
      measures_id,
      "FIXME",
      column.meas_ref);
    CHECK_H5(H5Gclose(measures_id));
  }
#endif

  CHECK_H5(H5Gclose(col_group_id));

  write_index_tree_to_attr<binary_index_tree_serdez>(
    column.index_tree(rt),
    table_id,
    colname,
    "index_tree");
}

void
hyperion::hdf5::write_table(
  Context ctx,
  Runtime* rt,
  const CXX_FILESYSTEM_NAMESPACE::path& path,
  hid_t loc_id,
  const Table& table,
  const std::unordered_set<std::string>& excluded_columns,
  bool with_data) {

  // open or create the group for the table
  auto tabname = table.name(ctx, rt);
  using_resource(
    [&]() {
      hid_t table_id;
      htri_t rc = H5Lexists(loc_id, tabname.c_str(), H5P_DEFAULT);
      if (rc == 0) {
        table_id =
          H5Gcreate(
            loc_id,
            tabname.c_str(),
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      } else {
        assert(rc > 0);
        table_id = H5Gopen(loc_id, tabname.c_str(), H5P_DEFAULT);
      }
      assert(table_id >= 0);
      return table_id;
    },
    [&](hid_t table_id) {
      // write axes datatype to table
      auto axes = AxesRegistrar::axes(table.axes_uid(ctx, rt));
      assert(axes);
      hid_t table_axes_dt = axes.value().h5_datatype;
      CHECK_H5(
        H5Tcommit(
          table_id,
          table_axes_dt_name,
          table_axes_dt,
          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

      // write index axes attribute to table
      htri_t rc = H5Aexists(table_id, table_index_axes_attr_name);
      if (rc > 0)
        CHECK_H5(H5Adelete(table_id, table_index_axes_attr_name));
      auto index_axes = table.index_axes(ctx, rt);
      using_resource(
        [&]() {
          hsize_t dims = index_axes.size();
          hid_t index_axes_ds = H5Screate_simple(1, &dims, NULL);
          assert(index_axes_ds >= 0);
          return index_axes_ds;
        },
        [&](hid_t index_axes_ds) {
          using_resource(
            [&]() {
              hid_t index_axes_id =
                H5Acreate(
                  table_id,
                  table_index_axes_attr_name,
                  table_axes_dt,
                  index_axes_ds,
                  H5P_DEFAULT, H5P_DEFAULT);
              assert(index_axes_id >= 0);
              return index_axes_id;
            },
            [&](hid_t index_axes_id) {
              std::vector<unsigned char> ax;
              ax.reserve(index_axes.size());
              std::copy(
                index_axes.begin(),
                index_axes.end(),
                std::back_inserter(ax));
              CHECK_H5(H5Awrite(index_axes_id, table_axes_dt, ax.data()));
            },
            [](hid_t index_axes_id) {
              CHECK_H5(H5Aclose(index_axes_id));
            });
        },
        [](hid_t index_axes_ds) {
          CHECK_H5(H5Sclose(index_axes_ds));
        });

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
              with_data);
        }
        rt->unmap_region(ctx, columns);
      }
      write_keywords(ctx, rt, table_id, table.keywords, with_data);
    },
    [](hid_t table_id) {
      CHECK_H5(H5Gclose(table_id));
    });
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
  if (!starts_with(name, HYPERION_NAMESPACE_PREFIX)) {
    H5O_info_t infobuf;
    CHECK_H5(H5Oget_info_by_name(loc_id, name, &infobuf, H5P_DEFAULT));
    if (infobuf.type == H5O_TYPE_DATASET)
      acc->push_back(name);
  }
  return 0;
}

static hyperion::TypeTag
read_dt_value(hid_t dt_id) {
  hyperion::TypeTag dt;
  // enumeration datatypes are converted by libhdf5 based on symbol names, which
  // ensures interoperability for hyperion HDF5 files written with one enumeration
  // definition and read with a different enumeration definition (for example,
  // in two hyperion codes built with and without HYPERION_USE_CASACORE)
  CHECK_H5(H5Aread(dt_id, H5T_NATIVE_INT, &dt));
  return dt;
}

hyperion::Keywords::kw_desc_t
hyperion::hdf5::init_keywords(
  Context ctx,
  Runtime* rt,
  hid_t loc_id) {

  std::vector<std::string> kw_names;
  hsize_t n = 0;
  CHECK_H5(
    H5Literate(
      loc_id,
      H5_INDEX_NAME,
      H5_ITER_INC,
      &n,
      acc_kw_names,
      &kw_names));

  if (kw_names.size() == 0)
    return {};

  return
    hyperion::map(
      kw_names,
      [&](const auto& nm) {
        hid_t dt_id =
          H5Aopen_by_name(
            loc_id,
            nm.c_str(),
            HYPERION_ATTRIBUTE_DT,
            H5P_DEFAULT, H5P_DEFAULT);
        assert(dt_id >= 0);
        hyperion::TypeTag dt = read_dt_value(dt_id);
        CHECK_H5(H5Aclose(dt_id));
        return std::make_tuple(nm, dt);
      });
}

#ifdef HYPERION_USE_CASACORE
template <typename T>
std::vector<T>
copy_mr_ds(hid_t ds) {

  hid_t spc = H5Dget_space(ds);
  assert(spc >= 0);
  int rank = H5Sget_simple_extent_ndims(spc);
  assert(rank > 0);
  hssize_t npts = H5Sget_simple_extent_npoints(spc);
  std::vector<T> result(npts);
  CHECK_H5(
    H5Dread(
      ds,
      H5DatatypeManager::datatype<ValueType<T>::DataType>(),
      H5S_ALL,
      H5S_ALL,
      H5P_DEFAULT,
      result.data()));
  CHECK_H5(H5Sclose(spc));
  return result;
}

template <int D, typename A, typename T>
static void
read_mr_region(
  Context ctx,
  Runtime* rt,
  hid_t ds,
  LogicalRegion region,
  FieldID fid) {

  std::vector<T> buff = copy_mr_ds<T>(ds);
  RegionRequirement req(region, WRITE_ONLY, EXCLUSIVE, region);
  req.add_field(fid);
  auto pr = rt->map_region(ctx, req);
  const A acc(pr, fid);
  Domain dom = rt->get_index_space_domain(region.get_index_space());
  Rect<D,coord_t> rect = dom.bounds<D,coord_t>();
  auto t = buff.begin();
  for (PointInRectIterator<D> pir(rect, false); pir(); pir++) {
    if (dom.contains(*pir))
      acc[*pir] = *t;
    ++t;
  }
  rt->unmap_region(ctx, pr);
}

static std::pair<std::string, MeasRef>
init_meas_ref(
  Context ctx,
  Runtime* rt,
  hid_t loc_id,
  const std::string& name,
  const std::optional<IndexTreeL>& metadata_tree,
  const std::optional<IndexTreeL>& value_tree,
  bool no_index) {

  if (!metadata_tree)
    return std::make_pair(name, MeasRef());

  std::array<LogicalRegion, 3> regions =
    MeasRef::create_regions(
      ctx,
      rt,
      metadata_tree.value(),
      value_tree.value(),
      no_index);
  LogicalRegion metadata_lr = regions[0];
  LogicalRegion values_lr = regions[1];
  LogicalRegion index_lr = regions[2];
  {
    // Read the datasets for the MeasRef values directly.
    {
      hid_t ds = H5Dopen(loc_id, HYPERION_MEAS_REF_MCLASS_DS, H5P_DEFAULT);
      assert(ds >= 0);

      switch (metadata_lr.get_index_space().get_dim()) {
#define W_MCLASS(D)                                                     \
        case D:                                                         \
          read_mr_region<                                               \
            D, \
            MeasRef::MeasureClassAccessor<WRITE_ONLY, D>, \
            MeasRef::MEASURE_CLASS_TYPE>( \
              ctx,                                                      \
              rt,                                                       \
              ds,                                                       \
              metadata_lr,                                              \
              MeasRef::MEASURE_CLASS_FID);                              \
          break;
        HYPERION_FOREACH_N_LESS_MAX(W_MCLASS);
#undef W_MCLASS
      default:
        assert(false);
        break;
      }
      CHECK_H5(H5Dclose(ds));
    }
    {
      hid_t ds = H5Dopen(loc_id, HYPERION_MEAS_REF_RTYPE_DS, H5P_DEFAULT);
      assert(ds >= 0);

      switch (metadata_lr.get_index_space().get_dim()) {
#define W_RTYPE(D)                                                      \
        case D:                                                         \
          read_mr_region<                                               \
            D, \
            MeasRef::RefTypeAccessor<WRITE_ONLY, D>, \
            MeasRef::REF_TYPE_TYPE>( \
              ctx,                                                      \
              rt,                                                       \
              ds,                                                       \
              metadata_lr,                                              \
              MeasRef::REF_TYPE_FID);                                   \
          break;
        HYPERION_FOREACH_N_LESS_MAX(W_RTYPE);
#undef W_RTYPE
      default:
        assert(false);
        break;
      }
      CHECK_H5(H5Dclose(ds));
    }
    {
      hid_t ds = H5Dopen(loc_id, HYPERION_MEAS_REF_NVAL_DS, H5P_DEFAULT);
      assert(ds >= 0);

      switch (metadata_lr.get_index_space().get_dim()) {
#define W_NVAL(D)                                                       \
        case D:                                                         \
          read_mr_region<                                               \
            D, \
            MeasRef::NumValuesAccessor<WRITE_ONLY, D>, \
            MeasRef::NUM_VALUES_TYPE>( \
              ctx,                                                      \
              rt,                                                       \
              ds,                                                       \
              metadata_lr,                                              \
              MeasRef::NUM_VALUES_FID);                                 \
          break;
        HYPERION_FOREACH_N_LESS_MAX(W_NVAL);
#undef W_NVAL
      default:
        assert(false);
        break;
      }
      CHECK_H5(H5Dclose(ds));
    }
  }
  if (values_lr != LogicalRegion::NO_REGION) {
    hid_t ds = H5Dopen(loc_id, HYPERION_MEAS_REF_VALUES_DS, H5P_DEFAULT);
    assert(ds >= 0);

    switch (values_lr.get_index_space().get_dim()) {
#define W_VALUES(D)                                                     \
      case D:                                                           \
        read_mr_region<                                                 \
          D, \
          MeasRef::ValueAccessor<WRITE_ONLY, D>, \
          MeasRef::VALUE_TYPE>( \
            ctx,                                                        \
            rt,                                                         \
            ds,                                                         \
            values_lr,                                                  \
            0);                                                         \
        break;
      HYPERION_FOREACH_N(W_VALUES);
#undef W_VALUES
    default:
      assert(false);
      break;
    }
    CHECK_H5(H5Dclose(ds));
  }
  if (index_lr != LogicalRegion::NO_REGION) {
    hid_t ds = H5Dopen(loc_id, HYPERION_MEAS_REF_INDEX_DS, H5P_DEFAULT);
    assert(ds >= 0);
    read_mr_region<
      1,
      MeasRef::MCodeAccessor<WRITE_ONLY>,
      MeasRef::M_CODE_TYPE>(ctx, rt, ds, index_lr, MeasRef::M_CODE_FID);
    CHECK_H5(H5Dclose(ds));
  }
  return std::make_pair(name, MeasRef(metadata_lr, values_lr, index_lr));
}

struct acc_meas_ref_ctx {
  Context ctx;
  Runtime* rt;
  bool has_index;
  std::unordered_map<std::string, MeasRef> acc;
};

static herr_t
acc_meas_ref(
  hid_t group,
  const char* name,
  const H5L_info_t* info,
  void* ctx) {

  acc_meas_ref_ctx* args = static_cast<acc_meas_ref_ctx*>(ctx);
  H5O_info_t infobuf;
  CHECK_H5(H5Oget_info_by_name(group, name, &infobuf, H5P_DEFAULT));
  if (infobuf.type == H5O_TYPE_GROUP) {
    hid_t mr_id = H5Gopen(group, name, H5P_DEFAULT);
    assert(mr_id >= 0);
    std::optional<IndexTreeL> metadata_tree;
    {
      std::optional<std::string> sid =
        read_index_tree_attr_metadata(mr_id, "metadata_index_tree");
      if (sid) {
        assert(sid.value() == "hyperion::hdf5::binary_index_tree_serdez");
        metadata_tree =
          read_index_tree_from_attr<binary_index_tree_serdez>(
            mr_id,
            "metadata_index_tree").value();
      }
    }
    std::optional<IndexTreeL> value_tree;
    {
      std::optional<std::string> sid =
        read_index_tree_attr_metadata(mr_id, "value_index_tree");
      if (sid) {
        assert(sid.value() == "hyperion::hdf5::binary_index_tree_serdez");
        value_tree =
          read_index_tree_from_attr<binary_index_tree_serdez>(
            mr_id,
            "value_index_tree");
      }
    }
    args->acc.insert(
      init_meas_ref(
        args->ctx,
        args->rt,
        mr_id,
        name,
        metadata_tree,
        value_tree,
        !args->has_index));
    CHECK_H5(H5Gclose(mr_id));
  }
  return 0;
}
#endif // HYPERION_USE_CASACORE

Column
hyperion::hdf5::init_column(
  Context ctx,
  Runtime* rt,
  const std::string& column_name,
  const std::string& axes_uid,
  hid_t loc_id,
  hid_t axes_dt,
  const std::string& name_prefix) {

  Column result;

  hyperion::TypeTag datatype = ValueType<int>::DataType;
  hid_t datatype_id = -1;
  std::vector<int> axes;
  hid_t axes_id = -1;
  hid_t axes_id_ds = -1;

  htri_t rc = H5Lexists(loc_id, HYPERION_COLUMN_DS, H5P_DEFAULT);
  if (rc > 0) {
    H5O_info_t infobuf;
    H5Oget_info_by_name(loc_id, HYPERION_COLUMN_DS, &infobuf, H5P_DEFAULT);
    if (infobuf.type == H5O_TYPE_DATASET) {
      {
        htri_t axes_exists = H5Aexists(loc_id, column_axes_attr_name);
        assert(axes_exists >= 0);
        if (axes_exists == 0)
          goto return_nothing;
        axes_id = H5Aopen(loc_id, column_axes_attr_name, H5P_DEFAULT);
        assert(axes_id >= 0);
        axes_id_ds = H5Aget_space(axes_id);
        assert(axes_id_ds >= 0);
        int ndims = H5Sget_simple_extent_ndims(axes_id_ds);
        if (ndims != 1)
          goto return_nothing;
        std::vector<unsigned char> ax(H5Sget_simple_extent_npoints(axes_id_ds));
        CHECK_H5(H5Aread(axes_id, axes_dt, ax.data()));
        axes.reserve(ax.size());
        std::copy(ax.begin(), ax.end(), std::back_inserter(axes));
      }
      {
        std::string datatype_name(HYPERION_ATTRIBUTE_DT);
        htri_t datatype_exists =
          H5Aexists_by_name(
            loc_id,
            HYPERION_COLUMN_DS,
            datatype_name.c_str(),
            H5P_DEFAULT);
        if (datatype_exists == 0)
          goto return_nothing;
        datatype_id =
          H5Aopen_by_name(
            loc_id,
            HYPERION_COLUMN_DS,
            datatype_name.c_str(),
            H5P_DEFAULT, H5P_DEFAULT);
        assert(datatype_id >= 0);
        datatype = read_dt_value(datatype_id);
      }
      std::optional<std::string> ref_column;
      {
        htri_t refcol_exists = H5Aexists(loc_id, column_refcol_attr_name);
        assert(refcol_exists >= 0);
        if (refcol_exists > 0){
          hid_t refcol_id =
            H5Aopen(loc_id, column_refcol_attr_name, H5P_DEFAULT);
          assert(refcol_id >= 0);
          hyperion::string s;
          CHECK_H5(
            H5Aread(
              refcol_id,
              H5DatatypeManager::datatype<HYPERION_TYPE_STRING>(),
              s.val));
          if (s.size() > 0)
            ref_column = s;
          CHECK_H5(H5Aclose(refcol_id));
        }
      }
      auto keywords = init_keywords(ctx, rt, loc_id);

#ifdef HYPERION_USE_CASACORE
      MeasRef mr;
      std::string mr_name;
      {
        std::unordered_map<std::string, MeasRef> mrs;
        htri_t rc = H5Lexists(loc_id, HYPERION_MEASURES_GROUP, H5P_DEFAULT);
        assert(rc >= 0);
        if (rc > 0) {
          hid_t measures_id =
            H5Gopen(loc_id, HYPERION_MEASURES_GROUP, H5P_DEFAULT);
          assert(measures_id >= 0);
          hsize_t position = 0;
          acc_meas_ref_ctx acc_meas_ref_ctx{ctx, rt, ref_column.has_value()};
          CHECK_H5(
            H5Literate(
              measures_id,
              H5_INDEX_NAME,
              H5_ITER_NATIVE,
              &position,
              acc_meas_ref,
              &acc_meas_ref_ctx));
          mrs = std::move(acc_meas_ref_ctx.acc);
          assert(mrs.size() == 1);
          mr = mrs.begin()->second;
          CHECK_H5(H5Gclose(measures_id));
        }
      }
#endif

      {
        std::optional<std::string> sid =
          read_index_tree_attr_metadata(loc_id, "index_tree");
        if (!sid
            || (sid.value() != "hyperion::hdf5::binary_index_tree_serdez"))
          goto return_nothing;
        std::optional<IndexTreeL> ixtree =
          read_index_tree_from_attr<binary_index_tree_serdez>(
            loc_id,
            "index_tree");
        assert(ixtree);
        auto itrank = ixtree.value().rank();
        IndexTreeL it;
        if (itrank && itrank.value() == axes.size())
          it = ixtree.value();
        result =
          Column::create(
            ctx,
            rt,
            column_name,
            axes_uid,
            axes,
            datatype,
            it,
#ifdef HYPERION_USE_CASACORE
            mr,
            ref_column,
#endif
            keywords,
            name_prefix);
      }
    }
  }

return_nothing:
  if (datatype_id >= 0)
    CHECK_H5(H5Aclose(datatype_id));
  if (axes_id_ds >= 0)
    CHECK_H5(H5Sclose(axes_id_ds));
  if (axes_id >= 0) 
    CHECK_H5(H5Aclose(axes_id));
  return result;
}

struct acc_col_ctx {
  const std::string& table_name;
  const std::unordered_set<std::string>* column_names;
  std::vector<hyperion::Column>* acc;
  std::string axes_uid;
  hid_t axes_dt;
  Runtime* rt;
  Context ctx;
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
          args->ctx,
          args->rt,
          name,
          args->axes_uid,
          col_group_id,
          args->axes_dt,
          args->table_name);
      args->acc->push_back(std::move(col));
      CHECK_H5(H5Gclose(col_group_id));
    }
  }
  return 0;
}

Table
hyperion::hdf5::init_table(
  Context ctx,
  Runtime* rt,
  const std::string& table_name,
  hid_t loc_id,
  const std::unordered_set<std::string>& column_names,
  const std::string& name_prefix) {

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
    hid_t dt = H5Topen(loc_id, table_axes_dt_name, H5P_DEFAULT);
    auto uid = AxesRegistrar::match_axes_datatype(dt);
    if (!uid)
      goto return_nothing;
    axes_uid = uid.value();
    axes_dt = AxesRegistrar::axes(axes_uid).value().h5_datatype;
    CHECK_H5(H5Tclose(dt));
  }
  {
    htri_t index_axes_exists = H5Aexists(loc_id, table_index_axes_attr_name);
    assert(index_axes_exists >= 0);
    if (index_axes_exists == 0)
      goto return_nothing;
    index_axes_id = H5Aopen(loc_id, table_index_axes_attr_name, H5P_DEFAULT);
    assert(index_axes_id >= 0);
    index_axes_id_ds = H5Aget_space(index_axes_id);
    assert(index_axes_id_ds >= 0);
    int ndims = H5Sget_simple_extent_ndims(index_axes_id_ds);
    if (ndims != 1)
      goto return_nothing;
    std::vector<unsigned char>
      ax(H5Sget_simple_extent_npoints(index_axes_id_ds));
    CHECK_H5(H5Aread(index_axes_id, axes_dt, ax.data()));
    index_axes.reserve(ax.size());
    std::copy(ax.begin(), ax.end(), std::back_inserter(index_axes));
  }
  {
    struct acc_col_ctx acc_col_ctx{
      table_name, &column_names, &cols, axes_uid, axes_dt, rt, ctx};
    hsize_t position = 0;
    CHECK_H5(
      H5Literate(
        loc_id,
        H5_INDEX_NAME,
        H5_ITER_NATIVE,
        &position,
        acc_col,
        &acc_col_ctx));
  }
  {
    auto keywords = init_keywords(ctx, rt, loc_id);

    result =
      Table::create(ctx, rt, table_name, axes_uid, index_axes, cols, keywords);
  }
return_nothing:
  if (index_axes_id_ds >= 0)
    CHECK_H5(H5Sclose(index_axes_id_ds));
  if (index_axes_id >= 0)
    CHECK_H5(H5Aclose(index_axes_id));
  return result;
}

Table
hyperion::hdf5::init_table(
  Context context,
  Runtime* runtime,
  const CXX_FILESYSTEM_NAMESPACE::path& file_path,
  const std::string& table_path,
  const std::unordered_set<std::string>& column_names,
  unsigned flags) {

  Table result;
  using_resource(
    [&]() {
      return H5Fopen(file_path.c_str(), flags, H5P_DEFAULT);
    },
    [&](hid_t fid) {
      if (fid >= 0) {
        using_resource(
          [&]() {
            return H5Gopen(fid, table_path.c_str(), H5P_DEFAULT);
          },
          [&](hid_t table_loc) {
            if (table_loc >= 0) {
              auto table_basename = table_path.rfind('/') + 1;
              result =
                init_table(
                  context,
                  runtime,
                  table_path.substr(table_basename),
                  table_loc,
                  column_names,
                  table_path.substr(0, table_basename));
            }
          },
          [](hid_t table_loc) {
            CHECK_H5(H5Gclose(table_loc));
          });
      }
    },
    [](hid_t fid) {
      CHECK_H5(H5Fclose(fid));
    });
  return result;
}

static herr_t
acc_table_paths(hid_t loc_id, const char* name, const H5L_info_t*, void* ctx) {

  std::unordered_set<std::string>* tblpaths =
    (std::unordered_set<std::string>*)(ctx);
  H5O_info_t infobuf;
  CHECK_H5( H5Oget_info_by_name(loc_id, name, &infobuf, H5P_DEFAULT));
  if (infobuf.type == H5O_TYPE_GROUP)
    tblpaths->insert(std::string("/") + name);
  return 0;
}

std::unordered_set<std::string>
hyperion::hdf5::get_table_paths(
  const CXX_FILESYSTEM_NAMESPACE::path& file_path) {

  std::unordered_set<std::string> result;
  hid_t fid = H5Fopen(file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (fid >= 0) {
    hsize_t n = 0;
    CHECK_H5(
      H5Literate(
        fid,
        H5_INDEX_NAME,
        H5_ITER_NATIVE,
        &n,
        acc_table_paths,
        &result));
    CHECK_H5(H5Fclose(fid));
  }
  return result;
}

static herr_t
acc_column_names(hid_t loc_id, const char* name, const H5L_info_t*, void* ctx) {
  std::unordered_set<std::string>* colnames =
    (std::unordered_set<std::string>*)(ctx);
  H5O_info_t infobuf;
  CHECK_H5(H5Oget_info_by_name(loc_id, name, &infobuf, H5P_DEFAULT));
  if (infobuf.type == H5O_TYPE_GROUP
      && (std::string(name).substr(0, sizeof(HYPERION_NAMESPACE_PREFIX) - 1)
          != HYPERION_NAMESPACE_PREFIX)) {
    hid_t gid = H5Gopen(loc_id, name, H5P_DEFAULT);
    assert(gid >= 0);
    htri_t has_col_ds = H5Oexists_by_name(gid, HYPERION_COLUMN_DS, H5P_DEFAULT);
    assert(has_col_ds >= 0);
    if (has_col_ds > 0) {
      CHECK_H5(
        H5Oget_info_by_name(gid, HYPERION_COLUMN_DS, &infobuf, H5P_DEFAULT));
      if (infobuf.type == H5O_TYPE_DATASET)
        colnames->insert(name);
    }
    CHECK_H5(H5Gclose(gid));
  }
  return 0;
}

std::unordered_set<std::string>
hyperion::hdf5::get_column_names(
  const CXX_FILESYSTEM_NAMESPACE::path& file_path,
  const std::string& table_path) {

  std::unordered_set<std::string> result;
  hid_t fid = H5Fopen(file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (fid >= 0) {
    hid_t tid = H5Gopen(fid, table_path.c_str(), H5P_DEFAULT);
    if (tid >= 0) {
      hsize_t n = 0;
      CHECK_H5(
        H5Literate(
          tid,
          H5_INDEX_NAME,
          H5_ITER_NATIVE,
          &n,
          acc_column_names,
          &result));
      CHECK_H5(H5Gclose(tid));
    }
    CHECK_H5(H5Fclose(fid));
  }
  return result;
}

std::unordered_map<std::string, std::string>
hyperion::hdf5::get_table_keyword_paths(
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
hyperion::hdf5::get_table_column_value_path(
  Context ctx,
  Runtime* rt,
  const Table& table,
  const std::string& colname) {

  return
    std::string("/") + table.name(ctx, rt) + "/"
    + colname + "/" + HYPERION_COLUMN_DS;
}

std::unordered_map<std::string, std::string>
hyperion::hdf5::get_table_column_keyword_paths(
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

PhysicalRegion
hyperion::hdf5::attach_keywords(
  Context ctx,
  Runtime* rt,
  const CXX_FILESYSTEM_NAMESPACE::path& file_path,
  const std::string& keywords_path,
  const Keywords& keywords,
  bool read_only) {

  assert(!keywords.is_empty());
  auto kws = keywords.values_lr;
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
  return rt->attach_external_resource(ctx, kws_attach);
}

PhysicalRegion
hyperion::hdf5::attach_column_values(
  Context ctx,
  Runtime* rt,
  const CXX_FILESYSTEM_NAMESPACE::path& file_path,
  const std::string& table_root,
  const Column& column,
  bool mapped,
  bool read_only) {

  assert(!column.is_empty());
  AttachLauncher attach(EXTERNAL_HDF5_FILE, column.values_lr, column.values_lr);
  attach.mapped = mapped;
  std::string col_path = table_root;
  if (col_path.back() != '/')
    col_path.push_back('/');
  col_path += column.name(ctx, rt) + "/" + HYPERION_COLUMN_DS;
  std::map<FieldID, const char*>
    fields{{Column::VALUE_FID, col_path.c_str()}};
  attach.attach_hdf5(
    file_path.c_str(),
    fields,
    read_only ? LEGION_FILE_READ_ONLY : LEGION_FILE_READ_WRITE);
  return rt->attach_external_resource(ctx, attach);
}

PhysicalRegion
hyperion::hdf5::attach_table_keywords(
  Context ctx,
  Runtime* rt,
  const CXX_FILESYSTEM_NAMESPACE::path& file_path,
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

void
hyperion::hdf5::release_table_column_values(
  Context ctx,
  Runtime* rt,
  const Table& table) {

  table.foreach_column(
    ctx,
    rt,
    [](Context c, Runtime* r, const Column& col) {
      ReleaseLauncher release(col.values_lr, col.values_lr);
      release.add_field(Column::VALUE_FID);
      r->issue_release(c, release);
    });
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
