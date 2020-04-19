/*
 * Copyright 2020 Associated Universities, Inc. Washington DC, USA.
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
#include <hyperion/utility.h>
#include <hyperion/tree_index_space.h>
#include <hyperion/MSTable.h>
#include <hyperion/Table.h>
#include <hyperion/ColumnSpace.h>
#include <hyperion/ColumnSpacePartition.h>
#include <hyperion/Column.h>
#include <hyperion/TableMapper.h>

#ifdef HYPERION_USE_HDF5
# include <hyperion/hdf5.h>
#endif // HYPERION_USE_HDF5

#ifdef HYPERION_USE_CASACORE
# include <hyperion/TableBuilder.h>
# include <hyperion/TableReadTask.h>
# include <hyperion/Measures.h>
#endif

# include <algorithm>
# include <cctype>

using namespace hyperion;
using namespace Legion;

void
hyperion::toupper(std::string& s) {
  std::transform(
    s.begin(),
    s.end(),
    s.begin(),
    [](unsigned char c){ return std::toupper(c); });
}

std::string
hyperion::toupper(const std::string& s) {
  std::string result;
  result.reserve(s.size());
  std::transform(
    s.begin(),
    s.end(),
    std::back_inserter(result),
    [](unsigned char c) { return std::toupper(c); });
  return result;
}

bool
hyperion::has_suffix(const std::string& str, const std::string& suffix) {
  typedef std::remove_reference_t<decltype(str)>::difference_type diff_t;
  return
    str.substr(std::max((diff_t)str.size() - (diff_t)suffix.size(), (diff_t)0))
    == suffix;
}

std::string
hyperion::add_name_prefix(const std::string& prefix, const std::string& str) {
  return
    ((prefix.size() > 0 && prefix.back() != '/') ? (prefix + "/") : prefix)
    + str;
}

unsigned
hyperion::min_divisor(
  size_t numerator,
  size_t min_quotient,
  unsigned max_divisor) {

  size_t quotient =
    std::max(min_quotient, (numerator + max_divisor - 1) / max_divisor);
  return (numerator + quotient - 1) / quotient;
}

IndexPartition
hyperion::partition_over_default_tunable(
  Context ctx,
  Runtime* rt,
  IndexSpace is,
  size_t min_block_size,
  Mapping::DefaultMapper::DefaultTunables tunable) {

  size_t num_subregions =
    rt->select_tunable_value(ctx, tunable).get_result<size_t>();
  auto dom = rt->get_index_space_domain(is);
  num_subregions =
    min_divisor(dom.get_volume(), min_block_size, num_subregions);
  Rect<1> color_rect(0, num_subregions - 1);
  IndexSpace color_is = rt->create_index_space(ctx, color_rect);
  return rt->create_equal_partition(ctx, is, color_is);
}

std::optional<int>
hyperion::column_is_axis(
  const std::vector<std::string>& axis_names,
  const std::string& colname,
  const std::vector<int>& axes) {

  auto colax =
    std::find_if(
      axes.begin(),
      axes.end(),
      [&axis_names, &colname](auto& ax) {
        return colname == axis_names[ax];
      });
  return (colax != axes.end()) ? *colax : std::optional<int>();
}

std::ostream&
operator<<(std::ostream& stream, const hyperion::string& str) {
  stream << str.val;
  return stream;
}

#define POINT_ADD_REDOP_IDENTITY(DIM)                       \
  const point_add_redop<DIM>::RHS                           \
  point_add_redop<DIM>::identity = Point<DIM>((coord_t)0);
HYPERION_FOREACH_N(POINT_ADD_REDOP_IDENTITY);
#undef POINT_ADD_REDOP_IDENTITY

CustomSerdezID hyperion::OpsManager::serdez_id_base;
ReductionOpID hyperion::OpsManager::reduction_id_base;

void
hyperion::OpsManager::register_ops(Runtime* rt) {
  static bool registered = false;
  if (registered)
    return;
  registered = true;

  serdez_id_base = rt->generate_library_serdez_ids("hyperion", NUM_SIDS);

  reduction_id_base =
    rt->generate_library_reduction_ids(
      "hyperion",
      POINT_ADD_REDOP_BASE + LEGION_MAX_DIM);

  Runtime::register_custom_serdez_op<index_tree_serdez>(
    serdez_id(INDEX_TREE_SID));

  Runtime::register_custom_serdez_op<
    vector_serdez<DomainPoint>>(serdez_id(V_DOMAIN_POINT_SID));
  Runtime::register_custom_serdez_op<
    string_serdez<std::string>>(serdez_id(STD_STRING_SID));

  Runtime::register_custom_serdez_op<
    acc_field_serdez<DataType<HYPERION_TYPE_STRING>::ValueType>>(
      serdez_id(ACC_FIELD_STRING_SID));
  Runtime::register_custom_serdez_op<
    acc_field_serdez<DataType<HYPERION_TYPE_BOOL>::ValueType>>(
      serdez_id(ACC_FIELD_BOOL_SID));
  Runtime::register_custom_serdez_op<
    acc_field_serdez<DataType<HYPERION_TYPE_CHAR>::ValueType>>(
      serdez_id(ACC_FIELD_CHAR_SID));
  Runtime::register_custom_serdez_op<
    acc_field_serdez<DataType<HYPERION_TYPE_UCHAR>::ValueType>>(
      serdez_id(ACC_FIELD_UCHAR_SID));
  Runtime::register_custom_serdez_op<
    acc_field_serdez<DataType<HYPERION_TYPE_SHORT>::ValueType>>(
      serdez_id(ACC_FIELD_SHORT_SID));
  Runtime::register_custom_serdez_op<
    acc_field_serdez<DataType<HYPERION_TYPE_USHORT>::ValueType>>(
      serdez_id(ACC_FIELD_USHORT_SID));
  Runtime::register_custom_serdez_op<
    acc_field_serdez<DataType<HYPERION_TYPE_INT>::ValueType>>(
      serdez_id(ACC_FIELD_INT_SID));
  Runtime::register_custom_serdez_op<
    acc_field_serdez<DataType<HYPERION_TYPE_UINT>::ValueType>>(
      serdez_id(ACC_FIELD_UINT_SID));
  Runtime::register_custom_serdez_op<
    acc_field_serdez<DataType<HYPERION_TYPE_FLOAT>::ValueType>>(
      serdez_id(ACC_FIELD_FLOAT_SID));
  Runtime::register_custom_serdez_op<
    acc_field_serdez<DataType<HYPERION_TYPE_DOUBLE>::ValueType>>(
      serdez_id(ACC_FIELD_DOUBLE_SID));
  Runtime::register_custom_serdez_op<
    acc_field_serdez<DataType<HYPERION_TYPE_COMPLEX>::ValueType>>(
      serdez_id(ACC_FIELD_COMPLEX_SID));
  Runtime::register_custom_serdez_op<
    acc_field_serdez<DataType<HYPERION_TYPE_DCOMPLEX>::ValueType>>(
      serdez_id(ACC_FIELD_DCOMPLEX_SID));

  Runtime::register_reduction_op<bool_or_redop>(
    reduction_id(BOOL_OR_REDOP));
  Runtime::register_reduction_op<coord_bor_redop>(
    reduction_id(COORD_BOR_REDOP));

#ifdef WITH_ACC_FIELD_REDOP_SERDEZ
  Runtime::register_reduction_op(
    reduction_id(ACC_FIELD_STRING_REDOP),
    Realm::ReductionOpUntyped::create_reduction_op<
    acc_field_redop<DataType<HYPERION_TYPE_STRING>::ValueType>>(),
    acc_field_redop<DataType<HYPERION_TYPE_STRING>::ValueType>::init_fn,
    acc_field_redop<DataType<HYPERION_TYPE_STRING>::ValueType>::fold_fn);
  Runtime::register_reduction_op(
    reduction_id(ACC_FIELD_BOOL_REDOP),
    Realm::ReductionOpUntyped::create_reduction_op<
    acc_field_redop<DataType<HYPERION_TYPE_BOOL>::ValueType>>(),
    acc_field_redop<DataType<HYPERION_TYPE_BOOL>::ValueType>::init_fn,
    acc_field_redop<DataType<HYPERION_TYPE_BOOL>::ValueType>::fold_fn);
  Runtime::register_reduction_op(
    reduction_id(ACC_FIELD_CHAR_REDOP),
    Realm::ReductionOpUntyped::create_reduction_op<
    acc_field_redop<DataType<HYPERION_TYPE_CHAR>::ValueType>>(),
    acc_field_redop<DataType<HYPERION_TYPE_CHAR>::ValueType>::init_fn,
    acc_field_redop<DataType<HYPERION_TYPE_CHAR>::ValueType>::fold_fn);
  Runtime::register_reduction_op(
    reduction_id(ACC_FIELD_UCHAR_REDOP),
    Realm::ReductionOpUntyped::create_reduction_op<
    acc_field_redop<DataType<HYPERION_TYPE_UCHAR>::ValueType>>(),
    acc_field_redop<DataType<HYPERION_TYPE_UCHAR>::ValueType>::init_fn,
    acc_field_redop<DataType<HYPERION_TYPE_UCHAR>::ValueType>::fold_fn);
  Runtime::register_reduction_op(
    reduction_id(ACC_FIELD_SHORT_REDOP),
    Realm::ReductionOpUntyped::create_reduction_op<
    acc_field_redop<DataType<HYPERION_TYPE_SHORT>::ValueType>>(),
    acc_field_redop<DataType<HYPERION_TYPE_SHORT>::ValueType>::init_fn,
    acc_field_redop<DataType<HYPERION_TYPE_SHORT>::ValueType>::fold_fn);
  Runtime::register_reduction_op(
    reduction_id(ACC_FIELD_USHORT_REDOP),
    Realm::ReductionOpUntyped::create_reduction_op<
    acc_field_redop<DataType<HYPERION_TYPE_USHORT>::ValueType>>(),
    acc_field_redop<DataType<HYPERION_TYPE_USHORT>::ValueType>::init_fn,
    acc_field_redop<DataType<HYPERION_TYPE_USHORT>::ValueType>::fold_fn);
  Runtime::register_reduction_op(
    reduction_id(ACC_FIELD_INT_REDOP),
    Realm::ReductionOpUntyped::create_reduction_op<
    acc_field_redop<DataType<HYPERION_TYPE_INT>::ValueType>>(),
    acc_field_redop<DataType<HYPERION_TYPE_INT>::ValueType>::init_fn,
    acc_field_redop<DataType<HYPERION_TYPE_INT>::ValueType>::fold_fn);
  Runtime::register_reduction_op(
    reduction_id(ACC_FIELD_UINT_REDOP),
    Realm::ReductionOpUntyped::create_reduction_op<
    acc_field_redop<DataType<HYPERION_TYPE_UINT>::ValueType>>(),
    acc_field_redop<DataType<HYPERION_TYPE_UINT>::ValueType>::init_fn,
    acc_field_redop<DataType<HYPERION_TYPE_UINT>::ValueType>::fold_fn);
  Runtime::register_reduction_op(
    reduction_id(ACC_FIELD_FLOAT_REDOP),
    Realm::ReductionOpUntyped::create_reduction_op<
    acc_field_redop<DataType<HYPERION_TYPE_FLOAT>::ValueType>>(),
    acc_field_redop<DataType<HYPERION_TYPE_FLOAT>::ValueType>::init_fn,
    acc_field_redop<DataType<HYPERION_TYPE_FLOAT>::ValueType>::fold_fn);
  Runtime::register_reduction_op(
    reduction_id(ACC_FIELD_DOUBLE_REDOP),
    Realm::ReductionOpUntyped::create_reduction_op<
    acc_field_redop<DataType<HYPERION_TYPE_DOUBLE>::ValueType>>(),
    acc_field_redop<DataType<HYPERION_TYPE_DOUBLE>::ValueType>::init_fn,
    acc_field_redop<DataType<HYPERION_TYPE_DOUBLE>::ValueType>::fold_fn);
  Runtime::register_reduction_op(
    reduction_id(ACC_FIELD_COMPLEX_REDOP),
    Realm::ReductionOpUntyped::create_reduction_op<
    acc_field_redop<DataType<HYPERION_TYPE_COMPLEX>::ValueType>>(),
    acc_field_redop<DataType<HYPERION_TYPE_COMPLEX>::ValueType>::init_fn,
    acc_field_redop<DataType<HYPERION_TYPE_COMPLEX>::ValueType>::fold_fn);
  Runtime::register_reduction_op(
    reduction_id(ACC_FIELD_DCOMPLEX_REDOP),
    Realm::ReductionOpUntyped::create_reduction_op<
    acc_field_redop<DataType<HYPERION_TYPE_DCOMPLEX>::ValueType>>(),
    acc_field_redop<DataType<HYPERION_TYPE_DCOMPLEX>::ValueType>::init_fn,
    acc_field_redop<DataType<HYPERION_TYPE_DCOMPLEX>::ValueType>::fold_fn);
#else
  Runtime::register_reduction_op<
    acc_field_redop<DataType<HYPERION_TYPE_STRING>::ValueType>>(
      reduction_id(ACC_FIELD_STRING_REDOP));
  Runtime::register_reduction_op<
    acc_field_redop<DataType<HYPERION_TYPE_BOOL>::ValueType>>(
      reduction_id(ACC_FIELD_BOOL_REDOP));
  Runtime::register_reduction_op<
    acc_field_redop<DataType<HYPERION_TYPE_CHAR>::ValueType>>(
      reduction_id(ACC_FIELD_CHAR_REDOP));
  Runtime::register_reduction_op<
    acc_field_redop<DataType<HYPERION_TYPE_UCHAR>::ValueType>>(
      reduction_id(ACC_FIELD_UCHAR_REDOP));
  Runtime::register_reduction_op<
    acc_field_redop<DataType<HYPERION_TYPE_SHORT>::ValueType>>(
      reduction_id(ACC_FIELD_SHORT_REDOP));
  Runtime::register_reduction_op<
    acc_field_redop<DataType<HYPERION_TYPE_USHORT>::ValueType>>(
      reduction_id(ACC_FIELD_USHORT_REDOP));
  Runtime::register_reduction_op<
    acc_field_redop<DataType<HYPERION_TYPE_INT>::ValueType>>(
      reduction_id(ACC_FIELD_INT_REDOP));
  Runtime::register_reduction_op<
    acc_field_redop<DataType<HYPERION_TYPE_UINT>::ValueType>>(
      reduction_id(ACC_FIELD_UINT_REDOP));
  Runtime::register_reduction_op<
    acc_field_redop<DataType<HYPERION_TYPE_FLOAT>::ValueType>>(
      reduction_id(ACC_FIELD_FLOAT_REDOP));
  Runtime::register_reduction_op<
    acc_field_redop<DataType<HYPERION_TYPE_DOUBLE>::ValueType>>(
      reduction_id(ACC_FIELD_DOUBLE_REDOP));
  Runtime::register_reduction_op<
    acc_field_redop<DataType<HYPERION_TYPE_COMPLEX>::ValueType>>(
      reduction_id(ACC_FIELD_COMPLEX_REDOP));
  Runtime::register_reduction_op<
    acc_field_redop<DataType<HYPERION_TYPE_DCOMPLEX>::ValueType>>(
      reduction_id(ACC_FIELD_DCOMPLEX_REDOP));
#endif // WITH_ACC_FIELD_REDOP_SERDEZ

#define REGISTER_POINT_ADD_REDOP(DIM)                   \
  Runtime::register_reduction_op<point_add_redop<DIM>>( \
    reduction_id(POINT_ADD_REDOP(DIM)));
  HYPERION_FOREACH_N(REGISTER_POINT_ADD_REDOP);
#undef REGISTER_POINT_ADD_REDOP
}

FieldID
hyperion::add_field(
  TypeTag datatype,
  FieldAllocator fa,
  FieldID field_id) {

  FieldID result;

#define ALLOC_FLD(tp)                           \
  case tp:                                      \
    result = fa.allocate_field(                 \
      sizeof(DataType<tp>::ValueType),          \
      field_id);                                \
    break;

  switch (datatype) {
    HYPERION_FOREACH_DATATYPE(ALLOC_FLD);
  default:
    assert(false);
    result = AUTO_GENERATE_ID;
    break;
  }
#undef ALLOC_FLD

  return result;
}

LayoutConstraintRegistrar&
hyperion::add_row_major_order_constraint(
  LayoutConstraintRegistrar& lc,
  unsigned rank) {

  std::vector<DimensionKind> dims(rank);
  std::generate(
    dims.rbegin(),
    dims.rend(),
    [n = 0]() mutable {
      return static_cast<legion_dimension_kind_t>(n++);
    });
  return lc.add_constraint(OrderingConstraint(dims, false));
}

Legion::MapperID hyperion::table_mapper;
Legion::LayoutConstraintID hyperion::soa_row_major_layout;
Legion::LayoutConstraintID hyperion::soa_column_major_layout;
Legion::LayoutConstraintID hyperion::aos_row_major_layout;
Legion::LayoutConstraintID hyperion::aos_column_major_layout;

void
hyperion::register_mapper(
  Machine machine,
  Runtime* rt,
  const std::set<Processor>& local_procs) {

  for (auto& proc : local_procs) {
    const size_t buffer_size = 64;
    char *name = static_cast<char*>(std::malloc(buffer_size * sizeof(char)));
    std::snprintf(
      name,
      buffer_size - 1,
      "TableMapper on Processor " IDFMT "",
      proc.id);
    rt->add_mapper(table_mapper, new TableMapper(machine, rt, proc, name), proc);
  }
}

void
hyperion::preregister_all() {

  {
    LayoutConstraintRegistrar
      registrar(FieldSpace::NO_SPACE, "soa_row_major");
    for (unsigned rank = 1; rank <= LEGION_MAX_DIM; ++rank) {
      std::vector<DimensionKind> dims(rank + 1);
      auto d = dims.rbegin();
      *d++ = DimensionKind::DIM_F;
      std::generate(
        d,
        dims.rend(),
        [n = static_cast<int>(DimensionKind::DIM_X)]() mutable {
          return static_cast<DimensionKind>(n++);
        });
      registrar.add_constraint(OrderingConstraint(dims, false));
    }
    soa_row_major_layout = Runtime::preregister_layout(registrar);
  }
  {
    LayoutConstraintRegistrar
      registrar(FieldSpace::NO_SPACE, "soa_column_major");
    for (unsigned rank = 1; rank <= LEGION_MAX_DIM; ++rank) {
      std::vector<DimensionKind> dims(rank + 1);
      auto d = dims.rbegin();
      *d++ = DimensionKind::DIM_F;
      std::generate(
        d,
        dims.rend(),
        [n = static_cast<int>(DimensionKind::DIM_X + rank - 1)]() mutable {
          return static_cast<DimensionKind>(n--);
        });
      registrar.add_constraint(OrderingConstraint(dims, false));
    }
    soa_column_major_layout = Runtime::preregister_layout(registrar);
  }
  {
    LayoutConstraintRegistrar
      registrar(FieldSpace::NO_SPACE, "aos_row_major");
    for (unsigned rank = 1; rank <= LEGION_MAX_DIM; ++rank) {
      std::vector<DimensionKind> dims(rank + 1);
      auto d = dims.begin();
      *d++ = DimensionKind::DIM_F;
      std::generate(
        d,
        dims.end(),
        [n = static_cast<int>(DimensionKind::DIM_X + rank - 1)]() mutable {
          return static_cast<DimensionKind>(n--);
        });
      registrar.add_constraint(OrderingConstraint(dims, false));
    }
    aos_row_major_layout = Runtime::preregister_layout(registrar);
  }
  {
    LayoutConstraintRegistrar
      registrar(FieldSpace::NO_SPACE, "aos_column_major");
    for (unsigned rank = 1; rank <= LEGION_MAX_DIM; ++rank) {
      std::vector<DimensionKind> dims(rank + 1);
      auto d = dims.begin();
      *d++ = DimensionKind::DIM_F;
      std::generate(
        d,
        dims.end(),
        [n = static_cast<int>(DimensionKind::DIM_X)]() mutable {
          return static_cast<DimensionKind>(n++);
        });
      registrar.add_constraint(OrderingConstraint(dims, false));
    }
    aos_column_major_layout = Runtime::preregister_layout(registrar);
  }

  table_mapper = Runtime::generate_static_mapper_id();
  Runtime::add_registration_callback(register_mapper);

#ifdef HYPERION_USE_HDF5
  H5DatatypeManager::preregister_datatypes();
#endif

#define REG_AXES(T) \
  AxesRegistrar::register_axes<typename MSTable<MS_##T>::Axes>();

  HYPERION_FOREACH_MS_TABLE(REG_AXES);
#undef REG_AXES

  TreeIndexSpaceTask::preregister_task();
  Table::preregister_tasks();
  ColumnSpace::preregister_tasks();
  ColumnSpacePartition::preregister_tasks();
  Column::preregister_tasks();
#ifdef HYPERION_USE_CASACORE
  TableReadTask::preregister_tasks();
#endif
}

void
hyperion::register_tasks(Context context, Runtime* runtime) {
  OpsManager::register_ops(runtime);
}

std::unordered_map<std::string, hyperion::AxesRegistrar::A>
hyperion::AxesRegistrar::axes_;

#ifdef HYPERION_USE_HDF5
void
hyperion::AxesRegistrar::register_axes(
  const std::string uid,
  const std::vector<std::string> names,
  hid_t hid) {
  A a{uid, names, hid};
  axes_[uid] = a;
}
#else // !HYPERION_USE_HDF5
void
hyperion::AxesRegistrar::register_axes(
  const std::string uid,
  const std::vector<std::string> names) {
  A a{uid, names};
  axes_[uid] = a;
}
#endif

std::optional<hyperion::AxesRegistrar::A>
hyperion::AxesRegistrar::axes(const std::string& uid) {
  return (axes_.count(uid) > 0) ? axes_[uid] : std::optional<A>();
}

#ifdef HYPERION_USE_HDF5
std::optional<std::string>
hyperion::AxesRegistrar::match_axes_datatype(hid_t hid) {
  auto ad =
    std::find_if(
      axes_.begin(),
      axes_.end(),
      [&hid](auto& ua) {
        return std::get<1>(ua).h5_datatype == hid;
      });
  return
    (ad != axes_.end()) ? std::get<0>(*ad) : std::optional<std::string>();
}
#endif // HYPERION_USE_HDF5

bool
hyperion::AxesRegistrar::in_range(
  const std::string& uid,
  const std::vector<int> xs) {

  auto oaxs = AxesRegistrar::axes(uid);
  if (!oaxs)
    return false;
  return
    std::all_of(
      xs.begin(),
      xs.end(),
      [m=oaxs.value().names.size()](auto& x) {
        return 0 <= x && static_cast<unsigned>(x) < m;
      });
}

#ifdef HYPERION_USE_HDF5
hid_t
hyperion::H5DatatypeManager::datatypes_[N_H5T_DATATYPES];

void
hyperion::H5DatatypeManager::preregister_datatypes() {
  datatypes_[BOOL_H5T] = H5T_NATIVE_HBOOL;
  datatypes_[CHAR_H5T] = H5T_NATIVE_SCHAR;
  datatypes_[UCHAR_H5T] = H5T_NATIVE_UCHAR;
  datatypes_[SHORT_H5T] = H5T_NATIVE_SHORT;
  datatypes_[USHORT_H5T] = H5T_NATIVE_USHORT;
  datatypes_[INT_H5T] = H5T_NATIVE_INT;
  datatypes_[UINT_H5T] = H5T_NATIVE_UINT;
  datatypes_[FLOAT_H5T] = H5T_NATIVE_FLOAT;
  datatypes_[DOUBLE_H5T] = H5T_NATIVE_DOUBLE;
  {
    hid_t dt = H5Tcreate(H5T_COMPOUND, 2 * sizeof(float));
    H5Tinsert(dt, "real", 0, H5T_NATIVE_FLOAT);
    H5Tinsert(dt, "imag", sizeof(float), H5T_NATIVE_FLOAT);
    datatypes_[COMPLEX_H5T] = dt;
  }
  {
    hid_t dt = H5Tcreate(H5T_COMPOUND, 2 * sizeof(double));
    H5Tinsert(dt, "real", 0, H5T_NATIVE_DOUBLE);
    H5Tinsert(dt, "imag", sizeof(double), H5T_NATIVE_DOUBLE);
    datatypes_[DCOMPLEX_H5T] = dt;
  }
  {
    hid_t dt = H5Tcopy(H5T_C_S1);
    H5Tset_size(dt, HYPERION_MAX_STRING_SIZE);
    datatypes_[STRING_H5T] = dt;
  }
  {
    hid_t dt = H5Tenum_create(H5T_NATIVE_UINT);

#define DTINSERT(T) do {                                            \
      unsigned val = HYPERION_TYPE_##T;                             \
      [[maybe_unused]] herr_t err = H5Tenum_insert(dt, #T, &val);   \
      assert(err >= 0);                                             \
    } while(0);
    HYPERION_FOREACH_BARE_DATATYPE(DTINSERT);
#undef DTINSERT
    datatypes_[DATATYPE_H5T] = dt;
  }

  datatypes_[FIELD_ID_H5T] = H5T_NATIVE_UINT;

#ifdef HYPERION_USE_CASACORE
  {
    hid_t dt = H5Tenum_create(H5T_NATIVE_UINT);
#define MCINSERT(MC) do {                                     \
      unsigned val = MC;                                      \
      [[maybe_unused]] herr_t err =                           \
        H5Tenum_insert(dt, MClassT<MC>::name.c_str(), &val);  \
      assert(err >= 0);                                       \
    } while(0);
    HYPERION_FOREACH_MCLASS(MCINSERT);
#undef MCINSERT
    datatypes_[MEASURE_CLASS_H5T] = dt;
  }
#endif
}

herr_t
hyperion::H5DatatypeManager::commit_derived(
  hid_t loc_id,
  hid_t lcpl_id,
  hid_t tcpl_id,
  hid_t tapl_id) {

  herr_t result =
    H5Tcommit(
      loc_id,
      "hyperion::complex",
      datatypes_[COMPLEX_H5T],
      lcpl_id,
      tcpl_id,
      tapl_id);
  if (result < 0)
    return result;

  result =
    H5Tcommit(
      loc_id,
      "hyperion::dcomplex",
      datatypes_[DCOMPLEX_H5T],
      lcpl_id,
      tcpl_id,
      tapl_id);
  if (result < 0)
    return result;

  result =
    H5Tcommit(
      loc_id,
      "hyperion::string",
      datatypes_[STRING_H5T],
      lcpl_id,
      tcpl_id,
      tapl_id);
  if (result < 0)
    return result;

  result =
    H5Tcommit(
      loc_id,
      "hyperion::TypeTag",
      datatypes_[DATATYPE_H5T],
      lcpl_id,
      tcpl_id,
      tapl_id);

  if (result < 0)
    return result;

  result =
    H5Tcommit(
      loc_id,
      "hyperion::MClass",
      datatypes_[MEASURE_CLASS_H5T],
      lcpl_id,
      tcpl_id,
      tapl_id);

  return result;
}

hid_t
hyperion::H5DatatypeManager::create(
  const CXX_FILESYSTEM_NAMESPACE::path& path,
  unsigned flags,
  hid_t fcpl_t,
  hid_t fapl_t) {

  hid_t result = H5Fcreate(path.c_str(), flags, fcpl_t, fapl_t);
  if (result >= 0) {
    [[maybe_unused]] herr_t rc = commit_derived(result);
    assert(rc >= 0);
  }
  return result;
}
#endif // HYPERION_USE_HDF5

IndexTreeL
hyperion::index_space_as_tree(Runtime* rt, IndexSpace is) {
  IndexTreeL result;
  Domain dom = rt->get_index_space_domain(is);
  switch (dom.get_dim()) {
#define TREE(N)                                         \
    case (N): {                                         \
      RectInDomainIterator<N> rid(dom);                 \
      while (rid()) {                                   \
        IndexTreeL t;                                   \
        for (size_t i = N; i > 0; --i) {                \
          t =                                           \
            IndexTreeL({                                \
                std::make_tuple(                        \
                  rid->lo[i - 1],                       \
                  rid->hi[i - 1] - rid->lo[i - 1] + 1,  \
                  t)});                                 \
        }                                               \
        result = result.merged_with(t);                 \
        rid++;                                          \
      }                                                 \
      break;                                            \
    }
    HYPERION_FOREACH_N(TREE)
#undef TREE
  default:
      assert(false);
    break;
  }
  return result;
}

#ifdef HYPERION_USE_CASACORE
std::pair<
  std::string,
  std::vector<
    std::tuple<
      ColumnSpace,
      bool,
      std::vector<std::pair<std::string, TableField>>>>>
hyperion::from_ms(
  Context ctx,
  Runtime* rt,
  const CXX_FILESYSTEM_NAMESPACE::path& path,
  const std::unordered_set<std::string>& column_selections) {

  std::string table_name = path.filename();

#define FROM_MS_TABLE(N)                        \
  do {                                          \
    if (table_name == MSTable<MS_##N>::name)    \
      return                                    \
        hyperion:: template from_ms<MS_##N>(    \
          ctx,                                  \
          rt,                                   \
          path,                                 \
          column_selections);                   \
  } while (0);

  HYPERION_FOREACH_MS_TABLE(FROM_MS_TABLE);

  // try to read as main table
  return hyperion:: template from_ms<MS_MAIN>(ctx, rt, path, column_selections);

#undef FROM_MS_TABLE
}
#endif // HYPERION_USE_CASACORE

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
