#pragma GCC visibility push(default)
#include<algorithm>
#pragma GCC visibility pop

#include "utility.h"
#include "Column.h"
#include "Table.h"
#include "TableReadTask.h"
#include "tree_index_space.h"
#include "MSTable.h"

#ifdef LEGMS_USE_HDF5
# include "legms_hdf5.h"
#endif // LEGMS_USE_HDF5

using namespace legms;
using namespace Legion;

std::optional<int>
legms::column_is_axis(
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
operator<<(std::ostream& stream, const legms::string& str) {
  stream << str.val;
  return stream;
}

#define POINT_ADD_REDOP_IDENTITY(DIM)                       \
  const point_add_redop<DIM>::RHS                           \
  point_add_redop<DIM>::identity = Point<DIM>((coord_t)0);
LEGMS_FOREACH_N(POINT_ADD_REDOP_IDENTITY);
#undef POINT_ADD_REDOP_IDENTITY

void
legms::OpsManager::preregister_ops() {
  Runtime::register_custom_serdez_op<index_tree_serdez>(
    INDEX_TREE_SID);

  Runtime::register_custom_serdez_op<
    vector_serdez<DomainPoint>>(V_DOMAIN_POINT_SID);
  Runtime::register_custom_serdez_op<
    string_serdez<std::string>>(STD_STRING_SID);

  Runtime::register_custom_serdez_op<
    acc_field_serdez<DataType<LEGMS_TYPE_STRING>::ValueType>>(
      ACC_FIELD_STRING_SID);
  Runtime::register_custom_serdez_op<
    acc_field_serdez<DataType<LEGMS_TYPE_BOOL>::ValueType>>(
      ACC_FIELD_BOOL_SID);
  Runtime::register_custom_serdez_op<
    acc_field_serdez<DataType<LEGMS_TYPE_CHAR>::ValueType>>(
      ACC_FIELD_CHAR_SID);
  Runtime::register_custom_serdez_op<
    acc_field_serdez<DataType<LEGMS_TYPE_UCHAR>::ValueType>>(
      ACC_FIELD_UCHAR_SID);
  Runtime::register_custom_serdez_op<
    acc_field_serdez<DataType<LEGMS_TYPE_SHORT>::ValueType>>(
      ACC_FIELD_SHORT_SID);
  Runtime::register_custom_serdez_op<
    acc_field_serdez<DataType<LEGMS_TYPE_USHORT>::ValueType>>(
      ACC_FIELD_USHORT_SID);
  Runtime::register_custom_serdez_op<
    acc_field_serdez<DataType<LEGMS_TYPE_INT>::ValueType>>(
      ACC_FIELD_INT_SID);
  Runtime::register_custom_serdez_op<
    acc_field_serdez<DataType<LEGMS_TYPE_UINT>::ValueType>>(
      ACC_FIELD_UINT_SID);
  Runtime::register_custom_serdez_op<
    acc_field_serdez<DataType<LEGMS_TYPE_FLOAT>::ValueType>>(
      ACC_FIELD_FLOAT_SID);
  Runtime::register_custom_serdez_op<
    acc_field_serdez<DataType<LEGMS_TYPE_DOUBLE>::ValueType>>(
      ACC_FIELD_DOUBLE_SID);
  Runtime::register_custom_serdez_op<
    acc_field_serdez<DataType<LEGMS_TYPE_COMPLEX>::ValueType>>(
      ACC_FIELD_COMPLEX_SID);
  Runtime::register_custom_serdez_op<
    acc_field_serdez<DataType<LEGMS_TYPE_DCOMPLEX>::ValueType>>(
      ACC_FIELD_DCOMPLEX_SID);

  Runtime::register_reduction_op<bool_or_redop>(BOOL_OR_REDOP);
  Runtime::register_reduction_op<coord_bor_redop>(COORD_BOR_REDOP);

#ifdef WITH_ACC_FIELD_REDOP_SERDEZ
  Runtime::register_reduction_op(
    ACC_FIELD_STRING_REDOP,
    Realm::ReductionOpUntyped::create_reduction_op<
    acc_field_redop<DataType<LEGMS_TYPE_STRING>::ValueType>>(),
    acc_field_redop<DataType<LEGMS_TYPE_STRING>::ValueType>::init_fn,
    acc_field_redop<DataType<LEGMS_TYPE_STRING>::ValueType>::fold_fn);
  Runtime::register_reduction_op(
    ACC_FIELD_BOOL_REDOP,
    Realm::ReductionOpUntyped::create_reduction_op<
    acc_field_redop<DataType<LEGMS_TYPE_BOOL>::ValueType>>(),
    acc_field_redop<DataType<LEGMS_TYPE_BOOL>::ValueType>::init_fn,
    acc_field_redop<DataType<LEGMS_TYPE_BOOL>::ValueType>::fold_fn);
  Runtime::register_reduction_op(
    ACC_FIELD_CHAR_REDOP,
    Realm::ReductionOpUntyped::create_reduction_op<
    acc_field_redop<DataType<LEGMS_TYPE_CHAR>::ValueType>>(),
    acc_field_redop<DataType<LEGMS_TYPE_CHAR>::ValueType>::init_fn,
    acc_field_redop<DataType<LEGMS_TYPE_CHAR>::ValueType>::fold_fn);
  Runtime::register_reduction_op(
    ACC_FIELD_UCHAR_REDOP,
    Realm::ReductionOpUntyped::create_reduction_op<
    acc_field_redop<DataType<LEGMS_TYPE_UCHAR>::ValueType>>(),
    acc_field_redop<DataType<LEGMS_TYPE_UCHAR>::ValueType>::init_fn,
    acc_field_redop<DataType<LEGMS_TYPE_UCHAR>::ValueType>::fold_fn);
  Runtime::register_reduction_op(
    ACC_FIELD_SHORT_REDOP,
    Realm::ReductionOpUntyped::create_reduction_op<
    acc_field_redop<DataType<LEGMS_TYPE_SHORT>::ValueType>>(),
    acc_field_redop<DataType<LEGMS_TYPE_SHORT>::ValueType>::init_fn,
    acc_field_redop<DataType<LEGMS_TYPE_SHORT>::ValueType>::fold_fn);
  Runtime::register_reduction_op(
    ACC_FIELD_USHORT_REDOP,
    Realm::ReductionOpUntyped::create_reduction_op<
    acc_field_redop<DataType<LEGMS_TYPE_USHORT>::ValueType>>(),
    acc_field_redop<DataType<LEGMS_TYPE_USHORT>::ValueType>::init_fn,
    acc_field_redop<DataType<LEGMS_TYPE_USHORT>::ValueType>::fold_fn);
  Runtime::register_reduction_op(
    ACC_FIELD_INT_REDOP,
    Realm::ReductionOpUntyped::create_reduction_op<
    acc_field_redop<DataType<LEGMS_TYPE_INT>::ValueType>>(),
    acc_field_redop<DataType<LEGMS_TYPE_INT>::ValueType>::init_fn,
    acc_field_redop<DataType<LEGMS_TYPE_INT>::ValueType>::fold_fn);
  Runtime::register_reduction_op(
    ACC_FIELD_UINT_REDOP,
    Realm::ReductionOpUntyped::create_reduction_op<
    acc_field_redop<DataType<LEGMS_TYPE_UINT>::ValueType>>(),
    acc_field_redop<DataType<LEGMS_TYPE_UINT>::ValueType>::init_fn,
    acc_field_redop<DataType<LEGMS_TYPE_UINT>::ValueType>::fold_fn);
  Runtime::register_reduction_op(
    ACC_FIELD_FLOAT_REDOP,
    Realm::ReductionOpUntyped::create_reduction_op<
    acc_field_redop<DataType<LEGMS_TYPE_FLOAT>::ValueType>>(),
    acc_field_redop<DataType<LEGMS_TYPE_FLOAT>::ValueType>::init_fn,
    acc_field_redop<DataType<LEGMS_TYPE_FLOAT>::ValueType>::fold_fn);
  Runtime::register_reduction_op(
    ACC_FIELD_DOUBLE_REDOP,
    Realm::ReductionOpUntyped::create_reduction_op<
    acc_field_redop<DataType<LEGMS_TYPE_DOUBLE>::ValueType>>(),
    acc_field_redop<DataType<LEGMS_TYPE_DOUBLE>::ValueType>::init_fn,
    acc_field_redop<DataType<LEGMS_TYPE_DOUBLE>::ValueType>::fold_fn);
  Runtime::register_reduction_op(
    ACC_FIELD_COMPLEX_REDOP,
    Realm::ReductionOpUntyped::create_reduction_op<
    acc_field_redop<DataType<LEGMS_TYPE_COMPLEX>::ValueType>>(),
    acc_field_redop<DataType<LEGMS_TYPE_COMPLEX>::ValueType>::init_fn,
    acc_field_redop<DataType<LEGMS_TYPE_COMPLEX>::ValueType>::fold_fn);
  Runtime::register_reduction_op(
    ACC_FIELD_DCOMPLEX_REDOP,
    Realm::ReductionOpUntyped::create_reduction_op<
    acc_field_redop<DataType<LEGMS_TYPE_DCOMPLEX>::ValueType>>(),
    acc_field_redop<DataType<LEGMS_TYPE_DCOMPLEX>::ValueType>::init_fn,
    acc_field_redop<DataType<LEGMS_TYPE_DCOMPLEX>::ValueType>::fold_fn);
#else
  Runtime::register_reduction_op<
    acc_field_redop<DataType<LEGMS_TYPE_STRING>::ValueType>>(
    ACC_FIELD_STRING_REDOP);
  Runtime::register_reduction_op<
    acc_field_redop<DataType<LEGMS_TYPE_BOOL>::ValueType>>(
    ACC_FIELD_BOOL_REDOP);
  Runtime::register_reduction_op<
    acc_field_redop<DataType<LEGMS_TYPE_CHAR>::ValueType>>(
    ACC_FIELD_CHAR_REDOP);
  Runtime::register_reduction_op<
    acc_field_redop<DataType<LEGMS_TYPE_UCHAR>::ValueType>>(
    ACC_FIELD_UCHAR_REDOP);
  Runtime::register_reduction_op<
    acc_field_redop<DataType<LEGMS_TYPE_SHORT>::ValueType>>(
    ACC_FIELD_SHORT_REDOP);
  Runtime::register_reduction_op<
    acc_field_redop<DataType<LEGMS_TYPE_USHORT>::ValueType>>(
    ACC_FIELD_USHORT_REDOP);
  Runtime::register_reduction_op<
    acc_field_redop<DataType<LEGMS_TYPE_INT>::ValueType>>(
    ACC_FIELD_INT_REDOP);
  Runtime::register_reduction_op<
    acc_field_redop<DataType<LEGMS_TYPE_UINT>::ValueType>>(
    ACC_FIELD_UINT_REDOP);
  Runtime::register_reduction_op<
    acc_field_redop<DataType<LEGMS_TYPE_FLOAT>::ValueType>>(
    ACC_FIELD_FLOAT_REDOP);
  Runtime::register_reduction_op<
    acc_field_redop<DataType<LEGMS_TYPE_DOUBLE>::ValueType>>(
    ACC_FIELD_DOUBLE_REDOP);
  Runtime::register_reduction_op<
    acc_field_redop<DataType<LEGMS_TYPE_COMPLEX>::ValueType>>(
    ACC_FIELD_COMPLEX_REDOP);
  Runtime::register_reduction_op<
    acc_field_redop<DataType<LEGMS_TYPE_DCOMPLEX>::ValueType>>(
    ACC_FIELD_DCOMPLEX_REDOP);
#endif // WITH_ACC_FIELD_REDOP_SERDEZ

#define REGISTER_POINT_ADD_REDOP(DIM)                   \
  Runtime::register_reduction_op<point_add_redop<DIM>>(POINT_ADD_REDOP(DIM));
  LEGMS_FOREACH_N(REGISTER_POINT_ADD_REDOP);
#undef REGISTER_POINT_ADD_REDOP
}

FieldID
legms::add_field(
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
    LEGMS_FOREACH_DATATYPE(ALLOC_FLD);
  default:
    assert(false);
    break;
  }
#undef ALLOC_FLD

  return result;
}

ProjectedIndexPartitionTask::ProjectedIndexPartitionTask(
  IndexSpace launch_space,
  LogicalPartition lp,
  LogicalRegion lr,
  args* global_arg)
  : IndexTaskLauncher(
    TASK_ID,
    launch_space,
    TaskArgument(
      global_arg,
      sizeof(ProjectedIndexPartitionTask::args)
      + global_arg->prjdim * sizeof(global_arg->dmap[0])),
    ArgumentMap()){

  add_region_requirement(
    RegionRequirement(lp, 0, WRITE_ONLY, EXCLUSIVE, lr));
  add_field(0, IMAGE_RANGES_FID);
}

void
ProjectedIndexPartitionTask::dispatch(Context ctx, Runtime* runtime) {
  runtime->execute_index_space(ctx, *this);
}

template <int IPDIM, int PRJDIM>
static void
pipt_impl(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context,
  Runtime *runtime) {

  const ProjectedIndexPartitionTask::args* targs =
    static_cast<const ProjectedIndexPartitionTask::args*>(task->args);
  Rect<PRJDIM> bounds = targs->bounds;

  const FieldAccessor<
    WRITE_ONLY,
    Rect<PRJDIM>,
    IPDIM,
    coord_t,
    Realm::AffineAccessor<Rect<PRJDIM>, IPDIM, coord_t>,
    false>
    image_ranges(regions[0], ProjectedIndexPartitionTask::IMAGE_RANGES_FID);

  DomainT<IPDIM> domain =
    runtime->get_index_space_domain(
      task->regions[0].region.get_index_space());
  for (PointInDomainIterator<IPDIM> pid(domain); pid(); pid++) {
    Rect<PRJDIM> r;
    for (size_t i = 0; i < PRJDIM; ++i)
      if (0 <= targs->dmap[i]) {
        r.lo[i] = r.hi[i] = pid[targs->dmap[i]];
      } else {
        r.lo[i] = bounds.lo[i];
        r.hi[i] = bounds.hi[i];
      }
    image_ranges[*pid] = r;
  }
}

void
ProjectedIndexPartitionTask::base_impl(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *rt) {

  const ProjectedIndexPartitionTask::args* targs =
    static_cast<const ProjectedIndexPartitionTask::args*>(task->args);
  IndexSpace is = task->regions[0].region.get_index_space();
  switch (is.get_dim() * LEGION_MAX_DIM + targs->prjdim) {
#define PIPT(I, P)                              \
    case (I * LEGION_MAX_DIM + P):              \
      pipt_impl<I, P>(task, regions, ctx, rt);  \
      break;
    LEGMS_FOREACH_NN(PIPT);
#undef PIPT
  default:
    assert(false);
    break;
  }
}

void
ProjectedIndexPartitionTask::preregister_task() {
  TASK_ID = Runtime::generate_static_task_id();
  TaskVariantRegistrar registrar(TASK_ID, TASK_NAME, false);
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  registrar.set_leaf();
  //registrar.set_idempotent();
  //registrar.set_replicable();
  Runtime::preregister_task_variant<base_impl>(registrar, TASK_NAME);
}

TaskID ProjectedIndexPartitionTask::TASK_ID;

IndexPartition
legms::projected_index_partition(
  Context ctx,
  Runtime* rt,
  IndexPartition ip,
  IndexSpace prj_is,
  const std::vector<int>& dmap) {

  if (prj_is == IndexSpace::NO_SPACE)
    return IndexPartition::NO_PART;

  switch (ip.get_dim() * LEGION_MAX_DIM + prj_is.get_dim()) {
#define PIP(I, P)                               \
    case (I * LEGION_MAX_DIM + P):              \
      return                                    \
        projected_index_partition(              \
          ctx,                                  \
          rt,                                   \
          IndexPartitionT<I>(ip),               \
          IndexSpaceT<P>(prj_is),               \
          dmap);                                \
      break;
    LEGMS_FOREACH_NN(PIP);
#undef PIP
  default:
    assert(false);
    return IndexPartition::NO_PART;
  }
}

LayoutConstraintRegistrar&
legms::add_row_major_order_constraint(
  LayoutConstraintRegistrar& lc,
  unsigned rank) {

  std::vector<DimensionKind> dims(rank);
  std::generate(
    dims.rbegin(),
    dims.rend(),
    [n = 0]() mutable {
      return static_cast<legion_dimension_kind_t>(n++);
    });
  return lc.add_constraint(OrderingConstraint(dims, true));
}

void
legms::preregister_all() {
  OpsManager::preregister_ops();
#ifdef LEGMS_USE_HDF5
  H5DatatypeManager::preregister_datatypes();
#endif

#define REG_AXES(T) \
  AxesRegistrar::register_axes<typename MSTable<MS_##T>::Axes>();

  LEGMS_FOREACH_MSTABLE(REG_AXES);
#undef REG_AXES

  TreeIndexSpaceTask::preregister_task();
  Table::preregister_tasks();
  ProjectedIndexPartitionTask::preregister_task();
#ifdef LEGMS_USE_CASACORE
  TableReadTask::preregister_task();
#endif
}

void
legms::register_tasks(Context context, Runtime* runtime) {
  Table::register_tasks(context, runtime);
}

std::unordered_map<std::string, legms::AxesRegistrar::A>
legms::AxesRegistrar::axes_;

#ifdef LEGMS_USE_HDF5
void
legms::AxesRegistrar::register_axes(
  const std::string uid,
  const std::vector<std::string> names,
  hid_t hid) {
  A a{uid, names, hid};
  axes_[uid] = a;
}
#else // !LEGMS_USE_HDF5
void
legms::AxesRegistrar::register_axes(
  const std::string uid,
  const std::vector<std::string> names) {
  A a{uid, names};
  axes_[uid] = a;
}
#endif

std::optional<legms::AxesRegistrar::A>
legms::AxesRegistrar::axes(const std::string& uid) {
  return (axes_.count(uid) > 0) ? axes_[uid] : std::optional<A>();
}

#ifdef LEGMS_USE_HDF5
std::optional<std::string>
legms::AxesRegistrar::match_axes_datatype(hid_t hid) {
  auto ad =
    std::find_if(
      axes_.begin(),
      axes_.end(),
      [&hid](auto& ua) {
        return H5Tequal(std::get<1>(ua).h5_datatype, hid) > 0;
      });
  return
    (ad != axes_.end()) ? std::get<0>(*ad) : std::optional<std::string>();
}
#endif // LEGMS_USE_HDF5

bool
legms::AxesRegistrar::in_range(
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

#ifdef LEGMS_USE_HDF5
hid_t
legms::H5DatatypeManager::datatypes_[DATATYPE_H5T + 1];

void
legms::H5DatatypeManager::preregister_datatypes() {
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
    H5Tset_size(dt, LEGMS_MAX_STRING_SIZE);
    datatypes_[STRING_H5T] = dt;
  }
  {
    hid_t dt = H5Tenum_create(H5T_NATIVE_UINT);

#define DTINSERT(T) do {                          \
      unsigned val = LEGMS_TYPE_##T;              \
      herr_t err = H5Tenum_insert(dt, #T, &val);  \
      assert(err >= 0);                           \
    } while(0);

    LEGMS_FOREACH_BARE_DATATYPE(DTINSERT);

    datatypes_[DATATYPE_H5T] = dt;
  }
}

herr_t
legms::H5DatatypeManager::commit_derived(
  hid_t loc_id,
  hid_t lcpl_id,
  hid_t tcpl_id,
  hid_t tapl_id) {

  herr_t result =
    H5Tcommit(
      loc_id,
      "legms::complex",
      datatypes_[COMPLEX_H5T],
      lcpl_id,
      tcpl_id,
      tapl_id);
  if (result < 0)
    return result;

  result =
    H5Tcommit(
      loc_id,
      "legms::dcomplex",
      datatypes_[DCOMPLEX_H5T],
      lcpl_id,
      tcpl_id,
      tapl_id);
  if (result < 0)
    return result;

  result =
    H5Tcommit(
      loc_id,
      "legms::string",
      datatypes_[STRING_H5T],
      lcpl_id,
      tcpl_id,
      tapl_id);
  if (result < 0)
    return result;

  result =
    H5Tcommit(
      loc_id,
      "legms::TypeTag",
      datatypes_[DATATYPE_H5T],
      lcpl_id,
      tcpl_id,
      tapl_id);
  return result;
}

hid_t
legms::H5DatatypeManager::create(
  const LEGMS_FS::path& path,
  unsigned flags,
  hid_t fcpl_t,
  hid_t fapl_t) {

  hid_t result = H5Fcreate(path.c_str(), flags, fcpl_t, fapl_t);
  if (result >= 0) {
    herr_t rc = commit_derived(result);
    assert(rc >= 0);
  }
  return result;
}
#endif // LEGMS_USE_HDF5

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
