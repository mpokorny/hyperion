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
#include <hyperion/synthesis/CFTable.h>
#include <hyperion/synthesis/CFPhysicalTable.h>
#include <hyperion/synthesis/FFT.h>
#include <hyperion/synthesis/GridCoordinateTable.h>
#include <hyperion/synthesis/PSTermTable.h>
#include <hyperion/synthesis/WTermTable.h>
#include <hyperion/synthesis/ATermZernikeModel.h>
#include <hyperion/synthesis/ATermIlluminationFunction.h>
#include <hyperion/synthesis/ATermTable.h>

#include <cstring>

#ifdef HYPERION_USE_OPENMP
# include <omp.h>
#endif

using namespace hyperion::synthesis;
using namespace hyperion;
using namespace Legion;

#if !HAVE_CXX17
constexpr const char*
hyperion::Axes<hyperion::synthesis::cf_table_axes_t>::uid;
constexpr const unsigned
hyperion::Axes<hyperion::synthesis::cf_table_axes_t>::num_axes;

constexpr const char*
hyperion::synthesis::cf_table_axis<CF_PS_SCALE>::name;
constexpr const char*
hyperion::synthesis::cf_table_axis<CF_BASELINE_CLASS>::name;
constexpr const char*
hyperion::synthesis::cf_table_axis<CF_FREQUENCY>::name;
constexpr const char*
hyperion::synthesis::cf_table_axis<CF_W>::name;
constexpr const char*
hyperion::synthesis::cf_table_axis<CF_PARALLACTIC_ANGLE>::name;
constexpr const char*
hyperion::synthesis::cf_table_axis<CF_STOKES_IN>::name;
constexpr const char*
hyperion::synthesis::cf_table_axis<CF_STOKES_OUT>::name;
constexpr const char*
hyperion::synthesis::cf_table_axis<CF_STOKES>::name;
constexpr const char*
hyperion::synthesis::cf_table_axis<CF_X>::name;
constexpr const char*
hyperion::synthesis::cf_table_axis<CF_Y>::name;

constexpr const Legion::FieldID CFTableBase::INDEX_VALUE_FID;
constexpr const Legion::FieldID CFTableBase::CF_VALUE_FID;
constexpr const char* CFTableBase::CF_VALUE_COLUMN_NAME;
constexpr const Legion::FieldID CFTableBase::CF_WEIGHT_FID;
constexpr const char* CFTableBase::CF_WEIGHT_COLUMN_NAME;

const constexpr char* CFTableBase::show_values_task_name;
const constexpr char* CFTableBase::init_index_column_task_name;

#endif // !HAVE_CXX17

Legion::TaskID CFTableBase::init_index_column_task_id;
Legion::TaskID CFTableBase::show_values_task_id;

const char*
hyperion::synthesis::cf_table_axis_name(cf_table_axes_t ax) {
  switch (ax) {
  case CF_PS_SCALE:
    return cf_table_axis<CF_PS_SCALE>::name;
  case CF_BASELINE_CLASS:
    return cf_table_axis<CF_BASELINE_CLASS>::name;
  case CF_FREQUENCY:
    return cf_table_axis<CF_FREQUENCY>::name;
  case CF_W:
    return cf_table_axis<CF_W>::name;
  case CF_PARALLACTIC_ANGLE:
    return cf_table_axis<CF_PARALLACTIC_ANGLE>::name;
  case CF_STOKES_OUT:
    return cf_table_axis<CF_STOKES_OUT>::name;
  case CF_STOKES_IN:
    return cf_table_axis<CF_STOKES_IN>::name;
  case CF_STOKES:
    return cf_table_axis<CF_STOKES>::name;
  case CF_X:
    return cf_table_axis<CF_X>::name;
  case CF_Y:
    return cf_table_axis<CF_Y>::name;
  case CF_ORDER0:
    return cf_table_axis<CF_ORDER0>::name;
  case CF_ORDER1:
    return cf_table_axis<CF_ORDER1>::name;
  }
  return nullptr;
}

template <typename T>
size_t
vector_serialized_size(const std::vector<T>& v) {
  return sizeof(size_t) + v.size() * sizeof(T);
}

template <typename T>
size_t
vector_serialize(const std::vector<T>& v, void* buff) {
  char* b = reinterpret_cast<char*>(buff);
  *reinterpret_cast<size_t*>(b) = v.size();
  b += sizeof(size_t);
  size_t sz = v.size() * sizeof(T);
  std::memcpy(reinterpret_cast<T*>(b), v.data(), sz);
  b += sz;
  return b - reinterpret_cast<char*>(buff);
}

template <typename T>
size_t
vector_deserialize(const void* buff, std::vector<T>& v) {
  const char* b = reinterpret_cast<const char*>(buff);
  size_t n = *reinterpret_cast<const size_t*>(b);
  b += sizeof(size_t);
  v.resize(n);
  size_t sz = n * sizeof(T);
  std::memcpy(v.data(), reinterpret_cast<const T*>(b), sz);
  b += sz;
  return b - reinterpret_cast<const char*>(buff);
}

size_t
CFTableBase::InitIndexColumnTaskArgs::serialized_size()
  const {

  return sizeof(Table::Desc)
    + vector_serialized_size(ps_scales)
    + vector_serialized_size(baseline_classes)
    + vector_serialized_size(frequencies)
    + vector_serialized_size(w_values)
    + vector_serialized_size(parallactic_angles)
    + vector_serialized_size(stokes_out_values)
    + vector_serialized_size(stokes_in_values)
    + vector_serialized_size(stokes_values);
}

size_t
CFTableBase::InitIndexColumnTaskArgs::serialize(
  void* buff) const {

  char* b = reinterpret_cast<char*>(buff);
  *reinterpret_cast<Table::Desc*>(b) = desc;
  b += sizeof(Table::Desc);
  b += vector_serialize(ps_scales, b);
  b += vector_serialize(baseline_classes, b);
  b += vector_serialize(frequencies, b);
  b += vector_serialize(w_values, b);
  b += vector_serialize(parallactic_angles, b);
  b += vector_serialize(stokes_out_values, b);
  b += vector_serialize(stokes_in_values, b);
  b += vector_serialize(stokes_values, b);
  return b - reinterpret_cast<char*>(buff);
}

size_t
CFTableBase::InitIndexColumnTaskArgs::deserialize(
  const void* buff) {

  const char* b = reinterpret_cast<const char*>(buff);
  desc = *reinterpret_cast<const Table::Desc*>(b);
  b += sizeof(Table::Desc);
  b += vector_deserialize(b, ps_scales);
  b += vector_deserialize(b, baseline_classes);
  b += vector_deserialize(b, frequencies);
  b += vector_deserialize(b, w_values);
  b += vector_deserialize(b, parallactic_angles);
  b += vector_deserialize(b, stokes_out_values);
  b += vector_deserialize(b, stokes_in_values);
  b += vector_deserialize(b, stokes_values);
  return b - reinterpret_cast<const char*>(buff);
}

const std::vector<std::string>
Axes<cf_table_axes_t>::names{
  cf_table_axis<CF_PS_SCALE>::name,
    cf_table_axis<CF_BASELINE_CLASS>::name,
    cf_table_axis<CF_FREQUENCY>::name,
    cf_table_axis<CF_W>::name,
    cf_table_axis<CF_PARALLACTIC_ANGLE>::name,
    cf_table_axis<CF_STOKES_OUT>::name,
    cf_table_axis<CF_STOKES_IN>::name,
    cf_table_axis<CF_STOKES>::name,
    cf_table_axis<CF_X>::name,
    cf_table_axis<CF_Y>::name
    };

#ifdef HYPERION_USE_HDF5
static hid_t
h5_axes_dt()  {
  hid_t result = H5Tenum_create(H5T_NATIVE_UCHAR);
  for (unsigned char a = 0;
       a <= static_cast<unsigned char>(Axes<cf_table_axes_t>::num_axes - 1);
       ++a) {
    [[maybe_unused]] herr_t err =
      H5Tenum_insert(result, Axes<cf_table_axes_t>::names[a].c_str(), &a);
    assert(err >= 0);
  }
  return result;
}

const hid_t
Axes<cf_table_axes_t>::h5_datatype =
  h5_axes_dt();
#endif

template <typename T>
static void
init_column(
  const std::vector<T>& vals,
  const PhysicalColumnTD<
    ValueType<T>::DataType, 1, 1, Legion::AffineAccessor>& col) {

  auto acc = col.template accessor<WRITE_ONLY>();
  for (PointInRectIterator<1> pir(col.rect()); pir(); pir++)
    acc[*pir] = vals[pir[0]];
}

void
CFTableBase::init_index_column_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  InitIndexColumnTaskArgs args;
  args.deserialize(task->args);

  auto ptcr =
    PhysicalTable::create(
      rt,
      args.desc,
      task->regions.begin(),
      task->regions.end(),
      regions.begin(),
      regions.end()).value();
#if HAVE_CXX17
  auto& [pt, rit, pit] = ptcr;
#else // !HAVE_CXX17
  auto& pt = std::get<0>(ptcr);
  auto& rit = std::get<1>(ptcr);
  auto& pit = std::get<2>(ptcr);
#endif // HAVE_CXX17
  assert(rit == task->regions.end());
  assert(pit == regions.end());

  auto cols = pt.columns();
  if (args.ps_scales.size() > 0)
    init_column<typename cf_table_axis<CF_PS_SCALE>::type>(
      args.ps_scales,
      *cols.at(cf_table_axis<CF_PS_SCALE>::name));
  if (args.baseline_classes.size() > 0)
    init_column<typename cf_table_axis<CF_BASELINE_CLASS>::type>(
      args.baseline_classes,
      *cols.at(cf_table_axis<CF_BASELINE_CLASS>::name));
  if (args.frequencies.size() > 0)
    init_column<typename cf_table_axis<CF_FREQUENCY>::type>(
      args.frequencies,
      *cols.at(cf_table_axis<CF_FREQUENCY>::name));
  if (args.w_values.size() > 0)
    init_column<typename cf_table_axis<CF_W>::type>(
      args.w_values,
      *cols.at(cf_table_axis<CF_W>::name));
  if (args.parallactic_angles.size() > 0)
    init_column<typename cf_table_axis<CF_PARALLACTIC_ANGLE>::type>(
      args.parallactic_angles,
      *cols.at(cf_table_axis<CF_PARALLACTIC_ANGLE>::name));
  if (args.stokes_out_values.size() > 0)
    init_column<typename cf_table_axis<CF_STOKES_OUT>::type>(
      args.stokes_out_values,
      *cols.at(cf_table_axis<CF_STOKES_OUT>::name));
  if (args.stokes_in_values.size() > 0)
    init_column<typename cf_table_axis<CF_STOKES_IN>::type>(
      args.stokes_in_values,
      *cols.at(cf_table_axis<CF_STOKES_IN>::name));
  if (args.stokes_values.size() > 0)
    init_column<typename cf_table_axis<CF_STOKES>::type>(
      args.stokes_values,
      *cols.at(cf_table_axis<CF_STOKES>::name));
}

void
CFTableBase::apply_fft(
  Context ctx,
  Runtime* rt,
  int sign,
  bool rotate_in,
  bool rotate_out,
  unsigned flags,
  double seconds,
  const ColumnSpacePartition& partition) const {

  FFT::Args args;
  args.desc.rank = 2;
  args.desc.precision =
    ((typeid(cf_fp_t) == typeid(float))
     ? FFT::Precision::SINGLE
     : FFT::Precision::DOUBLE);
  args.desc.transform = FFT::Type::C2C;
  args.desc.sign = sign;
  args.rotate_in = rotate_in;
  args.rotate_out = rotate_out;
  args.seconds = seconds;
  args.flags = flags;

  auto cols = columns();
  auto part =
    cols.at(CF_VALUE_COLUMN_NAME)
    .narrow_partition(ctx, rt, partition, {CF_X, CF_Y})
    .value_or(ColumnSpacePartition());
  for (auto& nm: {CF_VALUE_COLUMN_NAME, CF_WEIGHT_COLUMN_NAME}) {
    auto col = cols.at(nm);
    args.fid = col.fid;
    if (!part.is_valid()) {
      TaskLauncher
        fft(FFT::in_place_task_id, TaskArgument(&args, sizeof(args)));
      RegionRequirement
        req(col.region, LEGION_READ_WRITE, EXCLUSIVE, col.region);
      req.add_field(col.fid);
      fft.add_region_requirement(req);
      rt->execute_task(ctx, fft);
    } else {
      auto lp = rt->get_logical_partition(ctx, col.region, part.column_ip);
      IndexTaskLauncher fft(
        FFT::in_place_task_id,
        rt->get_index_partition_color_space(ctx, part.column_ip),
        TaskArgument(&args, sizeof(args)),
        ArgumentMap());
      RegionRequirement req(lp, 0, LEGION_READ_WRITE, EXCLUSIVE, col.region);
      req.add_field(col.fid);
      fft.add_region_requirement(req);
      rt->execute_index_space(ctx, fft);
    }
  }
  if (part != partition)
    part.destroy(ctx, rt);
}

void
CFTableBase::show_cf_values(
  Context ctx,
  Runtime* rt,
  const std::string& title) const {

  auto reqs =
    requirements(
      ctx,
      rt,
      ColumnSpacePartition(),
      {},
      Column::default_requirements_mapped);
  ShowValuesTaskArgs args;
  args.tdesc = std::get<2>(reqs);
  args.title = title;
  TaskLauncher task(show_values_task_id, TaskArgument(&args, sizeof(args)));
  for (auto& r : std::get<0>(reqs))
    task.add_region_requirement(r);
  rt->execute_task(ctx, task);
}

template <cf_table_axes_t T>
static void
show_index_valueT(const PhysicalColumn& col, Legion::coord_t i) {
  auto acc =
    col.accessor<
      LEGION_READ_ONLY,
      typename cf_table_axis<T>::type,
      1,
      coord_t,
      AffineAccessor>();
  std::cout << acc[i];
}

void
CFTableBase::show_index_value(const PhysicalColumn& col, Legion::coord_t i) {
  switch (static_cast<cf_table_axes_t>(col.axes()[0])) {
  case CF_PS_SCALE:
    return show_index_valueT<CF_PS_SCALE>(col, i);
  case CF_BASELINE_CLASS:
    return show_index_valueT<CF_BASELINE_CLASS>(col, i);
  case CF_FREQUENCY:
    return show_index_valueT<CF_FREQUENCY>(col, i);
  case CF_W:
    return show_index_valueT<CF_W>(col, i);
  case CF_PARALLACTIC_ANGLE:
    return show_index_valueT<CF_PARALLACTIC_ANGLE>(col, i);
  case CF_STOKES_OUT:
    return show_index_valueT<CF_STOKES_OUT>(col, i);
  case CF_STOKES_IN:
    return show_index_valueT<CF_STOKES_IN>(col, i);
  case CF_STOKES:
    return show_index_valueT<CF_STOKES>(col, i);
  default:
    assert(false);
    break;
  }
}

template <unsigned N>
static void
show_values(const PhysicalTable& pt) {
  auto columns = pt.columns();
  std::vector<cf_table_axes_t> index_axes;
  for (auto& ia : pt.index_axes())
    index_axes.push_back(static_cast<cf_table_axes_t>(ia));
  std::vector<std::shared_ptr<PhysicalColumn>> index_columns;
  for (auto& ia : index_axes)
    index_columns.push_back(columns.at(cf_table_axis_name(ia)));
  auto value_col =
    PhysicalColumnTD<
      ValueType<CFTableBase::cf_value_t>::DataType,
      N,
      N + 2,
      AffineAccessor>(*columns.at(CFTableBase::CF_VALUE_COLUMN_NAME));
  auto value_rect = value_col.rect();
  auto grid_size = value_rect.hi[N] - value_rect.lo[N] + 1;
  auto values = value_col.template accessor<LEGION_READ_ONLY>();
  Point<N + 2> index;
  for (size_t i = 0; i < N; ++i)
    index[i] = -1;
  PointInRectIterator<N + 2> pir(value_rect, false);
  while (pir()) {
    for (size_t i = 0; i < N; ++i)
      index[i] = pir[i];
    std::cout << "*** " << cf_table_axis_name(index_axes[0])
              << ": ";
    CFTableBase::show_index_value(*index_columns[0], pir[0]);
    for (size_t i = 1; i < N; ++i) {
      std::cout << "; " << cf_table_axis_name(index_axes[i])
                << ": ";
      CFTableBase::show_index_value(*index_columns[i], pir[i]);
    }
    std::cout << std::endl;
    for (coord_t i = 0; i < grid_size; ++i) {
      for (coord_t j = 0; j < grid_size; ++j)
        std::cout << values[*pir++] << " ";
      std::cout << std::endl;
    }
  }
}

void
CFTableBase::show_values_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  const ShowValuesTaskArgs& args =
    *static_cast<const ShowValuesTaskArgs*>(task->args);

  std::cout << "++++ " << args.title << " ++++" << std::endl;
  auto pt =
    PhysicalTable::create_all_unsafe(
      rt,
      {args.tdesc},
      task->regions,
      regions)[0];
  switch (pt.index_rank()) {
  case 1:
    show_values<1>(pt);
    break;
  case 2:
    show_values<2>(pt);
    break;
  case 3:
    show_values<3>(pt);
    break;
  case 4:
    show_values<4>(pt);
    break;
  case 5:
    show_values<5>(pt);
    break;
  case 6:
    show_values<6>(pt);
    break;
  default:
    assert(false);
    break;
  }
}

void
CFTableBase::preregister_all() {

#ifdef HYPERION_USE_OPENMP
  {
    auto rc = fftwf_init_threads();
    assert(rc != 0);
    std::cout << "omp_get_max_threads() -> " << omp_get_max_threads()
              << std::endl;
    fftwf_plan_with_nthreads(omp_get_max_threads());
  }
#endif

  AxesRegistrar::register_axes<cf_table_axes_t>();
  {
    // init_index_column_task
    init_index_column_task_id = Runtime::generate_static_task_id();
    TaskVariantRegistrar registrar(
      init_index_column_task_id,
      init_index_column_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_idempotent();
    registrar.set_leaf();
    Runtime::preregister_task_variant<CFTableBase::init_index_column_task>(
      registrar,
      init_index_column_task_name);
  }
  {
    // show_values_task
    show_values_task_id = Runtime::generate_static_task_id();
    TaskVariantRegistrar registrar(show_values_task_id, show_values_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<CFTableBase::show_values_task>(
      registrar,
      show_values_task_name);
  }

  // TODO: move these into a synthesis initialization function
  FFT::preregister_tasks();
  GridCoordinateTable::preregister_tasks();
  PSTermTable::preregister_tasks();
  WTermTable::preregister_tasks();
  ATermZernikeModel::preregister_tasks();
  ATermIlluminationFunction::preregister_tasks();
  ATermTable::preregister_tasks();
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
