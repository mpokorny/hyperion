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

#include <cstring>

using namespace hyperion::synthesis;
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

#endif // !HAVE_CXX17

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
hyperion::synthesis::CFTableBase::InitIndexColumnTaskArgs::serialized_size()
  const {

  return sizeof(hyperion::Table::Desc)
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
hyperion::synthesis::CFTableBase::InitIndexColumnTaskArgs::serialize(
  void* buff) const {

  char* b = reinterpret_cast<char*>(buff);
  *reinterpret_cast<hyperion::Table::Desc*>(b) = desc;
  b += sizeof(hyperion::Table::Desc);
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
hyperion::synthesis::CFTableBase::InitIndexColumnTaskArgs::deserialize(
  const void* buff) {

  const char* b = reinterpret_cast<const char*>(buff);
  desc = *reinterpret_cast<const hyperion::Table::Desc*>(b);
  b += sizeof(hyperion::Table::Desc);
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
hyperion::Axes<hyperion::synthesis::cf_table_axes_t>::names{
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
       a <= static_cast<unsigned char>(
         hyperion::Axes<cf_table_axes_t>::num_axes - 1);
       ++a) {
    [[maybe_unused]] herr_t err =
      H5Tenum_insert(
        result,
        hyperion::Axes<cf_table_axes_t>::names[a].c_str(),
        &a);
    assert(err >= 0);
  }
  return result;
}

const hid_t
hyperion::Axes<hyperion::synthesis::cf_table_axes_t>::h5_datatype =
  h5_axes_dt();
#endif

Legion::TaskID hyperion::synthesis::CFTableBase::init_index_column_task_id;

template <typename T>
static void
init_column(
  const std::vector<T>& vals,
  const hyperion::PhysicalColumnTD<
    hyperion::ValueType<T>::DataType, 1, 1, Legion::AffineAccessor>& col) {

  auto acc = col.template accessor<WRITE_ONLY>();
  for (PointInRectIterator<1> pir(col.rect()); pir(); pir++)
    acc[*pir] = vals[pir[0]];
}

void
hyperion::synthesis::CFTableBase::init_index_column_task(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  const InitIndexColumnTaskArgs& args =
    *reinterpret_cast<const InitIndexColumnTaskArgs*>(task->args);

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
CFTableBase::preregister_all() {

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
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
