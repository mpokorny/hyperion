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
#include <hyperion/testing/TestSuiteDriver.h>
#include <hyperion/testing/TestRecorder.h>
#include <hyperion/testing/TestExpression.h>

#include <hyperion/hdf5.h>
#include <hyperion/x/Table.h>
#include <hyperion/x/Column.h>

#ifdef HYPERION_USE_CASACORE
# include <hyperion/MeasRef.h>
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <hdf5.h>
#include <numeric>
#include <set>
#include <stdlib.h>
#include <unistd.h>

using namespace hyperion;
using namespace hyperion::hdf5;
using namespace Legion;

enum {
  HDF5_TEST_SUITE
};

enum struct Table0Axes {
  ROW = 0,
  X,
  Y,
  ZP
};

enum {
  COL_X,
  COL_Y,
  COL_Z
};

template <>
struct hyperion::Axes<Table0Axes> {
  static const constexpr char* uid = "Table0Axes";
  static const std::vector<std::string> names;
  static const unsigned num_axes = 4;
#ifdef HYPERION_USE_HDF5
  static const hid_t h5_datatype;
#endif
};

const std::vector<std::string>
hyperion::Axes<Table0Axes>::names{"ROW", "X", "Y", "ZP"};

hid_t
h5_dt() {
  hid_t result = H5Tenum_create(H5T_NATIVE_UCHAR);
  Table0Axes a = Table0Axes::ROW;
  herr_t err = H5Tenum_insert(result, "ROW", &a);
  assert(err >= 0);
  a = Table0Axes::X;
  err = H5Tenum_insert(result, "X", &a);
  a = Table0Axes::Y;
  err = H5Tenum_insert(result, "Y", &a);
  a = Table0Axes::ZP;
  err = H5Tenum_insert(result, "ZP", &a);
  return result;
}

const hid_t
hyperion::Axes<Table0Axes>::h5_datatype = h5_dt();

std::ostream&
operator<<(std::ostream& stream, const Table0Axes& ax) {
  switch (ax) {
  case Table0Axes::ROW:
    stream << "Table0Axes::ROW";
    break;
  case Table0Axes::X:
    stream << "Table0Axes::X";
    break;
  case Table0Axes::Y:
    stream << "Table0Axes::Y";
    break;
  case Table0Axes::ZP:
    stream << "Table0Axes::ZP";
    break;
  }
  return stream;
}

#define TABLE0_NUM_X 4
#define TABLE0_NUM_Y 3
#define TABLE0_NUM_ROWS (TABLE0_NUM_X * TABLE0_NUM_Y)
unsigned table0_x[TABLE0_NUM_ROWS] {
                   0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
unsigned table0_y[TABLE0_NUM_ROWS] {
                   0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
unsigned table0_z[2 * TABLE0_NUM_ROWS];

LogicalRegion
copy_region(Context context, Runtime* runtime, const PhysicalRegion& region) {
  LogicalRegion lr = region.get_logical_region();
  LogicalRegion result =
    runtime->create_logical_region(
      context,
      lr.get_index_space(),
      lr.get_field_space());
  std::vector<FieldID> instance_fields;
  runtime
    ->get_field_space_fields(context, lr.get_field_space(), instance_fields);
  std::set<FieldID>
    privilege_fields(instance_fields.begin(), instance_fields.end());
  CopyLauncher launcher;
  launcher.add_copy_requirements(
    RegionRequirement(
      lr,
      privilege_fields,
      instance_fields,
      READ_ONLY,
      EXCLUSIVE,
      lr),
    RegionRequirement(
      result,
      privilege_fields,
      instance_fields,
      WRITE_ONLY,
      EXCLUSIVE,
      result));
  runtime->issue_copy_operation(context, launcher);
  return result;
}

PhysicalRegion
attach_table0_col(
  Context context,
  Runtime* runtime,
  const x::Column& col,
  unsigned *base) {

  const Memory local_sysmem =
    Machine::MemoryQuery(Machine::get_machine())
    .has_affinity_to(runtime->get_executing_processor(context))
    .only_kind(Memory::SYSTEM_MEM)
    .first();

  AttachLauncher task(EXTERNAL_INSTANCE, col.vreq.region, col.vreq.region);
  task.attach_array_soa(base, false, {col.fid}, local_sysmem);
  PhysicalRegion result = runtime->attach_external_resource(context, task);
  AcquireLauncher acq(col.vreq.region, col.vreq.region, result);
  acq.add_field(col.fid);
  runtime->issue_acquire(context, acq);
  return result;
}

#define TE(f) testing::TestEval([&](){ return f; }, #f)

struct other_index_tree_serdez {

  static const constexpr char* id = "other_index_tree_serdez";

  static size_t
  serialized_size(const IndexTreeL& tree) {
    return tree.serialized_size();
  }

  static size_t
  serialize(const IndexTreeL& tree, void *buffer) {
    return tree.serialize(static_cast<char*>(buffer));
  }

  static size_t
  deserialize(IndexTreeL& tree, const void* buffer) {
    tree = IndexTreeL::deserialize(static_cast<const char*>(buffer));
    return tree.serialized_size();
  }
};

void
test_index_tree_attribute(
  hid_t fid,
  const std::string& dataset_name,
  testing::TestRecorder<READ_WRITE>& recorder,
  const IndexTreeL& tree,
  const std::string& tree_name) {

  write_index_tree_to_attr<binary_index_tree_serdez>(
    tree,
    fid,
    dataset_name,
    tree_name);

  hid_t ds = H5Dopen(fid, dataset_name.c_str(), H5P_DEFAULT);
  assert(ds >= 0);
  auto tree_md = read_index_tree_attr_metadata(ds, tree_name.c_str());
  recorder.assert_true(
    std::string("IndexTree attribute ") + tree_name + " metadata exists",
    tree_md.has_value());
  recorder.expect_true(
    std::string("IndexTree attribute ") + tree_name
    + " metadata has expected serializer id",
    TE(std::string(tree_md.value())) == binary_index_tree_serdez::id);
  auto optTree =
    read_index_tree_from_attr<binary_index_tree_serdez>(ds, tree_name.c_str());
  recorder.assert_true(
    std::string("IndexTree attribute ") + tree_name + " value exists",
    optTree.has_value());
  recorder.expect_true(
    std::string("IndexTree attribute ") + tree_name + " has expected value",
    TE(optTree.value()) == tree);

  auto optTree_bad =
    read_index_tree_from_attr<other_index_tree_serdez>(ds, tree_name.c_str());
  recorder.expect_false(
    std::string("Failure to read IndexTree attribute ") + tree_name
    + " with incorrect deserializer",
    optTree_bad.has_value());
  herr_t err = H5Dclose(ds);
  assert(err >= 0);
}

void
tree_tests(testing::TestRecorder<READ_WRITE>& recorder) {
  std::string fname = "h5.XXXXXX";
  int fd = mkstemp(fname.data());
  assert(fd != -1);
  close(fd);
  try {
    hid_t fid =
      H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    assert(fid >= 0);
    hsize_t sz = 1000;
    hid_t dsp = H5Screate_simple(1, &sz, &sz);
    assert(dsp >= 0);
    std::string dataset_name = "Albert";
    hid_t dset =
      H5Dcreate(
        fid,
        dataset_name.c_str(),
        H5T_NATIVE_DOUBLE,
        dsp,
        H5P_DEFAULT,
        H5P_DEFAULT,
        H5P_DEFAULT);
    assert(dset >= 0);
    H5Dclose(dset);

    test_index_tree_attribute(
      fid,
      dataset_name,
      recorder,
      IndexTreeL(HYPERION_LARGE_TREE_MIN / 2 + 1),
      "small-tree");

    IndexTreeL tree1(4);
    while (tree1.serialized_size() < HYPERION_LARGE_TREE_MIN)
      tree1 = IndexTreeL({{0, 1, tree1}});
    test_index_tree_attribute(fid, dataset_name, recorder, tree1, "large-tree");

    H5Fclose(fid);
    unlink(fname.c_str());
  } catch (...) {
    unlink(fname.c_str());
    throw;
  }
}

template <legion_privilege_mode_t MODE, typename FT, int N>
using FA =
  FieldAccessor<
  MODE,
  FT,
  N,
  coord_t,
  AffineAccessor<FT, N, coord_t>,
  true>;

template <size_t N>
bool
verify_col(
  Context ctx,
  Runtime* rt,
  const unsigned* expected,
  const PhysicalRegion& region,
  FieldID fid,
  const std::array<size_t, N>& dims) {

  LogicalRegion lr = copy_region(ctx, rt, region);
  RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
  req.add_field(fid);
  PhysicalRegion pr = rt->map_region(ctx, req);

  bool result = true;
  const FA<READ_ONLY, unsigned, N> acc(pr, fid);
  PointInRectIterator<N> pid(region, false);
  std::array<size_t, N> pt;
  pt.fill(0);
  size_t off = 0;
  while (result && pid() && pt[0] < dims[0]) {
    result = acc[*pid] == expected[off];
    pid++;
    ++off;
    size_t i = N - 1;
    ++pt[i];
    while (i > 0 && pt[i] == dims[i]) {
      pt[i] = 0;
      --i;
      ++pt[i];
    }
  }
  result = result && !pid();

  rt->unmap_region(ctx, pr);
  rt->destroy_logical_region(ctx, lr);
  return result;
}

void
table_tests(
  Context ctx,
  Runtime* rt,
  bool save_output_file,
  testing::TestRecorder<READ_WRITE>& recorder) {

  for (size_t i = 0; i < TABLE0_NUM_ROWS; ++i) {
    table0_z[2 * i] = table0_x[i];
    table0_z[2 * i + 1] = table0_y[i];
  }

  const float ms_vn = -42.1f;
  hyperion::string ms_nm("test");
  std::string fname = "h5.XXXXXX";
  
  auto xy_is = rt->create_index_space(ctx, Rect<1>(0, TABLE0_NUM_ROWS - 1));
  auto xy_space =
    x::ColumnSpace::create(
      ctx,
      rt,
      std::vector<Table0Axes>{Table0Axes::ROW},
      xy_is,
      false);

  auto z_is =
    rt->create_index_space(
      ctx,
      Rect<2>(Point<2>(0, 0), Point<2>(TABLE0_NUM_ROWS - 1, 1)));
  auto z_space =
    x::ColumnSpace::create(
      ctx,
      rt,
      std::vector<Table0Axes>{Table0Axes::ROW, Table0Axes::ZP},
      z_is,
      false);

#ifdef HYPERION_USE_CASACORE
  casacore::MeasRef<casacore::MEpoch> utc(casacore::MEpoch::UTC);
  casacore::MeasRef<casacore::MDirection>
    direction(casacore::MDirection::J2000);
  auto columnX_direction = MeasRef::create(ctx, rt, direction);
  auto columnZ_epoch = MeasRef::create(ctx, rt, utc);
  std::vector<std::pair<std::string, x::TableField>> xy_fields{
    {"X",
     x::TableField(HYPERION_TYPE_UINT, COL_X,
                   columnX_direction, std::nullopt, Keywords())},
    {"Y",
     x::TableField(HYPERION_TYPE_UINT, COL_Y,
                   MeasRef(), std::nullopt, Keywords())}
  };
  std::vector<std::pair<std::string, x::TableField>> z_fields{
    {"Z",
     x::TableField(HYPERION_TYPE_UINT, COL_Z,
                   columnZ_epoch, std::nullopt, Keywords())}
  };
#else
  std::vector<std::pair<std::string, x::TableField>> xy_fields{
    {"X", x::TableField(HYPERION_TYPE_UINT, COL_X, Keywords())},
    {"Y", x::TableField(HYPERION_TYPE_UINT, COL_Y, Keywords())}
  };
  std::vector<std::pair<std::string, x::TableField>> z_fields{
    {"Z", x::TableField(HYPERION_TYPE_UINT, COL_Z, Keywords())}
  };
#endif

  auto tb0 =
    x::Table::create(ctx, rt, {{xy_space, xy_fields}, {z_space, z_fields}});
  auto cols0 =
    x::Table::column_map(
      tb0.columns(ctx, rt).get<x::Table::columns_result_t>());

  int fd = mkstemp(fname.data());
  assert(fd != -1);
  if (save_output_file)
    std::cout << "test file name: " << fname << std::endl;
  close(fd);
  hid_t fid, root_loc;
  recorder.assert_no_throw(
    "Write to HDF5 file",
    testing::TestEval(
      [&]() {
        fid = H5DatatypeManager::create(fname, H5F_ACC_TRUNC);
        assert(fid >= 0);
        hid_t t1_loc =
          H5Gcreate(fid, "/table1", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        assert(t1_loc >= 0);
        write_xtable(ctx, rt, t1_loc, tb0);
        H5Gclose(t1_loc);
        root_loc = H5Gopen(fid, "/", H5P_DEFAULT);
        assert(root_loc >= 0);
        return true;
      }));

  {
    std::unordered_map<std::string, unsigned*> col0_arrays{
      {"X", table0_x},
      {"Y", table0_y},
      {"Z", table0_z}
    };
    std::unordered_map<std::string, PhysicalRegion> col0_prs;
    for (auto& c : {"X", "Y", "Z"}) {
      std::string cstr(c);
      col0_prs[cstr] =
        attach_table0_col(ctx, rt, cols0.at(cstr), col0_arrays.at(cstr));
    }

    auto [tb1, tb1_paths] = init_xtable(ctx, rt, root_loc, "table1");

    // read back metadata
    recorder.assert_false(
      "Table initialized from HDF5 is not empty",
      TE(tb1.is_empty()));
    auto cols1 =
      x::Table::column_map(
        tb1.columns(ctx, rt).get<x::Table::columns_result_t>());
    {
      recorder.assert_true(
        "Column X logically recreated",
        TE(cols1.count("X") > 0));
      auto cx = cols1.at("X");
      recorder.expect_true(
        "Column X has expected data type",
        TE(cx.dt == HYPERION_TYPE_UINT));
      recorder.expect_true(
        "Column X has expected FieldID",
        TE(cx.fid == COL_X));
      recorder.expect_true(
        "Column X has no measure reference column",
        TE(!cx.rc.has_value()));
      recorder.expect_true(
        "Column X has expected axes",
        TE(cx.csp.axes(ctx, rt)) ==
        map_to_int(std::vector<Table0Axes>{Table0Axes::ROW}));
      recorder.expect_true(
        "Column X has expected indexes",
        TE(index_space_as_tree(rt, cx.csp.column_is))
        == IndexTreeL(TABLE0_NUM_ROWS));
#ifdef HYPERION_USE_CASACORE
      recorder.expect_true(
        "Column X has expected measure",
        TE(cx.mr.equiv(ctx, rt, columnX_direction)));
#endif
    }
    {
      recorder.assert_true(
        "Column Y logically recreated",
        TE(cols1.count("Y") > 0));
      auto cy = cols1.at("Y");
      recorder.expect_true(
        "Column Y has expected data type",
        TE(cy.dt == HYPERION_TYPE_UINT));
      recorder.expect_true(
        "Column Y has expected FieldID",
        TE(cy.fid == COL_Y));
      recorder.expect_true(
        "Column Y has no measure reference column",
        TE(!cy.rc.has_value()));
      recorder.expect_true(
        "Column Y has expected axes",
        TE(cy.csp.axes(ctx, rt)) ==
        map_to_int(std::vector<Table0Axes>{Table0Axes::ROW}));
      recorder.expect_true(
        "Column Y has expected indexes",
        TE(index_space_as_tree(rt, cy.csp.column_is))
        == IndexTreeL(TABLE0_NUM_ROWS));
#ifdef HYPERION_USE_CASACORE
      recorder.expect_true(
        "Column Y has expected measure (none)",
        TE(cy.mr.is_empty()));
#endif
    }
    {
      recorder.assert_true(
        "Column Z logically recreated",
        TE(cols1.count("Z") > 0));
      auto cz = cols1.at("Z");
      recorder.expect_true(
        "Column Z has expected data type",
        TE(cz.dt == HYPERION_TYPE_UINT));
      recorder.expect_true(
        "Column Z has expected FieldID",
        TE(cz.fid == COL_Z));
      recorder.expect_true(
        "Column Z has no measure reference column",
        TE(!cz.rc.has_value()));
      recorder.expect_true(
        "Column Z has expected axes",
        TE(cz.csp.axes(ctx, rt)) ==
        map_to_int(std::vector<Table0Axes>{Table0Axes::ROW, Table0Axes::ZP}));
      recorder.expect_true(
        "Column Z has expected indexes",
        TE(index_space_as_tree(rt, cz.csp.column_is))
        == IndexTreeL({{TABLE0_NUM_ROWS, IndexTreeL(2)}}));
#ifdef HYPERION_USE_CASACORE
      recorder.expect_true(
        "Column Z has expected measure",
        TE(cz.mr.equiv(ctx, rt, columnZ_epoch)));
#endif
    }
    {
      auto pr =
        attach_xtable_columns(
          ctx,
          rt,
          fname,
          "/",
          tb1,
          {"X", "Y", "Z"},
          tb1_paths,
          true,
          true);
      recorder.expect_false(
        "Cannot attach columns with different ColumnSpaces at once",
        TE(pr.has_value()));
    }
    {
      std::unordered_map<std::string, std::string> some_paths(
        tb1_paths.begin(), tb1_paths.end());
      some_paths.erase("X");
      auto pr =
        attach_xtable_columns(
          ctx,
          rt,
          fname,
          "/",
          tb1,
          {"X", "Y"},
          some_paths,
          true,
          true);
      recorder.expect_false(
        "Cannot attach columns without associated paths",
        TE(pr.has_value()));
    }
    auto tb1_xy_pr =
      attach_xtable_columns(
        ctx,
        rt,
        fname,
        "/",
        tb1,
        {"X", "Y"},
        tb1_paths,
        false,
        true);
    recorder.expect_true(
      "Attach X and Y columns in HDF5 file",
      TE(tb1_xy_pr.has_value()));
    auto tb1_z_pr =
      attach_xtable_columns(
        ctx,
        rt,
        fname,
        "/",
        tb1,
        {"Z"},
        tb1_paths,
        false,
        true);
    recorder.expect_true(
      "Attach Z column in HDF5 file",
      TE(tb1_z_pr.has_value()));

    // copy values from tb0 to tb1
    {
      CopyLauncher copy;
      {
        auto src_lr = col0_prs["X"].get_logical_region();
        RegionRequirement srq(src_lr, READ_ONLY, EXCLUSIVE, src_lr);
        srq.add_field(cols0.at("X").fid);
        srq.add_field(cols0.at("Y").fid);
        auto dst_lr = tb1_xy_pr.value().get_logical_region();
        RegionRequirement drq(dst_lr, WRITE_ONLY, EXCLUSIVE, dst_lr);
        drq.add_field(cols0.at("X").fid);
        drq.add_field(cols0.at("Y").fid);
        copy.add_copy_requirements(srq, drq);
      }
      {
        auto src_lr = col0_prs["Z"].get_logical_region();
        RegionRequirement srq(src_lr, READ_ONLY, EXCLUSIVE, src_lr);
        srq.add_field(cols0.at("Z").fid);
        auto dst_lr = tb1_z_pr.value().get_logical_region();
        RegionRequirement drq(dst_lr, WRITE_ONLY, EXCLUSIVE, dst_lr);
        drq.add_field(cols0.at("Z").fid);
        copy.add_copy_requirements(srq, drq);
      }
      rt->issue_copy_operation(ctx, copy);
    }

    rt->detach_external_resource(ctx, tb1_xy_pr.value());
    rt->detach_external_resource(ctx, tb1_z_pr.value());
    tb1.destroy(ctx, rt);
    H5Gclose(root_loc);
    H5Fclose(fid);

    for (auto& c: {"X", "Y", "Z"})
      rt->detach_external_resource(ctx, col0_prs[c]);
  }

  {
    // reattach to HDF5 and verify column values
    fid = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    assert(fid >= 0);
    root_loc = H5Gopen(fid, "/", H5P_DEFAULT);
    auto [tb1, tb1_paths] = init_xtable(ctx, rt, root_loc, "table1");
    auto cols1 =
      x::Table::column_map(
        tb1.columns(ctx, rt).get<x::Table::columns_result_t>());
    auto tb1_xy_pr =
      attach_xtable_columns(
        ctx,
        rt,
        fname,
        "/",
        tb1,
        {"X", "Y"},
        tb1_paths,
        true,
        true);
    recorder.expect_true(
      "Re-attached X and Y columns in HDF5 file",
      TE(tb1_xy_pr.has_value()));
    auto tb1_z_pr =
      attach_xtable_columns(
        ctx,
        rt,
        fname,
        "/",
        tb1,
        {"Z"},
        tb1_paths,
        true,
        true);
    recorder.expect_true(
      "Re-attached Z column in HDF5 file",
      TE(tb1_z_pr.has_value()));
    recorder.expect_true(
      "Column X values read from HDF5 as expected",
      TE(verify_col<1>(
           ctx,
           rt,
           table0_x,
           tb1_xy_pr.value(),
           cols1.at("X").fid,
           {TABLE0_NUM_ROWS})));
    recorder.expect_true(
      "Column Y values read from HDF5 as expected",
      TE(verify_col<1>(
           ctx,
           rt,
           table0_y,
           tb1_xy_pr.value(),
           cols1.at("Y").fid,
           {TABLE0_NUM_ROWS})));
    recorder.expect_true(
      "Column Z values read from HDF5 as expected",
      TE(verify_col<2>(
           ctx,
           rt,
           table0_z,
           tb1_z_pr.value(),
           cols1.at("Z").fid,
           {TABLE0_NUM_ROWS, 2})));

    rt->detach_external_resource(ctx, tb1_xy_pr.value());
    rt->detach_external_resource(ctx, tb1_z_pr.value());
    tb1.destroy(ctx, rt);
    H5Gclose(root_loc);
    H5Fclose(fid);
  }
  tb0.destroy(ctx, rt);
  if (!save_output_file)
    CXX_FILESYSTEM_NAMESPACE::remove(fname);
}

void
hdf5_test_suite(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *runtime) {

  const InputArgs& args = Runtime::get_input_args();
  bool save_output_file = false;
  for (int i = 1; i < args.argc; ++i)
    save_output_file = std::string(args.argv[i]) == "--save-output";

  register_tasks(ctx, runtime);

  testing::TestLog<READ_WRITE> log(
    task->regions[0].region,
    regions[0],
    task->regions[1].region,
    regions[1],
    ctx,
    runtime);
  testing::TestRecorder<READ_WRITE> recorder(log);

  tree_tests(recorder);
  table_tests(ctx, runtime, save_output_file, recorder);
}

int
main(int argc, char* argv[]) {

  AxesRegistrar::register_axes<Table0Axes>();

  testing::TestSuiteDriver driver =
    testing::TestSuiteDriver::make<hdf5_test_suite>(
      HDF5_TEST_SUITE,
      "hdf5_test_suite");

  return driver.start(argc, argv);

}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
