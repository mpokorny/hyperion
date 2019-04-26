#include "legms_hdf5.h"

#include "TestSuiteDriver.h"
#include "TestRecorder.h"
#include "TestExpression.h"

#include <cassert>
#include <hdf5.h>
#include <numeric>
#include <stdlib.h>
#include <unistd.h>

using namespace legms;
using namespace legms::hdf5;
using namespace Legion;

enum {
  HDF5_TEST_SUITE,
};

enum struct Table0Axes {
  index = -1,
  ROW = 0,
  X,
  Y,
  ZP
};

std::ostream&
operator<<(std::ostream& stream, const Table0Axes& ax) {
  switch (ax) {
  case Table0Axes::index:
    stream << "Table0Axes::index";
    break;
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

ColumnT<Table0Axes>::Generator
table0_col(const std::string& name) {
  if (name == "X") {
    return
      [name](Context context, Runtime* runtime) {
        return
          std::make_unique<ColumnT<Table0Axes>>(
            context,
            runtime,
            name,
            ValueType<unsigned>::DataType,
            std::vector<Table0Axes>{Table0Axes::ROW},
            IndexTreeL(TABLE0_NUM_ROWS));
      };
  } else if (name == "Y"){
    return
      [name](Context context, Runtime* runtime) {
        return
          std::make_unique<ColumnT<Table0Axes>>(
            context,
            runtime,
            name,
            ValueType<unsigned>::DataType,
            std::vector<Table0Axes>{Table0Axes::ROW},
            IndexTreeL(TABLE0_NUM_ROWS),
            WithKeywords::kw_desc_t{{"perfect", ValueType<short>::DataType}});
      };
  } else /* name == "Z" */ {
    return
      [name](Context context, Runtime* runtime) {
        return
          std::make_unique<ColumnT<Table0Axes>>(
            context,
            runtime,
            name,
            ValueType<unsigned>::DataType,
            std::vector<Table0Axes>{Table0Axes::ROW, Table0Axes::ZP},
            IndexTreeL({{TABLE0_NUM_ROWS, IndexTreeL(2)}}));
      };
  }
}

PhysicalRegion
attach_table0_col(
  const ColumnT<Table0Axes>* col,
  unsigned *base,
  Context context,
  Runtime* runtime) {

  const Memory local_sysmem =
    Machine::MemoryQuery(Machine::get_machine())
    .has_affinity_to(runtime->get_executing_processor(context))
    .only_kind(Memory::SYSTEM_MEM)
    .first();

  AttachLauncher
    task(EXTERNAL_INSTANCE, col->logical_region(), col->logical_region());
  task.attach_array_soa(
    base,
    true,
    {Column::value_fid},
    local_sysmem);
  return runtime->attach_external_resource(context, task);
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
  testing::TestRecorder<WRITE_DISCARD>& recorder,
  const IndexTreeL& tree,
  const std::string& tree_name) {

  write_index_tree_to_attr<binary_index_tree_serdez>(
    tree,
    fid,
    dataset_name,
    tree_name);

  auto tree_md =
    read_index_tree_attr_metadata(fid, dataset_name.c_str(), tree_name.c_str());
  recorder.assert_true(
    std::string("IndexTree attribute ") + tree_name + " metadata exists",
    tree_md.has_value());
  recorder.expect_true(
    std::string("IndexTree attribute ") + tree_name
    + " metadata has expected serializer id",
    TE(tree_md.value())
    == std::string(binary_index_tree_serdez::id));
  auto optTree =
    read_index_tree_from_attr<binary_index_tree_serdez>(
      fid,
      dataset_name,
      tree_name);
  recorder.assert_true(
    std::string("IndexTree attribute ") + tree_name + " value exists",
    optTree.has_value());
  recorder.expect_true(
    std::string("IndexTree attribute ") + tree_name + " has expected value",
    TE(optTree.value()) == tree);

  auto optTree_bad =
    read_index_tree_from_attr<other_index_tree_serdez>(
      fid,
      dataset_name.c_str(),
      tree_name.c_str());
  recorder.expect_false(
    std::string("Failure to read IndexTree attribute ") + tree_name
    + " with incorrect deserializer",
    optTree_bad.has_value());
}

void
tree_tests(testing::TestRecorder<WRITE_DISCARD>& recorder) {
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
      IndexTreeL(large_tree_min / 2 + 1),
      "small-tree");

    IndexTreeL tree1(4);
    while (tree1.serialized_size() < large_tree_min)
      tree1 = IndexTreeL({{0, tree1}});
    test_index_tree_attribute(fid, dataset_name, recorder, tree1, "large-tree");

    H5Fclose(fid);
    unlink(fname.c_str());
  } catch (...) {
    unlink(fname.c_str());
    throw;
  }
}

void
table_tests(
  testing::TestRecorder<WRITE_DISCARD>& recorder,
  Context context,
  Runtime* runtime) {

  for (size_t i = 0; i < TABLE0_NUM_ROWS; ++i) {
    table0_z[2 * i] = table0_x[i];
    table0_z[2 * i + 1] = table0_y[i];
  }

  TableT<Table0Axes>
    table0(
      context,
      runtime,
      "table0",
      {static_cast<int>(Table0Axes::ROW)},
      {table0_col("X"),
       table0_col("Y"),
       table0_col("Z")},
      {{"MS_VERSION", ValueType<float>::DataType},
       {"NAME", ValueType<casacore::String>::DataType}});

  auto col_x =
    attach_table0_col(table0.columnT("X").get(), table0_x, context, runtime);
  auto col_y =
    attach_table0_col(table0.columnT("Y").get(), table0_y, context, runtime);
  auto col_z =
    attach_table0_col(table0.columnT("Z").get(), table0_z, context, runtime);

  {
    // initialize table0 keyword values
    RegionRequirement kw_req(
      table0.keywords_region(),
      WRITE_ONLY,
      EXCLUSIVE,
      table0.keywords_region());
    std::vector<FieldID> fids(2);
    std::iota(fids.begin(), fids.end(), 0);
    kw_req.add_fields(fids);
    PhysicalRegion kws = runtime->map_region(context, kw_req);
    const FieldAccessor<
      WRITE_ONLY,
      float,
      1,
      coord_t,
      AffineAccessor<float, 1, coord_t>,
      true> ms_version(kws, 0);
    const FieldAccessor<
      WRITE_ONLY,
      casacore::String,
      1,
      coord_t,
      AffineAccessor<casacore::String, 1, coord_t>,
      true> name(kws, 1);
    ms_version[0] = -42.1f;
    name[0] = "test";
    runtime->unmap_region(context, kws);
  }
  {
    // initialize column Y keyword value
    auto cy = table0.columnT("Y");
    RegionRequirement kw_req(
      cy->keywords_region(),
      WRITE_ONLY,
      EXCLUSIVE,
      cy->keywords_region());
    kw_req.add_field(0);
    PhysicalRegion kws = runtime->map_region(context, kw_req);
    const FieldAccessor<
      WRITE_ONLY,
      short,
      1,
      coord_t,
      AffineAccessor<short, 1, coord_t>,
      true> perfect(kws, 0);
    perfect[0] = 496;
    runtime->unmap_region(context, kws);
  }
  
  std::string fname = "h5.XXXXXX";
  int fd = mkstemp(fname.data());
  assert(fd != -1);
  std::cout << "test file name: " << fname << std::endl;
  close(fd);
  hid_t fid = H5DatatypeManager::create(fname, H5F_ACC_TRUNC);
  hid_t root_loc = H5Gopen(fid, "/", H5P_DEFAULT);
  assert(root_loc >= 0);
  write_table(fname, root_loc, &table0);
  H5Fclose(fid);

  runtime->detach_external_resource(context, col_x);
  runtime->detach_external_resource(context, col_y);
  runtime->detach_external_resource(context, col_z);
}

void
hdf5_test_suite(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *runtime) {

  register_tasks(runtime);

  testing::TestLog<WRITE_DISCARD> log(regions[0], regions[1], ctx, runtime);
  testing::TestRecorder<WRITE_DISCARD> recorder(log);

  tree_tests(recorder);
  table_tests(recorder, ctx, runtime);
}

int
main(int argc, char* argv[]) {

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
