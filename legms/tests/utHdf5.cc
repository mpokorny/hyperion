#include "legms_hdf5.h"

#include "TestSuiteDriver.h"
#include "TestRecorder.h"
#include "TestExpression.h"

#include <cassert>
#include <hdf5.h>
#include <stdlib.h>
#include <unistd.h>

using namespace legms;
using namespace legms::hdf5;
using namespace Legion;

enum {
  HDF5_TEST_SUITE,
};

#define TE(f) testing::TestEval([&](){ return f; }, #f)

void
tree_tests(
  hid_t fid,
  const std::string& dataset_name,
  testing::TestRecorder<READ_WRITE>& recorder,
  const IndexTreeL& tree,
  const std::string& tree_name) {

  write_index_tree_to_attr<binary_index_tree_serdez<Legion::coord_t>>(
    tree,
    fid,
    dataset_name.c_str(),
    tree_name.c_str());

  auto tree_md =
    read_index_tree_attr_metadata(fid, dataset_name.c_str(), tree_name.c_str());
  recorder.assert_true(
    std::string("IndexTree attribute ") + tree_name + " metadata exists",
    tree_md.has_value());
  recorder.expect_true(
    std::string("IndexTree attribute ") + tree_name
    + " metadata has expected serializer id",
    TE(tree_md.value())
    == binary_index_tree_serdez<Legion::coord_t>::id);
  auto optTree =
    read_index_tree_from_attr<binary_index_tree_serdez<Legion::coord_t>>(
      fid,
      dataset_name.c_str(),
      tree_name.c_str());
  recorder.assert_true(
    std::string("IndexTree attribute ") + tree_name + " value exists",
    optTree.has_value());
  recorder.expect_true(
    std::string("IndexTree attribute ") + tree_name + " has expected value",
    TE(optTree.value()) == tree);

  auto optTree_bad =
    read_index_tree_from_attr<binary_index_tree_serdez<unsigned char>>(
      fid,
      dataset_name.c_str(),
      tree_name.c_str());
  recorder.expect_false(
    std::string("Failure to read IndexTree attribute ") + tree_name
    + " with incorrect deserializer",
    optTree_bad.has_value());
}

void
hdf5_test_suite(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *runtime) {

  testing::TestLog<READ_WRITE> log(regions[0], regions[1], ctx, runtime);
  testing::TestRecorder<READ_WRITE> recorder(log);

  std::string fname = "tmpXXXXXX";
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

    tree_tests(fid, dataset_name, recorder, IndexTreeL(84000), "small-tree");

    IndexTreeL tree1(4);
    while (tree1.serialized_size() < 100000)
      tree1 = IndexTreeL({{0, tree1}});
    tree_tests(fid, dataset_name, recorder, tree1, "large-tree");

    H5Fclose(fid);
    unlink(fname.c_str());
  } catch (...) {
    unlink(fname.c_str());
    throw;
  }
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
