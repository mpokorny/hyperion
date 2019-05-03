#include "legms_hdf5.h"

#include "TestSuiteDriver.h"
#include "TestRecorder.h"
#include "TestExpression.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <hdf5.h>
#include <numeric>
#include <set>
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

template <>
struct legms::AxesUID<Table0Axes> {
  static constexpr const char* id = "Table0Axes";
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

  hid_t ds = H5Dopen(fid, dataset_name.c_str(), H5P_DEFAULT);
  assert(ds >= 0);
  auto tree_md = read_index_tree_attr_metadata(ds, tree_name.c_str());
  recorder.assert_true(
    std::string("IndexTree attribute ") + tree_name + " metadata exists",
    tree_md.has_value());
  recorder.expect_true(
    std::string("IndexTree attribute ") + tree_name
    + " metadata has expected serializer id",
    TE(tree_md.value())
    == std::string(binary_index_tree_serdez::id));
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
  const unsigned* expected,
  const PhysicalRegion& region,
  const std::array<size_t, N>& dims) {

  bool result = true;
  //const FA<READ_ONLY, unsigned, N> acc(region, Column::value_fid);
  const FieldAccessor<READ_ONLY, unsigned, N> acc(region, Column::value_fid);
  PointInDomainIterator<N> pid(region.get_bounds<N, Legion::coord_t>(), false);
  std::array<size_t, N> pt;
  pt.fill(0);
  size_t off = 0;
  while (result && pid() && pt[0] < dims[0]) {
    std::cout << "pid " << *pid
              << "; dims [";
    for (size_t i = 0; i < N; ++i)
      std::cout << dims[i] << ",";
    std::cout << "]"
              << "; off " << off
              << "; val " << acc[*pid]
              << "; exp " << expected[off]
              << std::endl;
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

  return result;
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

  const float ms_vn = -42.1f;
  const casacore::String ms_nm = "test";

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
    const FA<WRITE_ONLY, float, 1> ms_version(kws, 0);
    const FA<WRITE_ONLY, casacore::String, 1> name(kws, 1);
    ms_version[0] = ms_vn;
    name[0] = ms_nm;
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
    const FA<WRITE_ONLY, short, 1> perfect(kws, 0);
    perfect[0] = 496;
    runtime->unmap_region(context, kws);
  }

  // write HDF5 file
  std::string fname = "h5.XXXXXX";
  int fd = mkstemp(fname.data());
  assert(fd != -1);
  std::cout << "test file name: " << fname << std::endl;
  close(fd);
  recorder.assert_no_throw(
    "Write to HDF5 file",
    testing::TestEval(
      [&table0, &fname]() {
        hid_t fid = H5DatatypeManager::create(fname, H5F_ACC_TRUNC);
        hid_t root_loc = H5Gopen(fid, "/", H5P_DEFAULT);
        assert(root_loc >= 0);
        write_table(fname, root_loc, &table0);
        H5Fclose(fid);
        return true;
      }));

  runtime->detach_external_resource(context, col_x);
  runtime->detach_external_resource(context, col_y);
  runtime->detach_external_resource(context, col_z);

  // read back metadata
  auto table_ga = init_table(fname, "/table0", runtime, context);
  recorder.assert_true(
    "Table generator arguments read back from HDF5",
    TE(table_ga.has_value()));
  auto tb0 = table_ga.value().template operator()<Table0Axes>(context, runtime);
  recorder.assert_true(
    "Table recreated from generator arguments",
    TE(bool(tb0)));
  {
    recorder.assert_true(
      "Table has expected keywords",
      testing::TestEval(
        [&tb0]() {
          auto tbkw_v = tb0->keywords();
          std::set<std::tuple<std::string, casacore::DataType>>
            tbkw(tbkw_v.begin(), tbkw_v.end());
          std::set<std::tuple<std::string, casacore::DataType>>
            kw{{"MS_VERSION", ValueType<float>::DataType},
               {"NAME", ValueType<casacore::String>::DataType}};
          return tbkw == kw;
        }));
  }

  {
    auto cx = tb0->columnT("X");
    recorder.assert_true("Column X logically recreated", TE(bool(cx)));
    recorder.expect_true(
      "Column X has expected axes",
      TE(cx->axesT()) == std::vector<Table0Axes>{Table0Axes::ROW});
    recorder.expect_true(
      "Column X has expected indexes",
      TE(cx->index_tree()) == IndexTreeL(TABLE0_NUM_ROWS));
  }
  {
    auto cy = tb0->columnT("Y");
    recorder.assert_true("Column Y logically recreated", TE(bool(cy)));
    recorder.expect_true(
      "Column Y has expected axes",
      TE(cy->axesT()) == std::vector<Table0Axes>{Table0Axes::ROW});
    recorder.expect_true(
      "Column Y has expected indexes",
      TE(cy->index_tree()) == IndexTreeL(TABLE0_NUM_ROWS));
  }
  {
    auto cz = tb0->columnT("Z");
    recorder.assert_true("Column Z logically recreated", TE(bool(cz)));
    recorder.expect_true(
      "Column Z has expected axes",
      TE(cz->axesT())
      == std::vector<Table0Axes>{Table0Axes::ROW, Table0Axes::ZP});
    recorder.expect_true(
      "Column Z has expected indexes",
      TE(cz->index_tree()) == IndexTreeL({{TABLE0_NUM_ROWS, IndexTreeL(2)}}));
  }

  // attach to file, and read back keywords
  {
    auto tb_kws =
      attach_table_keywords(
        fname,
        "/",
        tb0.get(),
        runtime,
        context);
    recorder.assert_true(
      "Table keywords attached",
      TE(tb_kws.has_value()));
    // std::map<std::string, size_t> fids;
    // for (size_t i = 0; i < tb0->keywords().size(); ++i)
    //   fids[std::get<0>(tb0->keywords()[i])] = i;
    // auto prtkw = tb_kws.value();
    // recorder.expect_true(
    //   "Table has expected keyword values",
    //   testing::TestEval(
    //     [&prtkw, &fids, &ms_vn, &ms_nm]() {
    //       const FieldAccessor<READ_ONLY, float, 1>
    //         vn(prtkw, fids.at("MS_VERSION"));
    //       const FieldAccessor<READ_ONLY, casacore::String, 1>
    //         nm(prtkw, fids.at("NAME"));
    //       return vn[0] == ms_vn && nm[0] == ms_nm;
    //      }));
  }
  // attach to file, and read back values
  {
    auto tb_cols =
      attach_table_columns(
        fname,
        "/",
        tb0.get(),
        {"X", "Y", "Z"},
        runtime,
        context);
    recorder.assert_true(
      "Table columns attached",
      TE(
        std::all_of(
          tb_cols.begin(),
          tb_cols.end(),
          [](auto& pr2) {
            return std::get<0>(pr2).has_value();
          })));
    recorder.assert_true(
      "Column keywords attached only when present",
      TE(!std::get<1>(tb_cols[0]).has_value()
         && std::get<1>(tb_cols[1]).has_value()
         && !std::get<1>(tb_cols[2]).has_value()));
    // auto prx = std::get<0>(tb_cols[0]).value();
    // recorder.expect_true(
    //   "Column 'X' values as expected",
    //   TE(verify_col<1>(table0_x, prx, {TABLE0_NUM_ROWS})));
    // auto pry = std::get<0>(tb_cols[1]).value();
    // recorder.expect_true(
    //   "Column 'Y' values as expected",
    //   TE(verify_col<1>(table0_y, pry, {TABLE0_NUM_ROWS})));
    // auto prz = std::get<0>(tb_cols[2]).value();
    // recorder.expect_true(
    //   "Column 'Z' values as expected",
    //   TE(verify_col<2>(table0_z, prz, {TABLE0_NUM_ROWS, 2})));

    // recorder.expect_no_throw(
    //   "Table columns detached",
    //   testing::TestEval(
    //     [&tb_cols, &runtime, &context]() {
    //       std::for_each(
    //         tb_cols.begin(),
    //         tb_cols.end(),
    //         [&runtime, &context](auto& pr2) {
    //           runtime
    //             ->detach_external_resource(context, std::get<0>(pr2).value());
    //         });
    //       return true;
    //     }));
  }
}

void
hdf5_test_suite(
  const Task*,
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
