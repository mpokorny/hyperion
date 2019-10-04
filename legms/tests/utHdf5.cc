#include <legms/hdf5.h>
#include <legms/Table.h>
#include <legms/Column.h>

#include <legms/testing/TestSuiteDriver.h>
#include <legms/testing/TestRecorder.h>
#include <legms/testing/TestExpression.h>

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
  ROW = 0,
  X,
  Y,
  ZP
};


template <>
struct legms::Axes<Table0Axes> {
  static const constexpr char* uid = "Table0Axes";
  static const std::vector<std::string> names;
  static const unsigned num_axes = 4;
#ifdef LEGMS_USE_HDF5
  static const hid_t h5_datatype;
#endif
};

const std::vector<std::string>
legms::Axes<Table0Axes>::names{"ROW", "X", "Y", "ZP"};

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
legms::Axes<Table0Axes>::h5_datatype = h5_dt();

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
copy_region(const PhysicalRegion& region, Context context, Runtime* runtime) {
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

Column::Generator
table0_col(const std::string& name) {
  if (name == "X") {
    return
      [name](Context ctx, Runtime* rt, const std::string& name_prefix) {
        return
          Column::create(
            ctx,
            rt,
            name,
            std::vector<Table0Axes>{Table0Axes::ROW},
            ValueType<unsigned>::DataType,
            IndexTreeL(TABLE0_NUM_ROWS),
            {},
            name_prefix);
      };
  } else if (name == "Y"){
    return
      [name](Context ctx, Runtime* rt, const std::string& name_prefix) {
        return
          Column::create(
            ctx,
            rt,
            name,
            std::vector<Table0Axes>{Table0Axes::ROW},
            ValueType<unsigned>::DataType,
            IndexTreeL(TABLE0_NUM_ROWS),
            Keywords::kw_desc_t{{"perfect", ValueType<short>::DataType}},
            name_prefix);
      };
  } else /* name == "Z" */ {
    return
      [name](Context ctx, Runtime* rt, const std::string& name_prefix) {
        return
          Column::create(
            ctx,
            rt,
            name,
            std::vector<Table0Axes>{Table0Axes::ROW, Table0Axes::ZP},
            ValueType<unsigned>::DataType,
            IndexTreeL({{TABLE0_NUM_ROWS, IndexTreeL(2)}}),
            {},
            name_prefix);
      };
  }
}

PhysicalRegion
attach_table0_col(
  Context context,
  Runtime* runtime,
  const Column& col,
  unsigned *base) {

  const Memory local_sysmem =
    Machine::MemoryQuery(Machine::get_machine())
    .has_affinity_to(runtime->get_executing_processor(context))
    .only_kind(Memory::SYSTEM_MEM)
    .first();

  AttachLauncher task(EXTERNAL_INSTANCE, col.values_lr, col.values_lr);
  task.attach_array_soa(
    base,
    false,
    {Column::VALUE_FID},
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
      IndexTreeL(LEGMS_LARGE_TREE_MIN / 2 + 1),
      "small-tree");

    IndexTreeL tree1(4);
    while (tree1.serialized_size() < LEGMS_LARGE_TREE_MIN)
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
  const unsigned* expected,
  const PhysicalRegion& region,
  const std::array<size_t, N>& dims,
  Context context,
  Runtime* runtime) {

  LogicalRegion lr = copy_region(region, context, runtime);
  RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
  req.add_field(Column::VALUE_FID);
  PhysicalRegion pr = runtime->map_region(context, req);

  bool result = true;
  const FA<READ_ONLY, unsigned, N> acc(pr, Column::VALUE_FID);
  PointInDomainIterator<N> pid(region.get_bounds<N, Legion::coord_t>(), false);
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

  runtime->unmap_region(context, pr);
  runtime->destroy_logical_region(context, lr);
  return result;
}

void
table_tests(
  testing::TestRecorder<READ_WRITE>& recorder,
  Context ctx,
  Runtime* rt) {

  for (size_t i = 0; i < TABLE0_NUM_ROWS; ++i) {
    table0_z[2 * i] = table0_x[i];
    table0_z[2 * i + 1] = table0_y[i];
  }

  const float ms_vn = -42.1f;
  legms::string ms_nm("test");
  std::string fname = "h5.XXXXXX";
  {
    Table table0 =
      Table::create(
        ctx,
        rt,
        "table0",
        std::vector<Table0Axes>{Table0Axes::ROW},
        std::vector<Column::Generator>{
          table0_col("X"),
            table0_col("Y"),
            table0_col("Z")},
#ifdef LEGMS_USE_CASACORE
        {}, // FIXME
        MeasRefContainer(), // FIXME
#endif
        {{"MS_VERSION", ValueType<float>::DataType},
         {"NAME", ValueType<std::string>::DataType}});

    auto col_x =
      attach_table0_col(ctx, rt, table0.column(ctx, rt, "X"), table0_x);
    auto col_y =
      attach_table0_col(ctx, rt, table0.column(ctx, rt, "Y"), table0_y);
    auto col_z =
      attach_table0_col(ctx, rt, table0.column(ctx, rt, "Z"), table0_z);

    {
      // initialize table0 keyword values
      std::vector<FieldID> fids(2);
      std::iota(fids.begin(), fids.end(), 0);
      auto reqs = table0.keywords.requirements<WRITE_ONLY>(fids);
      auto prs =
        reqs.map(
          [&ctx, rt](const RegionRequirement& r){
            return rt->map_region(ctx, r);
          });
      Keywords::write<WRITE_ONLY>(prs, (FieldID)0, ms_vn);
      Keywords::write<WRITE_ONLY>(prs, (FieldID)1, ms_nm);
      prs.map(
        [&ctx, rt](const PhysicalRegion& p) {
          rt->unmap_region(ctx, p);
          return 0;
        });
    }
    {
      // initialize column Y keyword value
      auto cy = table0.column(ctx, rt, "Y");
      cy.keywords.write(ctx, rt, 0, (unsigned)496);
    }

    // write HDF5 file
    int fd = mkstemp(fname.data());
    assert(fd != -1);
    std::cout << "test file name: " << fname << std::endl;
    close(fd);
    recorder.assert_no_throw(
      "Write to HDF5 file",
      testing::TestEval(
        [&table0, &fname, &ctx, rt]() {
          hid_t fid = H5DatatypeManager::create(fname, H5F_ACC_TRUNC);
          hid_t root_loc = H5Gopen(fid, "/", H5P_DEFAULT);
          assert(root_loc >= 0);
          write_table(ctx, rt, fname, root_loc, table0);
          H5Gclose(root_loc);
          H5Fclose(fid);
          return true;
        }));

    rt->detach_external_resource(ctx, col_x);
    rt->detach_external_resource(ctx, col_y);
    rt->detach_external_resource(ctx, col_z);
    table0.destroy(ctx, rt);
  }

  {
    std::unordered_set<std::string> tblpaths = get_table_paths(fname);
    recorder.expect_true(
      "File contains single, written table",
      TE(tblpaths.count("/table0") == 1 && tblpaths.size() == 1));
  }
  {
    std::unordered_set<std::string> colnames =
      get_column_names(fname, "/table0");
    recorder.expect_true(
      "table0 contains expected column names",
      TE(colnames.count("X") == 1
         && colnames.count("Y") == 1
         && colnames.count("Z") == 1
         && colnames.size() == 3));
  }
  {
    // read back metadata
    auto tb0 = init_table(ctx, rt, fname, "/table0", {"X", "Y", "Z"});
    recorder.assert_false(
      "Table initialized from HDF5 is not empty",
      TE(tb0.is_empty(ctx, rt)));
    recorder.assert_true(
      "Table has expected keywords",
      testing::TestEval(
        [&tb0, &ctx, rt]() {
          auto keys = tb0.keywords.keys(rt);
          std::vector<FieldID> fids(keys.size());
          std::iota(fids.begin(), fids.end(), 0);
          std::set<std::tuple<std::string, legms::TypeTag>> tbkw;
          auto tts = tb0.keywords.value_types(ctx, rt, fids);
          for (size_t i = 0; i < tts.size(); ++i)
            tbkw.insert(make_tuple(keys[i], tts[i]));
          std::set<std::tuple<std::string, legms::TypeTag>>
            kw{{"MS_VERSION", ValueType<float>::DataType},
               {"NAME", ValueType<std::string>::DataType}};
          return tbkw == kw;
        }));

    {
      auto cx = tb0.column(ctx, rt, "X");
      recorder.assert_true(
        "Column X logically recreated",
        TE(!cx.is_empty()));
      recorder.expect_true(
        "Column X has expected axes",
        TE(cx.axes(ctx, rt)) ==
        map_to_int(std::vector<Table0Axes>{Table0Axes::ROW}));
      recorder.expect_true(
        "Column X has expected indexes",
        TE(cx.index_tree(rt)) == IndexTreeL(TABLE0_NUM_ROWS));
    }
    {
      auto cy = tb0.column(ctx, rt, "Y");
      recorder.assert_true(
        "Column Y logically recreated",
        TE(!cy.is_empty()));
      recorder.expect_true(
        "Column Y has expected axes",
        TE(cy.axes(ctx, rt)) ==
        map_to_int(std::vector<Table0Axes>{Table0Axes::ROW}));
      recorder.expect_true(
        "Column Y has expected indexes",
        TE(cy.index_tree(rt)) == IndexTreeL(TABLE0_NUM_ROWS));
    }
    {
      auto cz = tb0.column(ctx, rt, "Z");
      recorder.assert_true(
        "Column Z logically recreated",
        TE(!cz.is_empty()));
      recorder.expect_true(
        "Column Z has expected axes",
        TE(cz.axes(ctx, rt))
        == map_to_int(std::vector<Table0Axes>{Table0Axes::ROW, Table0Axes::ZP}));
      recorder.expect_true(
        "Column Z has expected indexes",
        TE(cz.index_tree(rt)) == IndexTreeL({{TABLE0_NUM_ROWS, IndexTreeL(2)}}));
    }

    // attach to file, and read back keywords
    // {
    //   auto tb_kws =
    //     attach_table_keywords(
    //       fname,
    //       "/",
    //       tb0.get(),
    //       rt,
    //       ctx);
    //   recorder.assert_true(
    //     "Table keywords attached",
    //     TE(tb_kws.has_value()));
    //   std::map<std::string, size_t> fids;
    //   for (size_t i = 0; i < tb0->keywords().size(); ++i)
    //     fids[std::get<0>(tb0->keywords()[i])] = i;
    //   recorder.expect_true(
    //     "Table has expected keyword values",
    //     testing::TestEval(
    //       [&tb_kws, &fids, &ms_vn, &ms_nm, ctx, rt]() {
    //         LogicalRegion kws = copy_region(tb_kws.value(), ctx, rt);
    //         RegionRequirement req(kws, READ_ONLY, EXCLUSIVE, kws);
    //         for (size_t i = 0; i < fids.size(); ++i)
    //           req.add_field(i);
    //         PhysicalRegion pr = rt->map_region(ctx, req);
    //         const FieldAccessor<READ_ONLY, float, 1>
    //           vn(pr, fids.at("MS_VERSION"));
    //         const FieldAccessor<READ_ONLY, casacore::String, 1>
    //           nm(pr, fids.at("NAME"));
    //         return vn[0] == ms_vn && nm[0] == ms_nm;
    //        }));
    // }
    // attach to file, and read back values
    {
      std::unordered_map<std::string,PhysicalRegion> tb_cols;
      for (auto& cn : tb0.column_names(ctx, rt)) {
        auto col = tb0.column(ctx, rt, cn);
        tb_cols[cn] = attach_column_values(ctx, rt, fname, "/table0", col);
      }
      recorder.expect_true(
        "All column values attached",
        testing::TestEval(
          [&tb0, &tb_cols, &ctx, rt]() {
            std::unordered_set<std::string> names;
            std::transform(
              tb_cols.begin(),
              tb_cols.end(),
              std::inserter(names, names.end()),
              [&tb0](auto& nm_pr) { return std::get<0>(nm_pr); });
            auto tbcns = tb0.column_names(ctx, rt);
            return (names ==
                    std::unordered_set<std::string>(tbcns.begin(), tbcns.end()));
          }));
      recorder.expect_true(
        "Column 'X' values as expected",
        TE(verify_col<1>(table0_x, tb_cols["X"], {TABLE0_NUM_ROWS}, ctx, rt)));
      recorder.expect_true(
        "Column 'Y' values as expected",
        TE(verify_col<1>(table0_y, tb_cols["Y"], {TABLE0_NUM_ROWS}, ctx, rt)));
      recorder.expect_true(
        "Column 'Z' values as expected",
        TE(verify_col<2>(table0_z, tb_cols["Z"], {TABLE0_NUM_ROWS, 2}, ctx, rt)));

      recorder.expect_no_throw(
        "Table columns detached",
        testing::TestEval(
          [&tb_cols, &ctx, rt]() {
            std::for_each(
              tb_cols.begin(),
              tb_cols.end(),
              [&rt, &ctx](auto& nm_pr) {
                rt->detach_external_resource(ctx, std::get<1>(nm_pr));
              });
            return true;
          }));
    }
    tb0.destroy(ctx, rt);
  }
}

void
hdf5_test_suite(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime *runtime) {

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
  table_tests(recorder, ctx, runtime);
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
