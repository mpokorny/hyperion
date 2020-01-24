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

#include <hyperion/utility.h>
#include <hyperion/x/Table.h>
#include <hyperion/x/Column.h>
#include <hyperion/x/ColumnSpace.h>

#include <algorithm>
#include <memory>
#include <ostream>
#include <vector>

using namespace hyperion;
using namespace Legion;

enum {
  REINDEXED_TEST_SUITE,
};

enum struct Table0Axes {
  ROW = 0,
  X,
  Y
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
  static const unsigned num_axes = 3;
#ifdef HYPERION_USE_HDF5
  static const hid_t h5_datatype;
#endif
};

const std::vector<std::string>
hyperion::Axes<Table0Axes>::names{"ROW", "X", "Y"};

#ifdef HYPERION_USE_HDF5
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
  return result;
}

const hid_t
hyperion::Axes<Table0Axes>::h5_datatype = h5_dt();
#endif

std::ostream&
operator<<(std::ostream& stream, const Table0Axes& ax) {
  switch (ax) {
  case Table0Axes::ROW:
    stream << "ROW";
    break;
  case Table0Axes::X:
    stream << "X";
    break;
  case Table0Axes::Y:
    stream << "Y";
    break;
  }
  return stream;
}

std::ostream&
operator<<(std::ostream& stream, const std::vector<Table0Axes>& axs) {
  stream << "[";
  const char* sep = "";
  std::for_each(
    axs.begin(),
    axs.end(),
    [&stream, &sep](auto& ax) {
      stream << sep << ax;
      sep = ",";
    });
  stream << "]";
  return stream;
}

#define TABLE0_NUM_X 4
#define OX 22
#define TABLE0_NUM_Y 3
#define OY 30
#define TABLE0_NUM_ROWS (TABLE0_NUM_X * TABLE0_NUM_Y)
unsigned table0_x[TABLE0_NUM_ROWS] {
                   OX + 0, OX + 0, OX + 0,
                     OX + 1, OX + 1, OX + 1,
                     OX + 2, OX + 2, OX + 2,
                     OX + 3, OX + 3, OX + 3};
unsigned table0_y[TABLE0_NUM_ROWS] {
                   OY + 0, OY + 1, OY + 2,
                     OY + 0, OY + 1, OY + 2,
                     OY + 0, OY + 1, OY + 2,
                     OY + 0, OY + 1, OY + 2};
unsigned table0_z[TABLE0_NUM_ROWS] {
                   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

Column::Generator
table0_col(
  const std::string& name
#ifdef HYPERION_USE_CASACORE
  , const std::unordered_map<std::string, MeasRef>& measures
  , const std::optional<std::string> &meas_name = std::nullopt
#endif
  ) {
  return
    [=](Context ctx, Runtime* rt, const std::string& name_prefix
#ifdef HYPERION_USE_CASACORE
        , const MeasRefContainer& table_mr
#endif
      ) {
#ifdef HYPERION_USE_CASACORE
      MeasRef mr;
      bool own_mr = false;
      if (meas_name) {
        auto mrs =
          MeasRefContainer::create(ctx, rt, measures, table_mr);
        std::tie(mr, own_mr) = mrs.lookup(ctx, rt, meas_name.value());
      }
#endif
      return
        Column::create(
          ctx,
          rt,
          name,
          std::vector<Table0Axes>{Table0Axes::ROW},
          ValueType<unsigned>::DataType,
          IndexTreeL(TABLE0_NUM_ROWS),
#ifdef HYPERION_USE_CASACORE
          mr,
          own_mr,
          meas_name.value_or(""),
#endif
          {},
          name_prefix);
    };
}

PhysicalRegion
attach_table0_col(
  Context ctx,
  Runtime* rt,
  const x::Column& col,
  unsigned *base) {

  const Memory local_sysmem =
    Machine::MemoryQuery(Machine::get_machine())
    .has_affinity_to(rt->get_executing_processor(ctx))
    .only_kind(Memory::SYSTEM_MEM)
    .first();

  AttachLauncher task(EXTERNAL_INSTANCE, col.vreq.region, col.vreq.region);
  task.attach_array_soa(base, false, {col.fid}, local_sysmem);
  PhysicalRegion result = rt->attach_external_resource(ctx, task);
  AcquireLauncher acq(col.vreq.region, col.vreq.region, result);
  acq.add_field(col.fid);
  rt->issue_acquire(ctx, acq);
  return result;
}

#define TE(f) testing::TestEval([&](){ return f; }, #f)

void
test_totally_reindexed_table(
  Context ctx,
  Runtime* rt,
  const x::Table& tb,
  bool x_before_y,
  const std::string& prefix,
  testing::TestRecorder<READ_WRITE>& recorder) {

  std::cout << "tb flr " << tb.fields_lr << std::endl; // FIXME: remove
  recorder.expect_true(
    prefix + " reindexed table is not empty",
    TE(!tb.is_empty()));

  std::vector<Table0Axes> ixax;
  if (x_before_y)
    ixax = {Table0Axes::X, Table0Axes::Y};
  else
    ixax = {Table0Axes::Y, Table0Axes::X};

  std::ostringstream oss;
  oss << ixax;

  recorder.expect_true(
    prefix + " reindexed table has " + oss.str() + " index axes",
    testing::TestEval(
      [&ctx, rt, &tb, &ixax]() {
        auto ax =
          tb.index_axes(ctx, rt)
          .template get_result<x::Table::index_axes_result_t>();
        auto axes = x::ColumnSpace::from_axis_vector(ax);
        return axes == map_to_int(ixax);
      }));

  auto cols =
    x::Table::column_map(
      tb.columns(ctx, rt)
      .template get_result<x::Table::columns_result_t>());
  {
    recorder.assert_true(
      prefix + " reindexed table has 'X' column",
      TE(cols.count("X") > 0));

    auto& cx = cols.at("X");
    {
      RegionRequirement
        req(cx.csp.metadata_lr, READ_ONLY, EXCLUSIVE, cx.csp.metadata_lr);
      req.add_field(x::ColumnSpace::AXIS_VECTOR_FID);
      req.add_field(x::ColumnSpace::AXIS_SET_UID_FID);
      req.add_field(x::ColumnSpace::INDEX_FLAG_FID);
      auto pr = rt->map_region(ctx, req);
      const x::ColumnSpace::AxisVectorAccessor<READ_ONLY>
        av(pr, x::ColumnSpace::AXIS_VECTOR_FID);
      const x::ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
        auid(pr, x::ColumnSpace::AXIS_SET_UID_FID);
      const x::ColumnSpace::IndexFlagAccessor<READ_ONLY>
        ifl(pr, x::ColumnSpace::INDEX_FLAG_FID);

      recorder.expect_true(
        prefix + " reindexed 'X' column has unchanged axis set uid",
        TE(auid[0] == std::string(Axes<Table0Axes>::uid)));
      recorder.expect_true(
        prefix + " reindexed 'X' column has only 'X' axis",
        TE(x::ColumnSpace::from_axis_vector(av[0])
           == map_to_int(std::vector<Table0Axes>{Table0Axes::X})));
      recorder.expect_true(
        prefix + " reindexed 'X' column has index flag set",
        TE(ifl[0]));
      rt->unmap_region(ctx, pr);
    }
    recorder.assert_true(
      prefix + " reindexed 'X' column has expected size",
      testing::TestEval(
        [&cx, &ctx, rt]() {
          auto is = cx.vreq.region.get_index_space();
          auto dom = rt->get_index_space_domain(ctx, is);
          Rect<1> r(dom.bounds<1,coord_t>());
          return r == Rect<1>(0, TABLE0_NUM_X - 1);
        }));
    recorder.expect_true(
      prefix + " reindexed 'X' column has expected values",
      testing::TestEval(
        [&cx, &ctx, rt]() {
          RegionRequirement
            req(cx.vreq.region, READ_ONLY, EXCLUSIVE, cx.vreq.region);
          req.add_field(cx.fid);
          PhysicalRegion pr = rt->map_region(ctx, req);
          const FieldAccessor<
            READ_ONLY, unsigned, 1, coord_t,
            AffineAccessor<unsigned, 1, coord_t>, true>
            x(pr, cx.fid);
          bool result =
            x[0] == OX && x[1] == OX + 1 && x[2] == OX + 2
            && x[3] == OX + 3;
          rt->unmap_region(ctx, pr);
          return result;
        }));
  }
  {
    recorder.assert_true(
      prefix + " reindexed table has 'Y' column",
      TE(cols.count("Y") > 0));

    auto& cy = cols.at("Y");
    {
      RegionRequirement
        req(cy.csp.metadata_lr, READ_ONLY, EXCLUSIVE, cy.csp.metadata_lr);
      req.add_field(x::ColumnSpace::AXIS_VECTOR_FID);
      req.add_field(x::ColumnSpace::AXIS_SET_UID_FID);
      req.add_field(x::ColumnSpace::INDEX_FLAG_FID);
      auto pr = rt->map_region(ctx, req);
      const x::ColumnSpace::AxisVectorAccessor<READ_ONLY>
        av(pr, x::ColumnSpace::AXIS_VECTOR_FID);
      const x::ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
        auid(pr, x::ColumnSpace::AXIS_SET_UID_FID);
      const x::ColumnSpace::IndexFlagAccessor<READ_ONLY>
        ifl(pr, x::ColumnSpace::INDEX_FLAG_FID);

      recorder.expect_true(
        prefix + " reindexed 'Y' column has unchanged axis set uid",
        TE(auid[0] == std::string(Axes<Table0Axes>::uid)));
      recorder.expect_true(
        prefix + " reindexed 'Y' column has only 'Y' axis",
        TE(x::ColumnSpace::from_axis_vector(av[0])
           == map_to_int(std::vector<Table0Axes>{Table0Axes::Y})));
      recorder.expect_true(
        prefix + " reindexed 'Y' column has index flag set",
        TE(ifl[0]));
      rt->unmap_region(ctx, pr);
    }
    recorder.assert_true(
      prefix + " reindexed 'Y' column has expected size",
      testing::TestEval(
        [&cy, &ctx, rt]() {
          auto is = cy.vreq.region.get_index_space();
          auto dom = rt->get_index_space_domain(ctx, is);
          Rect<1> r(dom.bounds<1,coord_t>());
          return r == Rect<1>(0, TABLE0_NUM_Y - 1);
        }));
    recorder.expect_true(
      prefix + " reindexed 'Y' column has expected values",
      testing::TestEval(
        [&cy, &ctx, rt]() {
          RegionRequirement
            req(cy.vreq.region, READ_ONLY, EXCLUSIVE, cy.vreq.region);
          req.add_field(cy.fid);
          PhysicalRegion pr = rt->map_region(ctx, req);
          const FieldAccessor<
            READ_ONLY, unsigned, 1, coord_t,
            AffineAccessor<unsigned, 1, coord_t>, true>
            y(pr, cy.fid);
          bool result =
            y[0] == OY && y[1] == OY + 1 && y[2] == OY + 2;
          rt->unmap_region(ctx, pr);
          return result;
        }));
  }
  {
    recorder.assert_true(
      prefix + " reindexed table has 'Z' column",
      TE(cols.count("Z") > 0));

    auto& cz = cols.at("Z");
    {
      RegionRequirement
        req(cz.csp.metadata_lr, READ_ONLY, EXCLUSIVE, cz.csp.metadata_lr);
      req.add_field(x::ColumnSpace::AXIS_VECTOR_FID);
      req.add_field(x::ColumnSpace::AXIS_SET_UID_FID);
      req.add_field(x::ColumnSpace::INDEX_FLAG_FID);
      auto pr = rt->map_region(ctx, req);
      const x::ColumnSpace::AxisVectorAccessor<READ_ONLY>
        av(pr, x::ColumnSpace::AXIS_VECTOR_FID);
      const x::ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
        auid(pr, x::ColumnSpace::AXIS_SET_UID_FID);
      const x::ColumnSpace::IndexFlagAccessor<READ_ONLY>
        ifl(pr, x::ColumnSpace::INDEX_FLAG_FID);

      recorder.expect_true(
        prefix + " reindexed 'Z' column has unchanged axis set uid",
        TE(auid[0] == std::string(Axes<Table0Axes>::uid)));
      recorder.expect_true(
        prefix + " reindexed 'Z' column has only " + oss.str() + " axes",
        TE(x::ColumnSpace::from_axis_vector(av[0]) == map_to_int(ixax)));
      recorder.expect_true(
        prefix + " reindexed 'Z' column does not have index flag set",
        TE(!ifl[0]));
      rt->unmap_region(ctx, pr);
    }
    recorder.assert_true(
      prefix + " reindexed 'Z' column has expected size",
      testing::TestEval(
        [&cz, &x_before_y, &ctx, rt]() {
          auto is = cz.vreq.region.get_index_space();
          auto dom = rt->get_index_space_domain(ctx, is);
          Rect<2> r(dom.bounds<2,coord_t>());
          coord_t r0, r1;
          if (x_before_y) {
            r0 = TABLE0_NUM_X - 1;
            r1 = TABLE0_NUM_Y - 1;
          } else {
            r1 = TABLE0_NUM_X - 1;
            r0 = TABLE0_NUM_Y - 1;
          }
          Rect<2> expected(Point<2>(0, 0), Point<2>(r0, r1));
          // FIXME: remove the followin
          std::cout << "expected " << expected << "; actual " << r << std::endl;
          return r == expected;
        }));
    recorder.expect_true(
      prefix + " reindexed 'Z' column has expected values",
      testing::TestEval(
        [&cz, &x_before_y, &ctx, rt]() {
          RegionRequirement
            req(cz.vreq.region, READ_ONLY, EXCLUSIVE, cz.vreq.region);
          req.add_field(cz.fid);
          PhysicalRegion pr = rt->map_region(ctx, req);
          const FieldAccessor<
            READ_ONLY, unsigned, 2, coord_t,
            AffineAccessor<unsigned, 2, coord_t>, true>
            z(pr, cz.fid);
          bool result = true;
          DomainT<2,coord_t> dom(pr);
          coord_t p0 = (x_before_y ? 0 : 1);
          coord_t p1 = 1 - p0;
          for (PointInDomainIterator<2> pid(dom); pid(); pid++)
            result = result &&
              z[*pid] == table0_z[pid[p0] * TABLE0_NUM_Y + pid[p1]];
          rt->unmap_region(ctx, pr);
          return result;
        }));
  }
}

void
reindexed_test_suite(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  register_tasks(ctx, rt);

  testing::TestRecorder<READ_WRITE> recorder(
    testing::TestLog<READ_WRITE>(
      task->regions[0].region,
      regions[0],
      task->regions[1].region,
      regions[1],
      ctx,
      rt));

  auto xyz_is = rt->create_index_space(ctx, Rect<1>(0, TABLE0_NUM_ROWS - 1));
  auto xyz_space =
    x::ColumnSpace::create(
      ctx,
      rt,
      std::vector<Table0Axes>{Table0Axes::ROW},
      xyz_is,
      false);

// #ifdef HYPERION_USE_CASACORE
//   casacore::MeasRef<casacore::MEpoch> tai(casacore::MEpoch::TAI);
//   casacore::MeasRef<casacore::MEpoch> utc(casacore::MEpoch::UTC);
//   auto table0_epoch = MeasRef::create(ctx, rt, tai);

//   casacore::MeasRef<casacore::MDirection>
//     direction(casacore::MDirection::J2000);
//   casacore::MeasRef<casacore::MFrequency>
//     frequency(casacore::MFrequency::GEO);
//   auto columnX_direction = MeasRef::create(ctx, rt, direction);
//   auto columnZ_epoch = MeasRef::create(ctx, rt, utc);
//   std::unordered_map<std::string, std::unordered_map<std::string, MeasRef>>
//     col_measures{
//     {"X", {{"DIRECTION", columnX_direction}}},
//     {"Y", {}},
//     {"Z", {{"EPOCH", columnZ_epoch}}}
//   };
//   std::vector<Column::Generator> column_generators{
//     table0_col("X", col_measures["X"], "DIRECTION"),
//     table0_col("Y", col_measures["Y"]),
//     table0_col("Z", col_measures["Z"], "EPOCH")
//   };
// #else
  std::vector<std::pair<std::string, x::TableField>> xyz_fields{
    {"X",
     x::TableField(HYPERION_TYPE_UINT, COL_X, MeasRef(), Keywords())},
    {"Y",
     x::TableField(HYPERION_TYPE_UINT, COL_Y, MeasRef(), Keywords())},
    {"Z",
     x::TableField(HYPERION_TYPE_UINT, COL_Z, MeasRef(), Keywords())}
  };
// #endif

  
  auto table0 = x::Table::create(ctx, rt, {{xyz_space, xyz_fields}});
  {
    std::unordered_map<std::string, PhysicalRegion> col_prs;
    {
      auto cols =
        x::Table::column_map(
          table0.columns(ctx, rt).get<x::Table::columns_result_t>());
      std::unordered_map<std::string, unsigned*> col_arrays{
        {"X", table0_x},
        {"Y", table0_y},
        {"Z", table0_z}
      };

      std::unordered_map<std::string, PhysicalRegion> col_prs;
      for (auto& c : {"X", "Y", "Z"}) {
        std::string cstr(c);
        col_prs[cstr] =
          attach_table0_col(ctx, rt, cols.at(cstr), col_arrays.at(cstr));
      }
    }
    // tests of complete, one-step reindexing
    {
      auto f =
        table0.reindexed(
          ctx,
          rt,
          std::vector<Table0Axes>{Table0Axes::X, Table0Axes::Y},
          false);

      auto tb = f.template get_result<x::Table::reindexed_result_t>();
      test_totally_reindexed_table(ctx, rt, tb, true, "Totally", recorder);
      //tb.destroy(ctx, rt); FIXME
    }
    // tests of two-step reindexing
    {
      x::Table tby;
      {
        auto f =
          table0.reindexed(
            ctx,
            rt,
            std::vector<Table0Axes>{Table0Axes::Y},
            true);
        tby = f.template get_result<x::Table::reindexed_result_t>();
      }
      {
        recorder.expect_true(
          "Partially reindexed table is not empty",
          TE(!tby.is_empty()));

        recorder.expect_true(
          "Partially reindexed table has ('Y', 'ROW') index axes",
          testing::TestEval(
            [&ctx, rt, &tby]() {
              auto ax =
                tby.index_axes(ctx, rt)
                .template get_result<x::Table::index_axes_result_t>();
              return x::ColumnSpace::from_axis_vector(ax) ==
                map_to_int(
                  std::vector<Table0Axes>{Table0Axes::Y, Table0Axes::ROW});
            }));

        auto cols =
          x::Table::column_map(
            tby.columns(ctx, rt)
            .template get_result<x::Table::columns_result_t>());
        {
          recorder.assert_true(
            "Partially reindexed table has 'Y' column",
            TE(cols.count("Y") > 0));

          auto& cy = cols.at("Y");
          {
            RegionRequirement
              req(cy.csp.metadata_lr, READ_ONLY, EXCLUSIVE, cy.csp.metadata_lr);
            req.add_field(x::ColumnSpace::AXIS_VECTOR_FID);
            req.add_field(x::ColumnSpace::AXIS_SET_UID_FID);
            req.add_field(x::ColumnSpace::INDEX_FLAG_FID);
            auto pr = rt->map_region(ctx, req);
            const x::ColumnSpace::AxisVectorAccessor<READ_ONLY>
              av(pr, x::ColumnSpace::AXIS_VECTOR_FID);
            const x::ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
              auid(pr, x::ColumnSpace::AXIS_SET_UID_FID);
            const x::ColumnSpace::IndexFlagAccessor<READ_ONLY>
              ifl(pr, x::ColumnSpace::INDEX_FLAG_FID);

            recorder.expect_true(
              "Partially reindexed 'Y' column has unchanged axis set uid",
              TE(auid[0] == std::string(Axes<Table0Axes>::uid)));
            recorder.expect_true(
              "Partially reindexed 'Y' column has only 'Y' axis",
              TE(x::ColumnSpace::from_axis_vector(av[0])
                 == map_to_int(std::vector<Table0Axes>{Table0Axes::Y})));
            recorder.expect_true(
              "Partially reindexed 'Y' column has index flag set",
              TE(ifl[0]));
            rt->unmap_region(ctx, pr);
          }
          recorder.assert_true(
            "Partially reindexed 'Y' column has expected size",
            testing::TestEval(
              [&cy, &ctx, rt]() {
                auto is = cy.vreq.region.get_index_space();
                auto dom = rt->get_index_space_domain(ctx, is);
                Rect<1> r(dom.bounds<1,coord_t>());
                return r == Rect<1>(0, TABLE0_NUM_Y - 1);
              }));
          recorder.expect_true(
            "Partially reindexed 'Y' column has expected values",
            testing::TestEval(
              [&cy, &ctx, rt]() {
                RegionRequirement
                  req(cy.vreq.region, READ_ONLY, EXCLUSIVE, cy.vreq.region);
                req.add_field(cy.fid);
                PhysicalRegion pr = rt->map_region(ctx, req);
                const FieldAccessor<
                  READ_ONLY, unsigned, 1, coord_t,
                  AffineAccessor<unsigned, 1, coord_t>, true>
                  y(pr, cy.fid);
                bool result =
                  y[0] == OY && y[1] == OY + 1 && y[2] == OY + 2;
                rt->unmap_region(ctx, pr);
                return result;
              }));
        }
        {
          recorder.assert_true(
            "Partially reindexed table has 'X' column",
            TE(cols.count("X") > 0));

          auto& cx = cols.at("X");
          {
            RegionRequirement
              req(cx.csp.metadata_lr, READ_ONLY, EXCLUSIVE, cx.csp.metadata_lr);
            req.add_field(x::ColumnSpace::AXIS_VECTOR_FID);
            req.add_field(x::ColumnSpace::AXIS_SET_UID_FID);
            req.add_field(x::ColumnSpace::INDEX_FLAG_FID);
            auto pr = rt->map_region(ctx, req);
            const x::ColumnSpace::AxisVectorAccessor<READ_ONLY>
              av(pr, x::ColumnSpace::AXIS_VECTOR_FID);
            const x::ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
              auid(pr, x::ColumnSpace::AXIS_SET_UID_FID);
            const x::ColumnSpace::IndexFlagAccessor<READ_ONLY>
              ifl(pr, x::ColumnSpace::INDEX_FLAG_FID);

            recorder.expect_true(
              "Partially reindexed 'X' column has unchanged axis set uid",
              TE(auid[0] == std::string(Axes<Table0Axes>::uid)));
            recorder.expect_true(
              "Partially reindexed 'X' column has ('Y', 'ROW') axes",
              TE(x::ColumnSpace::from_axis_vector(av[0])
                 == map_to_int(
                   std::vector<Table0Axes>{Table0Axes::Y, Table0Axes::ROW})));
            recorder.expect_true(
              "Partially reindexed 'X' column does not have index flag set",
              TE(!ifl[0]));
            rt->unmap_region(ctx, pr);
          }
          recorder.assert_true(
            "Partially reindexed 'X' column has expected size",
            testing::TestEval(
              [&cx, &ctx, rt]() {
                auto is = cx.vreq.region.get_index_space();
                auto dom = rt->get_index_space_domain(ctx, is);
                Rect<2> r(dom.bounds<2,coord_t>());
                return r == Rect<2>(
                  Point<2>(0, 0),
                  Point<2>(TABLE0_NUM_Y - 1, TABLE0_NUM_X - 1));
              }));
          recorder.expect_true(
            "Partially reindexed 'X' column has expected values",
            testing::TestEval(
              [&cx, &ctx, rt]() {
                RegionRequirement
                  req(cx.vreq.region, READ_ONLY, EXCLUSIVE, cx.vreq.region);
                req.add_field(cx.fid);
                PhysicalRegion pr = rt->map_region(ctx, req);
                const FieldAccessor<
                  READ_ONLY, unsigned, 2, coord_t,
                  AffineAccessor<unsigned, 2, coord_t>, true>
                  x(pr, cx.fid);
                bool result = true;
                DomainT<2,coord_t> dom(pr);
                for (PointInDomainIterator<2> pid(dom); pid(); pid++)
                  result = result && x[*pid] == OX + pid[1];
                rt->unmap_region(ctx, pr);
                return result;
              }));
        }
        {
          recorder.assert_true(
            "Partially reindexed table has 'Z' column",
            TE(cols.count("Z") > 0));

          auto& cz = cols.at("Z");
          {
            RegionRequirement
              req(cz.csp.metadata_lr, READ_ONLY, EXCLUSIVE, cz.csp.metadata_lr);
            req.add_field(x::ColumnSpace::AXIS_VECTOR_FID);
            req.add_field(x::ColumnSpace::AXIS_SET_UID_FID);
            req.add_field(x::ColumnSpace::INDEX_FLAG_FID);
            auto pr = rt->map_region(ctx, req);
            const x::ColumnSpace::AxisVectorAccessor<READ_ONLY>
              av(pr, x::ColumnSpace::AXIS_VECTOR_FID);
            const x::ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
              auid(pr, x::ColumnSpace::AXIS_SET_UID_FID);
            const x::ColumnSpace::IndexFlagAccessor<READ_ONLY>
              ifl(pr, x::ColumnSpace::INDEX_FLAG_FID);

            recorder.expect_true(
              "Partially reindexed 'Z' column has unchanged axis set uid",
              TE(auid[0] == std::string(Axes<Table0Axes>::uid)));
            recorder.expect_true(
              "Partially reindexed 'Z' column has only ('Y', 'ROW') axes",
              TE(x::ColumnSpace::from_axis_vector(av[0])
                 == map_to_int(
                   std::vector<Table0Axes>{Table0Axes::Y, Table0Axes::ROW})));
            recorder.expect_true(
              "Partially reindexed 'Z' column does not have index flag set",
              TE(!ifl[0]));
            rt->unmap_region(ctx, pr);
          }
          recorder.assert_true(
            "Partially reindexed 'Z' column has expected size",
            testing::TestEval(
              [&cz, &ctx, rt]() {
                auto is = cz.vreq.region.get_index_space();
                auto dom = rt->get_index_space_domain(ctx, is);
                Rect<2> r(dom.bounds<2,coord_t>());
                return
                  r ==
                  Rect<2>(
                    Point<2>(0, 0),
                    Point<2>(TABLE0_NUM_Y - 1, TABLE0_NUM_X - 1));
              }));
          recorder.expect_true(
            "Partially reindexed 'Z' column has expected values",
            testing::TestEval(
              [&cz, &ctx, rt]() {
                RegionRequirement
                  req(cz.vreq.region, READ_ONLY, EXCLUSIVE, cz.vreq.region);
                req.add_field(cz.fid);
                PhysicalRegion pr = rt->map_region(ctx, req);
                const FieldAccessor<
                  READ_ONLY, unsigned, 2, coord_t,
                  AffineAccessor<unsigned, 2, coord_t>, true>
                  z(pr, cz.fid);
                bool result = true;
                DomainT<2,coord_t> dom(pr);
                for (PointInDomainIterator<2> pid(dom); pid(); pid++)
                  result = result &&
                    z[*pid] == table0_z[pid[1] * TABLE0_NUM_Y + pid[0]];
                rt->unmap_region(ctx, pr);
                return result;
              }));
        }
      }
      // tests of further reindexing of partially indexed table
      x::Table tbyx;
      {
        auto f =
          tby.reindexed(
            ctx,
            rt,
            std::vector<Table0Axes>{Table0Axes::Y, Table0Axes::X},
            false);
        tbyx = f.template get_result<x::Table::reindexed_result_t>();
      }
      test_totally_reindexed_table(
        ctx,
        rt,
        tbyx,
        false,
        "Final, totally",
        recorder);
      //tbyx.destroy(ctx, rt); FIXME
      //tby.destroy(ctx, rt); FIXME
    }
  }
  //table0.destroy(ctx, rt); FIXME
}

int
main(int argc, char* argv[]) {

  AxesRegistrar::register_axes<Table0Axes>();

  testing::TestSuiteDriver driver =
    testing::TestSuiteDriver::make<reindexed_test_suite>(
      REINDEXED_TEST_SUITE,
      "reindexed_test_suite");

  return driver.start(argc, argv);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
