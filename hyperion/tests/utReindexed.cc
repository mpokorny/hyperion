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
#include <hyperion/testing/TestSuiteDriver.h>
#include <hyperion/testing/TestRecorder.h>

#include <hyperion/utility.h>
#include <hyperion/Table.h>
#include <hyperion/Column.h>
#include <hyperion/ColumnSpace.h>

#include <algorithm>
#include <functional>
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
  [[maybe_unused]] herr_t err = H5Tenum_insert(result, "ROW", &a);
  assert(err >= 0);
  a = Table0Axes::X;
  err = H5Tenum_insert(result, "X", &a);
  assert(err >= 0);
  a = Table0Axes::Y;
  err = H5Tenum_insert(result, "Y", &a);
  assert(err >= 0);
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

PhysicalRegion
attach_table0_col(Context ctx, Runtime* rt, const Column& col, unsigned *base) {

  const Memory local_sysmem =
    Machine::MemoryQuery(Machine::get_machine())
    .has_affinity_to(rt->get_executing_processor(ctx))
    .only_kind(Memory::SYSTEM_MEM)
    .first();

  AttachLauncher task(EXTERNAL_INSTANCE, col.region, col.region);
  task.attach_array_soa(base, false, {col.fid}, local_sysmem);
  PhysicalRegion result = rt->attach_external_resource(ctx, task);
  AcquireLauncher acq(col.region, col.region, result);
  acq.add_field(col.fid);
  rt->issue_acquire(ctx, acq);
  return result;
}

#if HAVE_CXX17
#define TE(f) testing::TestEval([&](){ return f; }, #f)
#else
#define TE(f) testing::TestEval<std::function<bool()>>([&](){ return f; }, #f)
#endif

void
test_totally_reindexed_table(
  Context ctx,
  Runtime* rt,
  const Table& tb,
  bool x_before_y,
  const std::string& prefix,
  testing::TestRecorder<READ_WRITE>& recorder) {

  auto ics = tb.index_column_space(ctx, rt);
  std::vector<Table0Axes> ixax;
  if (x_before_y)
    ixax = {Table0Axes::X, Table0Axes::Y};
  else
    ixax = {Table0Axes::Y, Table0Axes::X};

  std::ostringstream oss;
  oss << ixax;

  recorder.expect_true(
    prefix + " reindexed table has " + oss.str() + " index axes",
    testing::TestEval<std::function<bool()>>(
      [&ctx, rt, &ics, &ixax]() {
        auto axes = ics.axes(ctx, rt);
        return axes == map_to_int(ixax);
      }));

  auto cols = tb.columns();
  {
    recorder.assert_true(
      prefix + " reindexed table has 'X' column",
      TE(cols.count("X") > 0));

    auto& cx = cols.at("X");
    {
      RegionRequirement
        req(cx.cs.metadata_lr, READ_ONLY, EXCLUSIVE, cx.cs.metadata_lr);
      req.add_field(ColumnSpace::AXIS_VECTOR_FID);
      req.add_field(ColumnSpace::AXIS_SET_UID_FID);
      req.add_field(ColumnSpace::INDEX_FLAG_FID);
      auto pr = rt->map_region(ctx, req);
      const ColumnSpace::AxisVectorAccessor<READ_ONLY>
        av(pr, ColumnSpace::AXIS_VECTOR_FID);
      const ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
        auid(pr, ColumnSpace::AXIS_SET_UID_FID);
      const ColumnSpace::IndexFlagAccessor<READ_ONLY>
        ifl(pr, ColumnSpace::INDEX_FLAG_FID);

      recorder.expect_true(
        prefix + " reindexed 'X' column has unchanged axis set uid",
        TE(auid[0] == std::string(Axes<Table0Axes>::uid)));
      recorder.expect_true(
        prefix + " reindexed 'X' column has only 'X' axis",
        TE(ColumnSpace::from_axis_vector(av[0])
           == map_to_int(std::vector<Table0Axes>{Table0Axes::X})));
      recorder.expect_true(
        prefix + " reindexed 'X' column has index flag set",
        TE(ifl[0]));
      rt->unmap_region(ctx, pr);
    }
    recorder.assert_true(
      prefix + " reindexed 'X' column has expected size",
      testing::TestEval<std::function<bool()>>(
        [&cx, &ctx, rt]() {
          auto is = cx.region.get_index_space();
          auto dom = rt->get_index_space_domain(ctx, is);
          Rect<1> r(dom.bounds<1,coord_t>());
          return r == Rect<1>(0, TABLE0_NUM_X - 1);
        }));
    recorder.expect_true(
      prefix + " reindexed 'X' column has expected values",
      testing::TestEval<std::function<bool()>>(
        [&cx, &ctx, rt]() {
          RegionRequirement req(cx.region, READ_ONLY, EXCLUSIVE, cx.region);
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
        req(cy.cs.metadata_lr, READ_ONLY, EXCLUSIVE, cy.cs.metadata_lr);
      req.add_field(ColumnSpace::AXIS_VECTOR_FID);
      req.add_field(ColumnSpace::AXIS_SET_UID_FID);
      req.add_field(ColumnSpace::INDEX_FLAG_FID);
      auto pr = rt->map_region(ctx, req);
      const ColumnSpace::AxisVectorAccessor<READ_ONLY>
        av(pr, ColumnSpace::AXIS_VECTOR_FID);
      const ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
        auid(pr, ColumnSpace::AXIS_SET_UID_FID);
      const ColumnSpace::IndexFlagAccessor<READ_ONLY>
        ifl(pr, ColumnSpace::INDEX_FLAG_FID);

      recorder.expect_true(
        prefix + " reindexed 'Y' column has unchanged axis set uid",
        TE(auid[0] == std::string(Axes<Table0Axes>::uid)));
      recorder.expect_true(
        prefix + " reindexed 'Y' column has only 'Y' axis",
        TE(ColumnSpace::from_axis_vector(av[0])
           == map_to_int(std::vector<Table0Axes>{Table0Axes::Y})));
      recorder.expect_true(
        prefix + " reindexed 'Y' column has index flag set",
        TE(ifl[0]));
      rt->unmap_region(ctx, pr);
    }
    recorder.assert_true(
      prefix + " reindexed 'Y' column has expected size",
      testing::TestEval<std::function<bool()>>(
        [&cy, &ctx, rt]() {
          auto is = cy.region.get_index_space();
          auto dom = rt->get_index_space_domain(ctx, is);
          Rect<1> r(dom.bounds<1,coord_t>());
          return r == Rect<1>(0, TABLE0_NUM_Y - 1);
        }));
    recorder.expect_true(
      prefix + " reindexed 'Y' column has expected values",
      testing::TestEval<std::function<bool()>>(
        [&cy, &ctx, rt]() {
          RegionRequirement req(cy.region, READ_ONLY, EXCLUSIVE, cy.region);
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
        req(cz.cs.metadata_lr, READ_ONLY, EXCLUSIVE, cz.cs.metadata_lr);
      req.add_field(ColumnSpace::AXIS_VECTOR_FID);
      req.add_field(ColumnSpace::AXIS_SET_UID_FID);
      req.add_field(ColumnSpace::INDEX_FLAG_FID);
      auto pr = rt->map_region(ctx, req);
      const ColumnSpace::AxisVectorAccessor<READ_ONLY>
        av(pr, ColumnSpace::AXIS_VECTOR_FID);
      const ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
        auid(pr, ColumnSpace::AXIS_SET_UID_FID);
      const ColumnSpace::IndexFlagAccessor<READ_ONLY>
        ifl(pr, ColumnSpace::INDEX_FLAG_FID);

      recorder.expect_true(
        prefix + " reindexed 'Z' column has unchanged axis set uid",
        TE(auid[0] == std::string(Axes<Table0Axes>::uid)));
      recorder.expect_true(
        prefix + " reindexed 'Z' column has only " + oss.str() + " axes",
        TE(ColumnSpace::from_axis_vector(av[0]) == map_to_int(ixax)));
      recorder.expect_true(
        prefix + " reindexed 'Z' column does not have index flag set",
        TE(!ifl[0]));
      rt->unmap_region(ctx, pr);
    }
    recorder.assert_true(
      prefix + " reindexed 'Z' column has expected size",
      testing::TestEval<std::function<bool()>>(
        [&cz, &x_before_y, &ctx, rt]() {
          auto is = cz.region.get_index_space();
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
          return r == expected;
        }));
    recorder.expect_true(
      prefix + " reindexed 'Z' column has expected values",
      testing::TestEval<std::function<bool()>>(
        [&cz, &x_before_y, &ctx, rt]() {
          RegionRequirement req(cz.region, READ_ONLY, EXCLUSIVE, cz.region);
          req.add_field(cz.fid);
          PhysicalRegion pr = rt->map_region(ctx, req);
          const FieldAccessor<
            READ_ONLY, unsigned, 2, coord_t,
            AffineAccessor<unsigned, 2, coord_t>, true>
            z(pr, cz.fid);
          bool result = true;
          DomainT<2,coord_t> dom =
            rt->get_index_space_domain(cz.region.get_index_space());
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
    ColumnSpace::create(
      ctx,
      rt,
      std::vector<Table0Axes>{Table0Axes::ROW},
      xyz_is,
      false);

//#ifdef HYPERION_USE_CASACORE
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
//#endif
  std::vector<std::pair<std::string, TableField>> xyz_fields{
    {"X", TableField(HYPERION_TYPE_UINT, COL_X)},
    {"Y", TableField(HYPERION_TYPE_UINT, COL_Y)},
    {"Z", TableField(HYPERION_TYPE_UINT, COL_Z)}
  };

  auto table0 = Table::create(ctx, rt, xyz_space, {{xyz_space, xyz_fields}});
  {
    std::unordered_map<std::string, PhysicalRegion> col_prs;
    {
      auto cols = table0.columns();
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
      Future f =
        table0.reindexed(
          ctx,
          rt,
          std::vector<Table0Axes>{Table0Axes::X, Table0Axes::Y},
          false);

      auto tb = f.get_result<Table>();
      test_totally_reindexed_table(ctx, rt, tb, true, "Totally", recorder);
      //tb.destroy(ctx, rt); FIXME
    }
    // tests of two-step reindexing
    {
      Table tby;
      {
        Future f =
          table0.reindexed(
            ctx,
            rt,
            std::vector<Table0Axes>{Table0Axes::Y},
            true);
        tby = f.get_result<Table>();
      }
      {
        auto ics_y = tby.index_column_space(ctx, rt);

        recorder.expect_true(
          "Partially reindexed table has ('Y', 'ROW') index axes",
          testing::TestEval<std::function<bool()>>(
            [&ctx, rt, &ics_y]() {
              auto ax = ics_y.axes(ctx, rt);
              return ax ==
                map_to_int(
                  std::vector<Table0Axes>{Table0Axes::Y, Table0Axes::ROW});
            }));

        auto cols = tby.columns();
        {
          recorder.assert_true(
            "Partially reindexed table has 'Y' column",
            TE(cols.count("Y") > 0));

          auto& cy = cols.at("Y");
          {
            RegionRequirement
              req(cy.cs.metadata_lr, READ_ONLY, EXCLUSIVE, cy.cs.metadata_lr);
            req.add_field(ColumnSpace::AXIS_VECTOR_FID);
            req.add_field(ColumnSpace::AXIS_SET_UID_FID);
            req.add_field(ColumnSpace::INDEX_FLAG_FID);
            auto pr = rt->map_region(ctx, req);
            const ColumnSpace::AxisVectorAccessor<READ_ONLY>
              av(pr, ColumnSpace::AXIS_VECTOR_FID);
            const ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
              auid(pr, ColumnSpace::AXIS_SET_UID_FID);
            const ColumnSpace::IndexFlagAccessor<READ_ONLY>
              ifl(pr, ColumnSpace::INDEX_FLAG_FID);

            recorder.expect_true(
              "Partially reindexed 'Y' column has unchanged axis set uid",
              TE(auid[0] == std::string(Axes<Table0Axes>::uid)));
            recorder.expect_true(
              "Partially reindexed 'Y' column has only 'Y' axis",
              TE(ColumnSpace::from_axis_vector(av[0])
                 == map_to_int(std::vector<Table0Axes>{Table0Axes::Y})));
            recorder.expect_true(
              "Partially reindexed 'Y' column has index flag set",
              TE(ifl[0]));
            rt->unmap_region(ctx, pr);
          }
          recorder.assert_true(
            "Partially reindexed 'Y' column has expected size",
            testing::TestEval<std::function<bool()>>(
              [&cy, &ctx, rt]() {
                auto is = cy.region.get_index_space();
                auto dom = rt->get_index_space_domain(ctx, is);
                Rect<1> r(dom.bounds<1,coord_t>());
                return r == Rect<1>(0, TABLE0_NUM_Y - 1);
              }));
          recorder.expect_true(
            "Partially reindexed 'Y' column has expected values",
            testing::TestEval<std::function<bool()>>(
              [&cy, &ctx, rt]() {
                RegionRequirement
                  req(cy.region, READ_ONLY, EXCLUSIVE, cy.region);
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
              req(cx.cs.metadata_lr, READ_ONLY, EXCLUSIVE, cx.cs.metadata_lr);
            req.add_field(ColumnSpace::AXIS_VECTOR_FID);
            req.add_field(ColumnSpace::AXIS_SET_UID_FID);
            req.add_field(ColumnSpace::INDEX_FLAG_FID);
            auto pr = rt->map_region(ctx, req);
            const ColumnSpace::AxisVectorAccessor<READ_ONLY>
              av(pr, ColumnSpace::AXIS_VECTOR_FID);
            const ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
              auid(pr, ColumnSpace::AXIS_SET_UID_FID);
            const ColumnSpace::IndexFlagAccessor<READ_ONLY>
              ifl(pr, ColumnSpace::INDEX_FLAG_FID);

            recorder.expect_true(
               "Partially reindexed 'X' column has unchanged axis set uid",
              TE(auid[0] == std::string(Axes<Table0Axes>::uid)));
            recorder.expect_true(
              "Partially reindexed 'X' column has ('Y', 'ROW') axes",
              TE(ColumnSpace::from_axis_vector(av[0])
                 == map_to_int(
                   std::vector<Table0Axes>{Table0Axes::Y, Table0Axes::ROW})));
            recorder.expect_true(
              "Partially reindexed 'X' column does not have index flag set",
              TE(!ifl[0]));
            rt->unmap_region(ctx, pr);
          }
          recorder.assert_true(
            "Partially reindexed 'X' column has expected size",
            testing::TestEval<std::function<bool()>>(
              [&cx, &ctx, rt]() {
                auto is = cx.region.get_index_space();
                auto dom = rt->get_index_space_domain(ctx, is);
                Rect<2> r(dom.bounds<2,coord_t>());
                return r == Rect<2>(
                  Point<2>(0, 0),
                  Point<2>(TABLE0_NUM_Y - 1, TABLE0_NUM_X - 1));
              }));
          recorder.expect_true(
            "Partially reindexed 'X' column has expected values",
            testing::TestEval<std::function<bool()>>(
              [&cx, &ctx, rt]() {
                RegionRequirement
                  req(cx.region, READ_ONLY, EXCLUSIVE, cx.region);
                req.add_field(cx.fid);
                PhysicalRegion pr = rt->map_region(ctx, req);
                const FieldAccessor<
                  READ_ONLY, unsigned, 2, coord_t,
                  AffineAccessor<unsigned, 2, coord_t>, true>
                  x(pr, cx.fid);
                bool result = true;
                DomainT<2,coord_t> dom =
                  rt->get_index_space_domain(cx.region.get_index_space());
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
              req(cz.cs.metadata_lr, READ_ONLY, EXCLUSIVE, cz.cs.metadata_lr);
            req.add_field(ColumnSpace::AXIS_VECTOR_FID);
            req.add_field(ColumnSpace::AXIS_SET_UID_FID);
            req.add_field(ColumnSpace::INDEX_FLAG_FID);
            auto pr = rt->map_region(ctx, req);
            const ColumnSpace::AxisVectorAccessor<READ_ONLY>
              av(pr, ColumnSpace::AXIS_VECTOR_FID);
            const ColumnSpace::AxisSetUIDAccessor<READ_ONLY>
              auid(pr, ColumnSpace::AXIS_SET_UID_FID);
            const ColumnSpace::IndexFlagAccessor<READ_ONLY>
              ifl(pr, ColumnSpace::INDEX_FLAG_FID);

            recorder.expect_true(
              "Partially reindexed 'Z' column has unchanged axis set uid",
              TE(auid[0] == std::string(Axes<Table0Axes>::uid)));
            recorder.expect_true(
              "Partially reindexed 'Z' column has only ('Y', 'ROW') axes",
              TE(ColumnSpace::from_axis_vector(av[0])
                 == map_to_int(
                   std::vector<Table0Axes>{Table0Axes::Y, Table0Axes::ROW})));
            recorder.expect_true(
              "Partially reindexed 'Z' column does not have index flag set",
              TE(!ifl[0]));
            rt->unmap_region(ctx, pr);
          }
          recorder.assert_true(
            "Partially reindexed 'Z' column has expected size",
            testing::TestEval<std::function<bool()>>(
              [&cz, &ctx, rt]() {
                auto is = cz.region.get_index_space();
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
            testing::TestEval<std::function<bool()>>(
              [&cz, &ctx, rt]() {
                RegionRequirement
                  req(cz.region, READ_ONLY, EXCLUSIVE, cz.region);
                req.add_field(cz.fid);
                PhysicalRegion pr = rt->map_region(ctx, req);
                const FieldAccessor<
                  READ_ONLY, unsigned, 2, coord_t,
                  AffineAccessor<unsigned, 2, coord_t>, true>
                  z(pr, cz.fid);
                bool result = true;
                DomainT<2,coord_t> dom =
                  rt->get_index_space_domain(cz.region.get_index_space());
                for (PointInDomainIterator<2> pid(dom); pid(); pid++)
                  result = result &&
                    z[*pid] == table0_z[pid[1] * TABLE0_NUM_Y + pid[0]];
                rt->unmap_region(ctx, pr);
                return result;
              }));
        }
      }
      // tests of further reindexing of partially indexed table
      Table tbyx;
      {
        Future f =
          tby.reindexed(
            ctx,
            rt,
            std::vector<Table0Axes>{Table0Axes::Y, Table0Axes::X},
            false);
        tbyx = f.get_result<Table>();
      }
      test_totally_reindexed_table(
        ctx,
        rt,
        tbyx,
        false,
        "Final, totally",
        recorder);
      tbyx.destroy(ctx, rt);
      tby.destroy(ctx, rt);
    }
  }
  table0.destroy(ctx, rt);
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
