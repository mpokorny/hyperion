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
#include <hyperion/Measures.h>
#include <hyperion/MeasRef.h>

#include <casacore/measures/TableMeasures.h>

using namespace hyperion;

#define MCLASS_NAME(M) \
  const std::string hyperion::MClassT<M>::name = MClassT<M>::type::showMe();
HYPERION_FOREACH_MCLASS(MCLASS_NAME)
#undef MCLASS_NAME

CXX_OPTIONAL_NAMESPACE::optional<
  std::tuple<
    hyperion::MClass,
    std::vector<std::tuple<std::unique_ptr<casacore::MRBase>, unsigned>>,
    CXX_OPTIONAL_NAMESPACE::optional<std::string>>>
hyperion::get_meas_refs(
  const casacore::Table& table,
  const std::string& colname) {

  CXX_OPTIONAL_NAMESPACE::optional<
    std::tuple<
      MClass,
      std::vector<std::tuple<std::unique_ptr<casacore::MRBase>, unsigned>>,
      CXX_OPTIONAL_NAMESPACE::optional<std::string>>>
    result;

  CXX_OPTIONAL_NAMESPACE::optional<unsigned> measinfo_index;
  auto tcol = casacore::TableColumn(table, colname);
  auto kws = tcol.keywordSet();
  auto nf = kws.nfields();
  for (unsigned f = 0; !measinfo_index && f < nf; ++f) {
    std::string name = kws.name(f);
    auto dt = kws.dataType(f);
    if (name == "MEASINFO" && dt == casacore::DataType::TpRecord)
      measinfo_index = f;
  }
  if (measinfo_index) {
    result =
      std::make_tuple(
        M_NONE,
        std::vector<std::tuple<std::unique_ptr<casacore::MRBase>, unsigned>>(),
        CXX_OPTIONAL_NAMESPACE::optional<std::string>());
#if __cplusplus >= 201703L
    auto& [mc, mrbs, refcol] = result.value();
#else // !c++17
    auto& mc = std::get<0>(result.value());
    auto& mrbs = std::get<1>(result.value());
    auto& refcol = std::get<2>(result.value());
#endif // c++17
    casacore::TableMeasColumn tmc(table, colname);
    const casacore::TableMeasDescBase& tmd = tmc.measDesc();
    if (!tmd.isRefCodeVariable() && !tmd.isOffsetVariable()) {
      std::string mtype = tmd.type();
      // I'm not sure of how, exactly, one can decide whether a call to
      // casacore::ScalarMeasColumn or casacore::ArrayMeasColumn is required.
      // Simply relying on tmc.isScalar() does not always work (c.f, test MS
      // t0.ms, table OBSERVATION, column TIME_RANGE). The following definition
      // of GET_MR seems hacky, but it's the best solution I can come up with
      // currently. The alternative definition, ALT_GET_MR, looks better, and
      // although it doesn't work correctly, I'm leaving it here for reference
      // for future work.
      if (false) {}
#define GET_MR(MC)                                                    \
      else if (MClassT<MC>::name == mtype) {                          \
        mc = MC;                                                      \
        std::unique_ptr<casacore::MeasRef<MClassT<MC>::type>> mrb;    \
        try {                                                         \
          casacore::ScalarMeasColumn<MClassT<MC>::type>               \
            smc(table, colname);                                      \
          mrb =                                                       \
            std::make_unique<casacore::MeasRef<MClassT<MC>::type>>(   \
              smc.getMeasRef());                                      \
        } catch (const casacore::TableInvDT&) {                       \
          casacore::ArrayMeasColumn<MClassT<MC>::type>                \
            amc(table, colname);                                      \
          mrb =                                                       \
            std::make_unique<casacore::MeasRef<MClassT<MC>::type>>(   \
              amc.getMeasRef());                                      \
        }                                                             \
        mrbs.emplace_back(std::move(mrb), tmd.getRefCode());          \
      }
      HYPERION_FOREACH_MCLASS(GET_MR)
#undef GET_MR
#define ALT_GET_MR(MC)                                                \
      else if (MClassT<MC>::name == mtype) {                          \
        mc = MC;                                                      \
        std::unique_ptr<casacore::MeasRef<MClassT<MC>::type>> mrb;    \
        if (tmc.isScalar()) {                                         \
          casacore::ScalarMeasColumn<MClassT<MC>::type>               \
            smc(table, colname);                                      \
          mrb =                                                       \
            std::make_unique<casacore::MeasRef<MClassT<MC>::type>>(   \
              smc.getMeasRef());                                      \
        } else {                                                      \
          casacore::ArrayMeasColumn<MClassT<MC>::type>                \
            amc(table, colname);                                      \
          mrb =                                                       \
            std::make_unique<casacore::MeasRef<MClassT<MC>::type>>(   \
              amc.getMeasRef());                                      \
        }                                                             \
        mrbs.emplace_back(mrb, tmd.getRefCode());                     \
      }
      //HYPERION_FOREACH_MCLASS(ALT_GET_MR)
#undef ALT_GET_MR
      else { assert(false); }
    } else {
      auto mi = casacore::Record(kws.asRecord(measinfo_index.value()));
      if (mi.fieldNumber("RefOff") >= 0) {
        std::cerr << "Row measures with variable offsets are unsupported"
                  << std::endl;
        assert(false);
      }
      std::vector<unsigned> i2c;
      auto vrc = mi.fieldNumber("VarRefCol");
      if (vrc >= 0) {
        std::string mtype = mi.asString("type");
        mtype[0] = std::toupper(mtype[0]); // FIXME: case-insensitive?
        refcol = mi.asString(vrc);
        auto vrt = mi.fieldNumber("TabRefTypes");
        if (table.tableDesc().columnDesc(refcol.value()).dataType()
            == casacore::DataType::TpInt
            && vrt >= 0) {
          auto cs = mi.toArrayuInt("TabRefCodes").tovector();
          i2c.reserve(cs.size());
          for (size_t i = 0; i < cs.size(); ++i)
            i2c.emplace_back(cs[i]);
        } else {
          // support for string-valued measure reference columns isn't hard to
          // add, but leave it unimplemented for now
          std::cerr << "String-valued measure reference columns not supported"
                    << std::endl;
          assert(false);
        }
        if (false) {}
#define GET_MRS(MC)                                                     \
        else if (MClassT<MC>::name == mtype) {                          \
          mc = MC;                                                      \
          if (i2c.size() == 0) {                                        \
            int nall, nextra;                                           \
            const unsigned* codes;                                      \
            [[maybe_unused]] auto types =                               \
              MClassT<MC>::type::allMyTypes(nall, nextra, codes);       \
            i2c.reserve(nall);                                          \
            for (int i = 0; i < nall; ++i)                              \
              i2c.emplace_back(codes[i]);                               \
          }                                                             \
          for (auto& code : i2c) {                                      \
            auto tp = MClassT<MC>::type::castType(code);                \
            mrbs.emplace_back(                                          \
              std::make_unique<casacore::MeasRef<MClassT<MC>::type>>(tp), \
              code);                                                    \
          }                                                             \
        }
        HYPERION_FOREACH_MCLASS(GET_MRS)
#undef GET_MRS
        else { assert(false); }
      }
    }
  }
  return result;
}

std::tuple<std::string, hyperion::MeasRef>
hyperion::create_named_meas_refs(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const std::tuple<
    hyperion::MClass,
    std::vector<std::tuple<casacore::MRBase*, unsigned>>>& mrs) {

  std::tuple<std::string, MeasRef> result;
#if __cplusplus >= 201703L
  auto& [mc, mrbs] = mrs;
#else // !c++17
  auto& mc = std::get<0>(mrs);
  auto& mrbs = std::get<1>(mrs);
#endif // c++17
  switch (mc) {
#define NM(MC)                                          \
    case MC:                                            \
      std::get<0>(result) = toupper(MClassT<MC>::name); \
      break;
    HYPERION_FOREACH_MCLASS(NM)
#undef NM
    default:
      assert(false);
      break;
  }
  std::vector<std::tuple<casacore::MRBase*, unsigned>> pmrbs;
  pmrbs.reserve(mrbs.size());
  for (auto& mct : mrbs) {
#if __cplusplus >= 201703L
    auto& [mrb, code] = mct;
#else // !c++17
    auto& mrb = std::get<0>(mct);
    auto& code = std::get<1>(mct);
#endif // c++17
    pmrbs.emplace_back(mrb, code);
  }
  std::get<1>(result) = MeasRef::create(ctx, rt, pmrbs, mc);
  return result;
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
