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
#include <hyperion/Measures.h>
#include <hyperion/MeasRef.h>

#include <casacore/measures/TableMeasures.h>

using namespace hyperion;

#define MCLASS_NAME(M) \
  const std::string hyperion::MClassT<M>::name = MClassT<M>::type::showMe();
HYPERION_FOREACH_MCLASS(MCLASS_NAME)
#undef MCLASS_NAME

std::tuple<hyperion::MClass, std::vector<std::unique_ptr<casacore::MRBase>>>
hyperion::get_meas_refs(
  const casacore::Table& table,
  const std::optional<std::string>& colname) {

  std::tuple<MClass, std::vector<std::unique_ptr<casacore::MRBase>>> result;
  std::get<0>(result) = M_NONE;

  if (colname) {
    bool has_measinfo = false;
    {
      auto tcol = casacore::TableColumn(table, colname.value());
      auto kws = tcol.keywordSet();
      auto nf = kws.nfields();
      for (unsigned f = 0; !has_measinfo && f < nf; ++f) {
        std::string name = kws.name(f);
        auto dt = kws.dataType(f);
        has_measinfo =
          (name == "MEASINFO" && dt == casacore::DataType::TpRecord);
      }
    }
    if (has_measinfo) {
      casacore::TableMeasColumn tmc(table, colname.value());
      const casacore::TableMeasDescBase& tmd = tmc.measDesc();
      if (!tmd.isRefCodeVariable() && !tmd.isOffsetVariable()) {
        std::string mtype = tmd.type();
        if (false) {}
#define GET_MEAS_REF(MC)                                              \
        else if (MClassT<MC>::name == mtype) {                        \
          std::get<0>(result) = MC;                                   \
          if (tmc.isScalar()) {                                       \
            casacore::ScalarMeasColumn<MClassT<MC>::type>             \
              smc(table, colname.value());                            \
            std::get<1>(result).emplace_back(                         \
              std::make_unique<casacore::MeasRef<MClassT<MC>::type>>( \
                smc.getMeasRef()));                                   \
          } else {                                                    \
            casacore::ArrayMeasColumn<MClassT<MC>::type>              \
              amc(table, colname.value());                            \
            std::get<1>(result).emplace_back(                         \
              std::make_unique<casacore::MeasRef<MClassT<MC>::type>>( \
                amc.getMeasRef()));                                   \
          }                                                           \
        }
        HYPERION_FOREACH_MCLASS(GET_MEAS_REF)
#undef GET_MEAS_REF
        else { assert(false); }
      } else {
        std::cout << colname.value() << " has variable MeasRef" << std::endl;
      }
    }
  } else {
    bool has_measinfo = false;
    auto kws = table.keywordSet();
    auto nf = kws.nfields();
    for (unsigned f = 0; !has_measinfo && f < nf; ++f) {
      std::string name = kws.name(f);
      auto dt = kws.dataType(f);
      if (name == "MEASINFO" && dt == casacore::DataType::TpRecord) {
        has_measinfo = true;
        // NB: this doesn't occur in normal MSs, but it exists in the logical
        // model of hyperion Tables, so it's nice to have, but the
        // implementation doesn't use standard casacore TableMeasures methods
        // and is thus somewhat ad hoc
        casacore::MeasureHolder mh;
        casacore::String err;
        auto converted = mh.fromRecord(err, kws.asRecord(f));
        if (converted) {
          if (false) {}
#define MK_MR(MC)                                                       \
          else if (MClassT<MC>::holds(mh)) {                            \
            MClassT<MC>::type m = MClassT<MC>::get(mh);                 \
            std::get<0>(result) = MC;                                   \
            std::get<1>(result).emplace_back(                           \
              std::make_unique<casacore::MeasRef<MClassT<MC>::type>>(   \
                m.getRef()));                                           \
          }
          HYPERION_FOREACH_MCLASS(MK_MR)
#undef MK_MR
          else { assert(false); }
        }
      }
    }
  }
  return result;
}

std::unordered_map<std::string, hyperion::MeasRef>
hyperion::create_named_meas_refs(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const std::vector<
    std::tuple<MClass, std::vector<std::unique_ptr<casacore::MRBase>>>>& mrs) {

  std::unordered_map<std::string, MeasRef> result;
  std::for_each(
    mrs.begin(),
    mrs.end(),
    [&result, &ctx, rt](auto& mc_mrbs) {
      auto& [mc, mrbs] = mc_mrbs;
      std::string nm;
      switch (mc) {
#define NM(MC)                                  \
        case MC:                                \
          nm = toupper(MClassT<MC>::name);      \
          break;
        HYPERION_FOREACH_MCLASS(NM)
#undef NM
        default:
          assert(false);
          break;
      }
      std::vector<casacore::MRBase*> pmrbs;
      pmrbs.reserve(mrbs.size());
      for (auto& mrb : mrbs)
        pmrbs.push_back(mrb.get());
      result.emplace(nm, MeasRef::create(ctx, rt, pmrbs, mc));
    });
  return result;
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
