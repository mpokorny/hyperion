#include <legms/legms.h>
#include <legms/TableBuilder.h>

using namespace legms;

void
legms::initialize_keywords_from_ms(
  Legion::Context ctx,
  Legion::Runtime* rt,
  const LEGMS_FS::path& path,
  Table& table) {

  casacore::Table cctable(
    casacore::String((path.filename() == "MAIN") ? path.parent_path() : path),
    casacore::TableLock::PermanentLockingWait);
  auto kws = cctable.keywordSet();
  auto keys = table.keywords.keys(rt);
  std::vector<Legion::FieldID> fids(keys.size());
  std::iota(fids.begin(), fids.end(), 0);
  auto reqs = table.keywords.requirements(rt, fids, WRITE_ONLY);
  auto prs =
    reqs.value().map(
      [&ctx, rt](const Legion::RegionRequirement& r){
        return rt->map_region(ctx, r);
      });
  for (size_t i = 0; i < fids.size(); ++i) {
    casacore::RecordFieldId f = kws.fieldNumber(keys[i]);
    switch (kws.dataType(f)) {
#define WRITE_KWVAL(DT)                                               \
      case (DataType<DT>::CasacoreTypeTag): {                         \
        DataType<DT>::CasacoreType cv;                                \
        kws.get(f, cv);                                               \
        DataType<DT>::ValueType vv;                                   \
        DataType<DT>::from_casacore(vv, cv);                          \
        table.keywords.template write<WRITE_ONLY>(prs, fids[i], vv);  \
        break;                                                        \
      }
      LEGMS_FOREACH_RECORD_DATATYPE(WRITE_KWVAL)
#undef WRITE_KWVAL
    default:
        assert(false);
      break;
    }
  }
  prs.map(
    [&ctx, rt](const Legion::PhysicalRegion& p) {
      rt->unmap_region(ctx, p);
      return 0;
    });

}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
