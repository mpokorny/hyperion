#ifndef LEGMS_MS_MAIN_H_
#define LEGMS_MS_MAIN_H_

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

#include <casacore/casa/Utilities/DataType.h>
#include <casacore/tables/Tables.h>
#include "legion.h"

#include "utility.h"

namespace legms {
namespace ms {

class Table {
public:

  static Legion::IndexSpaceT<1>
  index_space(
    const casacore::Table& table,
    Legion::Context ctx,
    Legion::Runtime* runtime) {

    return runtime->create_index_space(
      ctx,
      Legion::Rect<1>(Legion::Point<1>(0), Legion::Point<1>(table.nrow() - 1)));
  }

  static std::tuple<
    Legion::FieldSpace,
    std::map<casacore::String, Legion::FieldID>>
  field_space(
    const casacore::Table& table,
    Legion::Context ctx,
    Legion::Runtime* runtime) {

    DataTypeSerdezManager::register_ops();
    auto tdesc = table.tableDesc();
    auto col_names = tdesc.columnNames();
    Legion::FieldSpace fs = runtime->create_field_space(ctx);
    auto fa = runtime->create_field_allocator(ctx, fs);
    std::vector<std::tuple<casacore::String, Legion::FieldID>> fids;
    std::transform(
      col_names.begin(),
      col_names.end(),
      std::back_inserter(fids),
      [&sdz, &tdesc, &fa](const auto& name) {
        return std::make_tuple(name, add_field(tdesc[name], sdz, fa));
      });
    std::map<casacore::String, Legion::FieldID> fidmap(
      [col_names](auto& s1, auto& s2) {
        auto end = col_names.end();
        auto i1 = std::find(col_names.begin(), end, s1);
        auto i2 = std::find(i1, end, s2);
        return i2 != end;
      });
    std::for_each(
      fids.begin(),
      fids.end(),
      [&fidmap](auto& [name, fid]) {
        fidmap[name] = fid;
      });
    return std::make_tuple(fs, fidmap);
  }

private:

  static Legion::FieldID
  add_field(
    const casacore::ColumnDesc& cdesc,
    Legion::FieldAllocator& fa) {

    // most fields are assumed to be in native data format, so there is no
    // "export" format dependency -- TODO: use format information in MS, and
    // develop serdez classes for all data types, so this function be used for
    // I/O
    std::optional<size_t> n;
    if (cdesc.is_scalar())
      n = 1;
    else if (cdesc.isFixedShape())
      n = cdesc.shape().size();

    switch (cdesc.dataType()) {
    case casacore::DataType::TpBool:
      if (n)
        return fa.allocate_field(n.value() * sizeof(casacore::Bool));
      else
        return fa.allocate_field(
          sizeof(std::vector<casacore::Bool>),
          DataTypeSerdezManager::CASACORE_BOOL_V_SID);
      break;

    case casacore::DataType::TpChar:
    case casacore::DataType::TpUChar:
      if (n)
        return fa.allocate_field(n.value() * sizeof(casacore::Char));
      else
        return fa.allocate_field(
          sizeof(std::vector<casacore::Char),
          DataTypeSerdezManager::CASACORE_CHAR_V_SID);
      break;

    case casacore::DataType::TpShort:
    case casacore::DataType::TpUShort:
      if (n)
        return fa.allocate_field(n.value() * sizeof(casacore::Short));
      else
        return fa.allocate_field(
          sizeof(std::vector<casacore::Short>),
          DataTypeSerdezManager::CASACORE_SHORT_V_SID);
      break;

    case casacore::DataType::TpInt:
    case casacore::DataType::TpUInt:
      if (n)
        return fa.allocate_field(n.value() * sizeof(casacore::Int));
      else
        return fa.allocate_field(
          sizeof(std::vector<casacore::Int>),
          DataTypeSerdezManager::CASACORE_INT_V_SID);
      break;

    case casacore::DataType::TpInt64:
      //case casacore::DataType::TpUInt64:
      if (n)
        return fa.allocate_field(n.value() * sizeof(casacore::Int64));
      else
        return fa.allocate_field(
          sizeof(std::vector<casacore::Int64>),
          DataTypeSerdezManager::CASACORE_INT64_V_SID);
      break;

    case casacore::DataType::TpFloat:
      if (n)
        return fa.allocate_field(n.value() * sizeof(casacore::Float));
      else
        return fa.allocate_field(
          sizeof(std::vector<casacore::Float>),
          DataTypeSerdezManager::CASACORE_FLOAT_V_SID);
      break;

    case casacore::DataType::TpDouble:
      if (n)
        return fa.allocate_field(n.value() * sizeof(casacore::Double));
      else
        return fa.allocate_field(
          sizeof(std::vector<casacore::Double>),
          DataTypeSerdezManager::CASACORE_DOUBLE_V_SID);
      break;

    case casacore::DataType::TpComplex:
      if (n)
        return fa.allocate_field(n.value() * sizeof(casacore::Complex));
      else
        return fa.allocate_field(
          sizeof(std::vector<casacore::Complex>),
          DataTypeSerdezManager::CASACORE_COMPLEX_V_SID);
      break;

    case casacore::DataType::TpDComplex:
      if (n)
        return fa.allocate_field(n.value() * sizeof(casacore::DComplex));
      else
        return fa.allocate_field(
          sizeof(std::vector<cascore::DComplex>),
          DataTypeSerdezManager::CASACORE_DCOMPLEX_V_SID);
      break;

    case casacore::DataType::TpString:
      if (n == 1)
        return fa.allocate_field(
          sizeof(casacore::String),
          DataTypeSerdezManager::CASACORE_STRING_SID);
      return fa.allocate_field(
        sizeof(std::vector<casacore::String>),
        DataTypeSerdezManager::CASACORE_STRING_ARRAY_SID);
      break;

    case casacore::DataType::TpQuantity:
      assert(false); // TODO: implement quantity-valued columns
      break;

    case casacore::DataType::TpRecord:
      assert(false); // TODO: implement record-valued columns
      break;

    case casacore::DataType::TpTable:
      assert(false); // TODO: implement table-valued columns
      break;

    default:
      assert(false);
      break;
    }
  }
};

} // end namespace ms
} // end namespace legms

#endif // LEGMS_MS_MAIN_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
