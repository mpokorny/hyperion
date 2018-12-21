#include "utility.h"

using namespace legms;
using namespace Legion;

std::once_flag SerdezManager::initialized;

FieldID
legms::add_field(
  casacore::DataType datatype,
  FieldAllocator fa,
  FieldID field_id) {

  FieldID result;

#define ALLOC_FLD(tp)                           \
  tp:                                           \
    result = fa.allocate_field(                 \
      sizeof(DataType<tp>::ValueType),          \
      field_id,                                 \
      DataType<tp>::serdez_id);

  switch (datatype) {

  case ALLOC_FLD(casacore::DataType::TpBool)
    break;

  case ALLOC_FLD(casacore::DataType::TpArrayBool)
    break;

  case ALLOC_FLD(casacore::DataType::TpChar)
    break;

  case ALLOC_FLD(casacore::DataType::TpArrayChar)
    break;

  case ALLOC_FLD(casacore::DataType::TpUChar)
    break;

  case ALLOC_FLD(casacore::DataType::TpArrayUChar)
    break;

  case ALLOC_FLD(casacore::DataType::TpShort)
    break;

  case ALLOC_FLD(casacore::DataType::TpArrayShort)
    break;

  case ALLOC_FLD(casacore::DataType::TpUShort)
    break;

  case ALLOC_FLD(casacore::DataType::TpArrayUShort)
    break;

  case ALLOC_FLD(casacore::DataType::TpInt)
    break;

  case ALLOC_FLD(casacore::DataType::TpArrayInt)
    break;

  case ALLOC_FLD(casacore::DataType::TpUInt)
    break;

  case ALLOC_FLD(casacore::DataType::TpArrayUInt)
    break;

  // case ALLOC_FLD(casacore::DataType::TpInt64)
  //   break;

  // case ALLOC_FLD(casacore::DataType::TpArrayInt64)
  //   break;

  case ALLOC_FLD(casacore::DataType::TpFloat)
    break;

  case ALLOC_FLD(casacore::DataType::TpArrayFloat)
    break;

  case ALLOC_FLD(casacore::DataType::TpDouble)
    break;

  case ALLOC_FLD(casacore::DataType::TpArrayDouble)
    break;

  case ALLOC_FLD(casacore::DataType::TpComplex)
    break;

  case ALLOC_FLD(casacore::DataType::TpArrayComplex)
    break;

  case ALLOC_FLD(casacore::DataType::TpDComplex)
    break;

  case ALLOC_FLD(casacore::DataType::TpArrayDComplex)
    break;

  case ALLOC_FLD(casacore::DataType::TpString)
    break;

  case ALLOC_FLD(casacore::DataType::TpArrayString)
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

#undef ALLOC_FLD

  return result;
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
