#include "utility.h"

using namespace legms;

std::once_flag SerdezManager::initialized;

Legion::FieldID
legms::add_field(
  casacore::DataType datatype,
  Legion::FieldAllocator fa,
  Legion::FieldID field_id) {

  Legion::FieldID result;

  switch (datatype) {

  case casacore::DataType::TpBool:
    result = fa.allocate_field(sizeof(casacore::Bool), field_id);
    break;

  case casacore::DataType::TpArrayBool:
    result = fa.allocate_field(
      sizeof(std::vector<casacore::Bool>),
      field_id,
      SerdezManager::CASACORE_BOOL_V_SID);
    break;

  case casacore::DataType::TpChar:
  case casacore::DataType::TpUChar:
    result = fa.allocate_field(sizeof(casacore::Char), field_id);
    break;

  case casacore::DataType::TpArrayChar:
  case casacore::DataType::TpArrayUChar:
    result = fa.allocate_field(
      sizeof(std::vector<casacore::Char>),
      field_id,
      SerdezManager::CASACORE_CHAR_V_SID);
    break;

  case casacore::DataType::TpShort:
  case casacore::DataType::TpUShort:
    result = fa.allocate_field(sizeof(casacore::Short), field_id);
    break;

  case casacore::DataType::TpArrayShort:
  case casacore::DataType::TpArrayUShort:
    result = fa.allocate_field(
      sizeof(std::vector<casacore::Short>),
      field_id,
      SerdezManager::CASACORE_SHORT_V_SID);
    break;

  case casacore::DataType::TpInt:
  case casacore::DataType::TpUInt:
    result = fa.allocate_field(sizeof(casacore::Int), field_id);
    break;

  case casacore::DataType::TpArrayInt:
  case casacore::DataType::TpArrayUInt:
    result = fa.allocate_field(
      sizeof(std::vector<casacore::Int>),
      field_id,
      SerdezManager::CASACORE_INT_V_SID);
    break;

  // case casacore::DataType::TpInt64:
  //   // case casacore::DataType::TpUInt64:
  //   result = fa.allocate_field(sizeof(casacore::Int64), field_id);
  //   break;

  // case casacore::DataType::TpArrayInt64:
  //   // case casacore::DataType::TpArrayUInt64:
  //   result = fa.allocate_field(
  //     sizeof(std::vector<casacore::Int64>),
  //     field_id,
  //     SerdezManager::CASACORE_INT64_V_SID);
  //   break;

  case casacore::DataType::TpFloat:
    result = fa.allocate_field(sizeof(casacore::Float), field_id);
    break;

  case casacore::DataType::TpArrayFloat:
    result = fa.allocate_field(
      sizeof(std::vector<casacore::Float>),
      field_id,
      SerdezManager::CASACORE_FLOAT_V_SID);
    break;

  case casacore::DataType::TpDouble:
    result = fa.allocate_field(sizeof(casacore::Double), field_id);
    break;

  case casacore::DataType::TpArrayDouble:
    result = fa.allocate_field(
      sizeof(std::vector<casacore::Double>),
      field_id,
      SerdezManager::CASACORE_DOUBLE_V_SID);
    break;

  case casacore::DataType::TpComplex:
    result = fa.allocate_field(sizeof(casacore::Complex), field_id);
    break;

  case casacore::DataType::TpArrayComplex:
    result = fa.allocate_field(
      sizeof(std::vector<casacore::Complex>),
      field_id,
      SerdezManager::CASACORE_COMPLEX_V_SID);
    break;

  case casacore::DataType::TpDComplex:
    result = fa.allocate_field(sizeof(casacore::DComplex), field_id);
    break;

  case casacore::DataType::TpArrayDComplex:
    result = fa.allocate_field(
      sizeof(std::vector<casacore::DComplex>),
      field_id,
      SerdezManager::CASACORE_DCOMPLEX_V_SID);
    break;

  case casacore::DataType::TpString:
    result = fa.allocate_field(
      sizeof(casacore::String),
      field_id,
      SerdezManager::CASACORE_STRING_SID);
    break;

  case casacore::DataType::TpArrayString:
    result = fa.allocate_field(
      sizeof(std::vector<casacore::String>),
      field_id,
      SerdezManager::CASACORE_STRING_ARRAY_SID);
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
  return result;
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:

