#ifndef LEGMS_MS_UTILITY_H_
#define LEGMS_MS_UTILITY_H_

#include <atomic>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>

#include <casacore/casa/Utilities/DataType.h>
#include "legion.h"
#include "IndexTree.h"

namespace legms {

typedef IndexTree<Legion::coord_t> IndexTreeL;

template <typename S>
class string_serdez {
  // this class only works for fixed size character encodings!

public:

  typedef S FIELD_TYPE;

  typedef typename S::size_type size_type;
  typedef typename S::value_type value_type;

  static const size_t MAX_SERIALIZED_SIZE =
    std::numeric_limits<size_type>::max();

  static size_t
  serialized_size(const S& val) {
    return sizeof(size_type) + val.size() * sizeof(value_type);
  }

  static size_t
  serialize(const S& val, void *buffer) {
    size_t result = serialized_size(val);
    auto nch = val.size();
    memcpy(static_cast<size_type *>(buffer), &nch, sizeof(nch));
    value_type* chbuf =
      reinterpret_cast<value_type *>(static_cast<size_type *>(buffer) + 1);
    memcpy(chbuf, val.data(), result - sizeof(size_type));
    return result;
  }

  static size_t
  deserialize(S& val, const void *buffer) {
    size_type nch = *static_cast<const size_type *>(buffer);
    val.reserve(nch);
    auto buf = const_cast<value_type*>(val.data());
    std::memcpy(
      buf,
      reinterpret_cast<const void*>(static_cast<const size_type *>(buffer) + 1),
      nch * sizeof(value_type));
    return serialized_size(val);
  }

  static void
  destroy(S&) {
  }
};

template <typename S>
class string_array_serdez {
  // this class only works for fixed size character encodings!

public:
  typedef std::vector<S> FIELD_TYPE;

  typedef typename S::size_type size_type;
  typedef typename S::value_type value_type;

  static const size_t MAX_SERIALIZED_SIZE =
    std::numeric_limits<size_type>::max();

  static size_t
  serialized_size(const std::vector<S>& val) {
    return
      std::accumulate(
        val.begin(),
        val.end(),
        sizeof(size_t),
        [](const size_t& acc, auto& s) {
          return acc + string_serdez<S>::serialized_size(s);
        });
  }

  static size_t
  serialize(const std::vector<S>& val, void *buffer) {
    *static_cast<size_t *>(buffer) = val.size();
    char* chbuf = reinterpret_cast<char *>(static_cast<size_t *>(buffer) + 1);
    std::for_each(
      val.begin(),
      val.end(),
      [&chbuf](auto& s) {
        chbuf += string_serdez<S>::serialize(s, chbuf);
      });
    return chbuf - static_cast<char *>(buffer);
  }

  static size_t
  deserialize(std::vector<S>& val, const void *buffer) {
    size_t ns = *static_cast<const size_t *>(buffer);
    const char* chbuf =
      reinterpret_cast<const char *>(static_cast<const size_t *>(buffer) + 1);
    for (size_t i = 0; i < ns; ++i) {
      S s;
      chbuf += string_serdez<S>::deserialize(s, chbuf);
      val.push_back(s);
    }
    return serialized_size(val);
  }

  static void
  destroy(std::vector<S>&) {
  }
};

template <typename T>
class vector_serdez {
public:

  typedef std::vector<T> FIELD_TYPE;

  static const size_t MAX_SERIALIZED_SIZE =
    std::numeric_limits<size_t>::max();

  static size_t
  serialized_size(const std::vector<T>& val) {
    return sizeof(size_t) + val.size() * sizeof(T);
  }

  static size_t
  serialize(const std::vector<T>& val, void *buffer) {
    *static_cast<size_t *>(buffer) = val.size();
    T* ts = reinterpret_cast<T *>(static_cast<size_t *>(buffer) + 1);
    memcpy(ts, val.data(), val.size() * sizeof(T));
    return serialized_size(val);
  }

  static size_t
  deserialize(std::vector<T>& val, const void *buffer) {
    size_t nt = *static_cast<const size_t *>(buffer);
    val.reserve(nt);
    std::memcpy(
      val.data(),
      reinterpret_cast<const T *>(static_cast<const size_t *>(buffer) + 1),
      nt * sizeof(T));
    return serialized_size(val);
  }

  static void
  destroy(std::vector<T>&) {
  }
};

class index_tree_serdez {
public:
  typedef IndexTreeL FIELD_TYPE;

  static const size_t MAX_SERIALIZED_SIZE = std::numeric_limits<size_t>::max();

  static size_t
  serialized_size(const IndexTreeL& val) {
    return val.serialized_size();
  }

  static size_t
  serialize(const IndexTreeL& val, void *buffer) {
    return val.serialize(reinterpret_cast<char*>(buffer));
  }

  static size_t
  deserialize(IndexTreeL& val, const void *buffer) {
    val = IndexTreeL::deserialize(static_cast<const char*>(buffer));
    return *reinterpret_cast<const size_t *>(buffer);
  }

  static void
  destroy(IndexTreeL&) {
  }
};

class SerdezManager {
public:

  static void
  register_ops() {
    std::call_once(
      initialized,
      []() {
        Legion::Runtime::register_custom_serdez_op<index_tree_serdez>(
          INDEX_TREE_SID);
        // Legion::Runtime::register_custom_serdez_op<
        //   vector_serdez<casacore::Bool>>(CASACORE_BOOL_V_SID);
        Legion::Runtime::register_custom_serdez_op<
          vector_serdez<casacore::Char>>(CASACORE_CHAR_V_SID);
        Legion::Runtime::register_custom_serdez_op<
          vector_serdez<casacore::Short>>(CASACORE_SHORT_V_SID);
        Legion::Runtime::register_custom_serdez_op<
          vector_serdez<casacore::Int>>(CASACORE_INT_V_SID);
        Legion::Runtime::register_custom_serdez_op<
          vector_serdez<casacore::Int64>>(CASACORE_INT64_V_SID);
        Legion::Runtime::register_custom_serdez_op<
          vector_serdez<casacore::Float>>(CASACORE_FLOAT_V_SID);
        Legion::Runtime::register_custom_serdez_op<
          vector_serdez<casacore::Double>>(CASACORE_DOUBLE_V_SID);
        Legion::Runtime::register_custom_serdez_op<
          vector_serdez<casacore::Complex>>(CASACORE_COMPLEX_V_SID);
        Legion::Runtime::register_custom_serdez_op<
          vector_serdez<casacore::DComplex>>(CASACORE_DCOMPLEX_V_SID);
        Legion::Runtime::register_custom_serdez_op<
          string_serdez<casacore::String>>(CASACORE_STRING_SID);
        Legion::Runtime::register_custom_serdez_op<
          string_array_serdez<std::vector<casacore::String>>>(
            CASACORE_STRING_ARRAY_SID);
      });
  }

  enum {
    INDEX_TREE_SID = 1,
    CASACORE_BOOL_V_SID,
    CASACORE_CHAR_V_SID,
    CASACORE_SHORT_V_SID,
    CASACORE_INT_V_SID,
    CASACORE_INT64_V_SID,
    CASACORE_FLOAT_V_SID,
    CASACORE_DOUBLE_V_SID,
    CASACORE_COMPLEX_V_SID,
    CASACORE_DCOMPLEX_V_SID,
    CASACORE_STRING_SID,
    CASACORE_STRING_ARRAY_SID
  };

private:

  static std::once_flag initialized;
};

Legion::FieldID
add_field(
  casacore::DataType datatype,
  Legion::FieldAllocator fa,
  Legion::FieldID field_id = AUTO_GENERATE_ID) {

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

  case casacore::DataType::TpInt64:
    // case casacore::DataType::TpUInt64:
    result = fa.allocate_field(sizeof(casacore::Int64), field_id);
    break;

  case casacore::DataType::TpArrayInt64:
    // case casacore::DataType::TpArrayUInt64:
    result = fa.allocate_field(
      sizeof(std::vector<casacore::Int64>),
      field_id,
      SerdezManager::CASACORE_INT64_V_SID);
    break;

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

}

#endif // LEGMS_MS_UTILITY_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:

