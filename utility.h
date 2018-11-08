#ifndef LEGMS_MS_UTILITY_H_
#define LEGMS_MS_UTILITY_H_

#include <atomic>
#include <cassert>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>

#include <casacore/casa/aipstype.h>
#include <casacore/casa/Arrays/IPosition.h>
#include <casacore/casa/BasicSL/String.h>
#include <casacore/casa/Utilities/DataType.h>
#include "legion.h"
#include "IndexTree.h"

namespace legms {

typedef IndexTree<Legion::coord_t> IndexTreeL;

template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

template <
  int ARRAY_DIM,
  int REGION_DIM,
  unsigned long HINT_DIM>
std::enable_if_t<REGION_DIM >= ARRAY_DIM>
pt2ipos(
  casacore::IPosition& ipos,
  const std::array<unsigned, HINT_DIM>& pv,
  const Legion::Point<REGION_DIM>& pt) {

  for (unsigned i = 0; i < ARRAY_DIM; ++i)
    ipos[pv[i]] = pt[i + REGION_DIM - ARRAY_DIM];
}

template <
  int ARRAY_DIM,
  int REGION_DIM,
  unsigned long HINT_DIM>
std::enable_if_t<!(REGION_DIM >= ARRAY_DIM)>
pt2ipos(
  casacore::IPosition&,
  const std::array<unsigned, HINT_DIM>&,
  const Legion::Point<REGION_DIM>&) {

  assert(false);
}

template <int REGION_DIM>
void
pt2ipos(
  casacore::IPosition& ipos,
  const unsigned* pv,
  const Legion::Point<REGION_DIM>& pt) {

  unsigned array_dim = ipos.size();
  assert(REGION_DIM >= array_dim);
  for (unsigned i = 0; i < array_dim; ++i)
    ipos[pv[i]] = pt[i + REGION_DIM - array_dim];
}

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
    size_type nch = val.size();
    memcpy(static_cast<size_type *>(buffer), &nch, sizeof(nch));
    if (nch > 0) {
      value_type* chbuf =
        reinterpret_cast<value_type *>(static_cast<size_type *>(buffer) + 1);
      memcpy(chbuf, val.data(), result - sizeof(size_type));
    }
    return result;
  }

  static size_t
  deserialize(S& val, const void *buffer) {
    size_type nch = *static_cast<const size_type *>(buffer);
    val.clear();
    if (nch > 0) {
      val.reserve(nch);
      val.append(
        reinterpret_cast<const value_type*>(
          static_cast<const size_type *>(buffer) + 1),
        nch);
    }
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
    const T* vs =
      reinterpret_cast<const T *>(static_cast<const size_t *>(buffer) + 1);
    val.insert(val.end(), vs, vs + nt);
    return serialized_size(val);
  }

  static void
  destroy(std::vector<T>&) {
  }
};

template <>
class vector_serdez<casacore::Bool> {
public:

  typedef std::vector<casacore::Bool> FIELD_TYPE;

  static const size_t MAX_SERIALIZED_SIZE =
    std::numeric_limits<size_t>::max();

  static size_t
  serialized_size(const std::vector<casacore::Bool>& val) {
    return sizeof(size_t) + (val.size() + 7) / 8;
  }

  static size_t
  serialize(const std::vector<casacore::Bool>& val, void *buffer) {
    *static_cast<size_t *>(buffer) = val.size();
    unsigned char* ts =
      reinterpret_cast<unsigned char *>(static_cast<size_t *>(buffer) + 1);
    unsigned bit = ((val.size() % 8) + 7) % 8;
    std::for_each(
      val.begin(),
      val.end(),
      [&ts, &bit](auto b) {
        *ts = (*ts << 1) | b;
        bit = (bit + 7) % 8;
        if (bit == 7)
          ++ts;
      });
    return serialized_size(val);
  }

  static size_t
  deserialize(std::vector<casacore::Bool>& val, const void *buffer) {
    size_t nt = *static_cast<const size_t *>(buffer);
    val.reserve(nt);
    const unsigned char* ts =
      reinterpret_cast<const unsigned char *>(
        static_cast<const size_t *>(buffer) + 1);
    unsigned bit = ((nt % 8) + 7) % 8;
    while (nt > 0) {
      val.push_back((*ts >> bit) & 1);
      bit = (bit + 7) % 8;
      if (bit == 7)
        ++ts;
      --nt;
    }
    return serialized_size(val);
  }

  static void
  destroy(std::vector<casacore::Bool>&) {
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
        Legion::Runtime::register_custom_serdez_op<
          vector_serdez<casacore::Bool>>(CASACORE_BOOL_V_SID);
        Legion::Runtime::register_custom_serdez_op<
          vector_serdez<casacore::Char>>(CASACORE_CHAR_V_SID);
        Legion::Runtime::register_custom_serdez_op<
          vector_serdez<casacore::Short>>(CASACORE_SHORT_V_SID);
        Legion::Runtime::register_custom_serdez_op<
          vector_serdez<casacore::Int>>(CASACORE_INT_V_SID);
        // Legion::Runtime::register_custom_serdez_op<
        //   vector_serdez<casacore::Int64>>(CASACORE_INT64_V_SID);
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
          string_array_serdez<casacore::String>>(
            CASACORE_STRING_ARRAY_SID);
      });
  }

  enum {
    INDEX_TREE_SID = 1,
    CASACORE_BOOL_V_SID,
    CASACORE_CHAR_V_SID,
    CASACORE_SHORT_V_SID,
    CASACORE_INT_V_SID,
    // CASACORE_INT64_V_SID,
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
  Legion::FieldID field_id = AUTO_GENERATE_ID);

template <casacore::DataType T>
struct DataType {

  //typedef X ValueType;
};

template <>
struct DataType<casacore::DataType::TpBool> {
  typedef casacore::Bool ValueType;
  constexpr static int serdez_id = 0;
};

template <>
struct DataType<casacore::DataType::TpArrayBool> {
  typedef std::vector<casacore::Bool> ValueType;
  constexpr static int serdez_id = SerdezManager::CASACORE_BOOL_V_SID;
};

template <>
struct DataType<casacore::DataType::TpChar> {
  typedef casacore::Char ValueType;
  constexpr static int serdez_id = 0;
};

template <>
struct DataType<casacore::DataType::TpArrayChar> {
  typedef std::vector<casacore::Char> ValueType;
  constexpr static int serdez_id = SerdezManager::CASACORE_CHAR_V_SID;
};

template <>
struct DataType<casacore::DataType::TpUChar> {
  typedef casacore::uChar ValueType;
  constexpr static int serdez_id = 0;
};

template <>
struct DataType<casacore::DataType::TpArrayUChar> {
  typedef std::vector<casacore::uChar> ValueType;
  constexpr static int serdez_id = SerdezManager::CASACORE_CHAR_V_SID;
};

template <>
struct DataType<casacore::DataType::TpShort> {
  typedef casacore::Short ValueType;
  constexpr static int serdez_id = 0;
};

template <>
struct DataType<casacore::DataType::TpArrayShort> {
  typedef std::vector<casacore::Short> ValueType;
  constexpr static int serdez_id = SerdezManager::CASACORE_SHORT_V_SID;
};

template <>
struct DataType<casacore::DataType::TpUShort> {
  typedef casacore::uShort ValueType;
  constexpr static int serdez_id = 0;
};

template <>
struct DataType<casacore::DataType::TpArrayUShort> {
  typedef std::vector<casacore::uShort> ValueType;
  constexpr static int serdez_id = SerdezManager::CASACORE_SHORT_V_SID;
};

template <>
struct DataType<casacore::DataType::TpInt> {
  typedef casacore::Int ValueType;
  constexpr static int serdez_id = 0;
};

template <>
struct DataType<casacore::DataType::TpArrayInt> {
  typedef std::vector<casacore::Int> ValueType;
  constexpr static int serdez_id = SerdezManager::CASACORE_INT_V_SID;
};

template <>
struct DataType<casacore::DataType::TpUInt> {
  typedef casacore::uInt ValueType;
  constexpr static int serdez_id = 0;
};

template <>
struct DataType<casacore::DataType::TpArrayUInt> {
  typedef std::vector<casacore::uInt> ValueType;
  constexpr static int serdez_id = SerdezManager::CASACORE_INT_V_SID;
};

template <>
struct DataType<casacore::DataType::TpFloat> {
  typedef casacore::Float ValueType;
  constexpr static int serdez_id = 0;
};

template <>
struct DataType<casacore::DataType::TpArrayFloat> {
  typedef std::vector<casacore::Float> ValueType;
  constexpr static int serdez_id = SerdezManager::CASACORE_FLOAT_V_SID;
};

template <>
struct DataType<casacore::DataType::TpDouble> {
  typedef casacore::Double ValueType;
  constexpr static int serdez_id = 0;
};

template <>
struct DataType<casacore::DataType::TpArrayDouble> {
  typedef std::vector<casacore::Double> ValueType;
  constexpr static int serdez_id = SerdezManager::CASACORE_DOUBLE_V_SID;
};

template <>
struct DataType<casacore::DataType::TpComplex> {
  typedef casacore::Complex ValueType;
  constexpr static int serdez_id = 0;
};

template <>
struct DataType<casacore::DataType::TpArrayComplex> {
  typedef std::vector<casacore::Complex> ValueType;
  constexpr static int serdez_id = SerdezManager::CASACORE_COMPLEX_V_SID;
};

template <>
struct DataType<casacore::DataType::TpDComplex> {
  typedef casacore::DComplex ValueType;
  constexpr static int serdez_id = 0;
};

template <>
struct DataType<casacore::DataType::TpArrayDComplex> {
  typedef std::vector<casacore::DComplex> ValueType;
  constexpr static int serdez_id = SerdezManager::CASACORE_DCOMPLEX_V_SID;
};

template <>
struct DataType<casacore::DataType::TpString> {
  typedef casacore::String ValueType;
  constexpr static int serdez_id = SerdezManager::CASACORE_STRING_SID;
};

template <>
struct DataType<casacore::DataType::TpArrayString> {
  typedef std::vector<casacore::String> ValueType;
  constexpr static int serdez_id = SerdezManager::CASACORE_STRING_ARRAY_SID;
};

template <typename T>
struct ValueType {
  // constexpr static const casacore::DataType DataType;
};

template <>
struct ValueType<casacore::Bool> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpBool;
};

template <>
struct ValueType<std::vector<casacore::Bool>> {
  constexpr static casacore::DataType DataType =
    casacore::DataType::TpArrayBool;
};

template <>
struct ValueType<casacore::Char> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpChar;
};

template <>
struct ValueType<std::vector<casacore::Char>> {
  constexpr static casacore::DataType DataType =
    casacore::DataType::TpArrayChar;
};

template <>
struct ValueType<casacore::uChar> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpUChar;
};

template <>
struct ValueType<std::vector<casacore::uChar>> {
  constexpr static casacore::DataType DataType =
    casacore::DataType::TpArrayUChar;
};

template <>
struct ValueType<casacore::Short> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpShort;
};

template <>
struct ValueType<std::vector<casacore::Short>> {
  constexpr static casacore::DataType DataType =
    casacore::DataType::TpArrayShort;
};

template <>
struct ValueType<casacore::uShort> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpUShort;
};

template <>
struct ValueType<std::vector<casacore::uShort>> {
  constexpr static casacore::DataType DataType =
    casacore::DataType::TpArrayUShort;
};

template <>
struct ValueType<casacore::Int> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpInt;
};

template <>
struct ValueType<std::vector<casacore::Int>> {
  constexpr static casacore::DataType DataType =
    casacore::DataType::TpArrayInt;
};

template <>
struct ValueType<casacore::uInt> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpUInt;
};

template <>
struct ValueType<std::vector<casacore::uInt>> {
  constexpr static casacore::DataType DataType =
    casacore::DataType::TpArrayUInt;
};

template <>
struct ValueType<casacore::Float> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpFloat;
};

template <>
struct ValueType<std::vector<casacore::Float>> {
  constexpr static casacore::DataType DataType =
    casacore::DataType::TpArrayFloat;
};

template <>
struct ValueType<casacore::Double> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpDouble;
};

template <>
struct ValueType<std::vector<casacore::Double>> {
  constexpr static casacore::DataType DataType =
    casacore::DataType::TpArrayDouble;
};

template <>
struct ValueType<casacore::Complex> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpComplex;
};

template <>
struct ValueType<std::vector<casacore::Complex>> {
  constexpr static casacore::DataType DataType =
    casacore::DataType::TpArrayComplex;
};

template <>
struct ValueType<casacore::DComplex> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpDComplex;
};

template <>
struct ValueType<std::vector<casacore::DComplex>> {
  constexpr static casacore::DataType DataType =
    casacore::DataType::TpArrayDComplex;
};

template <>
struct ValueType<casacore::String> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpString;
};

template <>
struct ValueType<std::vector<casacore::String>> {
  constexpr static casacore::DataType DataType =
    casacore::DataType::TpArrayString;
};

} // end namespace legms

#endif // LEGMS_MS_UTILITY_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
