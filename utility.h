#ifndef LEGMS_MS_UTILITY_H_
#define LEGMS_MS_UTILITY_H_

#include <atomic>
#include <cstring>
#include <limits>
#include <mutex>

#include "casa/casacore/util"
#include "legion.h"

namespace legms {
namespace ms {

template <typename S>
class string_serdez {
  // this class only works for fixed size character encodings!

public:
  typedef S FIELD_TYPE;

  typedef size_type typename S::size_type;
  typedef value_type typename S::value_type;

  static const size_t MAX_SERIALIZED_SIZE =
    std::numeric_limits<size_type>::max();

  static size_t
  serialized_size(const S& val) {
    return sizeof(size_type) + val.length() * sizeof(value_type);
  }

  static size_t
  serialize(const S& val, void *buffer) {
    size_t result = serialized_size(val);
    auto nch = val.length();
    memcpy(static_cast<size_type *>(buffer), &nch, sizeof(nch));
    value_type* chbuf =
      static_cast<value_type *>(static_cast<size_type *>(buffer) + 1);
    memcpy(chbuf, val.data(), result - sizeof(size_type));
    return result;
  }

  static size_t
  deserialize(S& val, const void *buffer) {
    size_type nch = *static_cast<const size_type *>(buffer);
    val.reserve(nch);
    val.insert(
      0,
      static_cast<const value_type *>(
        static_cast<const size_type *>(buffer) + 1),
      nch);
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

  typedef size_type typename S::size_type;
  typedef value_type typename S::value_type;

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
    char* chbuf = static_cast<char *>(static_cast<size_t *>(buffer) + 1);
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
      static_cast<const char *>(static_cast<const size_t *>(buffer) + 1);
    for (size_t i = 0; i < ns; ++i) {
      std::vector<S> s;
      chbuf += string_serdez<S>::deserialize(s, chbuf);
      val.push_back(s);
    }
    return serialized_size(val);
  }

  static void
  destroy(S&) {
  }
};

template <typename T>
class vector_serdez {
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
    T* ts = static_cast<T *>(static_cast<size_t *>(buffer) + 1);
    memcpy(ts, val.data(), val.size() * sizeof(T));
    return serialized_size(val);
  }

  static size_t
  deserialize(std::vector<T>& val, const void *buffer) {
    size_t nt = *static_cast<const size_t *>(buffer);
    val.reserve(nt);
    val.insert(
      0,
      static_cast<const T *>(static_cast<const size_t *>(buffer) + 1),
      nt);
    return serialized_size(val);
  }

  static void
  destroy(std::vector<T>&) {
  }
};

class DataTypeSerdezManager {
public:

  static void
  register_ops() {
    std::call_once(
      initialized,
      []() {
        Runtime::register_custom_serdez_op<
          vector_serdez<casacore::Bool>>(CASACORE_BOOL_V_SID);
        Runtime::register_custom_serdez_op<
          vector_serdez<casacore::Char>>(CASACORE_CHAR_V_SID);
        Runtime::register_custom_serdez_op<
          vector_serdez<casacore::Short>>(CASACORE_SHORT_V_SID);
        Runtime::register_custom_serdez_op<
          vector_serdez<casacore::Int>>(CASACORE_INT_V_SID);
        Runtime::register_custom_serdez_op<
          vector_serdez<casacore::Int64>>(CASACORE_INT64_V_SID);
        Runtime::register_custom_serdez_op<
          vector_serdez<casacore::Float>>(CASACORE_FLOAT_V_SID);
        Runtime::register_custom_serdez_op<
          vector_serdez<casacore::Double>>(CASACORE_DOUBLE_V_SID);
        Runtime::register_custom_serdez_op<
          vector_serdez<casacore::Complex>>(CASACORE_COMPLEX_V_SID);
        Runtime::register_custom_serdez_op<
          vector_serdez<casacore::DComplex>>(CASACORE_DCOMPLEX_V_SID);
        Runtime::register_custom_serdez_op<
          string_serdez<casacore::String>>(CASACORE_STRING_SID);
        Runtime::register_custom_serdez_op<
          string_array_serdez<std::vector<casacore::String>>>(
            CASACORE_STRING_ARRAY_SID);
      });
  }

  enum {
    CASACORE_BOOL_V_SID = 1,
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
  }

private:

  static std::once_flag initialized;
};

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
