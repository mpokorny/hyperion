#ifndef LEGMS_UTILITY_H_
#define LEGMS_UTILITY_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>

#ifndef LEGMS_MAX_DIM
# define LEGMS_MAX_DIM 4
#endif

#if USE_HDF
# include <hdf5.h>
#endif

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

template <typename D>
bool
has_unique_values(const std::vector<D>& axes) {
  std::vector<D> ax = axes;
  std::sort(ax.begin(), ax.end());
  std::unique(ax.begin(), ax.end());
  return ax.size() == axes.size();
}

template <typename D>
std::vector<int>
dimensions_map(const std::vector<D>& from, const std::vector<D>& to) {
  std::vector<int> result(from.size());
  for (size_t i = 0; i < from.size(); ++i) {
    result[i] = -1;
    for (size_t j = 0; result[i] == -1 && j < to.size(); ++j)
      if (from[i] == to[j])
        result[i] = j;
  }
  return result;
}

template <typename T>
class vector_serdez {
public:

  typedef std::vector<T> FIELD_TYPE;

  static const size_t MAX_SERIALIZED_SIZE =
    std::numeric_limits<size_t>::max();

  static size_t
  serialized_size(const FIELD_TYPE& val) {
    return sizeof(size_t) + val.size() * sizeof(T);
  }

  static size_t
  serialize(const FIELD_TYPE& val, void *buffer) {
    size_t result = serialized_size(val);
    size_t n = val.size();
    std::memcpy(static_cast<size_t *>(buffer), &n, sizeof(n));
    if (n > 0) {
      T* tbuf =
        reinterpret_cast<T *>(static_cast<size_t *>(buffer) + 1);
      std::memcpy(tbuf, val.data(), result - sizeof(size_t));
    }
    return result;
  }

  static size_t
  deserialize(FIELD_TYPE& val, const void *buffer) {
    size_t n = *static_cast<const size_t *>(buffer);
    val.clear();
    if (n > 0) {
      val.resize(n);
      std::memcpy(
        val.data(),
        reinterpret_cast<const T*>(static_cast<const size_t *>(buffer) + 1),
        n * sizeof(T));
    }
    return serialized_size(val);
  }

  static void
  destroy(FIELD_TYPE&) {
  }
};

template <>
class vector_serdez<bool> {
public:

  typedef std::vector<bool> FIELD_TYPE;

  static const size_t MAX_SERIALIZED_SIZE =
    std::numeric_limits<size_t>::max();

  static size_t
  serialized_size(const FIELD_TYPE& val) {
    return sizeof(size_t) + val.size() * sizeof(bool);
  }

  static size_t
  serialize(const FIELD_TYPE& val, void *buffer) {
    size_t result = serialized_size(val);
    size_t n = val.size();
    std::memcpy(static_cast<size_t *>(buffer), &n, sizeof(n));
    if (n > 0) {
      bool* bbuf = reinterpret_cast<bool *>(static_cast<size_t *>(buffer) + 1);
      for (size_t i = 0; i < n; ++i) {
        bool b = val[i];
        std::memcpy(bbuf++, &b, sizeof(bool));
      }
    }
    return result;
  }

  static size_t
  deserialize(FIELD_TYPE& val, const void *buffer) {
    size_t n = *static_cast<const size_t *>(buffer);
    val.clear();
    if (n > 0) {
      val.resize(n);
      const bool* bbuf =
        reinterpret_cast<const bool *>(static_cast<const size_t *>(buffer) + 1);
      for (size_t i = 0; i < n; ++i)
        val[i] = *bbuf++;
    }
    return serialized_size(val);
  }

  static void
  destroy(FIELD_TYPE&) {
  }
};

template <typename T>
class acc_field_serdez {
public:

  typedef std::vector<
  std::tuple<T, std::vector<Legion::DomainPoint>>> FIELD_TYPE;

  static const size_t MAX_SERIALIZED_SIZE =
    std::numeric_limits<size_t>::max();

  static size_t
  serialized_size(const FIELD_TYPE& val) {
    return
      std::accumulate(
        val.begin(),
        val.end(),
        sizeof(size_t),
        [](auto& acc, auto& t) {
          return
            acc
            + sizeof(T)
            + vector_serdez<Legion::DomainPoint>::serialized_size(
              std::get<1>(t));
        });
  }

  static size_t
  serialize(const FIELD_TYPE& val, void *buffer) {
    size_t result = serialized_size(val);
    size_t n = val.size();
    std::memcpy(static_cast<size_t *>(buffer), &n, sizeof(n));
    buffer = static_cast<void *>(static_cast<size_t*>(buffer) + 1);
    for (size_t i = 0; i < n; ++i) {
      auto& [t, rns] = val[i];
      T* tbuf = reinterpret_cast<T *>(buffer);
      std::memcpy(tbuf, &t, sizeof(T));
      buffer = static_cast<void *>(tbuf + 1);
      buffer =
        static_cast<char*>(buffer)
        + vector_serdez<Legion::DomainPoint>::serialize(rns, buffer);
    }
    return result;
  }

  static size_t
  deserialize(FIELD_TYPE& val, const void *buffer) {
    size_t n = *static_cast<const size_t *>(buffer);
    buffer = static_cast<const void *>(static_cast<const size_t*>(buffer) + 1);
    val.clear();
    val.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      const T* tbuf = reinterpret_cast<const T *>(buffer);
      buffer = static_cast<const void *>(tbuf + 1);
      std::vector<Legion::DomainPoint> rns;
      buffer =
        static_cast<const char*>(buffer)
        + vector_serdez<Legion::DomainPoint>::deserialize(rns, buffer);
      val.emplace_back(*tbuf, rns);
    }
    return serialized_size(val);
  }

  static void
  destroy(FIELD_TYPE&) {
  }
};

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
    std::memcpy(static_cast<size_type *>(buffer), &nch, sizeof(nch));
    if (nch > 0) {
      value_type* chbuf =
        reinterpret_cast<value_type *>(static_cast<size_type *>(buffer) + 1);
      std::memcpy(chbuf, val.data(), result - sizeof(size_type));
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

template <>
class acc_field_serdez<casacore::String> {
public:

  typedef std::vector<
  std::tuple<casacore::String, std::vector<Legion::DomainPoint>>> FIELD_TYPE;

  static const size_t MAX_SERIALIZED_SIZE =
    std::numeric_limits<size_t>::max();

  static size_t
  serialized_size(const FIELD_TYPE& val) {
    return
      std::accumulate(
        val.begin(),
        val.end(),
        sizeof(size_t),
        [](auto& acc, auto& t) {
          return
            acc
            + string_serdez<casacore::String>::serialized_size(std::get<0>(t))
            + vector_serdez<Legion::DomainPoint>::serialized_size(
              std::get<1>(t));
        });
  }

  static size_t
  serialize(const FIELD_TYPE& val, void *buffer) {
    size_t result = serialized_size(val);
    size_t n = val.size();
    std::memcpy(static_cast<size_t *>(buffer), &n, sizeof(n));
    buffer = static_cast<void *>(static_cast<size_t*>(buffer) + 1);
    char* buff = static_cast<char*>(buffer);
    for (size_t i = 0; i < n; ++i) {
      auto& [t, rns] = val[i];
      buff += string_serdez<casacore::String>::serialize(t, buff);
      buff += vector_serdez<Legion::DomainPoint>::serialize(rns, buff);
    }
    return result;
  }

  static size_t
  deserialize(FIELD_TYPE& val, const void *buffer) {
    size_t n = *static_cast<const size_t *>(buffer);
    buffer = static_cast<const void *>(static_cast<const size_t*>(buffer) + 1);
    val.clear();
    val.reserve(n);
    const char* buff = static_cast<const char*>(buffer);
    for (size_t i = 0; i < n; ++i) {
      casacore::String str;
      buff += string_serdez<casacore::String>::deserialize(str, buff);
      std::vector<Legion::DomainPoint> rns;
      buff += vector_serdez<Legion::DomainPoint>::deserialize(rns, buffer);
      val.emplace_back(str, rns);
    }
    return serialized_size(val);
  }

  static void
  destroy(FIELD_TYPE&) {
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

class bool_or_redop {
public:
  typedef bool LHS;
  typedef bool RHS;

  static void
  combine(LHS& lhs, RHS rhs) {
    lhs = lhs || rhs;
  }

  template <bool EXCL>
  static void
  apply(LHS& lhs, RHS rhs) {
    combine(lhs, rhs);
  }

  static constexpr const RHS identity = false;

  template <bool EXCL>
  static void
  fold(RHS& rhs1, RHS rhs2) {
    combine(rhs1, rhs2);
  }
};

template <typename T>
class acc_field_redop {
public:
  typedef std::vector<std::tuple<T, std::vector<Legion::DomainPoint>>> LHS;
  typedef std::vector<std::tuple<T, std::vector<Legion::DomainPoint>>> RHS;

  static void
  combine(LHS& lhs, RHS rhs) {
    std::for_each(
      rhs.begin(),
      rhs.end(),
      [&lhs](auto& t_rns) {
        auto& [t, rns] = t_rns;
        if (rns.size() > 0) {
          auto lb =
            std::lower_bound(
              lhs.begin(),
              lhs.end(),
              t,
              [](auto& l, auto& t) { return std::get<0>(l) < t; });
          if (lb != lhs.end() && std::get<0>(*lb) == t) {
            auto& lrns = std::get<1>(*lb);
            auto l = lrns.begin();
            auto r = rns.begin();
            while (r!= rns.end()) {
              if (*r < *l) {
                lrns.insert(l, *r);
                ++r;
              } else if (*r == *l) {
                ++r;
              } else {
                ++l;
              }
            }
          } else {
            lhs.insert(lb, t_rns);
          }
        }
      });
  }

  template <bool EXCL>
  static void
  apply(LHS& lhs, RHS rhs) {
    combine(lhs, rhs);
  }

  static const RHS identity;

  template <bool EXCL>
  static void
  fold(RHS& rhs1, RHS rhs2) {
    combine(rhs1, rhs2);
  }
};

template <typename T>
typename acc_field_redop<T>::RHS const acc_field_redop<T>::identity = {};

class OpsManager {
public:

  static void
  register_ops() {
    std::call_once(
      initialized,
      []() {
        Legion::Runtime::register_custom_serdez_op<index_tree_serdez>(
          INDEX_TREE_SID);

        Legion::Runtime::register_custom_serdez_op<
          vector_serdez<Legion::DomainPoint>>(V_DOMAIN_POINT_SID);

        Legion::Runtime::register_custom_serdez_op<
          acc_field_serdez<casacore::String>>(ACC_FIELD_STRING_SID);
        Legion::Runtime::register_custom_serdez_op<
          acc_field_serdez<casacore::Bool>>(ACC_FIELD_BOOL_SID);
        Legion::Runtime::register_custom_serdez_op<
          acc_field_serdez<casacore::Char>>(ACC_FIELD_CHAR_SID);
        Legion::Runtime::register_custom_serdez_op<
          acc_field_serdez<casacore::uChar>>(ACC_FIELD_UCHAR_SID);
        Legion::Runtime::register_custom_serdez_op<
          acc_field_serdez<casacore::Short>>(ACC_FIELD_SHORT_SID);
        Legion::Runtime::register_custom_serdez_op<
          acc_field_serdez<casacore::uShort>>(ACC_FIELD_USHORT_SID);
        Legion::Runtime::register_custom_serdez_op<
          acc_field_serdez<casacore::Int>>(ACC_FIELD_INT_SID);
        Legion::Runtime::register_custom_serdez_op<
          acc_field_serdez<casacore::uInt>>(ACC_FIELD_UINT_SID);
        Legion::Runtime::register_custom_serdez_op<
          acc_field_serdez<casacore::Float>>(ACC_FIELD_FLOAT_SID);
        Legion::Runtime::register_custom_serdez_op<
          acc_field_serdez<casacore::Double>>(ACC_FIELD_DOUBLE_SID);
        Legion::Runtime::register_custom_serdez_op<
          acc_field_serdez<casacore::Complex>>(ACC_FIELD_COMPLEX_SID);
        Legion::Runtime::register_custom_serdez_op<
          acc_field_serdez<casacore::DComplex>>(ACC_FIELD_DCOMPLEX_SID);

        Legion::Runtime::register_reduction_op<bool_or_redop>(BOOL_OR_REDOP);

        Legion::Runtime::register_custom_serdez_op<
          string_serdez<casacore::String>>(CASACORE_STRING_SID);
        Legion::Runtime::register_custom_serdez_op<
          vector_serdez<casacore::Bool>>(CASACORE_V_BOOL_SID);
        Legion::Runtime::register_custom_serdez_op<
          vector_serdez<casacore::Char>>(CASACORE_V_CHAR_SID);
        Legion::Runtime::register_custom_serdez_op<
          vector_serdez<casacore::uChar>>(CASACORE_V_UCHAR_SID);
        Legion::Runtime::register_custom_serdez_op<
          vector_serdez<casacore::Short>>(CASACORE_V_SHORT_SID);
        Legion::Runtime::register_custom_serdez_op<
          vector_serdez<casacore::uShort>>(CASACORE_V_USHORT_SID);
        Legion::Runtime::register_custom_serdez_op<
          vector_serdez<casacore::Int>>(CASACORE_V_INT_SID);
        Legion::Runtime::register_custom_serdez_op<
          vector_serdez<casacore::uInt>>(CASACORE_V_UINT_SID);
        Legion::Runtime::register_custom_serdez_op<
          vector_serdez<casacore::Float>>(CASACORE_V_FLOAT_SID);
        Legion::Runtime::register_custom_serdez_op<
          vector_serdez<casacore::Double>>(CASACORE_V_DOUBLE_SID);
        Legion::Runtime::register_custom_serdez_op<
          vector_serdez<casacore::Complex>>(CASACORE_V_COMPLEX_SID);
        Legion::Runtime::register_custom_serdez_op<
          vector_serdez<casacore::DComplex>>(CASACORE_V_DCOMPLEX_SID);

        Legion::Runtime::register_reduction_op<
          acc_field_redop<casacore::String>>(ACC_FIELD_STRING_REDOP);
        Legion::Runtime::register_reduction_op<
          acc_field_redop<casacore::Bool>>(ACC_FIELD_BOOL_REDOP);
        Legion::Runtime::register_reduction_op<
          acc_field_redop<casacore::Char>>(ACC_FIELD_CHAR_REDOP);
        Legion::Runtime::register_reduction_op<
          acc_field_redop<casacore::uChar>>(ACC_FIELD_UCHAR_REDOP);
        Legion::Runtime::register_reduction_op<
          acc_field_redop<casacore::Short>>(ACC_FIELD_SHORT_REDOP);
        Legion::Runtime::register_reduction_op<
          acc_field_redop<casacore::uShort>>(ACC_FIELD_USHORT_REDOP);
        Legion::Runtime::register_reduction_op<
          acc_field_redop<casacore::Int>>(ACC_FIELD_INT_REDOP);
        Legion::Runtime::register_reduction_op<
          acc_field_redop<casacore::uInt>>(ACC_FIELD_UINT_REDOP);
        Legion::Runtime::register_reduction_op<
          acc_field_redop<casacore::Float>>(ACC_FIELD_FLOAT_REDOP);
        Legion::Runtime::register_reduction_op<
          acc_field_redop<casacore::Double>>(ACC_FIELD_DOUBLE_REDOP);
        Legion::Runtime::register_reduction_op<
          acc_field_redop<casacore::Complex>>(ACC_FIELD_COMPLEX_REDOP);
        Legion::Runtime::register_reduction_op<
          acc_field_redop<casacore::DComplex>>(ACC_FIELD_DCOMPLEX_REDOP);
      });
  }

  enum {
    INDEX_TREE_SID = 1,

    V_DOMAIN_POINT_SID,

    ACC_FIELD_STRING_SID,
    ACC_FIELD_BOOL_SID,
    ACC_FIELD_CHAR_SID,
    ACC_FIELD_UCHAR_SID,
    ACC_FIELD_SHORT_SID,
    ACC_FIELD_USHORT_SID,
    ACC_FIELD_INT_SID,
    ACC_FIELD_UINT_SID,
    ACC_FIELD_FLOAT_SID,
    ACC_FIELD_DOUBLE_SID,
    ACC_FIELD_COMPLEX_SID,
    ACC_FIELD_DCOMPLEX_SID,

    CASACORE_STRING_SID,
    CASACORE_V_BOOL_SID,
    CASACORE_V_CHAR_SID,
    CASACORE_V_UCHAR_SID,
    CASACORE_V_SHORT_SID,
    CASACORE_V_USHORT_SID,
    CASACORE_V_INT_SID,
    CASACORE_V_UINT_SID,
    CASACORE_V_FLOAT_SID,
    CASACORE_V_DOUBLE_SID,
    CASACORE_V_COMPLEX_SID,
    CASACORE_V_DCOMPLEX_SID
  };

  enum {
    BOOL_OR_REDOP = 1,

    ACC_FIELD_STRING_REDOP,
    ACC_FIELD_BOOL_REDOP,
    ACC_FIELD_CHAR_REDOP,
    ACC_FIELD_UCHAR_REDOP,
    ACC_FIELD_SHORT_REDOP,
    ACC_FIELD_USHORT_REDOP,
    ACC_FIELD_INT_REDOP,
    ACC_FIELD_UINT_REDOP,
    ACC_FIELD_FLOAT_REDOP,
    ACC_FIELD_DOUBLE_REDOP,
    ACC_FIELD_COMPLEX_REDOP,
    ACC_FIELD_DCOMPLEX_REDOP,
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
  constexpr static const char* s = "Bool";
  constexpr static int id = 0;
  constexpr static int serdez_id = 0;
  constexpr static int v_serdez_id = OpsManager::CASACORE_V_BOOL_SID;
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_BOOL_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_BOOL_REDOP;
};

template <>
struct DataType<casacore::DataType::TpChar> {
  typedef casacore::Char ValueType;
  constexpr static const char* s = "Char";
  constexpr static int id = 1;
  constexpr static int serdez_id = 0;
  constexpr static int v_serdez_id = OpsManager::CASACORE_V_CHAR_SID;
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_CHAR_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_CHAR_REDOP;
};

template <>
struct DataType<casacore::DataType::TpUChar> {
  typedef casacore::uChar ValueType;
  constexpr static const char* s = "uChar";
  constexpr static int id = 2;
  constexpr static int serdez_id = 0;
  constexpr static int v_serdez_id = OpsManager::CASACORE_V_UCHAR_SID;
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_UCHAR_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_UCHAR_REDOP;
};

template <>
struct DataType<casacore::DataType::TpShort> {
  typedef casacore::Short ValueType;
  constexpr static const char* s = "Short";
  constexpr static int id = 3;
  constexpr static int serdez_id = 0;
  constexpr static int v_serdez_id = OpsManager::CASACORE_V_SHORT_SID;
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_SHORT_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_SHORT_REDOP;
};

template <>
struct DataType<casacore::DataType::TpUShort> {
  typedef casacore::uShort ValueType;
  constexpr static const char* s = "uShort";
  constexpr static int id = 4;
  constexpr static int serdez_id = 0;
  constexpr static int v_serdez_id = OpsManager::CASACORE_V_USHORT_SID;
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_USHORT_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_USHORT_REDOP;
};

template <>
struct DataType<casacore::DataType::TpInt> {
  typedef casacore::Int ValueType;
  constexpr static const char* s = "Int";
  constexpr static int id = 5;
  constexpr static int serdez_id = 0;
  constexpr static int v_serdez_id = OpsManager::CASACORE_V_INT_SID;
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_INT_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_INT_REDOP;
};

template <>
struct DataType<casacore::DataType::TpUInt> {
  typedef casacore::uInt ValueType;
  constexpr static const char* s = "uInt";
  constexpr static int id = 6;
  constexpr static int serdez_id = 0;
  constexpr static int v_serdez_id = OpsManager::CASACORE_V_UINT_SID;
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_UINT_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_UINT_REDOP;
};

template <>
struct DataType<casacore::DataType::TpFloat> {
  typedef casacore::Float ValueType;
  constexpr static const char* s = "Float";
  constexpr static int id = 7;
  constexpr static int serdez_id = 0;
  constexpr static int v_serdez_id = OpsManager::CASACORE_V_FLOAT_SID;
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_FLOAT_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_FLOAT_REDOP;
};

template <>
struct DataType<casacore::DataType::TpDouble> {
  typedef casacore::Double ValueType;
  constexpr static const char* s = "Double";
  constexpr static int id = 8;
  constexpr static int serdez_id = 0;
  constexpr static int v_serdez_id = OpsManager::CASACORE_V_DOUBLE_SID;
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_DOUBLE_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_DOUBLE_REDOP;
};

template <>
struct DataType<casacore::DataType::TpComplex> {
  typedef casacore::Complex ValueType;
  constexpr static const char* s = "Complex";
  constexpr static int id = 9;
  constexpr static int serdez_id = 0;
  constexpr static int v_serdez_id = OpsManager::CASACORE_V_COMPLEX_SID;
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_COMPLEX_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_COMPLEX_REDOP;
};

template <>
struct DataType<casacore::DataType::TpDComplex> {
  typedef casacore::DComplex ValueType;
  constexpr static const char* s = "DComplex";
  constexpr static int id = 10;
  constexpr static int serdez_id = 0;
  constexpr static int v_serdez_id = OpsManager::CASACORE_V_DCOMPLEX_SID;
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_DCOMPLEX_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_DCOMPLEX_REDOP;
};

template <>
struct DataType<casacore::DataType::TpString> {
  typedef casacore::String ValueType;
  constexpr static const char* s = "String";
  constexpr static int id = 11;
  constexpr static int serdez_id = OpsManager::CASACORE_STRING_SID;
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_STRING_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_STRING_REDOP;
};

#define NUM_CASACORE_DATATYPES (DataType<casacore::DataType::TpString>::id + 1)

#define FOREACH_DATATYPE(__func__)                  \
__func__(casacore::DataType::TpBool)                \
__func__(casacore::DataType::TpChar)                \
__func__(casacore::DataType::TpUChar)               \
__func__(casacore::DataType::TpShort)               \
__func__(casacore::DataType::TpUShort)              \
__func__(casacore::DataType::TpInt)                 \
__func__(casacore::DataType::TpUInt)                \
__func__(casacore::DataType::TpFloat)               \
__func__(casacore::DataType::TpDouble)              \
__func__(casacore::DataType::TpComplex)             \
__func__(casacore::DataType::TpDComplex)            \
__func__(casacore::DataType::TpString)

template <typename T>
struct ValueType {
  // constexpr static const casacore::DataType DataType;
};

template <>
struct ValueType<casacore::Bool> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpBool;
};

template <>
struct ValueType<casacore::Char> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpChar;
};

template <>
struct ValueType<casacore::uChar> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpUChar;
};

template <>
struct ValueType<casacore::Short> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpShort;
};

template <>
struct ValueType<casacore::uShort> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpUShort;
};

template <>
struct ValueType<casacore::Int> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpInt;
};

template <>
struct ValueType<casacore::uInt> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpUInt;
};

template <>
struct ValueType<casacore::Float> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpFloat;
};

template <>
struct ValueType<casacore::Double> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpDouble;
};

template <>
struct ValueType<casacore::Complex> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpComplex;
};

template <>
struct ValueType<casacore::DComplex> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpDComplex;
};

template <>
struct ValueType<casacore::String> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpString;
};

class ProjectedIndexPartitionTask
  : Legion::IndexTaskLauncher {
public:

  enum { IMAGE_RANGES_FID };

  struct args {
    Legion::Domain bounds;
    int prjdim;
    int dmap[];
  };

  constexpr static const char * const TASK_NAME =
    "legms::ProjectedIndexPartitionTask";
  static Legion::TaskID TASK_ID;

  ProjectedIndexPartitionTask(
    Legion::IndexSpace launch_space,
    Legion::LogicalPartition lp,
    Legion::LogicalRegion lr,
    args* global_arg);

  void
  dispatch(Legion::Context ctx, Legion::Runtime* runtime);

  static void
  base_impl(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *runtime);

  static void
  register_task(Legion::Runtime* runtime);
};

template <int IPDIM, int PRJDIM>
Legion::IndexPartitionT<PRJDIM>
projected_index_partition(
  Legion::Context ctx,
  Legion::Runtime* runtime,
  Legion::IndexPartitionT<IPDIM> ip,
  Legion::IndexSpaceT<PRJDIM> prj_is,
  const std::array<int, PRJDIM>& dmap) {

  assert(
    std::all_of(
      dmap.begin(),
      dmap.end(),
      [](auto d) { return -1 <= d && d < IPDIM; }));

  std::unique_ptr<ProjectedIndexPartitionTask::args> args(
    static_cast<ProjectedIndexPartitionTask::args*>(
      operator new(sizeof(ProjectedIndexPartitionTask::args)
                   + PRJDIM * sizeof(dmap[0]))));
  auto prj_domain = runtime->get_index_space_domain(ctx, prj_is);
  args->bounds = Legion::Rect<PRJDIM>(prj_domain.lo(), prj_domain.hi());
  args->prjdim = PRJDIM;
  std::memcpy(args->dmap, dmap.data(), PRJDIM * sizeof(dmap[0]));

  Legion::FieldSpace images_fs = runtime->create_field_space(ctx);
  {
    auto fa = runtime->create_field_allocator(ctx, images_fs);
    fa.allocate_field(
      sizeof(Legion::Rect<PRJDIM>),
      ProjectedIndexPartitionTask::IMAGE_RANGES_FID);
  }
  Legion::LogicalRegionT<IPDIM> images_lr(
    runtime->create_logical_region(
      ctx,
      runtime->get_parent_index_space(ip),
      images_fs));
  Legion::LogicalPartitionT<IPDIM> images_lp(
    runtime->get_logical_partition(ctx, images_lr, ip));

  Legion::IndexSpace ip_cs =
    runtime->get_index_partition_color_space_name(ctx, ip);

  ProjectedIndexPartitionTask
    fill_images(ip_cs, images_lp, images_lr, args.get());
  fill_images.dispatch(ctx, runtime);

  Legion::IndexPartitionT<PRJDIM> result(
    runtime->create_partition_by_image_range(
      ctx,
      prj_is,
      images_lp,
      images_lr,
      0,
      ip_cs));

  runtime->destroy_logical_partition(ctx, images_lp);
  runtime->destroy_logical_region(ctx, images_lr);
  runtime->destroy_field_space(ctx, images_fs);
  return result;
}

Legion::IndexPartition
projected_index_partition(
  Legion::Context ctx,
  Legion::Runtime* runtime,
  Legion::IndexPartition ip,
  Legion::IndexSpace prj_is,
  const std::vector<int>& dmap);

template <typename D>
struct AxisPartition {
  D dim;
  Legion::coord_t stride;
  Legion::coord_t offset;
  Legion::coord_t lo;
  Legion::coord_t hi;

  bool
  operator==(const AxisPartition& other) {
    return
      dim == other.dim
      && stride == other.stride
      && offset == other.offset
      && lo == other.lo
      && hi == other.hi;
  }

  bool
  operator!=(const AxisPartition& other) {
    return !operator==(other);
  }

  bool
  operator<(const AxisPartition& other) {
    if (dim < other.dim) return true;
    if (dim == other.dim) {
      if (stride < other.stride) return true;
      if (stride == other.stride) {
        if (offset < other.offset) return true;
        if (offset == other.offset) {
          if (lo < other.lo) return true;
          if (lo == other.lo)
            return hi < other.hi;
        }
      }
    }
    return false;
  }
};

Legion::IndexPartition
create_partition_on_axes(
  Legion::Context ctx,
  Legion::Runtime* runtime,
  Legion::IndexSpace is,
  const std::vector<AxisPartition<int>>& parts);

void
register_tasks(Legion::Runtime* runtime);

#if USE_HDF

// TODO: it might be nice to support use of types of IndexSpace descriptions
// other than IndexTree...this might require some sort of type registration
// interface, the descriptions would have to support a
// serialization/deserialization interface, and the type would have to be
// recorded in another HDF5 attribute

// FIXME: HDF5 call error handling

template <typename COORD_T>
void
write_to_attr(
  const IndexTree<COORD_T>& spec,
  hid_t loc_id,
  const std::string& obj_name,
  const std::string& attr_name) {

  // remove current attribute value (it's OK if some of these calls return an
  // error)
  H5Adelete_by_name(
    loc_id,
    obj_name.c_str(),
    attr_name.c_str(),
    H5P_DEFAULT);
  std::string attr_ds_name = obj_name + "-" + attr_name;
  H5Gunlink(loc_id, attr_ds_name.c_str());

  auto size = spec.serialized_size();
  auto buf = std::make_unique<char[]>(size);
  spec.serialize(buf.get());
  hsize_t value_dims = size;
  hid_t value_space_id = H5Screate_simple(1, &value_dims, &value_dims);
  if (size < (64 * (1 << 10))) {
    // small serialized size: save byte string as an attribute
    hid_t attr_type = H5T_NATIVE_UINT8;
    hid_t attr_id =
      H5Acreate_by_name(
        loc_id,
        obj_name.c_str(),
        attr_name.c_str(),
        attr_type,
        value_space_id,
        H5P_DEFAULT,
        H5P_DEFAULT,
        H5P_DEFAULT);
    assert(attr_id >= 0);
    herr_t rc = H5Awrite(attr_id, H5T_NATIVE_UINT8, buf.get());
    assert (rc >= 0);
  } else {
    // large serialized size: create a new dataset containing byte string, and
    // save reference to that dataset as attribute
    hid_t attr_ds =
      H5Dcreate(
        loc_id,
        attr_ds_name.c_str(),
        H5T_NATIVE_UINT8,
        value_space_id,
        H5P_DEFAULT,
        H5P_DEFAULT,
        H5P_DEFAULT);
    herr_t rc =
      H5Dwrite(
        attr_ds,
        H5T_NATIVE_UINT8,
        H5S_ALL,
        H5S_ALL,
        H5P_DEFAULT,
        buf.get());
    assert(rc >= 0);

    hsize_t ref_dims = 1;
    hid_t ref_space_id = H5Screate_simple(1, &ref_dims, &ref_dims);
    hid_t attr_type = H5T_STD_REF_OBJ;
    hid_t attr_id =
      H5Acreate_by_name(
        loc_id,
        obj_name.c_str(),
        attr_name.c_str(),
        attr_type,
        ref_space_id,
        H5P_DEFAULT,
        H5P_DEFAULT,
        H5P_DEFAULT);
    assert(attr_id >= 0);
    hobj_ref_t attr_ref;
    rc = H5Rcreate(&attr_ref, loc_id, attr_ds_name.c_str(), H5R_OBJECT, -1);
    assert (rc >= 0);
    rc = H5Awrite(attr_id, H5T_STD_REF_OBJ, &attr_ref);
    assert (rc >= 0);
    H5Sclose(ref_space_id);
  }
  H5Sclose(value_space_id);
}

template <typename COORD_T>
std::optional<IndexTree<COORD_T>>
read_from_attr(
  hid_t loc_id,
  const std::string& obj_name,
  const std::string& attr_name) {

  hid_t attr_id =
    H5Aopen_by_name(
      loc_id,
      obj_name.c_str(),
      attr_name.c_str(),
      H5P_DEFAULT,
      H5P_DEFAULT);

  std::optional<IndexTree<COORD_T>> result;
  if (attr_id < 0)
    return result;

  hid_t attr_type = H5Aget_type(attr_id);
  if (H5Tequal(attr_type, H5T_NATIVE_UINT8) > 0) {

    // serialized value was written into attribute
    hid_t attr_ds = H5Aget_space(attr_id);
    assert(attr_ds >= 0);
    hssize_t attr_sz = H5Sget_simple_extent_npoints(attr_ds);
    assert(attr_sz >= 0);
    H5Dclose(attr_ds);
    auto buf = std::make_unique<char[]>(static_cast<size_t>(attr_sz));
    herr_t rc = H5Aread(attr_id, H5T_NATIVE_UINT8, buf.get());
    assert(rc >= 0);
    result = IndexTree<COORD_T>::deserialize(buf.get());

  } else if (H5Tequal(attr_type, H5T_STD_REF_OBJ) > 0) {

    // serialized value is in a dataset referenced by attribute
    hobj_ref_t attr_ref;
    herr_t rc = H5Aread(attr_id, H5T_STD_REF_OBJ, &attr_ref);
    assert(rc >= 0);
    hid_t attr_ds =
      H5Rdereference2(loc_id, H5P_DEFAULT, H5R_OBJECT, &attr_ref);
    assert(attr_ds >= 0);
    hssize_t attr_sz = H5Sget_simple_extent_npoints(attr_ds);
    assert(attr_sz >= 0);
    auto buf = std::make_unique<char[]>(static_cast<size_t>(attr_sz));
    rc =
      H5Dread(
        attr_ds,
        H5T_NATIVE_UINT8,
        H5S_ALL,
        H5S_ALL,
        H5P_DEFAULT,
        buf.get());
    assert(rc >= 0);
    H5Dclose(attr_ds);
    result = IndexTree<COORD_T>::deserialize(buf.get());
  }
  H5Tclose(attr_type);
  return result;
}

#endif //USE_HDF

#if LEGMS_MAX_DIM == 1

#define LEGMS_FOREACH_N(__func__)               \
  __func__(1) 
#define LEGMS_FOREACH_NN(__func__)              \
  __func__(1,1)
#define LEGMS_FOREACH_MN(__func__)              \
  __func__(1,1)

#elif LEGMS_MAX_DIM == 2

#define LEGMS_FOREACH_N(__func__)               \
  __func__(1)                                   \
  __func__(2)
#define LEGMS_FOREACH_NN(__func__)              \
  __func__(1,1)                                 \
  __func__(1,2)                                 \
  __func__(2,1)                                 \
  __func__(2,2)
#define LEGMS_FOREACH_MN(__func__)              \
  __func__(1,1)                                 \
  __func__(1,2)                                 \
  __func__(2,2)

#elif LEGMS_MAX_DIM == 3

#define LEGMS_FOREACH_N(__func__)               \
  __func__(1)                                   \
  __func__(2)                                   \
  __func__(3)
#define LEGMS_FOREACH_NN(__func__)              \
  __func__(1,1)                                 \
  __func__(1,2)                                 \
  __func__(1,3)                                 \
  __func__(2,1)                                 \
  __func__(2,2)                                 \
  __func__(2,3)                                 \
  __func__(3,1)                                 \
  __func__(3,2)                                 \
  __func__(3,3)
#define LEGMS_FOREACH_MN(__func__)              \
  __func__(1,1)                                 \
  __func__(1,2)                                 \
  __func__(1,3)                                 \
  __func__(2,2)                                 \
  __func__(2,3)                                 \
  __func__(3,3)

#elif LEGMS_MAX_DIM == 4

#define LEGMS_FOREACH_N(__func__)               \
  __func__(1)                                   \
  __func__(2)                                   \
  __func__(3)                                   \
  __func__(4)
#define LEGMS_FOREACH_NN(__func__)              \
  __func__(1,1)                                 \
  __func__(1,2)                                 \
  __func__(1,3)                                 \
  __func__(1,4)                                 \
  __func__(2,1)                                 \
  __func__(2,2)                                 \
  __func__(2,3)                                 \
  __func__(2,4)                                 \
  __func__(3,1)                                 \
  __func__(3,2)                                 \
  __func__(3,3)                                 \
  __func__(3,4)                                 \
  __func__(4,1)                                 \
  __func__(4,2)                                 \
  __func__(4,3)                                 \
  __func__(4,4)
#define LEGMS_FOREACH_MN(__func__)              \
  __func__(1,1)                                 \
  __func__(1,2)                                 \
  __func__(1,3)                                 \
  __func__(1,4)                                 \
  __func__(2,2)                                 \
  __func__(2,3)                                 \
  __func__(2,4)                                 \
  __func__(3,3)                                 \
  __func__(3,4)                                 \
  __func__(4,4)

#elif LEGMS_MAX_DIM == 5

#define LEGMS_FOREACH_N(__func__)               \
  __func__(1)                                   \
  __func__(2)                                   \
  __func__(3)                                   \
  __func__(4)                                   \
  __func__(5)
#define LEGMS_FOREACH_NN(__func__)              \
  __func__(1,1)                                 \
  __func__(1,2)                                 \
  __func__(1,3)                                 \
  __func__(1,4)                                 \
  __func__(1,5)                                 \
  __func__(2,1)                                 \
  __func__(2,2)                                 \
  __func__(2,3)                                 \
  __func__(2,4)                                 \
  __func__(2,5)                                 \
  __func__(3,1)                                 \
  __func__(3,2)                                 \
  __func__(3,3)                                 \
  __func__(3,4)                                 \
  __func__(3,5)                                 \
  __func__(4,1)                                 \
  __func__(4,2)                                 \
  __func__(4,3)                                 \
  __func__(4,4)                                 \
  __func__(4,5)                                 \
  __func__(5,1)                                 \
  __func__(5,2)                                 \
  __func__(5,3)                                 \
  __func__(5,4)                                 \
  __func__(5,5)
#define LEGMS_FOREACH_MN(__func__)              \
  __func__(1,1)                                 \
  __func__(1,2)                                 \
  __func__(1,3)                                 \
  __func__(1,4)                                 \
  __func__(1,5)                                 \
  __func__(2,2)                                 \
  __func__(2,3)                                 \
  __func__(2,4)                                 \
  __func__(2,5)                                 \
  __func__(3,3)                                 \
  __func__(3,4)                                 \
  __func__(3,5)                                 \
  __func__(4,4)                                 \
  __func__(4,5)                                 \
  __func__(5,5)

#elif LEGMS_MAX_DIM == 6

#define LEGMS_FOREACH_N(__func__)               \
  __func__(1)                                   \
  __func__(2)                                   \
  __func__(3)                                   \
  __func__(4)                                   \
  __func__(5)                                   \
  __func__(6)
#define LEGMS_FOREACH_NN(__func__)              \
  __func__(1,1)                                 \
  __func__(1,2)                                 \
  __func__(1,3)                                 \
  __func__(1,4)                                 \
  __func__(1,5)                                 \
  __func__(1,6)                                 \
  __func__(2,1)                                 \
  __func__(2,2)                                 \
  __func__(2,3)                                 \
  __func__(2,4)                                 \
  __func__(2,5)                                 \
  __func__(2,6)                                 \
  __func__(3,1)                                 \
  __func__(3,2)                                 \
  __func__(3,3)                                 \
  __func__(3,4)                                 \
  __func__(3,5)                                 \
  __func__(3,6)                                 \
  __func__(4,1)                                 \
  __func__(4,2)                                 \
  __func__(4,3)                                 \
  __func__(4,4)                                 \
  __func__(4,5)                                 \
  __func__(4,6)                                 \
  __func__(5,1)                                 \
  __func__(5,2)                                 \
  __func__(5,3)                                 \
  __func__(5,4)                                 \
  __func__(5,5)                                 \
  __func__(5,6)                                 \
  __func__(6,1)                                 \
  __func__(6,2)                                 \
  __func__(6,3)                                 \
  __func__(6,4)                                 \
  __func__(6,5)                                 \
  __func__(6,6)
#define LEGMS_FOREACH_MN(__func__)              \
  __func__(1,1)                                 \
  __func__(1,2)                                 \
  __func__(1,3)                                 \
  __func__(1,4)                                 \
  __func__(1,5)                                 \
  __func__(1,6)                                 \
  __func__(2,2)                                 \
  __func__(2,3)                                 \
  __func__(2,4)                                 \
  __func__(2,5)                                 \
  __func__(2,6)                                 \
  __func__(3,3)                                 \
  __func__(3,4)                                 \
  __func__(3,5)                                 \
  __func__(3,6)                                 \
  __func__(4,4)                                 \
  __func__(4,5)                                 \
  __func__(4,6)                                 \
  __func__(5,5)                                 \
  __func__(5,6)                                 \
  __func__(6,6)

#elif LEGMS_MAX_DIM == 7

#define LEGMS_FOREACH_N(__func__)               \
  __func__(1)                                   \
  __func__(2)                                   \
  __func__(3)                                   \
  __func__(4)                                   \
  __func__(5)                                   \
  __func__(6)                                   \
  __func__(7)
#define LEGMS_FOREACH_NN(__func__)              \
  __func__(1,1)                                 \
  __func__(1,2)                                 \
  __func__(1,3)                                 \
  __func__(1,4)                                 \
  __func__(1,5)                                 \
  __func__(1,6)                                 \
  __func__(1,7)                                 \
  __func__(2,1)                                 \
  __func__(2,2)                                 \
  __func__(2,3)                                 \
  __func__(2,4)                                 \
  __func__(2,5)                                 \
  __func__(2,6)                                 \
  __func__(2,7)                                 \
  __func__(3,1)                                 \
  __func__(3,2)                                 \
  __func__(3,3)                                 \
  __func__(3,4)                                 \
  __func__(3,5)                                 \
  __func__(3,6)                                 \
  __func__(3,7)                                 \
  __func__(4,1)                                 \
  __func__(4,2)                                 \
  __func__(4,3)                                 \
  __func__(4,4)                                 \
  __func__(4,5)                                 \
  __func__(4,6)                                 \
  __func__(4,7)                                 \
  __func__(5,1)                                 \
  __func__(5,2)                                 \
  __func__(5,3)                                 \
  __func__(5,4)                                 \
  __func__(5,5)                                 \
  __func__(5,6)                                 \
  __func__(5,7)                                 \
  __func__(6,1)                                 \
  __func__(6,2)                                 \
  __func__(6,3)                                 \
  __func__(6,4)                                 \
  __func__(6,5)                                 \
  __func__(6,6)                                 \
  __func__(6,7)                                 \
  __func__(7,1)                                 \
  __func__(7,2)                                 \
  __func__(7,3)                                 \
  __func__(7,4)                                 \
  __func__(7,5)                                 \
  __func__(7,6)                                 \
  __func__(7,7)
#define LEGMS_FOREACH_MN(__func__)              \
  __func__(1,1)                                 \
  __func__(1,2)                                 \
  __func__(1,3)                                 \
  __func__(1,4)                                 \
  __func__(1,5)                                 \
  __func__(1,6)                                 \
  __func__(1,7)                                 \
  __func__(2,2)                                 \
  __func__(2,3)                                 \
  __func__(2,4)                                 \
  __func__(2,5)                                 \
  __func__(2,6)                                 \
  __func__(2,7)                                 \
  __func__(3,3)                                 \
  __func__(3,4)                                 \
  __func__(3,5)                                 \
  __func__(3,6)                                 \
  __func__(3,7)                                 \
  __func__(4,4)                                 \
  __func__(4,5)                                 \
  __func__(4,6)                                 \
  __func__(4,7)                                 \
  __func__(5,5)                                 \
  __func__(5,6)                                 \
  __func__(5,7)                                 \
  __func__(6,6)                                 \
  __func__(6,7)                                 \
  __func__(7,7)

#elif LEGMS_MAX_DIM == 8

#define LEGMS_FOREACH_N(__func__)               \
  __func__(1)                                   \
  __func__(2)                                   \
  __func__(3)                                   \
  __func__(4)                                   \
  __func__(5)                                   \
  __func__(6)                                   \
  __func__(7)                                   \
  __func__(8)
#define LEGMS_FOREACH_NN(__func__)              \
  __func__(1,1)                                 \
  __func__(1,2)                                 \
  __func__(1,3)                                 \
  __func__(1,4)                                 \
  __func__(1,5)                                 \
  __func__(1,6)                                 \
  __func__(1,7)                                 \
  __func__(1,8)                                 \
  __func__(2,1)                                 \
  __func__(2,2)                                 \
  __func__(2,3)                                 \
  __func__(2,4)                                 \
  __func__(2,5)                                 \
  __func__(2,6)                                 \
  __func__(2,7)                                 \
  __func__(2,8)                                 \
  __func__(3,1)                                 \
  __func__(3,2)                                 \
  __func__(3,3)                                 \
  __func__(3,4)                                 \
  __func__(3,5)                                 \
  __func__(3,6)                                 \
  __func__(3,7)                                 \
  __func__(3,8)                                 \
  __func__(4,1)                                 \
  __func__(4,2)                                 \
  __func__(4,3)                                 \
  __func__(4,4)                                 \
  __func__(4,5)                                 \
  __func__(4,6)                                 \
  __func__(4,7)                                 \
  __func__(4,8)                                 \
  __func__(5,1)                                 \
  __func__(5,2)                                 \
  __func__(5,3)                                 \
  __func__(5,4)                                 \
  __func__(5,5)                                 \
  __func__(5,6)                                 \
  __func__(5,7)                                 \
  __func__(5,8)                                 \
  __func__(6,1)                                 \
  __func__(6,2)                                 \
  __func__(6,3)                                 \
  __func__(6,4)                                 \
  __func__(6,5)                                 \
  __func__(6,6)                                 \
  __func__(6,7)                                 \
  __func__(6,8)                                 \
  __func__(7,1)                                 \
  __func__(7,2)                                 \
  __func__(7,3)                                 \
  __func__(7,4)                                 \
  __func__(7,5)                                 \
  __func__(7,6)                                 \
  __func__(7,7)                                 \
  __func__(7,8)                                 \
  __func__(8,1)                                 \
  __func__(8,2)                                 \
  __func__(8,3)                                 \
  __func__(8,4)                                 \
  __func__(8,5)                                 \
  __func__(8,6)                                 \
  __func__(8,7)                                 \
  __func__(8,8)
#define LEGMS_FOREACH_MN(__func__)              \
  __func__(1,1)                                 \
  __func__(1,2)                                 \
  __func__(1,3)                                 \
  __func__(1,4)                                 \
  __func__(1,5)                                 \
  __func__(1,6)                                 \
  __func__(1,7)                                 \
  __func__(1,8)                                 \
  __func__(2,2)                                 \
  __func__(2,3)                                 \
  __func__(2,4)                                 \
  __func__(2,5)                                 \
  __func__(2,6)                                 \
  __func__(2,7)                                 \
  __func__(2,8)                                 \
  __func__(3,3)                                 \
  __func__(3,4)                                 \
  __func__(3,5)                                 \
  __func__(3,6)                                 \
  __func__(3,7)                                 \
  __func__(3,8)                                 \
  __func__(4,4)                                 \
  __func__(4,5)                                 \
  __func__(4,6)                                 \
  __func__(4,7)                                 \
  __func__(4,8)                                 \
  __func__(5,5)                                 \
  __func__(5,6)                                 \
  __func__(5,7)                                 \
  __func__(5,8)                                 \
  __func__(6,6)                                 \
  __func__(6,7)                                 \
  __func__(6,8)                                 \
  __func__(7,7)                                 \
  __func__(7,8)                                 \
  __func__(8,8)

#elif LEGMS_MAX_DIM == 9

#define LEGMS_FOREACH_N(__func__)               \
  __func__(1)                                   \
  __func__(2)                                   \
  __func__(3)                                   \
  __func__(4)                                   \
  __func__(5)                                   \
  __func__(6)                                   \
  __func__(7)                                   \
  __func__(8)                                   \
  __func__(9)
#define LEGMS_FOREACH_NN(__func__)              \
  __func__(1,1)                                 \
  __func__(1,2)                                 \
  __func__(1,3)                                 \
  __func__(1,4)                                 \
  __func__(1,5)                                 \
  __func__(1,6)                                 \
  __func__(1,7)                                 \
  __func__(1,8)                                 \
  __func__(1,9)                                 \
  __func__(2,1)                                 \
  __func__(2,2)                                 \
  __func__(2,3)                                 \
  __func__(2,4)                                 \
  __func__(2,5)                                 \
  __func__(2,6)                                 \
  __func__(2,7)                                 \
  __func__(2,8)                                 \
  __func__(2,9)                                 \
  __func__(3,1)                                 \
  __func__(3,2)                                 \
  __func__(3,3)                                 \
  __func__(3,4)                                 \
  __func__(3,5)                                 \
  __func__(3,6)                                 \
  __func__(3,7)                                 \
  __func__(3,8)                                 \
  __func__(3,9)                                 \
  __func__(4,1)                                 \
  __func__(4,2)                                 \
  __func__(4,3)                                 \
  __func__(4,4)                                 \
  __func__(4,5)                                 \
  __func__(4,6)                                 \
  __func__(4,7)                                 \
  __func__(4,8)                                 \
  __func__(4,9)                                 \
  __func__(5,1)                                 \
  __func__(5,2)                                 \
  __func__(5,3)                                 \
  __func__(5,4)                                 \
  __func__(5,5)                                 \
  __func__(5,6)                                 \
  __func__(5,7)                                 \
  __func__(5,8)                                 \
  __func__(5,9)                                 \
  __func__(6,1)                                 \
  __func__(6,2)                                 \
  __func__(6,3)                                 \
  __func__(6,4)                                 \
  __func__(6,5)                                 \
  __func__(6,6)                                 \
  __func__(6,7)                                 \
  __func__(6,8)                                 \
  __func__(6,9)                                 \
  __func__(7,1)                                 \
  __func__(7,2)                                 \
  __func__(7,3)                                 \
  __func__(7,4)                                 \
  __func__(7,5)                                 \
  __func__(7,6)                                 \
  __func__(7,7)                                 \
  __func__(7,8)                                 \
  __func__(7,9)                                 \
  __func__(8,1)                                 \
  __func__(8,2)                                 \
  __func__(8,3)                                 \
  __func__(8,4)                                 \
  __func__(8,5)                                 \
  __func__(8,6)                                 \
  __func__(8,7)                                 \
  __func__(8,8)                                 \
  __func__(8,9)                                 \
  __func__(9,1)                                 \
  __func__(9,2)                                 \
  __func__(9,3)                                 \
  __func__(9,4)                                 \
  __func__(9,5)                                 \
  __func__(9,6)                                 \
  __func__(9,7)                                 \
  __func__(9,8)                                 \
  __func__(9,9)
#define LEGMS_FOREACH_MN(__func__)              \
  __func__(1,1)                                 \
  __func__(1,2)                                 \
  __func__(1,3)                                 \
  __func__(1,4)                                 \
  __func__(1,5)                                 \
  __func__(1,6)                                 \
  __func__(1,7)                                 \
  __func__(1,8)                                 \
  __func__(1,9)                                 \
  __func__(2,2)                                 \
  __func__(2,3)                                 \
  __func__(2,4)                                 \
  __func__(2,5)                                 \
  __func__(2,6)                                 \
  __func__(2,7)                                 \
  __func__(2,8)                                 \
  __func__(2,9)                                 \
  __func__(3,3)                                 \
  __func__(3,4)                                 \
  __func__(3,5)                                 \
  __func__(3,6)                                 \
  __func__(3,7)                                 \
  __func__(3,8)                                 \
  __func__(3,9)                                 \
  __func__(4,4)                                 \
  __func__(4,5)                                 \
  __func__(4,6)                                 \
  __func__(4,7)                                 \
  __func__(4,8)                                 \
  __func__(4,9)                                 \
  __func__(5,5)                                 \
  __func__(5,6)                                 \
  __func__(5,7)                                 \
  __func__(5,8)                                 \
  __func__(5,9)                                 \
  __func__(6,6)                                 \
  __func__(6,7)                                 \
  __func__(6,8)                                 \
  __func__(6,9)                                 \
  __func__(7,7)                                 \
  __func__(7,8)                                 \
  __func__(7,9)                                 \
  __func__(8,8)                                 \
  __func__(8,9)                                 \
  __func__(9,9)

#else
#error "Unsupported LEGMS_MAX_DIM"
#endif

} // end namespace legms

#endif // LEGMS_UTILITY_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
