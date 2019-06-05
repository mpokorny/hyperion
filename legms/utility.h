#ifndef LEGMS_UTILITY_H_
#define LEGMS_UTILITY_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <complex>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>

#include "legms.h"
#include "IndexTree.h"

#ifdef USE_HDF5
# include <hdf5.h>
# include <experimental/filesystem>
#endif

#ifdef USE_CASACORE
# include <casacore/casa/aipstype.h>
# include <casacore/casa/Arrays/IPosition.h>
# include <casacore/casa/BasicSL/String.h>
# include <casacore/casa/Utilities/DataType.h>
#endif

namespace legms {

typedef IndexTree<Legion::coord_t> IndexTreeL;

template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

struct string {

  string() {
    val[0] = '\0';
  }

  string(const std::string& s) {
    std::strncpy(val, s.c_str(), sizeof(val));
    val[sizeof(val) - 1] = '\0';
  }

  string(const char* s) {
    std::strncpy(val, s, sizeof(val));
    val[sizeof(val) - 1] = '\0';  
  }

  char val[LEGMS_MAX_STRING_SIZE];

  bool
  operator==(const string& other) {
    return std::strncmp(val, other.val, sizeof(val)) == 0;
  }

  bool
  operator!=(const string& other) {
    return std::strncmp(val, other.val, sizeof(val)) != 0;
  }

  bool
  operator<(const string& other) {
    return std::strncmp(val, other.val, sizeof(val)) < 0;
  }
};

template <typename F>
bool
operator<(const std::complex<F>& a, const std::complex<F>& b) {
  if (a.real() < b.real())
    return true;
  if (a.real() > b.real())
    return false;
  return a.imag() < b.imag();
}

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

#ifdef USE_CASACORE
# define TTDEF(dt) dt = casacore::DataType::dt
#else
# define TTDEF(dt) dt
#endif
enum TypeTag : unsigned char {
  TTDEF(TpBool),
  TTDEF(TpChar),
  TTDEF(TpUChar),
  TTDEF(TpShort),
  TTDEF(TpUShort),
  TTDEF(TpInt),
  TTDEF(TpUInt),
  TTDEF(TpFloat),
  TTDEF(TpDouble),
  TTDEF(TpComplex),
  TTDEF(TpDComplex),
  TTDEF(TpString)
};
#undef TTDEF

// uid of axes
template <typename T>
struct AxesUID {
  // static const char* id;
};

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

  static const size_t MAX_SERIALIZED_SIZE = 1000;

  static const size_t MAX_CHARLEN = MAX_SERIALIZED_SIZE - sizeof(size_t);

  static size_t
  serialized_size(const S& val) {
    assert(val.size() < MAX_CHARLEN);
    return sizeof(size_t) + val.size() + 1;
  }

  static size_t
  serialize(const S& val, void *buffer) {
    assert(val.size() < MAX_CHARLEN);
    char* buff = static_cast<char*>(buffer);
    *reinterpret_cast<size_t*>(buff) = val.size();
    buff += sizeof(size_t);
    std::strcpy(buff, val.c_str());
    buff += val.size();
    *buff = '\0';
    buff += 1;
    return buff - static_cast<char*>(buffer);
  }

  static size_t
  deserialize(S& val, const void *buffer) {
    const char* buff = static_cast<const char*>(buffer);
    const size_t& len = *reinterpret_cast<const size_t*>(buff);
    val.resize(len);
    buff += sizeof(size_t);
    std::strncpy(val.data(), buff, len);
    buff += len + 1;
    return buff - static_cast<const char*>(buffer);
  }

  static void
  destroy(S&) {
  }
};

template <>
class acc_field_serdez<legms::string> {
public:

  typedef std::vector<
  std::tuple<legms::string, std::vector<Legion::DomainPoint>>> FIELD_TYPE;

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
            + sizeof(legms::string)
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
      std::memcpy(buff, t.val, sizeof(legms::string));
      buff += sizeof(legms::string);
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
      legms::string str;
      std::memcpy(str.val, buff, sizeof(legms::string));
      buff += sizeof(legms::string);
      std::vector<Legion::DomainPoint> rns;
      buff += vector_serdez<Legion::DomainPoint>::deserialize(rns, buff);
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
struct acc_field_redop_rhs {
  std::vector<std::tuple<T, std::vector<Legion::DomainPoint>>> v;

  size_t
  legion_buffer_size(void) const {
    return acc_field_serdez<T>::serialized_size(v);
  }

  size_t
  legion_serialize(void* buffer) const {
    return acc_field_serdez<T>::serialize(v, buffer);
  }

  size_t
  legion_deserialize(const void* buffer) {
    return acc_field_serdez<T>::deserialize(v, buffer);
  }
};

template <typename T>
class acc_field_redop {
public:
  typedef std::vector<std::tuple<T, std::vector<Legion::DomainPoint>>> LHS;
  typedef acc_field_redop_rhs<T> RHS;

  static void
  combine(LHS& lhs, const RHS& rhs) {
    std::for_each(
      rhs.v.begin(),
      rhs.v.end(),
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
              if (l != lrns.end()) {
                if (*r < *l) {
                  lrns.insert(l, *r);
                  ++r;
                } else if (*r == *l) {
                  ++r;
                } else {
                  ++l;
                }
              } else {
                lrns.push_back(*r);
                ++r;
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
  apply(LHS& lhs, const RHS& rhs) {
    combine(lhs, rhs);
  }

  static const RHS identity;

  template <bool EXCL>
  static void
  fold(RHS& rhs1, const RHS& rhs2) {
    combine(rhs1.v, rhs2);
  }

  static void
  init_fn(
    const Legion::ReductionOp*,
    void *& state,
    size_t& sz __attribute__((unused))) {
    assert(sz >= sizeof(RHS));
    ::new(state) RHS;
  }

  static void
  fold_fn(
    const Legion::ReductionOp* reduction_op,
    void *& state,
    size_t& sz __attribute__((unused)),
    const void* result) {
    RHS rhs;
    rhs.legion_deserialize(result);
    reduction_op->fold(state, &rhs, 1, true);
  }
};

template <typename T>
typename acc_field_redop<T>::RHS const acc_field_redop<T>::identity =
  acc_field_redop_rhs<T>{{}};

template <TypeTag T>
struct DataType {

  //typedef X ValueType;
};

struct OpsManager {
public:

  static void
  register_ops();

  enum {
    INDEX_TREE_SID = 1,

    V_DOMAIN_POINT_SID,

    STD_STRING_SID,

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
};

#ifdef USE_HDF5
class H5DatatypeManager {
public:

  // TODO: add support for non-native types in HDF5 files

  enum {
    BOOL_H5T = 0,
    CHAR_H5T,
    UCHAR_H5T,
    SHORT_H5T,
    USHORT_H5T,
    INT_H5T,
    UINT_H5T,
    FLOAT_H5T,
    DOUBLE_H5T,
    COMPLEX_H5T,
    DCOMPLEX_H5T,
    STRING_H5T,
    DATATYPE_H5T,
  };

  static void
  register_datatypes();

  static const hid_t*
  datatypes() {
    return datatypes_;
  }

  template <TypeTag DT>
  static hid_t
  datatype();

  static herr_t
  commit_derived(
    hid_t loc_id,
    hid_t lcpl_id = H5P_DEFAULT,
    hid_t tcpl_id = H5P_DEFAULT,
    hid_t tapl_id = H5P_DEFAULT);

  static hid_t
  create(
    const std::experimental::filesystem::path& path,
    unsigned flags,
    hid_t fcpl_t = H5P_DEFAULT,
    hid_t fapl_t = H5P_DEFAULT);

private:

  static hid_t datatypes_[DATATYPE_H5T + 1];
};
#endif

Legion::FieldID
add_field(
  TypeTag datatype,
  Legion::FieldAllocator fa,
  Legion::FieldID field_id = AUTO_GENERATE_ID);

template <>
struct DataType<TypeTag::TpBool> {
  typedef bool ValueType;
  constexpr static const char* s = "bool";
  constexpr static int id = 0;
  constexpr static size_t serdez_size = sizeof(ValueType);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_BOOL_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_BOOL_REDOP;
#ifdef USE_CASACORE
  typedef casacore::Bool CasacoreType;
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
#endif
#ifdef USE_HDF5
  constexpr static hid_t h5t_id = H5DatatypeManager::BOOL_H5T;
#endif
  static bool equiv(const ValueType& a, const ValueType& b) {
    return a == b;
  }
};

template <>
struct DataType<TypeTag::TpChar> {
  typedef char ValueType;
  constexpr static const char* s = "char";
  constexpr static int id = 1;
  constexpr static size_t serdez_size = sizeof(ValueType);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_CHAR_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_CHAR_REDOP;
#ifdef USE_CASACORE
  typedef casacore::Char CasacoreType;
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
#endif
#ifdef USE_HDF5
  constexpr static hid_t h5t_id = H5DatatypeManager::CHAR_H5T;
#endif
  static bool equiv(const ValueType& a, const ValueType& b) {
    return a == b;
  }
};

template <>
struct DataType<TypeTag::TpUChar> {
  typedef unsigned char ValueType;
  constexpr static const char* s = "uChar";
  constexpr static int id = 2;
  constexpr static size_t serdez_size = sizeof(ValueType);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_UCHAR_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_UCHAR_REDOP;
#ifdef USE_CASACORE
  typedef casacore::uChar CasacoreType;
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
#endif
#ifdef USE_HDF5
  constexpr static hid_t h5t_id = H5DatatypeManager::UCHAR_H5T;
#endif
  static bool equiv(const ValueType& a, const ValueType& b) {
    return a == b;
  }
};

template <>
struct DataType<TypeTag::TpShort> {
  typedef short ValueType;
  constexpr static const char* s = "short";
  constexpr static int id = 3;
  constexpr static size_t serdez_size = sizeof(ValueType);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_SHORT_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_SHORT_REDOP;
#ifdef USE_CASACORE
  typedef casacore::Short CasacoreType;
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
#endif
#ifdef USE_HDF5
  constexpr static hid_t h5t_id = H5DatatypeManager::SHORT_H5T;
#endif
  static bool equiv(const ValueType& a, const ValueType& b) {
    return a == b;
  }
};

template <>
struct DataType<TypeTag::TpUShort> {
  typedef unsigned short ValueType;
  constexpr static const char* s = "uShort";
  constexpr static int id = 4;
  constexpr static size_t serdez_size = sizeof(ValueType);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_USHORT_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_USHORT_REDOP;
#ifdef USE_CASACORE
  typedef casacore::uShort CasacoreType;
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
#endif
#ifdef USE_HDF5
  constexpr static hid_t h5t_id = H5DatatypeManager::USHORT_H5T;
#endif
  static bool equiv(const ValueType& a, const ValueType& b) {
    return a == b;
  }
};

template <>
struct DataType<TypeTag::TpInt> {
  typedef int ValueType;
  constexpr static const char* s = "int";
  constexpr static int id = 5;
  constexpr static size_t serdez_size = sizeof(ValueType);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_INT_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_INT_REDOP;
#ifdef USE_CASACORE
  typedef casacore::Int CasacoreType;
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
#endif
#ifdef USE_HDF5
  constexpr static hid_t h5t_id = H5DatatypeManager::INT_H5T;
#endif
  static bool equiv(const ValueType& a, const ValueType& b) {
    return a == b;
  }
};

template <>
struct DataType<TypeTag::TpUInt> {
  typedef unsigned int ValueType;
  constexpr static const char* s = "uInt";
  constexpr static int id = 6;
  constexpr static size_t serdez_size = sizeof(ValueType);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_UINT_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_UINT_REDOP;
#ifdef USE_CASACORE
  typedef casacore::uInt CasacoreType;
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
#endif
#ifdef USE_HDF5
  constexpr static hid_t h5t_id = H5DatatypeManager::UINT_H5T;
#endif
  static bool equiv(const ValueType& a, const ValueType& b) {
    return a == b;
  }
};

template <>
struct DataType<TypeTag::TpFloat> {
  typedef float ValueType;
  constexpr static const char* s = "float";
  constexpr static int id = 7;
  constexpr static size_t serdez_size = sizeof(ValueType);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_FLOAT_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_FLOAT_REDOP;
#ifdef USE_CASACORE
  typedef casacore::Float CasacoreType;
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
#endif
#ifdef USE_HDF5
  constexpr static hid_t h5t_id = H5DatatypeManager::FLOAT_H5T;
#endif
  static bool equiv(const ValueType& a, const ValueType& b) {
    return a == b;
  }
};

template <>
struct DataType<TypeTag::TpDouble> {
  typedef double ValueType;
  constexpr static const char* s = "double";
  constexpr static int id = 8;
  constexpr static size_t serdez_size = sizeof(ValueType);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_DOUBLE_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_DOUBLE_REDOP;
#ifdef USE_CASACORE
  typedef casacore::Double CasacoreType;
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
#endif
#ifdef USE_HDF5
  constexpr static hid_t h5t_id = H5DatatypeManager::DOUBLE_H5T;
#endif
  static bool equiv(const ValueType& a, const ValueType& b) {
    return a == b;
  }
};

template <>
struct DataType<TypeTag::TpComplex> {
  typedef std::complex<float> ValueType;
  constexpr static const char* s = "complex";
  constexpr static int id = 9;
  constexpr static size_t serdez_size = sizeof(ValueType);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_COMPLEX_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_COMPLEX_REDOP;
#ifdef USE_CASACORE
  typedef casacore::Complex CasacoreType;
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
#endif
#ifdef USE_HDF5
  constexpr static hid_t h5t_id = H5DatatypeManager::COMPLEX_H5T;
#endif
  static bool equiv(const ValueType& a, const ValueType& b) {
    return a == b;
  }
};

template <>
struct DataType<TypeTag::TpDComplex> {
  typedef std::complex<double> ValueType;
  constexpr static const char* s = "dComplex";
  constexpr static int id = 10;
  constexpr static size_t serdez_size = sizeof(ValueType);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_DCOMPLEX_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_DCOMPLEX_REDOP;
#ifdef USE_CASACORE
  typedef casacore::DComplex CasacoreType;
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
#endif
#ifdef USE_HDF5
  constexpr static hid_t h5t_id = H5DatatypeManager::DCOMPLEX_H5T;
#endif
  static bool equiv(const ValueType& a, const ValueType& b) {
    return a == b;
  }
};

template <>
struct DataType<TypeTag::TpString> {
  typedef legms::string ValueType;
  constexpr static const char* s = "string";
  constexpr static int id = 11;
  constexpr static size_t serdez_size = sizeof(ValueType);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_STRING_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_STRING_REDOP;
#ifdef USE_CASACORE
  typedef casacore::String CasacoreType;
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    std::strncpy(v.val, c.c_str(), sizeof(v.val));
    v.val[sizeof(v.val) - 1] = '\0';
  }
#endif
#ifdef USE_HDF5
  constexpr static hid_t h5t_id = H5DatatypeManager::STRING_H5T;
#endif
  static bool equiv(const ValueType& a, const ValueType& b) {
    return std::strcmp(a.val, b.val) == 0;
  }
};

#ifdef USE_HDF5
template <TypeTag DT>
hid_t
H5DatatypeManager::datatype() {
  return datatypes()[DataType<DT>::h5t_id];
}
#endif

#define NUM_LEGMS_DATATYPES (DataType<legms::TypeTag::TpString>::id + 1)

#define FOREACH_DATATYPE(__func__)              \
  __func__(legms::TypeTag::TpBool)              \
  __func__(legms::TypeTag::TpChar)              \
  __func__(legms::TypeTag::TpUChar)             \
  __func__(legms::TypeTag::TpShort)             \
  __func__(legms::TypeTag::TpUShort)            \
  __func__(legms::TypeTag::TpInt)               \
  __func__(legms::TypeTag::TpUInt)              \
  __func__(legms::TypeTag::TpFloat)             \
  __func__(legms::TypeTag::TpDouble)            \
  __func__(legms::TypeTag::TpComplex)           \
  __func__(legms::TypeTag::TpDComplex)          \
  __func__(legms::TypeTag::TpString)

#define FOREACH_BARE_DATATYPE(__func__)  \
  __func__(TpBool)          \
  __func__(TpChar)          \
  __func__(TpUChar)         \
  __func__(TpShort)         \
  __func__(TpUShort)        \
  __func__(TpInt)           \
  __func__(TpUInt)          \
  __func__(TpFloat)         \
  __func__(TpDouble)        \
  __func__(TpComplex)       \
  __func__(TpDComplex)      \
  __func__(TpString)

template <typename T>
struct ValueType {
  // constexpr static const TypeTag DataType;
};

#define VT(dt)                                  \
  template <>                                   \
  struct ValueType<DataType<dt>::ValueType> {   \
    constexpr static TypeTag DataType = dt;   \
  };

FOREACH_DATATYPE(VT)

template <>
struct ValueType<std::string> {
  constexpr static TypeTag DataType = TypeTag::TpString;
};
#undef VT

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

#if LEGION_MAX_DIM == 1

#define LEGMS_FOREACH_N(__func__)               \
  __func__(1) 
#define LEGMS_FOREACH_NN(__func__)              \
  __func__(1,1)
#define LEGMS_FOREACH_MN(__func__)              \
  __func__(1,1)

#elif LEGION_MAX_DIM == 2

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

#elif LEGION_MAX_DIM == 3

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

#elif LEGION_MAX_DIM == 4

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

#elif LEGION_MAX_DIM == 5

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

#elif LEGION_MAX_DIM == 6

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

#elif LEGION_MAX_DIM == 7

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

#elif LEGION_MAX_DIM == 8

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

#elif LEGION_MAX_DIM == 9

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
#error "Unsupported LEGION_MAX_DIM"
#endif

} // end namespace legms

std::ostream&
operator<<(std::ostream& stream, const legms::string& str);

#endif // LEGMS_UTILITY_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
