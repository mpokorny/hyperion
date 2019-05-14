#ifndef LEGMS_UTILITY_H_
#define LEGMS_UTILITY_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>

#ifdef USE_HDF5
# include <hdf5.h>
# include <experimental/filesystem>
#endif

#include <casacore/casa/aipstype.h>
#include <casacore/casa/Arrays/IPosition.h>
#include <casacore/casa/BasicSL/String.h>
#include <casacore/casa/Utilities/DataType.h>

#include "legms.h"

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
class acc_field_serdez<std::string> {
public:

  typedef std::vector<
  std::tuple<std::string, std::vector<Legion::DomainPoint>>> FIELD_TYPE;

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
            + string_serdez<std::string>::serialized_size(std::get<0>(t))
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
      buff += string_serdez<std::string>::serialize(t, buff);
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
      std::string str;
      buff += string_serdez<std::string>::deserialize(str, buff);
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

class OpsManager {
public:

  static void
  register_ops() {
    Legion::Runtime::register_custom_serdez_op<index_tree_serdez>(
      INDEX_TREE_SID);

    Legion::Runtime::register_custom_serdez_op<
      vector_serdez<Legion::DomainPoint>>(V_DOMAIN_POINT_SID);
    Legion::Runtime::register_custom_serdez_op<
      string_serdez<std::string>>(STD_STRING_SID);

    Legion::Runtime::register_custom_serdez_op<
      acc_field_serdez<std::string>>(ACC_FIELD_STRING_SID);
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

    Legion::Runtime::register_reduction_op(
      ACC_FIELD_STRING_REDOP,
      Realm::ReductionOpUntyped::create_reduction_op<
      acc_field_redop<std::string>>(),
      acc_field_redop<std::string>::init_fn,
      acc_field_redop<std::string>::fold_fn);
    Legion::Runtime::register_reduction_op(
      ACC_FIELD_BOOL_REDOP,
      Realm::ReductionOpUntyped::create_reduction_op<
      acc_field_redop<casacore::Bool>>(),
      acc_field_redop<casacore::Bool>::init_fn,
      acc_field_redop<casacore::Bool>::fold_fn);
    Legion::Runtime::register_reduction_op(
      ACC_FIELD_CHAR_REDOP,
      Realm::ReductionOpUntyped::create_reduction_op<
      acc_field_redop<casacore::Char>>(),
      acc_field_redop<casacore::Char>::init_fn,
      acc_field_redop<casacore::Char>::fold_fn);
    Legion::Runtime::register_reduction_op(
      ACC_FIELD_UCHAR_REDOP,
      Realm::ReductionOpUntyped::create_reduction_op<
      acc_field_redop<casacore::uChar>>(),
      acc_field_redop<casacore::uChar>::init_fn,
      acc_field_redop<casacore::uChar>::fold_fn);
    Legion::Runtime::register_reduction_op(
      ACC_FIELD_SHORT_REDOP,
      Realm::ReductionOpUntyped::create_reduction_op<
      acc_field_redop<casacore::Short>>(),
      acc_field_redop<casacore::Short>::init_fn,
      acc_field_redop<casacore::Short>::fold_fn);
    Legion::Runtime::register_reduction_op(
      ACC_FIELD_USHORT_REDOP,
      Realm::ReductionOpUntyped::create_reduction_op<
      acc_field_redop<casacore::uShort>>(),
      acc_field_redop<casacore::uShort>::init_fn,
      acc_field_redop<casacore::uShort>::fold_fn);
    Legion::Runtime::register_reduction_op(
      ACC_FIELD_INT_REDOP,
      Realm::ReductionOpUntyped::create_reduction_op<
      acc_field_redop<casacore::Int>>(),
      acc_field_redop<casacore::Int>::init_fn,
      acc_field_redop<casacore::Int>::fold_fn);
    Legion::Runtime::register_reduction_op(
      ACC_FIELD_UINT_REDOP,
      Realm::ReductionOpUntyped::create_reduction_op<
      acc_field_redop<casacore::uInt>>(),
      acc_field_redop<casacore::uInt>::init_fn,
      acc_field_redop<casacore::uInt>::fold_fn);
    Legion::Runtime::register_reduction_op(
      ACC_FIELD_FLOAT_REDOP,
      Realm::ReductionOpUntyped::create_reduction_op<
      acc_field_redop<casacore::Float>>(),
      acc_field_redop<casacore::Float>::init_fn,
      acc_field_redop<casacore::Float>::fold_fn);
    Legion::Runtime::register_reduction_op(
      ACC_FIELD_DOUBLE_REDOP,
      Realm::ReductionOpUntyped::create_reduction_op<
      acc_field_redop<casacore::Double>>(),
      acc_field_redop<casacore::Double>::init_fn,
      acc_field_redop<casacore::Double>::fold_fn);
    Legion::Runtime::register_reduction_op(
      ACC_FIELD_COMPLEX_REDOP,
      Realm::ReductionOpUntyped::create_reduction_op<
      acc_field_redop<casacore::Complex>>(),
      acc_field_redop<casacore::Complex>::init_fn,
      acc_field_redop<casacore::Complex>::fold_fn);
    Legion::Runtime::register_reduction_op(
      ACC_FIELD_DCOMPLEX_REDOP,
      Realm::ReductionOpUntyped::create_reduction_op<
      acc_field_redop<casacore::DComplex>>(),
      acc_field_redop<casacore::DComplex>::init_fn,
      acc_field_redop<casacore::DComplex>::fold_fn);
  }

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
    CASACORE_BOOL_H5T = 0,
    CASACORE_CHAR_H5T,
    CASACORE_UCHAR_H5T,
    CASACORE_SHORT_H5T,
    CASACORE_USHORT_H5T,
    CASACORE_INT_H5T,
    CASACORE_UINT_H5T,
    CASACORE_FLOAT_H5T,
    CASACORE_DOUBLE_H5T,
    CASACORE_COMPLEX_H5T,
    CASACORE_DCOMPLEX_H5T,
    CASACORE_STRING_H5T,
    CASACORE_DATATYPE_H5T,
  };

  static void
  register_datatypes();

  static const hid_t*
  datatypes() {
    return datatypes_;
  }

  template <casacore::DataType DT>
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

  static hid_t datatypes_[CASACORE_STRING_H5T + 1];
};
#endif

Legion::FieldID
add_field(
  casacore::DataType datatype,
  Legion::FieldAllocator fa,
  Legion::FieldID field_id = AUTO_GENERATE_ID);

template <casacore::DataType T>
struct DataType {

  //typedef X ValueType;
};

struct string {
  char val[LEGMS_MAX_STRING_SIZE];
};

template <>
struct DataType<casacore::DataType::TpBool> {
  typedef casacore::Bool ValueType;
  typedef casacore::Bool CasacoreType;
  constexpr static const char* s = "Bool";
  constexpr static int id = 0;
  constexpr static size_t serdez_size = sizeof(casacore::Bool);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_BOOL_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_BOOL_REDOP;
#ifdef USE_HDF5
  constexpr static hid_t h5t_native_id = H5DatatypeManager::CASACORE_BOOL_H5T;
#endif
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
};

template <>
struct DataType<casacore::DataType::TpChar> {
  typedef casacore::Char ValueType;
  typedef casacore::Char CasacoreType;
  constexpr static const char* s = "Char";
  constexpr static int id = 1;
  constexpr static size_t serdez_size = sizeof(casacore::Char);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_CHAR_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_CHAR_REDOP;
#ifdef USE_HDF5
  constexpr static hid_t h5t_native_id = H5DatatypeManager::CASACORE_CHAR_H5T;
#endif
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
};

template <>
struct DataType<casacore::DataType::TpUChar> {
  typedef casacore::uChar ValueType;
  typedef casacore::uChar CasacoreType;
  constexpr static const char* s = "uChar";
  constexpr static int id = 2;
    constexpr static size_t serdez_size = sizeof(casacore::uChar);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_UCHAR_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_UCHAR_REDOP;
#ifdef USE_HDF5
  constexpr static hid_t h5t_native_id = H5DatatypeManager::CASACORE_UCHAR_H5T;
#endif
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
};

template <>
struct DataType<casacore::DataType::TpShort> {
  typedef casacore::Short ValueType;
  typedef casacore::Short CasacoreType;
  constexpr static const char* s = "Short";
  constexpr static int id = 3;
  constexpr static size_t serdez_size = sizeof(casacore::Short);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_SHORT_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_SHORT_REDOP;
#ifdef USE_HDF5
  constexpr static hid_t h5t_native_id = H5DatatypeManager::CASACORE_SHORT_H5T;
#endif
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
};

template <>
struct DataType<casacore::DataType::TpUShort> {
  typedef casacore::uShort ValueType;
  typedef casacore::uShort CasacoreType;
  constexpr static const char* s = "uShort";
  constexpr static int id = 4;
  constexpr static size_t serdez_size = sizeof(casacore::uShort);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_USHORT_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_USHORT_REDOP;
#ifdef USE_HDF5
  constexpr static hid_t h5t_native_id = H5DatatypeManager::CASACORE_USHORT_H5T;
#endif
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
};

template <>
struct DataType<casacore::DataType::TpInt> {
  typedef casacore::Int ValueType;
  typedef casacore::Int CasacoreType;
  constexpr static const char* s = "Int";
  constexpr static int id = 5;
    constexpr static size_t serdez_size = sizeof(casacore::Int);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_INT_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_INT_REDOP;
#ifdef USE_HDF5
  constexpr static hid_t h5t_native_id = H5DatatypeManager::CASACORE_INT_H5T;
#endif
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
};

template <>
struct DataType<casacore::DataType::TpUInt> {
  typedef casacore::uInt ValueType;
  typedef casacore::uInt CasacoreType;
  constexpr static const char* s = "uInt";
  constexpr static int id = 6;
    constexpr static size_t serdez_size = sizeof(casacore::uInt);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_UINT_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_UINT_REDOP;
#ifdef USE_HDF5
  constexpr static hid_t h5t_native_id = H5DatatypeManager::CASACORE_UINT_H5T;
#endif
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
};

template <>
struct DataType<casacore::DataType::TpFloat> {
  typedef casacore::Float ValueType;
  typedef casacore::Float CasacoreType;
  constexpr static const char* s = "Float";
  constexpr static int id = 7;
    constexpr static size_t serdez_size = sizeof(casacore::Float);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_FLOAT_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_FLOAT_REDOP;
#ifdef USE_HDF5
  constexpr static hid_t h5t_native_id = H5DatatypeManager::CASACORE_FLOAT_H5T;
#endif
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
};

template <>
struct DataType<casacore::DataType::TpDouble> {
  typedef casacore::Double ValueType;
  typedef casacore::Double CasacoreType;
  constexpr static const char* s = "Double";
  constexpr static int id = 8;
    constexpr static size_t serdez_size = sizeof(casacore::Double);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_DOUBLE_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_DOUBLE_REDOP;
#ifdef USE_HDF5
  constexpr static hid_t h5t_native_id = H5DatatypeManager::CASACORE_DOUBLE_H5T;
#endif
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
};

template <>
struct DataType<casacore::DataType::TpComplex> {
  typedef casacore::Complex ValueType;
  typedef casacore::Complex CasacoreType;
  constexpr static const char* s = "Complex";
  constexpr static int id = 9;
    constexpr static size_t serdez_size = sizeof(casacore::Complex);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_COMPLEX_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_COMPLEX_REDOP;
#ifdef USE_HDF5
  constexpr static hid_t h5t_native_id =
    H5DatatypeManager::CASACORE_COMPLEX_H5T;
#endif
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
};

template <>
struct DataType<casacore::DataType::TpDComplex> {
  typedef casacore::DComplex ValueType;
  typedef casacore::DComplex CasacoreType;
  constexpr static const char* s = "DComplex";
  constexpr static int id = 10;
    constexpr static size_t serdez_size = sizeof(casacore::DComplex);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_DCOMPLEX_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_DCOMPLEX_REDOP;
#ifdef USE_HDF5
  constexpr static hid_t h5t_native_id =
    H5DatatypeManager::CASACORE_DCOMPLEX_H5T;
#endif
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
};

template <>
struct DataType<casacore::DataType::TpString> {
  typedef string ValueType;
  typedef casacore::String CasacoreType;
  constexpr static const char* s = "String";
  constexpr static int id = 11;
    constexpr static size_t serdez_size = LEGMS_MAX_STRING_SIZE;
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_STRING_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_STRING_REDOP;
#ifdef USE_HDF5
  constexpr static hid_t h5t_native_id =
    H5DatatypeManager::CASACORE_STRING_H5T;
#endif
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    std::strncpy(v.val, c.c_str(), sizeof(v.val));
    v.val[sizeof(v.val) - 1] = '\0';
  }
};

#ifdef USE_HDF5
template <casacore::DataType DT>
hid_t
H5DatatypeManager::datatype() {
  return datatypes()[DataType<DT>::h5t_native_id];
}
#endif

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
struct ValueType<std::string> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpString;
};

template <>
struct ValueType<string> {
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
