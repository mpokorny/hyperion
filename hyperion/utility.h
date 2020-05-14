/*
 * Copyright 2020 Associated Universities, Inc. Washington DC, USA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef HYPERION_UTILITY_H_
#define HYPERION_UTILITY_H_

#include <hyperion/hyperion.h>
#include <hyperion/IndexTree.h>
#include <hyperion/utility_c.h>

#pragma GCC visibility push(default)
# include <mappers/default_mapper.h>

# include <algorithm>
# include <array>
# include <atomic>
# include <cassert>
# include <complex>
# include <cstring>
# include <limits>
# include <memory>
# include <numeric>
# include <optional>
# include <set>
# include <type_traits>
# include <unordered_map>
# include <unordered_set>

# include CXX_FILESYSTEM_HEADER

# ifdef HYPERION_USE_HDF5
#  include <hdf5.h>
# endif

# ifdef HYPERION_USE_CASACORE
#  include <casacore/casa/aipstype.h>
#  include <casacore/casa/Arrays/IPosition.h>
#  include <casacore/casa/BasicSL/String.h>
#  include <casacore/casa/Utilities/DataType.h>
# endif
#pragma GCC visibility pop

// The following reiterates a constant in Realm -- unfortunately there is no way
// to either import or discover its value. Hopefully the impact is minor, as
// this value should only be required by serdez functions, and Realm has
// implemented a runtime check.
#define HYPERION_IB_MAX_SIZE (63 * (((size_t)1) << 20))

namespace hyperion {

class Table;
class ColumnSpace;
class TableField;

typedef IndexTree<Legion::coord_t> IndexTreeL;

template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

template <typename T, typename F>
std::optional<std::invoke_result_t<F, T>>
map(const std::optional<T>& ot, F f) {
  return
    ot
    ? std::make_optional(f(ot.value()))
    : std::optional<std::invoke_result_t<F, T>>();
}

template <typename T, typename F>
std::invoke_result_t<F, T>
flatMap(const std::optional<T>& ot, F f) {
  return ot ? f(ot.value()) : std::invoke_result_t<F, T>();
}

template <template <typename> typename C, typename T, typename F>
C<std::invoke_result_t<F, T>>
map(const C<T>& ts, F f) {
  C<std::invoke_result_t<F, T>> result;
  std::transform(
    ts.begin(),
    ts.end(),
    std::inserter(result, result.end()),
    [&f](const auto& t) { return f(t); });
  return result;
}

template <template <typename> typename C, typename T>
C<int>
map_to_int(const C<T>& ts) {
  return map(ts, [](const auto& t) { return static_cast<int>(t); });
}

void
toupper(std::string& s);

std::string
toupper(const std::string& s);

bool
has_suffix(const std::string& str, const std::string& suffix);

std::string
add_name_prefix(const std::string& prefix, const std::string& str);

template <typename D>
struct Axes {
  static const char* uid;
  static const std::vector<std::string> names;
  static const unsigned num_axes;
#ifdef HYPERION_USE_HDF5
  static const hid_t h5_datatype;
#endif
};

HYPERION_LOCAL std::optional<int>
column_is_axis(
  const std::vector<std::string>& axis_names,
  const std::string& colname,
  const std::vector<int>& axes);

template <size_t N>
char *
fstrcpy(char(& dest)[N], const std::string& src) {
  auto n = std::min(N - 1, src.size());
  std::strncpy(&dest[0], src.c_str(), n);
  dest[n] = '\0';
  return &dest[0];
}

template <size_t N>
char *
fstrcpy(char(& dest)[N], const char* src) {
  auto n = std::min(N - 1, std::strlen(src));
  std::strncpy(&dest[0], src, n);
  dest[n] = '\0';
  return &dest[0];
}

HYPERION_API unsigned
min_divisor(
  size_t numerator,
  size_t min_quotient,
  unsigned max_divisor);

// this returns an IndexPartition with a new IndexSpace as a color space, which
// the caller must eventually destroy
HYPERION_API Legion::IndexPartition
partition_over_default_tunable(
  Legion::Context ctx,
  Legion::Runtime* rt,
  Legion::IndexSpace is,
  size_t min_block_size,
  Legion::Mapping::DefaultMapper::DefaultTunables tunable);

HYPERION_API IndexTreeL
index_space_as_tree(Legion::Runtime* rt, Legion::IndexSpace is);

#ifdef HYPERION_USE_CASACORE
HYPERION_API std::tuple<
  std::string,
  ColumnSpace,
  std::vector<
    std::tuple<ColumnSpace, std::vector<std::pair<std::string, TableField>>>>>
from_ms(
  Legion::Context ctx,
  Legion::Runtime* runtime,
  const CXX_FILESYSTEM_NAMESPACE::path& path,
  const std::unordered_set<std::string>& column_selections);
#endif // HYPERION_USE_CASACORE

template <
  typename OPEN,
  typename F,
  typename CLOSE,
  std::enable_if_t<
    !std::is_void_v<
      std::invoke_result_t<F, std::invoke_result_t<OPEN>&>>,
    int> = 0>
std::invoke_result_t<F, std::invoke_result_t<OPEN>&>
using_resource(OPEN open, F f, CLOSE close) {
  auto r = open();
  std::invoke_result_t<F, std::invoke_result_t<OPEN>&> result;
  try {
    result = f(r);
  } catch (...) {
    close(r);
    throw;
  }
  close(r);
  return result;
}

template <
  typename OPEN,
  typename F,
  typename CLOSE,
  std::enable_if_t<
    std::is_void_v<
      std::invoke_result_t<F, std::invoke_result_t<OPEN>&>>,
    int> = 0>
void
using_resource(OPEN open, F f, CLOSE close) {
  auto r = open();
  try {
    f(r);
  } catch (...) {
    close(r);
    throw;
  }
  close(r);
}

struct HYPERION_API string {

  string() {
    val[0] = '\0';
  }

  string(const std::string& s) {
    fstrcpy(val, s);
  }

  string(const char* s) {
    fstrcpy(val, s);
  }

  string&
  operator=(const std::string& s) {
    fstrcpy(val, s);
    return *this;
  }

  string&
  operator=(const char* s) {
    fstrcpy(val, s);
    return *this;
  }

  char val[HYPERION_MAX_STRING_SIZE];

  operator std::string() const {
    return std::string(val);
  }

  size_t
  size() const {
    return std::strlen(val);
  };

  bool
  operator==(const string& other) const {
    return std::strncmp(val, other.val, sizeof(val)) == 0;
  }

  bool
  operator!=(const string& other) const {
    return std::strncmp(val, other.val, sizeof(val)) != 0;
  }

  bool
  operator<(const string& other) const {
    return std::strncmp(val, other.val, sizeof(val)) < 0;
  }
};

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

typedef ::type_tag_t TypeTag;

template <typename T>
class vector_serdez {
public:

  typedef std::vector<T> FIELD_TYPE;

  static const size_t MAX_SERIALIZED_SIZE = HYPERION_IB_MAX_SIZE;

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
      for (size_t i = 0; i < n; ++i)
        *tbuf++ = val[i];
    }
    return result;
  }

  static size_t
  deserialize(FIELD_TYPE& val, const void *buffer) {
    ::new(&val) FIELD_TYPE;
    size_t n = *static_cast<const size_t *>(buffer);
    if (n > 0) {
      val.resize(n);
      const T* tbuf =
        reinterpret_cast<const T*>(static_cast<const size_t *>(buffer) + 1);
      for (size_t i = 0; i < n; ++i)
        val[i] = *tbuf++;
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

  static const size_t MAX_SERIALIZED_SIZE = HYPERION_IB_MAX_SIZE;

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
    ::new(&val) FIELD_TYPE;
    size_t n = *static_cast<const size_t *>(buffer);
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

  static const size_t MAX_SERIALIZED_SIZE = HYPERION_IB_MAX_SIZE;

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
    ::new(&val) FIELD_TYPE;
    size_t n = *static_cast<const size_t *>(buffer);
    buffer = static_cast<const void *>(static_cast<const size_t*>(buffer) + 1);
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

  static const size_t MAX_CHARLEN = MAX_SERIALIZED_SIZE - 1;

  static size_t
  serialized_size(const S& val) {
    assert(val.size() < MAX_CHARLEN);
    return val.size() + 1;
  }

  static size_t
  serialize(const S& val, void *buffer) {
    assert(val.size() < MAX_CHARLEN);
    char* buff = static_cast<char*>(buffer);
    if (val.size() > 0) {
      std::strcpy(buff, val.c_str());
      buff += val.size();
    }
    *buff = '\0';
    buff += 1;
    return buff - static_cast<char*>(buffer);
  }

  static size_t
  deserialize(S& val, const void *buffer) {
    const char* buff = static_cast<const char*>(buffer);
    ::new(&val) S;
    val.assign(buff);
    buff += val.size() + 1;
    return buff - static_cast<const char*>(buffer);
  }

  static void
  destroy(S&) {
  }
};

template <>
class acc_field_serdez<hyperion::string> {
public:

  typedef std::vector<
  std::tuple<hyperion::string, std::vector<Legion::DomainPoint>>> FIELD_TYPE;

  static const size_t MAX_SERIALIZED_SIZE = HYPERION_IB_MAX_SIZE;

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
            + sizeof(hyperion::string)
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
      std::memcpy(buff, t.val, sizeof(hyperion::string));
      buff += sizeof(hyperion::string);
      buff += vector_serdez<Legion::DomainPoint>::serialize(rns, buff);
    }
    return result;
  }

  static size_t
  deserialize(FIELD_TYPE& val, const void *buffer) {
    ::new(&val) FIELD_TYPE;
    size_t n = *static_cast<const size_t *>(buffer);
    buffer = static_cast<const void *>(static_cast<const size_t*>(buffer) + 1);
    val.reserve(n);
    const char* buff = static_cast<const char*>(buffer);
    for (size_t i = 0; i < n; ++i) {
      hyperion::string str;
      std::memcpy(str.val, buff, sizeof(hyperion::string));
      buff += sizeof(hyperion::string);
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

class HYPERION_API index_tree_serdez {
public:
  typedef IndexTreeL FIELD_TYPE;

  static const size_t MAX_SERIALIZED_SIZE = HYPERION_IB_MAX_SIZE;

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

class HYPERION_LOCAL bool_or_redop {
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
    if (lhs.size() == 0) {
      lhs = rhs.v;
    } else {
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
                [](auto& a, auto& b) { return std::get<0>(a) < b; });
            if (lb != lhs.end() && std::get<0>(*lb) == t) {
              auto& lrns = std::get<1>(*lb);
              auto l = lrns.begin();
              auto r = rns.begin();
              while (r != rns.end() && l != lrns.end()) {
                if (*r < *l) {
                  // the following lrns.insert() call may invalidate the "l"
                  // iterator; to compensate, save the iterator's offset, do the
                  // insertion, and then recreate the iterator
                  auto lo = std::distance(l, lrns.begin());
                  lrns.insert(l, *r);
                  l = lrns.begin() + lo + 1;
                  ++r;
                } else if (*r == *l) {
                  ++r;
                } else {
                  ++l;
                }
              }
              std::copy(r, rns.end(), std::back_inserter(lrns));
            } else {
              lhs.insert(lb, t_rns);
            }
          }
        });
    }
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

#ifdef WITH_ACC_FIELD_REDOP_SERDEZ
  static void
  init_fn(const Legion::ReductionOp*, void *& state, size_t& sz) {
    RHS rhs;
    state = realloc(state, rhs.legion_buffer_size());
    sz = rhs.legion_serialize(state);
  }

  static void
  fold_fn(
    const Legion::ReductionOp* reduction_op,
    void *& state,
    size_t& sz,
    const void* result) {
    RHS rhs, lhs;
    rhs.legion_deserialize(result);
    lhs.legion_deserialize(state);
    reduction_op->fold(&lhs, &rhs, 1, true);
    sz = lhs.legion_buffer_size();
    state = realloc(state, sz);
    lhs.legion_serialize(state);
  }
#endif // WITH_ACC_FIELD_REDOP_SERDEZ
};

template <typename T>
typename acc_field_redop<T>::RHS const acc_field_redop<T>::identity =
  acc_field_redop_rhs<T>{{}};

struct HYPERION_API coord_bor_redop {

  typedef coord_t LHS;
  typedef coord_t RHS;

  static const constexpr coord_t identity = 0;

  template <bool EXCLUSIVE>
  static void
  apply(LHS& lhs, RHS rhs);

  template <bool EXCLUSIVE>
  static void
  fold(RHS& rhs1, RHS rhs2);
};

template <>
inline void
coord_bor_redop::apply<true>(LHS& lhs, RHS rhs) {
  lhs |= rhs;
}
template <>
inline void
coord_bor_redop::apply<false>(LHS& lhs, RHS rhs) {
  __atomic_or_fetch(&lhs, rhs, __ATOMIC_ACQUIRE);
}

template <>
inline void
coord_bor_redop::fold<true>(RHS& rhs1, RHS rhs2) {
  rhs1 |= rhs2;
}
template <>
inline void
coord_bor_redop::fold<false>(RHS& rhs1, RHS rhs2) {
  __atomic_or_fetch(&rhs1, rhs2, __ATOMIC_ACQUIRE);
}

template <TypeTag T>
struct DataType {

  //typedef X ValueType;
};

class HYPERION_API OpsManager {
public:

  enum {
    INDEX_TREE_SID = 0,

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
    ACC_FIELD_RECT2_SID,
    ACC_FIELD_RECT3_SID,

    NUM_SIDS
  };

  enum {
    BOOL_OR_REDOP = 0,

    COORD_BOR_REDOP,

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
    ACC_FIELD_RECT2_REDOP,
    ACC_FIELD_RECT3_REDOP,

    // this must be at the end
    POINT_ADD_REDOP_BASE
  };

  static void
  register_ops(Legion::Runtime* rt);

  static Legion::CustomSerdezID
  serdez_id_base;

  static inline int
  POINT_ADD_REDOP(int dim) {
    return POINT_ADD_REDOP_BASE + dim - 1;
  }

  static Legion::CustomSerdezID
  serdez_id(int sid) {
    return serdez_id_base + sid;
  }

  static Legion::ReductionOpID
  reduction_id_base;

  static Legion::ReductionOpID
  reduction_id(int redop) {
    return reduction_id_base + redop;
  }
};

template <int DIM>
class point_add_redop {
public:
  typedef Legion::Point<DIM> LHS;
  typedef Legion::Point<DIM> RHS;

  static const RHS identity;

  template <bool EXCL>
  static void
  apply(LHS& lhs, RHS rhs) {
    lhs += rhs;
  }

  template <bool EXCL>
  static void
  fold(RHS& rhs1, RHS rhs2) {
    rhs1 += rhs2;
  }
};

#ifdef HYPERION_USE_HDF5
class HYPERION_API H5DatatypeManager {
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
    FIELD_ID_H5T,
    RECT2_H5T,
    RECT3_H5T,
#ifdef HYPERION_USE_CASACORE
    MEASURE_CLASS_H5T,
#endif
    N_H5T_DATATYPES
  };

  static void
  preregister_datatypes();

  static const hid_t*
  datatypes() {
    return datatypes_;
  }

  template <TypeTag DT>
  static constexpr hid_t
  datatype();

  static herr_t
  commit_derived(
    hid_t loc_id,
    hid_t lcpl_id = H5P_DEFAULT,
    hid_t tcpl_id = H5P_DEFAULT,
    hid_t tapl_id = H5P_DEFAULT);

  static hid_t
  create(
    const CXX_FILESYSTEM_NAMESPACE::path& path,
    unsigned flags,
    hid_t fcpl_t = H5P_DEFAULT,
    hid_t fapl_t = H5P_DEFAULT);

private:

  static hid_t datatypes_[N_H5T_DATATYPES];
};
#endif

class HYPERION_API AxesRegistrar {
public:

  struct A {
    std::string uid;
    std::vector<std::string> names;
#ifdef HYPERION_USE_HDF5
    hid_t h5_datatype;
#endif
  };

#ifdef HYPERION_USE_HDF5
  static void
  register_axes(
    const std::string uid,
    const std::vector<std::string> names,
    hid_t hid);

  template <typename D>
  static void
  register_axes() {
    register_axes(Axes<D>::uid, Axes<D>::names, Axes<D>::h5_datatype);
  }
#else // !HYPERION_USE_HDF5
  static void
  register_axes(
    const std::string uid,
    const std::vector<std::string> names);

  template <typename D>
  static void
  register_axes() {
    register_axes(Axes<D>::uid, Axes<D>::names);
  }
#endif

  static std::optional<A>
  axes(const std::string& uid);

#ifdef HYPERION_USE_HDF5
  static std::optional<std::string>
  match_axes_datatype(hid_t hid);
#endif // HYPERION_USE_HDF5

  static bool
  in_range(const std::string& uid, const std::vector<int> xs);

private:

  static std::unordered_map<std::string, A> axes_;
};

HYPERION_API Legion::FieldID
add_field(
  TypeTag datatype,
  Legion::FieldAllocator fa,
  Legion::FieldID field_id = AUTO_GENERATE_ID);

template <>
struct DataType<HYPERION_TYPE_BOOL> {
  typedef bool ValueType;
  constexpr static const char* s = "bool";
  constexpr static int id = 0;
  constexpr static size_t serdez_size = sizeof(ValueType);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_BOOL_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_BOOL_REDOP;
#ifdef HYPERION_USE_CASACORE
  typedef casacore::Bool CasacoreType;
  constexpr static casacore::DataType CasacoreTypeTag =
    casacore::DataType::TpBool;
  constexpr static casacore::DataType CasacoreArrayTypeTag =
    casacore::DataType::TpArrayBool;
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
#endif
#ifdef HYPERION_USE_HDF5
  constexpr static hid_t h5t_id = H5DatatypeManager::BOOL_H5T;
#endif
  static bool equiv(const ValueType& a, const ValueType& b) {
    return a == b;
  }
};

template <>
struct DataType<HYPERION_TYPE_CHAR> {
  typedef char ValueType;
  constexpr static const char* s = "char";
  constexpr static int id = 1;
  constexpr static size_t serdez_size = sizeof(ValueType);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_CHAR_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_CHAR_REDOP;
#ifdef HYPERION_USE_CASACORE
  typedef casacore::Char CasacoreType;
  constexpr static casacore::DataType CasacoreTypeTag =
    casacore::DataType::TpChar;
  constexpr static casacore::DataType CasacoreArrayTypeTag =
    casacore::DataType::TpArrayChar;
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
#endif
#ifdef HYPERION_USE_HDF5
  constexpr static hid_t h5t_id = H5DatatypeManager::CHAR_H5T;
#endif
  static bool equiv(const ValueType& a, const ValueType& b) {
    return a == b;
  }
};

template <>
struct DataType<HYPERION_TYPE_UCHAR> {
  typedef unsigned char ValueType;
  constexpr static const char* s = "uChar";
  constexpr static int id = 2;
  constexpr static size_t serdez_size = sizeof(ValueType);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_UCHAR_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_UCHAR_REDOP;
#ifdef HYPERION_USE_CASACORE
  typedef casacore::uChar CasacoreType;
  constexpr static casacore::DataType CasacoreTypeTag =
    casacore::DataType::TpUChar;
  constexpr static casacore::DataType CasacoreArrayTypeTag =
    casacore::DataType::TpArrayUChar;
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
#endif
#ifdef HYPERION_USE_HDF5
  constexpr static hid_t h5t_id = H5DatatypeManager::UCHAR_H5T;
#endif
  static bool equiv(const ValueType& a, const ValueType& b) {
    return a == b;
  }
};

template <>
struct DataType<HYPERION_TYPE_SHORT> {
  typedef short ValueType;
  constexpr static const char* s = "short";
  constexpr static int id = 3;
  constexpr static size_t serdez_size = sizeof(ValueType);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_SHORT_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_SHORT_REDOP;
#ifdef HYPERION_USE_CASACORE
  typedef casacore::Short CasacoreType;
  constexpr static casacore::DataType CasacoreTypeTag =
    casacore::DataType::TpShort;
  constexpr static casacore::DataType CasacoreArrayTypeTag =
    casacore::DataType::TpArrayShort;
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
#endif
#ifdef HYPERION_USE_HDF5
  constexpr static hid_t h5t_id = H5DatatypeManager::SHORT_H5T;
#endif
  static bool equiv(const ValueType& a, const ValueType& b) {
    return a == b;
  }
};

template <>
struct DataType<HYPERION_TYPE_USHORT> {
  typedef unsigned short ValueType;
  constexpr static const char* s = "uShort";
  constexpr static int id = 4;
  constexpr static size_t serdez_size = sizeof(ValueType);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_USHORT_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_USHORT_REDOP;
#ifdef HYPERION_USE_CASACORE
  typedef casacore::uShort CasacoreType;
  constexpr static casacore::DataType CasacoreTypeTag =
    casacore::DataType::TpUShort;
  constexpr static casacore::DataType CasacoreArrayTypeTag =
    casacore::DataType::TpArrayUShort;
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
#endif
#ifdef HYPERION_USE_HDF5
  constexpr static hid_t h5t_id = H5DatatypeManager::USHORT_H5T;
#endif
  static bool equiv(const ValueType& a, const ValueType& b) {
    return a == b;
  }
};

template <>
struct DataType<HYPERION_TYPE_INT> {
  typedef int ValueType;
  constexpr static const char* s = "int";
  constexpr static int id = 5;
  constexpr static size_t serdez_size = sizeof(ValueType);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_INT_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_INT_REDOP;
#ifdef HYPERION_USE_CASACORE
  typedef casacore::Int CasacoreType;
  constexpr static casacore::DataType CasacoreTypeTag =
    casacore::DataType::TpInt;
  constexpr static casacore::DataType CasacoreArrayTypeTag =
    casacore::DataType::TpArrayInt;
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
#endif
#ifdef HYPERION_USE_HDF5
  constexpr static hid_t h5t_id = H5DatatypeManager::INT_H5T;
#endif
  static bool equiv(const ValueType& a, const ValueType& b) {
    return a == b;
  }
};

template <>
struct DataType<HYPERION_TYPE_UINT> {
  typedef unsigned int ValueType;
  constexpr static const char* s = "uInt";
  constexpr static int id = 6;
  constexpr static size_t serdez_size = sizeof(ValueType);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_UINT_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_UINT_REDOP;
#ifdef HYPERION_USE_CASACORE
  typedef casacore::uInt CasacoreType;
  constexpr static casacore::DataType CasacoreTypeTag =
    casacore::DataType::TpUInt;
  constexpr static casacore::DataType CasacoreArrayTypeTag =
    casacore::DataType::TpArrayUInt;
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
#endif
#ifdef HYPERION_USE_HDF5
  constexpr static hid_t h5t_id = H5DatatypeManager::UINT_H5T;
#endif
  static bool equiv(const ValueType& a, const ValueType& b) {
    return a == b;
  }
};

template <>
struct DataType<HYPERION_TYPE_FLOAT> {
  typedef float ValueType;
  constexpr static const char* s = "float";
  constexpr static int id = 7;
  constexpr static size_t serdez_size = sizeof(ValueType);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_FLOAT_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_FLOAT_REDOP;
#ifdef HYPERION_USE_CASACORE
  typedef casacore::Float CasacoreType;
  constexpr static casacore::DataType CasacoreTypeTag =
    casacore::DataType::TpFloat;
  constexpr static casacore::DataType CasacoreArrayTypeTag =
    casacore::DataType::TpArrayFloat;
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
#endif
#ifdef HYPERION_USE_HDF5
  constexpr static hid_t h5t_id = H5DatatypeManager::FLOAT_H5T;
#endif
  static bool equiv(const ValueType& a, const ValueType& b) {
    return a == b;
  }
};

template <>
struct DataType<HYPERION_TYPE_DOUBLE> {
  typedef double ValueType;
  constexpr static const char* s = "double";
  constexpr static int id = 8;
  constexpr static size_t serdez_size = sizeof(ValueType);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_DOUBLE_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_DOUBLE_REDOP;
#ifdef HYPERION_USE_CASACORE
  typedef casacore::Double CasacoreType;
  constexpr static casacore::DataType CasacoreTypeTag =
    casacore::DataType::TpDouble;
  constexpr static casacore::DataType CasacoreArrayTypeTag =
    casacore::DataType::TpArrayDouble;
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
#endif
#ifdef HYPERION_USE_HDF5
  constexpr static hid_t h5t_id = H5DatatypeManager::DOUBLE_H5T;
#endif
  static bool equiv(const ValueType& a, const ValueType& b) {
    return a == b;
  }
};

template <>
struct DataType<HYPERION_TYPE_COMPLEX> {
  typedef std::complex<float> ValueType;
  constexpr static const char* s = "complex";
  constexpr static int id = 9;
  constexpr static size_t serdez_size = sizeof(ValueType);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_COMPLEX_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_COMPLEX_REDOP;
#ifdef HYPERION_USE_CASACORE
  typedef casacore::Complex CasacoreType;
  constexpr static casacore::DataType CasacoreTypeTag =
    casacore::DataType::TpComplex;
  constexpr static casacore::DataType CasacoreArrayTypeTag =
    casacore::DataType::TpArrayComplex;
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
#endif
#ifdef HYPERION_USE_HDF5
  constexpr static hid_t h5t_id = H5DatatypeManager::COMPLEX_H5T;
#endif
  static bool equiv(const ValueType& a, const ValueType& b) {
    return a == b;
  }
};

template <>
struct DataType<HYPERION_TYPE_DCOMPLEX> {
  typedef std::complex<double> ValueType;
  constexpr static const char* s = "dComplex";
  constexpr static int id = 10;
  constexpr static size_t serdez_size = sizeof(ValueType);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_DCOMPLEX_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_DCOMPLEX_REDOP;
#ifdef HYPERION_USE_CASACORE
  typedef casacore::DComplex CasacoreType;
  constexpr static casacore::DataType CasacoreTypeTag =
    casacore::DataType::TpDComplex;
  constexpr static casacore::DataType CasacoreArrayTypeTag =
    casacore::DataType::TpArrayDComplex;
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    v = c;
  }
#endif
#ifdef HYPERION_USE_HDF5
  constexpr static hid_t h5t_id = H5DatatypeManager::DCOMPLEX_H5T;
#endif
  static bool equiv(const ValueType& a, const ValueType& b) {
    return a == b;
  }
};

template <>
struct DataType<HYPERION_TYPE_STRING> {
  typedef hyperion::string ValueType;
  constexpr static const char* s = "string";
  constexpr static int id = 11;
  constexpr static size_t serdez_size = sizeof(ValueType);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_STRING_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_STRING_REDOP;
#ifdef HYPERION_USE_CASACORE
  typedef casacore::String CasacoreType;
  constexpr static casacore::DataType CasacoreTypeTag =
    casacore::DataType::TpString;
  constexpr static casacore::DataType CasacoreArrayTypeTag =
    casacore::DataType::TpArrayString;
  static void
  from_casacore(ValueType& v, const CasacoreType& c) {
    fstrcpy(v.val, c);
  }
#endif
#ifdef HYPERION_USE_HDF5
  constexpr static hid_t h5t_id = H5DatatypeManager::STRING_H5T;
#endif
  static bool equiv(const ValueType& a, const ValueType& b) {
    return std::strcmp(a.val, b.val) == 0;
  }
};

template <>
struct DataType<HYPERION_TYPE_RECT2> {
  typedef Legion::Rect<2> ValueType;
  constexpr static const char* s = "rect2";
  constexpr static int id = static_cast<int>(HYPERION_TYPE_RECT2);
  constexpr static size_t serdez_size = sizeof(ValueType);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_RECT2_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_RECT2_REDOP;
#ifdef HYPERION_USE_HDF5
  constexpr static hid_t h5t_id = H5DatatypeManager::RECT2_H5T;
#endif
  static bool equiv(const ValueType& a, const ValueType& b) {
    return a == b;
  }
};

template <>
struct DataType<HYPERION_TYPE_RECT3> {
  typedef Legion::Rect<3> ValueType;
  constexpr static const char* s = "rect3";
  constexpr static int id = static_cast<int>(HYPERION_TYPE_RECT3);
  constexpr static size_t serdez_size = sizeof(ValueType);
  constexpr static int af_serdez_id = OpsManager::ACC_FIELD_RECT3_SID;
  constexpr static int af_redop_id = OpsManager::ACC_FIELD_RECT3_REDOP;
#ifdef HYPERION_USE_HDF5
  constexpr static hid_t h5t_id = H5DatatypeManager::RECT3_H5T;
#endif
  static bool equiv(const ValueType& a, const ValueType& b) {
    return a == b;
  }
};


#ifdef HYPERION_USE_HDF5
template <TypeTag DT>
constexpr hid_t
H5DatatypeManager::datatype() {
  return datatypes()[DataType<DT>::h5t_id];
}
#endif

#define NUM_HYPERION_DATATYPES (DataType<HYPERION_TYPE_STRING>::id + 1)

#define HYPERION_FOREACH_DATATYPE(__func__)              \
  __func__(HYPERION_TYPE_BOOL)              \
  __func__(HYPERION_TYPE_CHAR)              \
  __func__(HYPERION_TYPE_UCHAR)             \
  __func__(HYPERION_TYPE_SHORT)             \
  __func__(HYPERION_TYPE_USHORT)            \
  __func__(HYPERION_TYPE_INT)               \
  __func__(HYPERION_TYPE_UINT)              \
  __func__(HYPERION_TYPE_FLOAT)             \
  __func__(HYPERION_TYPE_DOUBLE)            \
  __func__(HYPERION_TYPE_COMPLEX)           \
  __func__(HYPERION_TYPE_DCOMPLEX)          \
  __func__(HYPERION_TYPE_STRING)

#define HYPERION_FOREACH_RECORD_DATATYPE(__func__)  \
  __func__(HYPERION_TYPE_BOOL)                     \
  __func__(HYPERION_TYPE_UCHAR)                    \
  __func__(HYPERION_TYPE_SHORT)                    \
  __func__(HYPERION_TYPE_INT)                      \
  __func__(HYPERION_TYPE_UINT)                     \
  __func__(HYPERION_TYPE_FLOAT)                    \
  __func__(HYPERION_TYPE_DOUBLE)                   \
  __func__(HYPERION_TYPE_COMPLEX)                  \
  __func__(HYPERION_TYPE_DCOMPLEX)                 \
  __func__(HYPERION_TYPE_STRING)

#define HYPERION_FOREACH_BARE_DATATYPE(__func__)  \
  __func__(BOOL)          \
  __func__(CHAR)          \
  __func__(UCHAR)         \
  __func__(SHORT)         \
  __func__(USHORT)        \
  __func__(INT)           \
  __func__(UINT)          \
  __func__(FLOAT)         \
  __func__(DOUBLE)        \
  __func__(COMPLEX)       \
  __func__(DCOMPLEX)      \
  __func__(STRING)

template <typename T>
struct ValueType {
  // constexpr static const TypeTag DataType;
};

#define VT(dt)                                  \
  template <>                                   \
  struct ValueType<DataType<dt>::ValueType> {   \
    constexpr static TypeTag DataType = dt;   \
  };

HYPERION_FOREACH_DATATYPE(VT)

template <>
struct ValueType<std::string> {
  constexpr static TypeTag DataType = HYPERION_TYPE_STRING;
};
#undef VT

struct HYPERION_API AxisPartition {
  string axes_uid;
  int dim;
  // stride between start of partition sub-spaces
  Legion::coord_t stride;
  // offset of start of first partition
  Legion::coord_t offset;
  // extent of partition
  std::array<Legion::coord_t, 2> extent;
  // axis limits
  std::array<Legion::coord_t, 2> limits;

  bool
  operator==(const AxisPartition& other) {
    // limits values is not compared, as it depends only on value of dim
    return
      axes_uid == other.axes_uid
      && dim == other.dim
      && stride == other.stride
      && offset == other.offset
      && extent == other.extent;
  }

  bool
  operator!=(const AxisPartition& other) {
    return !operator==(other);
  }

  bool
  operator<(const AxisPartition& other) {
    if (axes_uid != other.axes_uid) return false;
    if (dim < other.dim) return true;
    if (dim == other.dim) {
      if (stride < other.stride) return true;
      if (stride == other.stride) {
        if (offset < other.offset) return true;
        if (offset == other.offset)
          return extent < other.extent;
      }
    }
    return false;
  }
};

HYPERION_API Legion::LayoutConstraintRegistrar&
add_row_major_order_constraint(
  Legion::LayoutConstraintRegistrar& lc,
  unsigned rank);

HYPERION_API void
register_mapper(
  Legion::Machine machine,
  Legion::Runtime* rt,
  const std::set<Legion::Processor>& local_procs);

HYPERION_API void
preregister_all();

// FIXME: rename this function
HYPERION_API void
register_tasks(Legion::Context context, Legion::Runtime* runtime);

HYPERION_API extern Legion::MapperID table_mapper;
HYPERION_API extern Legion::LayoutConstraintID soa_row_major_layout;
HYPERION_API extern Legion::LayoutConstraintID soa_column_major_layout;
HYPERION_API extern Legion::LayoutConstraintID aos_row_major_layout;
HYPERION_API extern Legion::LayoutConstraintID aos_column_major_layout;

#if LEGION_MAX_DIM == 1

#define HYPERION_FOREACH_N(__func__)            \
  __func__(1)
#define HYPERION_FOREACH_NN(__func__)           \
  __func__(1,1)
#define HYPERION_FOREACH_MN(__func__)           \
  __func__(1,1)
#define HYPERION_FOREACH_LMN(__func__)          \
  __func__(1,1,1)
#define HYPERION_FOREACH_LESS_MAX_N(__func__)

#elif LEGION_MAX_DIM == 2

#define HYPERION_FOREACH_N(__func__)            \
  __func__(1)                                   \
  __func__(2)
#define HYPERION_FOREACH_NN(__func__)           \
  __func__(1,1)                                 \
  __func__(1,2)                                 \
  __func__(2,1)                                 \
  __func__(2,2)
#define HYPERION_FOREACH_MN(__func__)           \
  __func__(1,1)                                 \
  __func__(1,2)                                 \
  __func__(2,2)
#define HYPERION_FOREACH_LMN(__func__)          \
  __func__(1,1,1)                               \
  __func__(1,1,2)                               \
  __func__(1,2,2)                               \
  __func__(2,2,2)
#define HYPERION_FOREACH_N_LESS_MAX(__func__)   \
  __func__(1)

#elif LEGION_MAX_DIM == 3

#define HYPERION_FOREACH_N(__func__)            \
  __func__(1)                                   \
  __func__(2)                                   \
  __func__(3)
#define HYPERION_FOREACH_NN(__func__)           \
  __func__(1,1)                                 \
  __func__(1,2)                                 \
  __func__(1,3)                                 \
  __func__(2,1)                                 \
  __func__(2,2)                                 \
  __func__(2,3)                                 \
  __func__(3,1)                                 \
  __func__(3,2)                                 \
  __func__(3,3)
#define HYPERION_FOREACH_MN(__func__)           \
  __func__(1,1)                                 \
  __func__(1,2)                                 \
  __func__(1,3)                                 \
  __func__(2,2)                                 \
  __func__(2,3)                                 \
  __func__(3,3)
#define HYPERION_FOREACH_LMN(__func__)          \
  __func__(1,1,1)                               \
  __func__(1,1,2)                               \
  __func__(1,1,3)                               \
  __func__(1,2,2)                               \
  __func__(2,2,2)                               \
  __func__(1,2,3)                               \
  __func__(2,2,3)                               \
  __func__(1,3,3)                               \
  __func__(2,3,3)                               \
  __func__(3,3,3)
#define HYPERION_FOREACH_N_LESS_MAX(__func__)   \
  __func__(1)                                   \
  __func__(2)

#elif LEGION_MAX_DIM == 4

#define HYPERION_FOREACH_N(__func__)            \
  __func__(1)                                   \
  __func__(2)                                   \
  __func__(3)                                   \
  __func__(4)
#define HYPERION_FOREACH_NN(__func__)           \
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
#define HYPERION_FOREACH_MN(__func__)           \
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
#define HYPERION_FOREACH_LMN(__func__)          \
  __func__(1,1,1)                               \
  __func__(1,1,2)                               \
  __func__(1,1,3)                               \
  __func__(1,1,4)                               \
  __func__(1,2,2)                               \
  __func__(2,2,2)                               \
  __func__(1,2,3)                               \
  __func__(2,2,3)                               \
  __func__(1,2,4)                               \
  __func__(2,2,4)                               \
  __func__(1,3,3)                               \
  __func__(2,3,3)                               \
  __func__(3,3,3)                               \
  __func__(1,3,4)                               \
  __func__(2,3,4)                               \
  __func__(3,3,4)                               \
  __func__(1,4,4)                               \
  __func__(2,4,4)                               \
  __func__(3,4,4)                               \
  __func__(4,4,4)
#define HYPERION_FOREACH_N_LESS_MAX(__func__)   \
  __func__(1)                                   \
  __func__(2)                                   \
  __func__(3)

#elif LEGION_MAX_DIM == 5

#define HYPERION_FOREACH_N(__func__)            \
  __func__(1)                                   \
  __func__(2)                                   \
  __func__(3)                                   \
  __func__(4)                                   \
  __func__(5)
#define HYPERION_FOREACH_NN(__func__)           \
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
#define HYPERION_FOREACH_MN(__func__)           \
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
#define HYPERION_FOREACH_LMN(__func__)          \
  __func__(1,1,1)                               \
  __func__(1,1,2)                               \
  __func__(1,1,3)                               \
  __func__(1,1,4)                               \
  __func__(1,1,5)                               \
  __func__(1,2,2)                               \
  __func__(2,2,2)                               \
  __func__(1,2,3)                               \
  __func__(2,2,3)                               \
  __func__(1,2,4)                               \
  __func__(2,2,4)                               \
  __func__(1,2,5)                               \
  __func__(2,2,5)                               \
  __func__(1,3,3)                               \
  __func__(2,3,3)                               \
  __func__(3,3,3)                               \
  __func__(1,3,4)                               \
  __func__(2,3,4)                               \
  __func__(3,3,4)                               \
  __func__(1,3,5)                               \
  __func__(2,3,5)                               \
  __func__(3,3,5)                               \
  __func__(1,4,4)                               \
  __func__(2,4,4)                               \
  __func__(3,4,4)                               \
  __func__(4,4,4)                               \
  __func__(1,4,5)                               \
  __func__(2,4,5)                               \
  __func__(3,4,5)                               \
  __func__(4,4,5)                               \
  __func__(1,5,5)                               \
  __func__(2,5,5)                               \
  __func__(3,5,5)                               \
  __func__(4,5,5)                               \
  __func__(5,5,5)
#define HYPERION_FOREACH_N_LESS_MAX(__func__)   \
  __func__(1)                                   \
  __func__(2)                                   \
  __func__(3)                                   \
  __func__(4)

#elif LEGION_MAX_DIM == 6

#define HYPERION_FOREACH_N(__func__)            \
  __func__(1)                                   \
  __func__(2)                                   \
  __func__(3)                                   \
  __func__(4)                                   \
  __func__(5)                                   \
  __func__(6)
#define HYPERION_FOREACH_NN(__func__)           \
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
#define HYPERION_FOREACH_MN(__func__)           \
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
#define HYPERION_FOREACH_LMN(__func__)          \
  __func__(1,1,1)                               \
  __func__(1,1,2)                               \
  __func__(1,1,3)                               \
  __func__(1,1,4)                               \
  __func__(1,1,5)                               \
  __func__(1,1,6)                               \
  __func__(1,2,2)                               \
  __func__(2,2,2)                               \
  __func__(1,2,3)                               \
  __func__(2,2,3)                               \
  __func__(1,2,4)                               \
  __func__(2,2,4)                               \
  __func__(1,2,5)                               \
  __func__(2,2,5)                               \
  __func__(1,2,6)                               \
  __func__(2,2,6)                               \
  __func__(1,3,3)                               \
  __func__(2,3,3)                               \
  __func__(3,3,3)                               \
  __func__(1,3,4)                               \
  __func__(2,3,4)                               \
  __func__(3,3,4)                               \
  __func__(1,3,5)                               \
  __func__(2,3,5)                               \
  __func__(3,3,5)                               \
  __func__(1,3,6)                               \
  __func__(2,3,6)                               \
  __func__(3,3,6)                               \
  __func__(1,4,4)                               \
  __func__(2,4,4)                               \
  __func__(3,4,4)                               \
  __func__(4,4,4)                               \
  __func__(1,4,5)                               \
  __func__(2,4,5)                               \
  __func__(3,4,5)                               \
  __func__(4,4,5)                               \
  __func__(4,4,6)                               \
  __func__(1,5,5)                               \
  __func__(2,5,5)                               \
  __func__(3,5,5)                               \
  __func__(4,5,5)                               \
  __func__(5,5,5)                               \
  __func__(5,5,6)                               \
  __func__(1,6,6)                               \
  __func__(2,6,6)                               \
  __func__(3,6,6)                               \
  __func__(4,6,6)                               \
  __func__(5,6,6)                               \
  __func__(6,6,6)
#define HYPERION_FOREACH_N_LESS_MAX(__func__)   \
  __func__(1)                                   \
  __func__(2)                                   \
  __func__(3)                                   \
  __func__(4)                                   \
  __func__(5)

#elif LEGION_MAX_DIM == 7

#define HYPERION_FOREACH_N(__func__)            \
  __func__(1)                                   \
  __func__(2)                                   \
  __func__(3)                                   \
  __func__(4)                                   \
  __func__(5)                                   \
  __func__(6)                                   \
  __func__(7)
#define HYPERION_FOREACH_NN(__func__)           \
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
#define HYPERION_FOREACH_MN(__func__)           \
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
#define HYPERION_FOREACH_LMN(__func__)          \
  __func__(1,1,1)                               \
  __func__(1,1,2)                               \
  __func__(1,1,3)                               \
  __func__(1,1,4)                               \
  __func__(1,1,5)                               \
  __func__(1,1,6)                               \
  __func__(1,1,7)                               \
  __func__(1,2,2)                               \
  __func__(2,2,2)                               \
  __func__(1,2,3)                               \
  __func__(2,2,3)                               \
  __func__(1,2,4)                               \
  __func__(2,2,4)                               \
  __func__(1,2,5)                               \
  __func__(2,2,5)                               \
  __func__(1,2,6)                               \
  __func__(2,2,6)                               \
  __func__(1,2,7)                               \
  __func__(2,2,7)                               \
  __func__(1,3,3)                               \
  __func__(2,3,3)                               \
  __func__(3,3,3)                               \
  __func__(1,3,4)                               \
  __func__(2,3,4)                               \
  __func__(3,3,4)                               \
  __func__(1,3,5)                               \
  __func__(2,3,5)                               \
  __func__(3,3,5)                               \
  __func__(1,3,6)                               \
  __func__(2,3,6)                               \
  __func__(3,3,6)                               \
  __func__(1,3,7)                               \
  __func__(2,3,7)                               \
  __func__(3,3,7)                               \
  __func__(1,4,4)                               \
  __func__(2,4,4)                               \
  __func__(3,4,4)                               \
  __func__(4,4,4)                               \
  __func__(1,4,5)                               \
  __func__(2,4,5)                               \
  __func__(3,4,5)                               \
  __func__(4,4,5)                               \
  __func__(1,4,6)                               \
  __func__(2,4,6)                               \
  __func__(3,4,6)                               \
  __func__(4,4,6)                               \
  __func__(1,4,7)                               \
  __func__(2,4,7)                               \
  __func__(3,4,7)                               \
  __func__(4,4,7)                               \
  __func__(1,5,5)                               \
  __func__(2,5,5)                               \
  __func__(3,5,5)                               \
  __func__(4,5,5)                               \
  __func__(5,5,5)                               \
  __func__(1,5,6)                               \
  __func__(2,5,6)                               \
  __func__(3,5,6)                               \
  __func__(4,5,6)                               \
  __func__(5,5,6)                               \
  __func__(1,5,7)                               \
  __func__(2,5,7)                               \
  __func__(3,5,7)                               \
  __func__(4,5,7)                               \
  __func__(5,5,7)                               \
  __func__(1,6,6)                               \
  __func__(2,6,6)                               \
  __func__(3,6,6)                               \
  __func__(4,6,6)                               \
  __func__(5,6,6)                               \
  __func__(6,6,6)                               \
  __func__(1,6,7)                               \
  __func__(2,6,7)                               \
  __func__(3,6,7)                               \
  __func__(4,6,7)                               \
  __func__(5,6,7)                               \
  __func__(6,6,7)                               \
  __func__(1,7,7)                               \
  __func__(2,7,7)                               \
  __func__(3,7,7)                               \
  __func__(4,7,7)                               \
  __func__(5,7,7)                               \
  __func__(6,7,7)                               \
  __func__(7,7,7)
#define HYPERION_FOREACH_N_LESS_MAX(__func__)   \
  __func__(1)                                   \
  __func__(2)                                   \
  __func__(3)                                   \
  __func__(4)                                   \
  __func__(5)                                   \
  __func__(6)

#elif LEGION_MAX_DIM == 8

#define HYPERION_FOREACH_N(__func__)            \
  __func__(1)                                   \
  __func__(2)                                   \
  __func__(3)                                   \
  __func__(4)                                   \
  __func__(5)                                   \
  __func__(6)                                   \
  __func__(7)                                   \
  __func__(8)
#define HYPERION_FOREACH_NN(__func__)           \
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
#define HYPERION_FOREACH_MN(__func__)           \
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
#define HYPERION_FOREACH_LMN(__func__)          \
  __func__(1,1,1)                               \
  __func__(1,1,2)                               \
  __func__(1,1,3)                               \
  __func__(1,1,4)                               \
  __func__(1,1,5)                               \
  __func__(1,1,6)                               \
  __func__(1,1,7)                               \
  __func__(1,1,8)                               \
  __func__(1,2,2)                               \
  __func__(2,2,2)                               \
  __func__(1,2,3)                               \
  __func__(2,2,3)                               \
  __func__(1,2,4)                               \
  __func__(2,2,4)                               \
  __func__(1,2,5)                               \
  __func__(2,2,5)                               \
  __func__(1,2,6)                               \
  __func__(2,2,6)                               \
  __func__(1,2,7)                               \
  __func__(2,2,7)                               \
  __func__(1,2,8)                               \
  __func__(2,2,8)                               \
  __func__(1,3,3)                               \
  __func__(2,3,3)                               \
  __func__(3,3,3)                               \
  __func__(1,3,4)                               \
  __func__(2,3,4)                               \
  __func__(3,3,4)                               \
  __func__(1,3,5)                               \
  __func__(2,3,5)                               \
  __func__(3,3,5)                               \
  __func__(1,3,6)                               \
  __func__(2,3,6)                               \
  __func__(3,3,6)                               \
  __func__(1,3,7)                               \
  __func__(2,3,7)                               \
  __func__(3,3,7)                               \
  __func__(1,3,8)                               \
  __func__(2,3,8)                               \
  __func__(3,3,8)                               \
  __func__(1,4,4)                               \
  __func__(2,4,4)                               \
  __func__(3,4,4)                               \
  __func__(4,4,4)                               \
  __func__(1,4,5)                               \
  __func__(2,4,5)                               \
  __func__(3,4,5)                               \
  __func__(4,4,5)                               \
  __func__(1,4,6)                               \
  __func__(2,4,6)                               \
  __func__(3,4,6)                               \
  __func__(4,4,6)                               \
  __func__(1,4,7)                               \
  __func__(2,4,7)                               \
  __func__(3,4,7)                               \
  __func__(4,4,7)                               \
  __func__(1,4,8)                               \
  __func__(2,4,8)                               \
  __func__(3,4,8)                               \
  __func__(4,4,8)                               \
  __func__(1,5,5)                               \
  __func__(2,5,5)                               \
  __func__(3,5,5)                               \
  __func__(4,5,5)                               \
  __func__(5,5,5)                               \
  __func__(1,5,6)                               \
  __func__(2,5,6)                               \
  __func__(3,5,6)                               \
  __func__(4,5,6)                               \
  __func__(5,5,6)                               \
  __func__(1,5,7)                               \
  __func__(2,5,7)                               \
  __func__(3,5,7)                               \
  __func__(4,5,7)                               \
  __func__(5,5,7)                               \
  __func__(1,5,8)                               \
  __func__(2,5,8)                               \
  __func__(3,5,8)                               \
  __func__(4,5,8)                               \
  __func__(5,5,8)                               \
  __func__(1,6,6)                               \
  __func__(2,6,6)                               \
  __func__(3,6,6)                               \
  __func__(4,6,6)                               \
  __func__(5,6,6)                               \
  __func__(6,6,6)                               \
  __func__(1,6,7)                               \
  __func__(2,6,7)                               \
  __func__(3,6,7)                               \
  __func__(4,6,7)                               \
  __func__(5,6,7)                               \
  __func__(6,6,7)                               \
  __func__(1,6,8)                               \
  __func__(2,6,8)                               \
  __func__(3,6,8)                               \
  __func__(4,6,8)                               \
  __func__(5,6,8)                               \
  __func__(6,6,8)                               \
  __func__(1,7,7)                               \
  __func__(2,7,7)                               \
  __func__(3,7,7)                               \
  __func__(4,7,7)                               \
  __func__(5,7,7)                               \
  __func__(6,7,7)                               \
  __func__(7,7,7)                               \
  __func__(1,7,8)                               \
  __func__(2,7,8)                               \
  __func__(3,7,8)                               \
  __func__(4,7,8)                               \
  __func__(5,7,8)                               \
  __func__(6,7,8)                               \
  __func__(7,7,8)                               \
  __func__(1,8,8)                               \
  __func__(2,8,8)                               \
  __func__(3,8,8)                               \
  __func__(4,8,8)                               \
  __func__(5,8,8)                               \
  __func__(6,8,8)                               \
  __func__(7,8,8)                               \
  __func__(8,8,8)
#define HYPERION_FOREACH_N_LESS_MAX(__func__)   \
  __func__(1)                                   \
  __func__(2)                                   \
  __func__(3)                                   \
  __func__(4)                                   \
  __func__(5)                                   \
  __func__(6)                                   \
  __func__(7)

#elif LEGION_MAX_DIM == 9

#define HYPERION_FOREACH_N(__func__)            \
  __func__(1)                                   \
  __func__(2)                                   \
  __func__(3)                                   \
  __func__(4)                                   \
  __func__(5)                                   \
  __func__(6)                                   \
  __func__(7)                                   \
  __func__(8)                                   \
  __func__(9)
#define HYPERION_FOREACH_NN(__func__)           \
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
#define HYPERION_FOREACH_MN(__func__)           \
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
#define HYPERION_FOREACH_LMN(__func__)          \
  __func__(1,1,1)                               \
  __func__(1,1,2)                               \
  __func__(1,1,3)                               \
  __func__(1,1,4)                               \
  __func__(1,1,5)                               \
  __func__(1,1,6)                               \
  __func__(1,1,7)                               \
  __func__(1,1,8)                               \
  __func__(1,1,9)                               \
  __func__(1,2,2)                               \
  __func__(2,2,2)                               \
  __func__(1,2,3)                               \
  __func__(2,2,3)                               \
  __func__(1,2,4)                               \
  __func__(2,2,4)                               \
  __func__(1,2,5)                               \
  __func__(2,2,5)                               \
  __func__(1,2,6)                               \
  __func__(2,2,6)                               \
  __func__(1,2,7)                               \
  __func__(2,2,7)                               \
  __func__(1,2,8)                               \
  __func__(2,2,8)                               \
  __func__(1,2,9)                               \
  __func__(2,2,9)                               \
  __func__(1,3,3)                               \
  __func__(2,3,3)                               \
  __func__(3,3,3)                               \
  __func__(1,3,4)                               \
  __func__(2,3,4)                               \
  __func__(3,3,4)                               \
  __func__(1,3,5)                               \
  __func__(2,3,5)                               \
  __func__(3,3,5)                               \
  __func__(1,3,6)                               \
  __func__(2,3,6)                               \
  __func__(3,3,6)                               \
  __func__(1,3,7)                               \
  __func__(2,3,7)                               \
  __func__(3,3,7)                               \
  __func__(1,3,8)                               \
  __func__(2,3,8)                               \
  __func__(3,3,8)                               \
  __func__(1,3,9)                               \
  __func__(2,3,9)                               \
  __func__(3,3,9)                               \
  __func__(1,4,4)                               \
  __func__(2,4,4)                               \
  __func__(3,4,4)                               \
  __func__(4,4,4)                               \
  __func__(1,4,5)                               \
  __func__(2,4,5)                               \
  __func__(3,4,5)                               \
  __func__(4,4,5)                               \
  __func__(1,4,6)                               \
  __func__(2,4,6)                               \
  __func__(3,4,6)                               \
  __func__(4,4,6)                               \
  __func__(1,4,7)                               \
  __func__(2,4,7)                               \
  __func__(3,4,7)                               \
  __func__(4,4,7)                               \
  __func__(1,4,8)                               \
  __func__(2,4,8)                               \
  __func__(3,4,8)                               \
  __func__(4,4,8)                               \
  __func__(1,4,9)                               \
  __func__(2,4,9)                               \
  __func__(3,4,9)                               \
  __func__(4,4,9)                               \
  __func__(1,5,5)                               \
  __func__(2,5,5)                               \
  __func__(3,5,5)                               \
  __func__(4,5,5)                               \
  __func__(5,5,5)                               \
  __func__(1,5,6)                               \
  __func__(2,5,6)                               \
  __func__(3,5,6)                               \
  __func__(4,5,6)                               \
  __func__(5,5,6)                               \
  __func__(1,5,7)                               \
  __func__(2,5,7)                               \
  __func__(3,5,7)                               \
  __func__(4,5,7)                               \
  __func__(5,5,7)                               \
  __func__(1,5,8)                               \
  __func__(2,5,8)                               \
  __func__(3,5,8)                               \
  __func__(4,5,8)                               \
  __func__(5,5,8)                               \
  __func__(1,5,9)                               \
  __func__(2,5,9)                               \
  __func__(3,5,9)                               \
  __func__(4,5,9)                               \
  __func__(5,5,9)                               \
  __func__(1,6,6)                               \
  __func__(2,6,6)                               \
  __func__(3,6,6)                               \
  __func__(4,6,6)                               \
  __func__(5,6,6)                               \
  __func__(6,6,6)                               \
  __func__(1,6,7)                               \
  __func__(2,6,7)                               \
  __func__(3,6,7)                               \
  __func__(4,6,7)                               \
  __func__(5,6,7)                               \
  __func__(6,6,7)                               \
  __func__(1,6,8)                               \
  __func__(2,6,8)                               \
  __func__(3,6,8)                               \
  __func__(4,6,8)                               \
  __func__(5,6,8)                               \
  __func__(6,6,8)                               \
  __func__(1,6,9)                               \
  __func__(2,6,9)                               \
  __func__(3,6,9)                               \
  __func__(4,6,9)                               \
  __func__(5,6,9)                               \
  __func__(6,6,9)                               \
  __func__(1,7,7)                               \
  __func__(2,7,7)                               \
  __func__(3,7,7)                               \
  __func__(4,7,7)                               \
  __func__(5,7,7)                               \
  __func__(6,7,7)                               \
  __func__(7,7,7)                               \
  __func__(1,7,8)                               \
  __func__(2,7,8)                               \
  __func__(3,7,8)                               \
  __func__(4,7,8)                               \
  __func__(5,7,8)                               \
  __func__(6,7,8)                               \
  __func__(7,7,8)                               \
  __func__(1,7,9)                               \
  __func__(2,7,9)                               \
  __func__(3,7,9)                               \
  __func__(4,7,9)                               \
  __func__(5,7,9)                               \
  __func__(6,7,9)                               \
  __func__(7,7,9)                               \
  __func__(1,8,8)                               \
  __func__(2,8,8)                               \
  __func__(3,8,8)                               \
  __func__(4,8,8)                               \
  __func__(5,8,8)                               \
  __func__(6,8,8)                               \
  __func__(7,8,8)                               \
  __func__(8,8,8)                               \
  __func__(1,8,9)                               \
  __func__(2,8,9)                               \
  __func__(3,8,9)                               \
  __func__(4,8,9)                               \
  __func__(5,8,9)                               \
  __func__(6,8,9)                               \
  __func__(7,8,9)                               \
  __func__(8,8,9)                               \
  __func__(1,9,9)                               \
  __func__(2,9,9)                               \
  __func__(3,9,9)                               \
  __func__(4,9,9)                               \
  __func__(5,9,9)                               \
  __func__(6,9,9)                               \
  __func__(7,9,9)                               \
  __func__(8,9,9)                               \
  __func__(9,9,9)
#define HYPERION_FOREACH_N_LESS_MAX(__func__)   \
  __func__(1)                                   \
  __func__(2)                                   \
  __func__(3)                                   \
  __func__(4)                                   \
  __func__(5)                                   \
  __func__(6)                                   \
  __func__(7)                                   \
  __func__(8)

#else
#error "Unsupported LEGION_MAX_DIM"
#endif

#define DEFINE_POINT_ADD_REDOP(DIM)                             \
  template <>                                                   \
  class point_add_redop<DIM> {                                  \
  public:                                                       \
  typedef Legion::Point<DIM> LHS;                               \
  typedef Legion::Point<DIM> RHS;                               \
                                                                \
  static const RHS identity;                                    \
                                                                \
  template <bool EXCL>                                          \
  static void                                                   \
  apply(LHS& lhs, RHS rhs);                                     \
                                                                \
  template <bool EXCL>                                          \
  static void                                                   \
  fold(RHS& rhs1, RHS rhs2);                                    \
  };                                                            \
  template <>                                                   \
  inline void                                                   \
  point_add_redop<DIM>::apply<true>(LHS& lhs, RHS rhs) {        \
    lhs += rhs;                                                 \
  }                                                             \
  template <>                                                   \
  inline void                                                   \
  point_add_redop<DIM>::apply<false>(LHS& lhs, RHS rhs) {       \
    for (int i = 0; i < DIM; ++i)                               \
      __atomic_fetch_add(&lhs[i], rhs[i], __ATOMIC_ACQUIRE);    \
  }                                                             \
  template <>                                                   \
  inline void                                                   \
  point_add_redop<DIM>::fold<true>(RHS& rhs1, RHS rhs2) {       \
    rhs1 += rhs2;                                               \
  }                                                             \
  template <>                                                   \
  inline void                                                   \
  point_add_redop<DIM>::fold<false>(RHS& rhs1, RHS rhs2) {      \
    for (int i = 0; i < DIM; ++i)                               \
      __atomic_fetch_add(&rhs1[i], rhs2[i], __ATOMIC_ACQUIRE);  \
  }
HYPERION_FOREACH_N(DEFINE_POINT_ADD_REDOP);
#undef DEFINE_POINT_ADD_REDOP

} // end namespace hyperion

HYPERION_API std::ostream&
operator<<(std::ostream& stream, const hyperion::string& str);

#ifndef HYPERION_USE_CASACORE
// casacore has an equivalent, no need to define another
namespace std {
template <typename F>
bool
operator<(const std::complex<F>& a, const std::complex<F>& b) {
  if (a.real() < b.real())
    return true;
  if (a.real() > b.real())
    return false;
  return a.imag() < b.imag();
}
} // end namespace std
#endif

namespace hyperion {
template <int N, typename C>
bool
operator<(const Legion::Point<N, C>& a, const Legion::Point<N, C>& b) {
  bool result = true;
  for (size_t i = 0; result && i < N; ++i)
    result = a[i] < b[i];
  return result;
}
template <int N, typename C>
bool
operator<(const Legion::Rect<N, C>& a, const Legion::Rect<N, C>& b) {
  if (a.lo < b.lo)
    return true;
  if (b.lo < a.lo)
    return false;
  return a.hi < b.hi;
}
} // end namespace std

#endif // HYPERION_UTILITY_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
