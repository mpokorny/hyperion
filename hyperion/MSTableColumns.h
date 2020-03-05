/*
 * Copyright 2019 Associated Universities, Inc. Washington DC, USA.
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
#ifndef HYPERION_MS_TABLE_COLUMNS_H_
#define HYPERION_MS_TABLE_COLUMNS_H_

#include <hyperion/MSTableColumns_c.h>
#include <hyperion/MSTable.h>
#ifdef HYPERION_USE_CASACORE
# include <hyperion/MeasRef.h>
#endif
#include <hyperion/Table.h>

#pragma GCC visibility push(default)
# ifdef HYPERION_USE_CASACORE
#  include <casacore/measures/Measures/MeasRef.h>
# endif

# include <algorithm>
# include <array>
# include <cstring>
# include <iterator>
# include <optional>
# include <unordered_map>
# include <variant>
# include <vector>
#pragma GCC visibility pop

namespace hyperion {

struct MSTableColumnsBase {

  template <TypeTag T, int N, legion_privilege_mode_t MODE, bool CHECK_BOUNDS>
  using FieldAccessor =
    Legion::FieldAccessor<
      MODE,
      typename DataType<T>::ValueType,
      N,
      Legion::coord_t,
      Legion::AffineAccessor<
        typename DataType<T>::ValueType,
        N,
        Legion::coord_t>,
      CHECK_BOUNDS>;

#ifdef HYPERION_USE_CASACORE
  template <typename M>
  using simple_mr_t = std::shared_ptr<casacore::MeasRef<M>>;

  template <typename M>
  using ref_mr_t = std::tuple<
    std::vector<std::shared_ptr<casacore::MeasRef<M>>>,
    std::unordered_map<unsigned, unsigned>,
    Legion::PhysicalRegion,
    Legion::FieldID>;

  template <typename M>
  using mr_t = std::variant<simple_mr_t<M>, ref_mr_t<M>>;
#endif

  struct Regions {
    Legion::PhysicalRegion values;
#ifdef HYPERION_USE_CASACORE
    std::vector<Legion::PhysicalRegion> meas_refs;
    std::optional<std::tuple<Legion::PhysicalRegion, Legion::FieldID>>
    ref_column;
#endif // HYPERION_USE_CASACORE
  };

  struct RegionsInfo {
    size_t values;
    Legion::FieldID fid;
#ifdef HYPERION_USE_CASACORE
    unsigned meas_refs_size;
    size_t meas_refs_0;
    bool has_ref_column;
    size_t ref_column;
    Legion::FieldID ref_column_fid;
#endif // HYPERION_USE_CASACORE

    Regions
    regions(const std::vector<Legion::PhysicalRegion>& prs) const;
  };

  typedef std::unordered_map<std::string, MSTableColumnsBase::RegionsInfo>
  region_infos_t;

  static std::tuple<std::unique_ptr<char[]>, size_t>
  serialize_region_infos(const region_infos_t& infos) {
    size_t size = 0;
    size += sizeof(unsigned);
    for (auto& [nm, ri] : infos)
      size += sizeof(unsigned) + nm.size() + 1 + sizeof(ri);

    std::unique_ptr<char[]> buffer = std::make_unique<char[]>(size);
    char* b = reinterpret_cast<char*>(buffer.get());
    *reinterpret_cast<unsigned*>(b) = infos.size();
    b += sizeof(unsigned);
    for (auto& [nm, ri] : infos) {
      unsigned sz = nm.size() + 1;
      *reinterpret_cast<unsigned*>(b) = sz;
      b += sizeof(unsigned);
      std::strncpy(b, nm.c_str(), sz);
      b += sz;
      *reinterpret_cast<region_infos_t::mapped_type*>(b) = ri;
      b += sizeof(region_infos_t::mapped_type);
    }
    return {std::move(buffer), size};
  }

  static region_infos_t
  deserialize_region_infos(const void* buffer) {
    region_infos_t result;
    const char* b = reinterpret_cast<const char*>(buffer);
    unsigned n = *reinterpret_cast<const unsigned*>(b);
    b += sizeof(unsigned);
    for (unsigned i = 0; i < n; ++i) {
      unsigned sz = *reinterpret_cast<const unsigned*>(b);
      b += sizeof(unsigned);
      std::string nm = b;
      b += sz;
      result[nm] = *reinterpret_cast<const region_infos_t::mapped_type*>(b);
      b += sizeof(region_infos_t::mapped_type);
    }
    return result;
  }

  static std::unordered_map<std::string, Regions>
  regions(
    const region_infos_t& infos,
    const std::vector<Legion::PhysicalRegion>& prs) {

    std::unordered_map<std::string, Regions> result;
    for (auto& [nm, ri] : infos)
      result[nm] = ri.regions(prs);
    return result;
  }

  static std::tuple<std::vector<Legion::RegionRequirement>, region_infos_t>
  requirements(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const Table& table,
    const std::vector<std::string>& cols,
    legion_privilege_mode_t mode) {

    std::vector<Legion::RegionRequirement> reqs;
    region_infos_t infos;

    auto tbl_columns =
      table.columns(ctx, rt).get_result<Table::columns_result_t>();

    // add measure reference columns to requirements
    std::set<std::string> cols_w_refs;
    for (auto& [csp, ixcs, vlr, nm_tfs] : tbl_columns.fields) {
      for (auto& [nm, tf] : nm_tfs) {
        std::string nmstr(nm.val);
        auto cp = std::find(cols.begin(), cols.end(), nmstr);
        if (cp != cols.end()) {
          cols_w_refs.insert(nmstr);
#ifdef HYPERION_USE_CASACORE
          if (tf.rc)
            cols_w_refs.insert(tf.rc.value());
#endif // HYPERION_USE_CASACORE
        }
      }
    }
    // compute requirements for all columns in cols_w_refs
#ifdef HYPERION_USE_CASACORE
    std::map<std::string, std::string> ref_column_names;
    std::map<std::string, unsigned> ref_column_req_indexes;
#endif // HYPERION_USE_CASACORE
    for (auto& [csp, ixcs, vlr, nm_tfs] : tbl_columns.fields) {
      std::vector<Legion::FieldID> fids;
      std::vector<std::tuple<std::string, TableField>> fields;
      for (auto& [nm, tf] : nm_tfs) {
        if (cols_w_refs.count(nm) > 0) {
          fids.push_back(tf.fid);
          fields.emplace_back(nm, tf);
        }
      }
      if (fids.size() > 0) {
        Legion::RegionRequirement req(vlr, mode, EXCLUSIVE, vlr);
        req.add_fields(fids);
        size_t values_idx = reqs.size();
        reqs.push_back(req);
        for (auto& [nm, tf] : fields) {
          RegionsInfo info;
          info.values = values_idx;
          info.fid = tf.fid;
#ifdef HYPERION_USE_CASACORE
          ref_column_req_indexes[nm] = values_idx;
          if (!tf.mr.is_empty()) {
            auto [mreq, vreq, oireq] = tf.mr.requirements(mode);
            info.meas_refs_size = 2 + (oireq.has_value() ? 1 : 0);
            info.meas_refs_0 = reqs.size();
            reqs.push_back(mreq);
            reqs.push_back(vreq);
            if (oireq)
              reqs.push_back(oireq.value());
          } else {
            info.meas_refs_size = 0;
          }
          if (tf.rc) {
            info.has_ref_column = true;
            ref_column_names[nm] = tf.rc.value();
          } else {
            info.has_ref_column = false;
          }
#endif // HYPERION_USE_CASACORE
          infos[nm] = info;
        }
      }
    }
#ifdef HYPERION_USE_CASACORE
    // write reference column requirement indexes
    for (auto& [nm, ri] : infos) {
      if (ri.has_ref_column)
        ri.ref_column = ref_column_req_indexes.at(ref_column_names.at(nm));
    }
#endif
    return {reqs, infos};
  }

#ifdef HYPERION_USE_CASACORE
  template <typename M>
  static mr_t<M>
  create_mr(Legion::Runtime* rt, const Regions& r) {
    MeasRef::DataRegions drs;
    drs.metadata = r.meas_refs[0];
    drs.values = r.meas_refs[1];
    if (r.meas_refs.size() > 2)
      drs.index = r.meas_refs[2];
    auto [mrs, index] = MeasRef::make<M>(rt, drs);
    if (r.ref_column) {
      assert(index.size() > 0);
      auto& [pr, fid] = r.ref_column.value();
      return std::make_tuple(mrs, index, pr, fid);
    } else {
      return mrs[0];
    }
  }
#endif //HYPERION_USE_CASACORE
};

template <
  typename M,
  int ROW_RANK,
  int COL_RANK,
  legion_privilege_mode_t MODE=READ_ONLY,
  bool CHECK_BOUNDS=false>
class ColumnMeasure { // TODO: a better name
public:

  static_assert(ROW_RANK <= COL_RANK);

  ColumnMeasure(const MSTableColumnsBase::mr_t<M>* mr) {
    std::visit(overloaded {
        [this](MSTableColumnsBase::simple_mr_t<M>& mr) {
          m_mr = mr;
          m_convert.setOut(*m_mr);
        },
        [this](MSTableColumnsBase::ref_mr_t<M>& mr) {
          auto& [mrs, rmap, rcodes_pr, fid] = mr;
          m_mrv =
            std::make_tuple(
              mrs,
              rmap,
              MSTableColumnsBase::FieldAccessor<
                HYPERION_TYPE_INT, // TODO: parameterize for string values
                ROW_RANK,
                MODE,
                CHECK_BOUNDS>(rcodes_pr, fid));
        }
      },
      *mr);
  }

  typename M::convert&
  convert_at(const Legion::Point<COL_RANK, Legion::coord_t>& pt) {
    if (m_mrv) {
      auto& [mrs, rmap, rcodes] = m_mrv.value();
      m_convert.setOut(
        *mrs[
          rmap.at(
            rcodes[
              reinterpret_cast<Legion::Point<ROW_RANK, Legion::coord_t>&>(
                pt)])]);
    }
    return m_convert;
  }

  casacore::MeasRef<M>&
  meas_ref_at(const Legion::Point<COL_RANK, Legion::coord_t>& pt) const {
    if (m_mrv) {
      auto& [mrs, rmap, rcodes] = m_mrv.value();
      m_mr =
        mrs[
          rmap.at(
            rcodes[
              reinterpret_cast<Legion::Point<ROW_RANK, Legion::coord_t>>(
                pt)])];
    }
    return *m_mr;
  }

private:
  std::shared_ptr<casacore::MeasRef<M>> m_mr;

  std::optional<
    std::tuple<
      std::vector<std::shared_ptr<casacore::MeasRef<M>>>,
      std::unordered_map<unsigned, unsigned>,
      MSTableColumnsBase::FieldAccessor<
        HYPERION_TYPE_INT,
        ROW_RANK,
        MODE,
        CHECK_BOUNDS>>>
  m_mrv;

  typename M::Convert m_convert;
};

template <MSTables T>
struct MSTableDefs {
  typedef void col_t;
  static const unsigned num_cols;
  static const char* column_names[];
  static const unsigned element_ranks[];
  static const unsigned fid_base;
  static const std::unordered_map<col_t, const char*> units;
  static const std::map<col_t, const char*> measure_names;
};

#define MSTDEF(T, t)                                                    \
  template <>                                                           \
  struct HYPERION_API MSTableDefs<MS_##T> {                             \
    typedef ms_##t##_col_t col_t;                                       \
    static const constexpr unsigned num_cols =                          \
      MS_##T##_NUM_COLS;                                                \
    static const constexpr std::array<const char*, num_cols> column_names = \
      MS_##T##_COLUMN_NAMES;                                            \
    static const constexpr std::array<unsigned, num_cols> element_ranks = \
      MS_##T##_COLUMN_ELEMENT_RANKS;                                    \
    static const constexpr unsigned fid_base =                          \
      MS_##T##_COL_FID_BASE;                                            \
    static const std::unordered_map<col_t, const char*> units;          \
    static const std::map<col_t, const char*> measure_names;            \
  };

HYPERION_FOREACH_MS_TABLE_Tt(MSTDEF);

#ifdef FOO

template <MSTables T>
struct MSTableColumns {
  //typedef ColEnums;
  static const std::array<const char*, 0> column_names;
};

template <MSTables T>
const std::array<const char*, 0> MSTableColumns<T>::column_names;

#define MSTC(T, t)                                                      \
  template <>                                                           \
  struct HYPERION_API MSTableColumns<MS_##T> {                          \
    typedef ms_##t##_col_t col_t;                                       \
    static const constexpr std::array<const char*, MS_##T##_NUM_COLS>   \
      column_names = MS_##T##_COLUMN_NAMES;                             \
    static const constexpr std::array<unsigned, MS_##T##_NUM_COLS>      \
      element_ranks = MS_##T##_COLUMN_ELEMENT_RANKS;                    \
    static constexpr Legion::FieldID fid(col_t c) {                     \
      return c + MS_##T##_COL_FID_BASE;                                 \
    }                                                                   \
    static const std::unordered_map<col_t, const char*> units;          \
    static const std::map<col_t, const char*> measure_names;            \
    static std::optional<col_t>                                         \
    lookup_col(const std::string& nm) {                                 \
      auto col =                                                        \
        std::find_if(                                                   \
          column_names.begin(),                                         \
          column_names.end(),                                           \
          [&nm](std::string cn) {                                       \
            return cn == nm;                                            \
          });                                                           \
      if (col != column_names.end())                                    \
        return                                                          \
          static_cast<col_t>(std::distance(column_names.begin(), col)); \
      return std::nullopt;                                              \
    }                                                                   \
    typedef std::array<                                                 \
      std::optional<MSTableColumnsBase::RegionsInfo>, \
      MS_##T##_NUM_COLS> \
      region_infos_t;                                                   \
    static std::unordered_map<col_t, MSTableColumnsBase::Regions>       \
      regions(                                                          \
        const region_infos_t& infos,                                    \
        const std::vector<Legion::PhysicalRegion>& prs) {               \
      std::unordered_map<col_t, MSTableColumnsBase::Regions> result;    \
      for (unsigned i = 0; i < MS_##T##_NUM_COLS; ++i) {                \
        if (infos[i])                                                   \
          result[static_cast<col_t>(i)] =                               \
            infos[i].value().regions(prs);                              \
      }                                                                 \
      return result;                                                    \
    }                                                                   \
  };
HYPERION_FOREACH_MS_TABLE_Tt(MSTC);
#else
template <MSTables T>
struct MSTableColumns {
  typedef typename MSTableDefs<T>::col_t col_t;

  static const constexpr std::array<const char*, MSTableDefs<T>::num_cols>
    column_names =
    MSTableDefs<T>::column_names;

  static const constexpr std::array<unsigned, MSTableDefs<T>::num_cols>
    element_ranks =
    MSTableDefs<T>::element_ranks;

  static constexpr Legion::FieldID fid(col_t c) {
    return c + MSTableDefs<T>::fid_base;
  }

  static const std::unordered_map<col_t, const char*> units;

  static const std::map<col_t, const char*> measure_names;

  static std::optional<col_t>
  lookup_col(const std::string& nm) {
    auto col =
      std::find_if(
        column_names.begin(),
        column_names.end(),
        [&nm](std::string cn) {
          return cn == nm;
        });
    if (col != column_names.end())
      return static_cast<col_t>(std::distance(column_names.begin(), col));
    return std::nullopt;
  }
};

template <MSTables T>
const std::unordered_map<typename MSTableColumns<T>::col_t, const char*>
MSTableColumns<T>::units =
  MSTableDefs<T>::units;

template <MSTables T>
const std::map<typename MSTableColumns<T>::col_t, const char*>
MSTableColumns<T>::measure_names =
  MSTableDefs<T>::measure_names;

#endif // FOO

#define HYPERION_COLUMN_NAME(T, C)              \
  MSTableColumns<MS_##T>::column_names[               \
    MSTableColumns<MS_##T>::col_t::MS_##T##_COL_##C]

} // end namespace hyperion

#endif // HYPERION_MS_TABLE_COLUMNS_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
