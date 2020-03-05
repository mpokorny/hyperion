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
# ifdef HYPERION_USE_CASACORE
#  include <hyperion/MeasRef.h>
# endif

#pragma GCC visibility push(default)
# ifdef HYPERION_USE_CASACORE
#  include <casacore/measures/Measures/MeasRef.h>
# endif

# include <algorithm>
# include <array>
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
