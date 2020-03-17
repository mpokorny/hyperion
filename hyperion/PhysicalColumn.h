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
#ifndef HYPERION_PHYSICAL_COLUMN_H_
#define HYPERION_PHYSICAL_COLUMN_H_

#include <hyperion/hyperion.h>
#include <hyperion/Keywords.h>
#ifdef HYPERION_USE_CASACORE
# include <hyperion/MeasRef.h>
#endif // HYPERION_USE_CASACORE

#pragma GCC visibility push(default)
# include <any>
# include <memory>
# include <optional>
# include <unordered_map>
# include <variant>
# include <vector>
#pragma GCC visibility pop

namespace hyperion {

class HYPERION_API PhysicalColumn {
class Column;
public:

#ifdef HYPERION_USE_CASACORE
  std::vector<std::shared_ptr<casacore::MRBase>>
  mrbs() const;

  typedef std::shared_ptr<casacore::MRBase> simple_mrb_t;

  typedef std::tuple<
    std::vector<std::shared_ptr<casacore::MRBase>>,
    std::unordered_map<unsigned, unsigned>,
    Legion::PhysicalRegion,
    Legion::FieldID> ref_mrb_t;

  typedef std::variant<simple_mrb_t, ref_mrb_t> mrb_t;
#endif

  PhysicalColumn(
    Legion::Runtime* rt,
    hyperion::TypeTag dt,
    Legion::FieldID fid,
    unsigned index_rank,
    const Legion::PhysicalRegion& metadata,
    const Legion::LogicalRegion& parent,
    const std::optional<Legion::PhysicalRegion>& values,
    const std::optional<Keywords::pair<Legion::PhysicalRegion>>& kws
#ifdef HYPERION_USE_CASACORE
    , const std::optional<MeasRef::DataRegions>& mr_drs
    , const std::optional<std::tuple<std::string, PhysicalColumn>>& refcol
#endif // HYPERION_USE_CASACORE
    )
    : m_dt(dt)
    , m_fid(fid)
    , m_index_rank(index_rank)
    , m_metadata(metadata)
    , m_parent(parent)
    , m_values(values)
    , m_kws(kws)
#ifdef HYPERION_USE_CASACORE
    , m_mr_drs(mr_drs)
#endif // HYPERION_USE_CASACORE
    {
      if (m_kws)
        m_kw_map = Keywords::to_map(rt, m_kws.value());
#ifdef HYPERION_USE_CASACORE
      update_refcol(rt, refcol);
#endif // HYPERION_USE_CASACORE
    }

  Column
  column() const;

  template <
    Legion::PrivilegeMode MODE,
    typename FT,
    int N,
    typename COORD_T = Legion::coord_t,
    typename A = Legion::AffineAccessor<FT, N, COORD_T>,
    bool CHECK_BOUNDS = false>
  Legion::FieldAccessor<MODE, FT, N, COORD_T, A, CHECK_BOUNDS>
  accessor() const {
    return
      Legion::FieldAccessor<MODE, FT, N, COORD_T, A, CHECK_BOUNDS>(
        m_values.value(),
        m_fid);
  }

  std::optional<std::any>
  kw(const std::string& key) const;

#ifdef HYPERION_USE_CASACORE
  void
  update_refcol(
    Legion::Runtime* rt,
    const std::optional<std::tuple<std::string, PhysicalColumn>>& refcol);
#endif

  Legion::LogicalRegion
  create_index(Legion::Context ctx, Legion::Runtime* rt) const;

protected:

  friend class PhysicalTable;

  hyperion::TypeTag m_dt;

  Legion::FieldID m_fid;

  unsigned m_index_rank;

  Legion::PhysicalRegion m_metadata;

  Legion::LogicalRegion m_parent;

  // allow an optional values region, to support a PhysicalColumn without values
  // (e.g, some Table index column spaces)
  std::optional<Legion::PhysicalRegion> m_values;

  std::optional<Keywords::pair<Legion::PhysicalRegion>> m_kws;

  std::unordered_map<std::string, std::any> m_kw_map;

#ifdef HYPERION_USE_CASACORE
  std::optional<MeasRef::DataRegions> m_mr_drs;

  std::optional<std::string> m_refcol_name;

  std::optional<mrb_t> m_mrb;
#endif // HYPERION_USE_CASACORE
};

} // end namespace hyperion

#endif // HYPERION_PHYSICAL_COLUMN_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
