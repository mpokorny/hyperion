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
#ifndef HYPERION_MEAS_REF_DICT_H_
#define HYPERION_MEAS_REF_DICT_H_

#include <hyperion/hyperion.h>
#include <hyperion/utility.h>
#include <hyperion/MeasRef.h>

#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <variant>

namespace hyperion {

class HYPERION_API MeasRefDict {
  // this class is meant to be instantiated and destroyed within a single task,
  // instances should not escape their enclosing context
public:

  typedef std::variant<
    std::shared_ptr<casacore::MeasRef<casacore::MBaseline>>,
    std::shared_ptr<casacore::MeasRef<casacore::MDirection>>,
    std::shared_ptr<casacore::MeasRef<casacore::MDoppler>>,
    std::shared_ptr<casacore::MeasRef<casacore::MEarthMagnetic>>,
    std::shared_ptr<casacore::MeasRef<casacore::MEpoch>>,
    std::shared_ptr<casacore::MeasRef<casacore::MFrequency>>,
    std::shared_ptr<casacore::MeasRef<casacore::MPosition>>,
    std::shared_ptr<casacore::MeasRef<casacore::MRadialVelocity>>,
    std::shared_ptr<casacore::MeasRef<casacore::Muvw>>> Ref;

  template <template <typename> typename C>
  MeasRefDict(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const C<const MeasRef*>& refs)
    : m_ctx(ctx)
    , m_rt(rt)
    , m_has_physical_regions(false) {

    for (auto& mr : refs)
      m_meas_refs[mr->name(ctx, rt)] = mr;
    add_tags();
  }

  template <typename MRIter>
  MeasRefDict(
    Legion::Context ctx,
    Legion::Runtime* rt,
    MRIter begin,
    MRIter end)
    : m_ctx(ctx)
    , m_rt(rt)
    , m_has_physical_regions(false) {

    std::for_each(
      begin,
      end,
      [this, &ctx, rt](const MeasRef* mr) {
        m_meas_refs[mr->name(ctx, rt)] = mr;
      });
    add_tags();
  }

  std::unordered_set<std::string>
  names() const;

  std::unordered_set<std::string>
  tags() const;

  std::optional<Ref>
  get(const std::string& name) const;

  std::optional<const MeasRef*>
  get_mr(const std::string& name) const;

  template <MClass M>
  static bool
  holds(const Ref& ref) {
    return
      std::holds_alternative<
        std::shared_ptr<casacore::MeasRef<typename MClassT<M>::type>>>(ref);
  }

  template <MClass M>
  static std::shared_ptr<casacore::MeasRef<typename MClassT<M>::type>>
  get(const Ref& ref) {
    return
      std::get<std::shared_ptr<casacore::MeasRef<typename MClassT<M>::type>>>(
        ref);
  }

protected:

  friend class MeasRefContainer;

  MeasRefDict(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::vector<
      std::tuple<std::string, MeasRef::DataRegions>>& named_prs)
  : m_ctx(ctx)
  , m_rt(rt)
  , m_has_physical_regions(true) {

    for (auto& [nm, prs] : named_prs)
      m_meas_refs[nm] = prs;
  }

private:

  void
  add_tags();

  Legion::Context m_ctx;

  Legion::Runtime* m_rt;

  bool m_has_physical_regions;

  // use pointers to MeasRef, rather than MeasRef values, even though values
  // would be cheap to copy, to emphasize that MeasRefDict instances depend on
  // the context in which they are created
  std::unordered_map<
    std::string,
    std::variant<const MeasRef*, MeasRef::DataRegions>> m_meas_refs;

  mutable std::unordered_map<std::string, Ref> m_refs;
};

} // end namespace hyperion

#endif // HYPERION_MEAS_REF_DICT_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
