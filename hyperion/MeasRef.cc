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
#include <hyperion/MeasRef.h>
#include <hyperion/IndexTree.h>
#include <hyperion/tree_index_space.h>
#include <hyperion/utility.h>

#include <cassert>
#include <stack>
#include <unordered_map>
#include <variant>

using namespace hyperion;
using namespace Legion;

static IndexTreeL
extend_index_ranks(const IndexTreeL& tree, unsigned rank) {
  IndexTreeL result;
  auto it = IndexTreeIterator(tree);
  while (it) {
    std::vector<coord_t> coords = *it;
    IndexTreeL ch;
    for (size_t i = rank; i > 0; --i) {
      if (i > coords.size()) {
        ch = IndexTreeL({{0, 1, ch}});
      } else {
        ch = IndexTreeL({{coords.back(), 1, ch}});
        coords.pop_back();
      }
    }
    assert(coords.size() == 0);
    result = result.merged_with(ch);
    ++it;
  }
  assert(result.rank());
  assert(result.rank().value() == rank);
  return result;
}

struct MeasureIndexTrees {
  std::optional<IndexTreeL> metadata_tree;
  std::optional<IndexTreeL> value_tree;
};

static MeasureIndexTrees
measure_index_trees(
  const std::variant<const casacore::Measure*, casacore::MRBase*>& vm) {

  std::unordered_map<MeasRef::ArrayComponent, const casacore::Measure*>
    components;

  casacore::MRBase* ref_base;
  std::visit(
    overloaded {
      [&components, &ref_base](const casacore::Measure* ms) {
        assert(ms != nullptr);
        components[MeasRef::ArrayComponent::VALUE] = ms;
        ref_base = ms->getRefPtr();
      },
      [&components, &ref_base](casacore::MRBase* mr) {
        components[MeasRef::ArrayComponent::VALUE] = nullptr;
        ref_base = mr;
      }
    },
    vm);

  assert(ref_base != nullptr);
  auto offset = ref_base->offset();
  if (offset != nullptr)
    components[MeasRef::ArrayComponent::OFFSET] = offset;
  auto frame = ref_base->getFrame();
  auto epoch = frame.epoch();
  if (epoch != nullptr)
    components[MeasRef::ArrayComponent::EPOCH] = epoch;
  auto position = frame.position();
  if (position != nullptr)
    components[MeasRef::ArrayComponent::POSITION] = position;
  auto direction = frame.direction();
  if (direction != nullptr)
    components[MeasRef::ArrayComponent::DIRECTION] = direction;
  auto radial_velocity = frame.radialVelocity();
  if (radial_velocity != nullptr)
    components[MeasRef::ArrayComponent::RADIAL_VELOCITY] = radial_velocity;
  auto comet = frame.comet();
  assert(comet == nullptr);

  MeasureIndexTrees result;
  for (auto& c_m : components) {
    auto& [c, m] = c_m;
    MeasureIndexTrees ctrees;
    switch (c) {
    case MeasRef::ArrayComponent::VALUE:
      ctrees.metadata_tree = IndexTreeL();
      if (m != nullptr)
        ctrees.value_tree = IndexTreeL(m->getData()->getVector().size());
      break;
    default:
      ctrees = measure_index_trees(m);
      break;
    }
    if (ctrees.metadata_tree) {
      auto md = IndexTreeL({{c, 1, ctrees.metadata_tree.value()}});
      if (result.metadata_tree)
        result.metadata_tree = result.metadata_tree.value().merged_with(md);
      else
        result.metadata_tree = md;
    }
    if (ctrees.value_tree) {
      auto v = IndexTreeL({{c, 1, ctrees.value_tree.value()}});
      if (result.value_tree)
        result.value_tree = result.value_tree.value().merged_with(v);
      else
        result.value_tree = v;
    }
  }
  if (result.metadata_tree && !result.metadata_tree.value().rank()) {
    auto md = result.metadata_tree.value();
    result.metadata_tree = extend_index_ranks(md, md.height() + 1);
  }
  if (result.value_tree && !result.value_tree.value().rank()) {
    auto v = result.value_tree.value();
    result.value_tree = extend_index_ranks(v, v.height() + 1);
  }
  return result;
}

template <int D>
static void
initialize_vm(
  PhysicalRegion value_pr,
  PhysicalRegion metadata_pr,
  const std::tuple<MClass, casacore::MRBase*>& mrb) {

  // TODO: remove bounds check on the following accessors
  const MeasRef::ValueAccessor<WRITE_ONLY, D+1>
    vals(value_pr, 0);
  const MeasRef::RefTypeAccessor<WRITE_ONLY, D>
    rtypes(metadata_pr, MeasRef::REF_TYPE_FID);
  const MeasRef::MeasureClassAccessor<WRITE_ONLY, D>
    mclasses(metadata_pr, MeasRef::MEASURE_CLASS_FID);
  const MeasRef::NumValuesAccessor<WRITE_ONLY, D>
    numvals(metadata_pr, MeasRef::NUM_VALUES_FID);

  Point<D + 1> p1;
  Point<D> p;

  std::stack<
    std::tuple<
      std::variant<
        const casacore::Measure*,
        std::tuple<MClass, casacore::MRBase*>>,
      MeasRef::ArrayComponent>> ms;
  ms.push(std::make_tuple(mrb, MeasRef::ArrayComponent::VALUE));
  while (!ms.empty()) {
    auto& [m, c] = ms.top();

    auto ms_size = ms.size();
    auto level = ms_size - 1;
    for (unsigned j = level + 1; j < D; ++j)
      p[j] = p1[j] = 0;
    p1[D] = 0;

    casacore::MRBase* ref_base = nullptr;
    if (c == MeasRef::ArrayComponent::VALUE) {
      // the measure value itself
      p[level] = p1[level] = MeasRef::ArrayComponent::VALUE;
      std::visit(
        overloaded {
          [level, &p, &p1, &vals, &numvals, &mclasses, &ref_base, &m]
          (const casacore::Measure* measure) {
            auto mvals = measure->getData()->getVector();
            for (unsigned j = 0; j < mvals.size(); ++j) {
              p1[level + 1] = j;
              vals[p1] = mvals[j];
            }
            numvals[p] = mvals.size();
            std::string name = measure->tellMe();
            if (name == "") assert(false);
#define SET_MCLASS(M)                           \
            else if (name == MClassT<M>::name)  \
              mclasses[p] = M;
            HYPERION_FOREACH_MCLASS(SET_MCLASS)
#undef SET_MCLASS
            else assert(false);
            ref_base = measure->getRefPtr();
            m = std::make_tuple(static_cast<MClass>(mclasses[p]), ref_base);
          },
          [&p, &numvals, &mclasses, &ref_base]
          (std::tuple<MClass, casacore::MRBase*>& kr) {
            auto& [k, r] = kr;
            mclasses[p] = k;
            numvals[p] = 0;
            ref_base = r;
          }
        },
        m);
      assert(ref_base != nullptr);
      rtypes[p] = ref_base->getType();
      c = MeasRef::ArrayComponent::OFFSET;
    } else {
      ref_base =
        std::get<1>(std::get<std::tuple<MClass, casacore::MRBase*>>(m));
    }
    assert(ref_base != nullptr);

    {
      auto frame = ref_base->getFrame();
      while (ms_size == ms.size()
             && c < MeasRef::ArrayComponent::NUM_COMPONENTS) {
        const casacore::Measure* cm;
        switch (c) {
        case MeasRef::ArrayComponent::VALUE:
          assert(false);
          cm = nullptr;
          break;
        case MeasRef::ArrayComponent::OFFSET:
          cm = ref_base->offset();
          break;
        case MeasRef::ArrayComponent::EPOCH:
          cm = frame.epoch();
          break;
        case MeasRef::ArrayComponent::POSITION:
          cm = frame.position();
          break;
        case MeasRef::ArrayComponent::DIRECTION:
          cm = frame.direction();
          break;
        case MeasRef::ArrayComponent::RADIAL_VELOCITY:
          cm = frame.radialVelocity();
          break;
        default:
          assert(false);
          break;
        }
        p[level] = p1[level] = c;
        c = (MeasRef::ArrayComponent)((unsigned)c + 1);
        if (cm != nullptr)
          ms.push(std::make_tuple(cm, MeasRef::ArrayComponent::VALUE));
      }
    }
    if (ms_size == ms.size())
      ms.pop();
  }
}

template <int D>
static void
initialize(
  MeasRef::DataRegions prs,
  const std::tuple<MClass, casacore::MRBase*>& mrb) {

  initialize_vm<D>(prs.values.value(), prs.metadata, mrb);
}

template <>
void
initialize<1>(
  MeasRef::DataRegions prs,
  const std::tuple<MClass, casacore::MRBase*>& mrb) {

  if (prs.values) {
    initialize_vm<1>(prs.values.value(), prs.metadata, mrb);
  } else {

    Point<1> p;

    auto& [mclass, ref_base] = mrb;
    MeasRef::ArrayComponent c = MeasRef::ArrayComponent::VALUE;
    while (c < MeasRef::ArrayComponent::NUM_COMPONENTS) {
      p[0] = c;
      switch (c) {
      case MeasRef::ArrayComponent::VALUE: {
        const MeasRef::RefTypeAccessor<WRITE_ONLY, 1>
          rtypes(prs.metadata, MeasRef::REF_TYPE_FID);
        const MeasRef::MeasureClassAccessor<WRITE_ONLY, 1>
          mclasses(prs.metadata, MeasRef::MEASURE_CLASS_FID);
        const MeasRef::NumValuesAccessor<WRITE_ONLY, 1>
          numvals(prs.metadata, MeasRef::NUM_VALUES_FID);

        mclasses[p] = mclass;
        numvals[p] = 0;
        rtypes[p] = ref_base->getType();
        break;
      }
      case MeasRef::ArrayComponent::OFFSET:
        assert(ref_base->offset() == nullptr);
        break;
      case MeasRef::ArrayComponent::EPOCH:
        assert(ref_base->getFrame().epoch() == nullptr);
        break;
      case MeasRef::ArrayComponent::POSITION:
        assert(ref_base->getFrame().position() == nullptr);
        break;
      case MeasRef::ArrayComponent::DIRECTION:
        assert(ref_base->getFrame().direction() == nullptr);
        break;
      case MeasRef::ArrayComponent::RADIAL_VELOCITY:
        assert(ref_base->getFrame().radialVelocity() == nullptr);
        break;
      default:
        assert(false);
        break;
      }
      c = (MeasRef::ArrayComponent)((unsigned)c + 1);
    }
  }
}

template <int D>
static std::unique_ptr<casacore::MRBase>
instantiate_vm(
  PhysicalRegion value_pr,
  PhysicalRegion metadata_pr,
  Domain metadata_domain) {

  // TODO: remove bounds check on the following accessors
  const MeasRef::ValueAccessor<READ_ONLY, D+1>
    vals(value_pr, 0);
  const MeasRef::RefTypeAccessor<READ_ONLY, D>
    rtypes(metadata_pr, MeasRef::REF_TYPE_FID);
  const MeasRef::MeasureClassAccessor<READ_ONLY, D>
    mclasses(metadata_pr, MeasRef::MEASURE_CLASS_FID);
  const MeasRef::NumValuesAccessor<READ_ONLY, D>
    numvals(metadata_pr, MeasRef::NUM_VALUES_FID);

  Point<D + 1> p1;
  Point<D> p;

  std::variant<
    std::unique_ptr<casacore::Measure>,
    std::unique_ptr<casacore::MRBase>> result
    = std::unique_ptr<casacore::Measure>();
  std::stack<
    std::tuple<
      std::unique_ptr<casacore::MeasValue>,
      std::unique_ptr<casacore::MRBase>,
      MClass,
      MeasRef::ArrayComponent>> ms;
#define PUSH_NEW(s) (s).push(                   \
    std::make_tuple(                            \
      std::unique_ptr<casacore::MeasValue>(),   \
      std::unique_ptr<casacore::MRBase>(),      \
      MClass::M_NONE,                           \
      MeasRef::ArrayComponent::VALUE))

  PUSH_NEW(ms);
  while (!ms.empty()) {
    auto& [v, r, k, c] = ms.top();

    auto ms_size = ms.size();
    auto level = ms_size - 1;
    for (unsigned j = level + 1; j < D; ++j)
      p[j] = p1[j] = 0;
    p1[D] = 0;

    if (c == MeasRef::ArrayComponent::VALUE) {
      // the measure value itself
      p[level] = p1[level] = MeasRef::ArrayComponent::VALUE;
      k = (MClass)mclasses[p];
      casacore::Vector<MeasRef::VALUE_TYPE> mvals(numvals[p]);
      for (unsigned i = 0; i < mvals.size(); ++i) {
        p1[level + 1] = i;
        mvals[i] = vals[p1];
      }
      switch (k) {
#define VR(M)                                                       \
        case M:                                                     \
          if (mvals.size() > 0)                                     \
            v = std::make_unique<MClassT<M>::type::MVType>(mvals);  \
          r = std::make_unique<MClassT<M>::type::Ref>(rtypes[p]);   \
          break;
        HYPERION_FOREACH_MCLASS(VR)
#undef VR
      default:
        assert(false);
        break;
      }
      c = MeasRef::ArrayComponent::OFFSET;
      p[level] = p1[level] = c;
      if (metadata_domain.contains(p))
        PUSH_NEW(ms);
    }

    while (ms_size == ms.size()
           && c < MeasRef::ArrayComponent::NUM_COMPONENTS) {
      assert(
        std::holds_alternative<std::unique_ptr<casacore::Measure>>(result));
      std::unique_ptr<casacore::Measure> cm =
        std::move(std::get<std::unique_ptr<casacore::Measure>>(result));
      switch (c) {
      case MeasRef::ArrayComponent::VALUE:
        assert(false);
        break;
      case MeasRef::ArrayComponent::OFFSET:
        if (cm) {
          switch (k) {
#define SET_OFFSET(M)                                               \
            case M:                                                 \
              dynamic_cast<MClassT<M>::type::Ref*>(r.get())         \
                ->set(*dynamic_cast<MClassT<M>::type*>(cm.get()));  \
              break;
            HYPERION_FOREACH_MCLASS(SET_OFFSET)
#undef SET_OFFSET
          default:
            assert(false);
            break;
          }
        }
        break;
      case MeasRef::ArrayComponent::EPOCH:
        if (cm)
          r->getFrame().resetEpoch(*cm);
        break;
      case MeasRef::ArrayComponent::POSITION:
        if (cm)
          r->getFrame().resetPosition(*cm);
        break;
      case MeasRef::ArrayComponent::DIRECTION:
        if (cm)
          r->getFrame().resetDirection(*cm);
        break;
      case MeasRef::ArrayComponent::RADIAL_VELOCITY:
        if (cm)
          r->getFrame().resetRadialVelocity(*cm);
        break;
      default:
        assert(false);
        break;
      }
      c = (MeasRef::ArrayComponent)((unsigned)c + 1);
      p[level] = p1[level] = c;
      if (metadata_domain.contains(p))
        PUSH_NEW(ms);
    }

    if (ms_size == ms.size()) {
      switch (k) {
#define SET_RESULT(M)                                               \
        case M:                                                     \
          if (v)                                                    \
            result =                                                \
              std::make_unique<MClassT<M>::type>(                   \
                *dynamic_cast<MClassT<M>::type::MVType*>(v.get()),  \
                *dynamic_cast<MClassT<M>::type::Ref*>(r.get()));    \
          else                                                      \
            result = std::move(r);                                  \
          break;
        HYPERION_FOREACH_MCLASS(SET_RESULT)
#undef SET_RESULT
      default:
        assert(false);
        break;
      }
      ms.pop();
    }
  }
  assert(std::holds_alternative<std::unique_ptr<casacore::MRBase>>(result));
  return std::move(std::get<std::unique_ptr<casacore::MRBase>>(result));
#undef PUSH_NEW
}

template <int D>
static std::unique_ptr<casacore::MRBase>
instantiate(
  MeasRef::DataRegions prs,
  Domain metadata_domain) {

  return instantiate_vm<D>(prs.values.value(), prs.metadata, metadata_domain);
}

template <>
std::unique_ptr<casacore::MRBase>
instantiate<1>(
  MeasRef::DataRegions prs,
  Domain metadata_domain) {

  if (prs.values) {
    return instantiate_vm<1>(prs.values.value(), prs.metadata, metadata_domain);
  } else {

    const MeasRef::RefTypeAccessor<READ_ONLY, 1>
      rtypes(prs.metadata, MeasRef::REF_TYPE_FID);
    const MeasRef::MeasureClassAccessor<READ_ONLY, 1>
      mclasses(prs.metadata, MeasRef::MEASURE_CLASS_FID);
    const MeasRef::NumValuesAccessor<READ_ONLY, 1>
      numvals(prs.metadata, MeasRef::NUM_VALUES_FID);

    Point<1> p;

    std::unique_ptr<casacore::MRBase> result;
    MeasRef::ArrayComponent c = MeasRef::ArrayComponent::VALUE;
    while (c < MeasRef::ArrayComponent::NUM_COMPONENTS) {
      p[0] = c;
      switch (c) {
      case MeasRef::ArrayComponent::VALUE:
        assert(numvals[p] == 0);
        switch ((MClass)mclasses[p]) {
#define MRB(M)                                                    \
          case M:                                                 \
            result =                                              \
              std::make_unique<MClassT<M>::type::Ref>(rtypes[p]); \
            break;
          HYPERION_FOREACH_MCLASS(MRB)
#undef MRB
        default:
          assert(false);
          break;
        }
        break;
      default:
        assert(!metadata_domain.contains(p));
        break;
      }
      c = (MeasRef::ArrayComponent)((unsigned)c + 1);
    }
    return result;
  }
}

template <int DIM>
static void
show_index_space(Runtime* rt, IndexSpaceT<DIM> is) {
  std::ostringstream oss;
  for (PointInDomainIterator<DIM> pid(rt->get_index_space_domain(is));
       pid();
       pid++)
    oss << *pid << " ";
  oss << std::endl;
  std::cout << oss.str();
}

std::tuple<RegionRequirement, std::optional<RegionRequirement>>
MeasRef::requirements(legion_privilege_mode_t mode) const {

  RegionRequirement mreq(metadata_lr, mode, EXCLUSIVE, metadata_lr);
  mreq.add_field(MEASURE_CLASS_FID);
  mreq.add_field(REF_TYPE_FID);
  mreq.add_field(NUM_VALUES_FID);
  std::optional<RegionRequirement> vreq;
  if (values_lr != LogicalRegion::NO_REGION)
    vreq =
      RegionRequirement(
        values_lr,
        {0},
        {0},
        mode,
        EXCLUSIVE,
        values_lr);
  return std::make_tuple(mreq, vreq);
}

MClass
MeasRef::mclass(Legion::Context ctx, Legion::Runtime* rt) const {
  RegionRequirement req(metadata_lr, READ_ONLY, EXCLUSIVE, metadata_lr);
  req.add_field(MEASURE_CLASS_FID);
  auto pr = rt->map_region(ctx, req);
  auto result = mclass(pr);
  rt->unmap_region(ctx, pr);
  return result;
}

MClass
MeasRef::mclass(Legion::PhysicalRegion pr) {
  switch (pr.get_logical_region().get_index_space().get_dim()) {
#define MC(D)                                   \
    case D: {                                   \
      const MeasureClassAccessor<READ_ONLY, D>  \
        mc(pr, MEASURE_CLASS_FID);              \
      Point<D> p = Point<D>::ZEROES();          \
      return (MClass)mc[p];                     \
      break;                                    \
    }
    HYPERION_FOREACH_N_LESS_MAX(MC)
#undef MC
    default:
      assert(false);
      break;
  }
}

bool
MeasRef::equiv(Context ctx, Runtime* rt, const MeasRef& other) const {
  if (metadata_lr == other.metadata_lr &&
      values_lr == other.values_lr)
    return true;
  assert(metadata_lr != other.metadata_lr);
  assert((values_lr != other.values_lr)
         || (values_lr == LogicalRegion::NO_REGION
             && other.values_lr == LogicalRegion::NO_REGION));
  if (metadata_lr == LogicalRegion::NO_REGION
      || other.metadata_lr == LogicalRegion::NO_REGION)
    return false;
  DataRegions pr_x;
  {
    RegionRequirement
      req(metadata_lr, READ_ONLY, EXCLUSIVE, metadata_lr);
    req.add_field(MEASURE_CLASS_FID);
    req.add_field(REF_TYPE_FID);
    req.add_field(NUM_VALUES_FID);
    pr_x.metadata = rt->map_region(ctx, req);
  }
  if (values_lr != LogicalRegion::NO_REGION) {
    RegionRequirement req(values_lr, READ_ONLY, EXCLUSIVE, values_lr);
    req.add_field(0);
    pr_x.values = rt->map_region(ctx, req);
  }
  DataRegions pr_y;
  {
    RegionRequirement
      req(other.metadata_lr, READ_ONLY, EXCLUSIVE, other.metadata_lr);
    req.add_field(MEASURE_CLASS_FID);
    req.add_field(REF_TYPE_FID);
    req.add_field(NUM_VALUES_FID);
    pr_y.metadata = rt->map_region(ctx, req);
  }
  if (other.values_lr != LogicalRegion::NO_REGION) {
    RegionRequirement
      req(other.values_lr, READ_ONLY, EXCLUSIVE, other.values_lr);
    req.add_field(0);
    pr_y.values = rt->map_region(ctx, req);
  }

  bool result = equiv(rt, pr_x, pr_y);

  rt->unmap_region(ctx, pr_x.metadata);
  if (pr_x.values)
    rt->unmap_region(ctx, pr_x.values.value());
  rt->unmap_region(ctx, pr_y.metadata);
  if (pr_y.values)
    rt->unmap_region(ctx, pr_y.values.value());
  return result;
}

bool
MeasRef::equiv(Runtime* rt, const DataRegions& x, const DataRegions& y) {

  bool result = x.values.has_value() == y.values.has_value();
  if (result) {
    if (!x.values)
      return true;
    IndexSpace is = x.metadata.get_logical_region().get_index_space();
    switch (is.get_dim()) {
#define CMP(D)                                                          \
      case D: {                                                         \
        const MeasureClassAccessor<READ_ONLY, D>                        \
          mclass_x(x.metadata, MEASURE_CLASS_FID);                      \
        const MeasureClassAccessor<READ_ONLY, D>                        \
          mclass_y(y.metadata, MEASURE_CLASS_FID);                      \
        const RefTypeAccessor<READ_ONLY, D>                             \
          rtype_x(x.metadata, REF_TYPE_FID);                            \
        const RefTypeAccessor<READ_ONLY, D>                             \
          rtype_y(y.metadata, REF_TYPE_FID);                            \
        const NumValuesAccessor<READ_ONLY, D>                           \
          nvals_x(x.metadata, NUM_VALUES_FID);                          \
        const NumValuesAccessor<READ_ONLY, D>                           \
          nvals_y(y.metadata, NUM_VALUES_FID);                          \
        for (PointInDomainIterator<D> pid(rt->get_index_space_domain(is)); \
             result && pid();                                           \
             pid++)                                                     \
          result =                                                      \
            mclass_x[*pid] == mclass_y[*pid]                            \
            && rtype_x[*pid] == rtype_y[*pid]                           \
            && nvals_x[*pid] == nvals_y[*pid];                          \
        if (x.values) {                                                 \
          auto pvx = x.values.value();                                  \
          auto pvy = y.values.value();                                  \
          const ValueAccessor<READ_ONLY, D + 1> values_x(pvx, 0);       \
          const ValueAccessor<READ_ONLY, D + 1> values_y(pvy, 0);       \
          for (PointInDomainIterator<D + 1>                             \
                 pid(                                                   \
                   rt->get_index_space_domain(                          \
                     pvx.get_logical_region().get_index_space()));      \
               result && pid();                                         \
               pid++)                                                   \
            result = values_x[*pid] == values_y[*pid];                  \
        }                                                               \
        break;                                                          \
      }
      HYPERION_FOREACH_N_LESS_MAX(CMP);
#undef CMP
    }
  }
  return result;
}

MeasRef
MeasRef::clone(Context ctx, Runtime* rt) const {
  if (is_empty())
    return MeasRef();

  DataRegions drs;
  {
    RegionRequirement req(metadata_lr, READ_ONLY, EXCLUSIVE, metadata_lr);
    req.add_field(MEASURE_CLASS_FID);
    req.add_field(REF_TYPE_FID);
    req.add_field(NUM_VALUES_FID);
    drs.metadata = rt->map_region(ctx, req);
  }
  if (values_lr != LogicalRegion::NO_REGION) {
    RegionRequirement req(values_lr, READ_ONLY, EXCLUSIVE, values_lr);
    req.add_field(0);
    drs.values = rt->map_region(ctx, req);
  }
  auto result = clone(ctx, rt, drs);
  rt->unmap_region(ctx, drs.metadata);
  if (drs.values)
    rt->unmap_region(ctx, drs.values.value());
  return result;
}

MeasRef
MeasRef::clone(Context ctx, Runtime* rt, const DataRegions& drs) {
  MeasRef result;
  switch (mclass(drs.metadata)) {
#define CLONE(MC)                                       \
  case MC: {                                            \
    auto mr = make<MClassT<MC>::type>(rt, drs).value(); \
    result = create(ctx, rt, *mr);                      \
    break;                                              \
  }
  HYPERION_FOREACH_MCLASS(CLONE);
#undef CLONE
  default:
    assert(false);
    break;
  }
  return result;
}

std::array<LogicalRegion, 2>
MeasRef::create_regions(
  Context ctx,
  Runtime* rt,
  const IndexTreeL& metadata_tree,
  const IndexTreeL& value_tree) {

  LogicalRegion metadata_lr;
  {
    IndexSpace is = tree_index_space(metadata_tree, ctx, rt);
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    fa.allocate_field(sizeof(MEASURE_CLASS_TYPE), MEASURE_CLASS_FID);
    fa.allocate_field(sizeof(REF_TYPE_TYPE), REF_TYPE_FID);
    fa.allocate_field(sizeof(NUM_VALUES_TYPE), NUM_VALUES_FID);
    metadata_lr = rt->create_logical_region(ctx, is, fs);
  }

  LogicalRegion values_lr;
  {
    IndexSpace is = tree_index_space(value_tree, ctx, rt);
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    fa.allocate_field(sizeof(VALUE_TYPE), 0);
    values_lr = rt->create_logical_region(ctx, is, fs);
  }

  return {metadata_lr, values_lr};
}

MeasRef
MeasRef::create(
  Context ctx,
  Runtime *rt,
  const std::vector<casacore::MRBase*>& mrbs,
  MClass klass) {

  MeasureIndexTrees index_trees;
  {
    std::vector<MeasureIndexTrees> itrees;
    for (auto& mrb : mrbs)
      itrees.push_back(measure_index_trees(mrb));
    std::vector<std::tuple<coord_t, IndexTreeL>> md_v;
    std::vector<std::tuple<coord_t, IndexTreeL>> val_v;
    for (auto& it : itrees) {
      md_v.emplace_back(1, it.metadata_tree.value());
      val_v.emplace_back(
        1,
        it.value_tree ? it.value_tree.value() : IndexTreeL());
    }
    index_trees.metadata_tree = IndexTreeL(md_v);
    index_trees.value_tree = IndexTreeL(val_v);
  }

  std::array<LogicalRegion, 2> regions =
    create_regions(
      ctx,
      rt,
      index_trees.metadata_tree.value(),
      index_trees.value_tree.value());

  LogicalRegion metadata_lr = regions[0];
  LogicalRegion values_lr = regions[1];
  {
    std::optional<PhysicalRegion> values_pr;
    if (values_lr != LogicalRegion::NO_REGION) {
      RegionRequirement
        values_req(values_lr, WRITE_ONLY, EXCLUSIVE, values_lr);
      values_req.add_field(0);
      values_pr = rt->map_region(ctx, values_req);
    }

    RegionRequirement
      metadata_req(metadata_lr, WRITE_ONLY, EXCLUSIVE, metadata_lr);
    metadata_req.add_field(MEASURE_CLASS_FID);
    metadata_req.add_field(REF_TYPE_FID);
    metadata_req.add_field(NUM_VALUES_FID);
    auto metadata_pr = rt->map_region(ctx, metadata_req);

    switch (metadata_lr.get_dim()) {
      // FIXME
#define INIT(D)                                                 \
      case D: {                                                 \
        auto km = std::make_tuple(klass, mrbs[0]);              \
        initialize<D>(DataRegions{metadata_pr, values_pr}, km); \
        break;                                                  \
      }
      HYPERION_FOREACH_N_LESS_MAX(INIT);
#undef INIT
    default:
      assert(false);
      break;
    }

    rt->unmap_region(ctx, metadata_pr);
    if (values_pr)
      rt->unmap_region(ctx, values_pr.value());
  }

  return MeasRef(metadata_lr, values_lr);
}

std::vector<std::unique_ptr<casacore::MRBase>>
MeasRef::make(Context ctx, Runtime* rt) const {

  std::optional<PhysicalRegion> values_pr;
  if (values_lr != LogicalRegion::NO_REGION) {
    RegionRequirement
      values_req(values_lr, READ_ONLY, EXCLUSIVE, values_lr);
    values_req.add_field(0);
    values_pr = rt->map_region(ctx, values_req);
  }
  RegionRequirement
    metadata_req(metadata_lr, READ_ONLY, EXCLUSIVE, metadata_lr);
  metadata_req.add_field(MEASURE_CLASS_FID);
  metadata_req.add_field(REF_TYPE_FID);
  metadata_req.add_field(NUM_VALUES_FID);
  auto metadata_pr = rt->map_region(ctx, metadata_req);

  auto result = make(rt, DataRegions{metadata_pr, values_pr});

  rt->unmap_region(ctx, metadata_pr);
  if (values_pr)
    rt->unmap_region(ctx, values_pr.value());

  return result;
}

std::vector<std::unique_ptr<casacore::MRBase>>
MeasRef::make(Legion::Runtime* rt, DataRegions prs) {

  std::vector<std::unique_ptr<casacore::MRBase>> result;
  switch (prs.metadata.get_logical_region().get_dim()) {
#define INST(D)                                                    \
    case D:                                                        \
      result.emplace_back(                                         \
        instantiate<D>(                                            \
          prs,                                                     \
          rt->get_index_space_domain(                              \
            prs.metadata.get_logical_region().get_index_space()))); \
      break;
    HYPERION_FOREACH_N_LESS_MAX(INST);
#undef INST
  default:
    assert(false);
    break;
  }
  return result;
}

void
MeasRef::destroy(Context ctx, Runtime* rt) {
  for (auto&lr : {&metadata_lr, &values_lr}) {
    if (*lr != LogicalRegion::NO_REGION) {
      rt->destroy_field_space(ctx, lr->get_field_space());
      rt->destroy_index_space(ctx, lr->get_index_space());
      *lr = LogicalRegion::NO_REGION;
    }
  }
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
