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
#include <hyperion/MeasRef.h>
#include <hyperion/IndexTree.h>
#include <hyperion/tree_index_space.h>
#include <hyperion/utility.h>

#include <cassert>
#include <stack>
#if HAVE_CXX17
#include <variant>
#endif

using namespace hyperion;
using namespace Legion;

static IndexTreeL
extend_index_ranks(const IndexTreeL& tree, unsigned rank) {
  IndexTreeL result;
  IndexTreeIterator<coord_t> it(tree);
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
  CXX_OPTIONAL_NAMESPACE::optional<IndexTreeL> metadata_tree;
  CXX_OPTIONAL_NAMESPACE::optional<IndexTreeL> value_tree;
};

#if HAVE_CXX17
typedef std::variant<const casacore::Measure*, casacore::MRBase*>
measure_index_trees_arg_t;
#else // !HAVE_CXX17
typedef struct {
  bool is_measure;
  union {
    const casacore::Measure* measure;
    casacore::MRBase* mrbase;
  };
} measure_index_trees_arg_t;
#endif // HAVE_CXX17

static MeasureIndexTrees
measure_index_trees(const measure_index_trees_arg_t& vm) {

  std::unordered_map<MeasRef::ArrayComponent, const casacore::Measure*>
    components;

  casacore::MRBase* ref_base;
#if HAVE_CXX17
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
#else // !HAVE_CXX17
  if (vm.is_measure) {
    assert(vm.measure != nullptr);
    components[MeasRef::ArrayComponent::VALUE] = vm.measure;
    ref_base = vm.measure->getRefPtr();
  } else {
    components[MeasRef::ArrayComponent::VALUE] = nullptr;
    ref_base = vm.mrbase;
  }
#endif // HAVE_CXX17

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
  [[maybe_unused]] auto comet = frame.comet();
  assert(comet == nullptr);

  MeasureIndexTrees result;
  for (auto& c_m : components) {
#if HAVE_CXX17
    auto& [c, m] = c_m;
#else // !HAVE_CXX17
    auto& c = std::get<0>(c_m);
    auto& m = std::get<1>(c_m);
#endif // HAVE_CXX17
    MeasureIndexTrees ctrees;
    switch (c) {
    case MeasRef::ArrayComponent::VALUE:
      ctrees.metadata_tree = IndexTreeL();
      if (m != nullptr)
        ctrees.value_tree = IndexTreeL(m->getData()->getVector().size());
      break;
    default:
#if HAVE_CXX17
      ctrees = measure_index_trees(m);
#else // !HAVE_CXX17
      {
        measure_index_trees_arg_t arg;
        arg.is_measure = true;
        arg.measure = m;
        ctrees = measure_index_trees(arg);
      }
#endif // HAVE_CXX17
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

#if HAVE_CXX17
typedef std::variant<
  const casacore::Measure*,
  std::tuple<MClass, casacore::MRBase*>> msr_t;
#else // !HAVE_CXX17
typedef struct {
  bool is_measure;
  union {
    const casacore::Measure* measure;
    std::tuple<MClass, casacore::MRBase*> mc_mrbase;
  };
} msr_t;
#endif // HAVE_CXX17

template <int D>
static void
initialize(
  MeasRef::DataRegions prs,
  const std::vector<casacore::MRBase*>& mrbs,
  MClass klass) {

  // TODO: remove bounds check on the following accessors
  const MeasRef::ValueAccessor<WRITE_ONLY, D + 1, true>
    vals(prs.values, 0);
  const MeasRef::RefTypeAccessor<WRITE_ONLY, D, true>
    rtypes(prs.metadata, MeasRef::REF_TYPE_FID);
  const MeasRef::MeasureClassAccessor<WRITE_ONLY, D, true>
    mclasses(prs.metadata, MeasRef::MEASURE_CLASS_FID);
  const MeasRef::NumValuesAccessor<WRITE_ONLY, D, true>
    numvals(prs.metadata, MeasRef::NUM_VALUES_FID);

  for (size_t i = 0; i < mrbs.size(); ++i) {
    Point<D + 1> p1;
    Point<D> p;
    p[0] = p1[0] = i;

    std::stack<std::tuple<msr_t, MeasRef::ArrayComponent>> ms;
    {
#if HAVE_CXX17
      msr_t msr;
      msr = std::tuple<MClass, casacore::MRBase*>(klass, mrbs[i]);
#else // !HAVE_CXX17
      msr_t msr{false};
      msr.mc_mrbase = {klass, mrbs[i]};
#endif // HAVE_CXX17
      ms.emplace(msr, MeasRef::ArrayComponent::VALUE);
    }
    while (!ms.empty()) {
#if HAVE_CXX17
      auto& [m, c] = ms.top();
#else // !HAVE_CXX17
      auto& m = std::get<0>(ms.top());
      auto& c = std::get<1>(ms.top());
#endif // HAVE_CXX17

      const auto level = ms.size();
      for (unsigned j = level + 1; j < D; ++j)
        p[j] = p1[j] = 0;
      p1[D] = 0;

      casacore::MRBase* ref_base = nullptr;
      if (c == MeasRef::ArrayComponent::VALUE) {
        // the measure value itself
        p[level] = p1[level] = MeasRef::ArrayComponent::VALUE;
#if HAVE_CXX17
        std::visit(
          overloaded {
            [level, &p, &p1, &vals, &numvals, &mclasses, &ref_base, &m]
            (const casacore::Measure* measure) {
              auto mvals = measure->getData()->getVector();
              for (unsigned j = 0; j < mvals.size(); ++j) {
                p1[level + 1] = j;
                vals.write(p1, mvals[j]);
              }
              numvals.write(p, mvals.size());
              std::string name = measure->tellMe();
              MeasRef::MEASURE_CLASS_TYPE mtype;
              if (name == "") assert(false);
#define SET_MCLASS(M)                             \
              else if (name == MClassT<M>::name)  \
                mtype = M;
              HYPERION_FOREACH_MCLASS(SET_MCLASS)
#undef SET_MCLASS
              else assert(false);
              ref_base = measure->getRefPtr();
              mclasses.write(p, mtype);
              m = std::make_tuple(static_cast<MClass>(mtype), ref_base);
            },
            [&p, &numvals, &mclasses, &ref_base]
            (std::tuple<MClass, casacore::MRBase*>& kr) {
              auto& [k, r] = kr;
              mclasses.write(p, k);
              numvals.write(p, 0);
              ref_base = r;
            }
          },
          m);
#else // !HAVE_CXX17
        if (m.is_measure) {
          auto mvals = m.measure->getData()->getVector();
          for (unsigned j = 0; j < mvals.size(); ++j) {
            p1[level + 1] = j;
            vals.write(p1, mvals[j]);
          }
          numvals.write(p, mvals.size());
          std::string name = m.measure->tellMe();
          MeasRef::MEASURE_CLASS_TYPE mtype;
          if (name == "") assert(false);
#define SET_MCLASS(M)                           \
          else if (name == MClassT<M>::name)    \
            mtype = M;
          HYPERION_FOREACH_MCLASS(SET_MCLASS)
#undef SET_MCLASS
          else assert(false);
          ref_base = m.measure->getRefPtr();
          mclasses.write(p, mtype);
          m.is_measure = false;
          m.mc_mrbase = {static_cast<MClass>(mtype), ref_base};
        } else {
          auto& k = std::get<0>(m.mc_mrbase);
          auto& r = std::get<1>(m.mc_mrbase);
          mclasses.write(p, k);
          numvals.write(p, 0);
          ref_base = r;
        }
#endif // HAVE_CXX17
        assert(ref_base != nullptr);
        rtypes.write(p, ref_base->getType());
        c = MeasRef::ArrayComponent::OFFSET;
      } else {
#if HAVE_CXX17
        ref_base =
          std::get<1>(std::get<std::tuple<MClass, casacore::MRBase*>>(m));
#else // !HAVE_CXX17
        assert(!m.is_measure);
        ref_base = std::get<1>(m.mc_mrbase);
#endif // HAVE_CXX17
      }
      assert(ref_base != nullptr);

      {
        auto frame = ref_base->getFrame();
        while (ms.size() == level
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
          if (cm != nullptr) {
#if HAVE_CXX17
            msr_t msr;
            msr = cm;
#else // !HAVE_CXX17
            msr_t msr{true};
            msr.measure = cm;
#endif //  HAVE_CXX17
            ms.push(std::make_tuple(msr, MeasRef::ArrayComponent::VALUE));
          }
        }
      }
      if (ms.size() == level)
        ms.pop();
    }
  }
}

template <>
void
initialize<1>(
  MeasRef::DataRegions prs,
  const std::vector<casacore::MRBase*>& mrbs,
  MClass klass) {
  // TODO: this should never be called, it might be better to remove it from the
  // macro expansion in the switch statement in create() that's used to select
  // the call to initialize<D>()
  assert(false);
}

#if HAVE_CXX17
typedef std::variant<
  std::unique_ptr<casacore::Measure>,
  std::unique_ptr<casacore::MRBase>> msrp_t;
#else // !HAVE_CXX17
typedef struct {
  bool is_measure;
  std::unique_ptr<casacore::Measure> measure;
  std::unique_ptr<casacore::MRBase> mrbase;
} msrp_t;
#endif // HAVE_CXX17

template <int D>
static std::vector<std::unique_ptr<casacore::MRBase>>
instantiate(MeasRef::DataRegions prs, Domain metadata_domain) {

  // TODO: remove bounds check on the following accessors
  const MeasRef::ValueAccessor<READ_ONLY, D + 1, true>
    vals(prs.values, 0);
  const MeasRef::RefTypeAccessor<READ_ONLY, D, true>
    rtypes(prs.metadata, MeasRef::REF_TYPE_FID);
  const MeasRef::MeasureClassAccessor<READ_ONLY, D, true>
    mclasses(prs.metadata, MeasRef::MEASURE_CLASS_FID);
  const MeasRef::NumValuesAccessor<READ_ONLY, D, true>
    numvals(prs.metadata, MeasRef::NUM_VALUES_FID);

  size_t n = metadata_domain.hi()[0] + 1;
  std::vector<std::unique_ptr<casacore::MRBase>> result(n);
  for (size_t i = 0; i < n; ++i) {

    Point<D + 1> p1;
    Point<D> p;
    p[0] = p1[0] = i;

#if HAVE_CXX17
    msrp_t vmrb = std::unique_ptr<casacore::Measure>();
#else // !HAVE_CXX17
    msrp_t vmrb{true};
#endif // HAVE_CXX17
    std::stack<
      std::tuple<
        std::unique_ptr<casacore::MeasValue>,
        std::unique_ptr<casacore::MRBase>,
        MClass,
        MeasRef::ArrayComponent>> ms;
#define PUSH_NEW(s) (s).push(                   \
      std::make_tuple(                          \
        std::unique_ptr<casacore::MeasValue>(), \
        std::unique_ptr<casacore::MRBase>(),    \
        MClass::M_NONE,                         \
        MeasRef::ArrayComponent::VALUE))

    PUSH_NEW(ms);
    while (!ms.empty()) {
#if HAVE_CXX17
      auto& [v, r, k, c] = ms.top();
#else // !HAVE_CXX17
      auto& v = std::get<0>(ms.top());
      auto& r = std::get<1>(ms.top());
      auto& k = std::get<2>(ms.top());
      auto& c = std::get<3>(ms.top());
#endif // HAVE_CXX17

      const auto level = ms.size();
      for (unsigned j = level + 1; j < D; ++j)
        p[j] = p1[j] = 0;
      p1[D] = 0;

      if (c == MeasRef::ArrayComponent::VALUE) {
        // the measure value itself
        p[level] = p1[level] = MeasRef::ArrayComponent::VALUE;
        k = (MClass)mclasses.read(p);
        casacore::Vector<MeasRef::VALUE_TYPE> mvals(numvals.read(p));
        for (unsigned i = 0; i < mvals.size(); ++i) {
          p1[level + 1] = i;
          mvals[i] = vals.read(p1);
        }
        switch (k) {
#define VR(M)                                                           \
          case M:                                                       \
            if (mvals.size() > 0)                                       \
              v = std::make_unique<MClassT<M>::type::MVType>(mvals);    \
            r = std::make_unique<MClassT<M>::type::Ref>(rtypes.read(p)); \
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

      while (ms.size() == level
             && c < MeasRef::ArrayComponent::NUM_COMPONENTS) {
#if HAVE_CXX17
        assert(
          std::holds_alternative<std::unique_ptr<casacore::Measure>>(vmrb));
        std::unique_ptr<casacore::Measure> cm =
          std::move(std::get<std::unique_ptr<casacore::Measure>>(vmrb));
#else // !HAVE_CXX17
        assert(vmrb.is_measure);
        std::unique_ptr<casacore::Measure> cm = std::move(vmrb.measure);
#endif // HAVE_CXX17
        switch (c) {
        case MeasRef::ArrayComponent::VALUE:
          assert(false);
          break;
        case MeasRef::ArrayComponent::OFFSET:
          if (cm) {
            switch (k) {
#define SET_OFFSET(M)                                                 \
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

      if (ms.size() == level) {
        switch (k) {
#if HAVE_CXX17
#define SET_VMRB(M)                                                   \
          case M:                                                     \
            if (v)                                                    \
              vmrb =                                                  \
                std::make_unique<MClassT<M>::type>(                   \
                  *dynamic_cast<MClassT<M>::type::MVType*>(v.get()),  \
                  *dynamic_cast<MClassT<M>::type::Ref*>(r.get()));    \
            else                                                      \
              vmrb = std::move(r);                                    \
            break;
#else // !HAVE_CXX17
#define SET_VMRB(M)                                                   \
          case M: {                                                   \
            msrp_t msrp;                                              \
            if (v) {                                                  \
              msrp.is_measure = true;                                 \
              msrp.measure =                                          \
                std::make_unique<MClassT<M>::type>(                   \
                  *dynamic_cast<MClassT<M>::type::MVType*>(v.get()),  \
                  *dynamic_cast<MClassT<M>::type::Ref*>(r.get()));    \
              msrp.mrbase.reset();                                    \
            } else {                                                  \
              msrp.is_measure = false;                                \
              msrp.mrbase = std::move(r);                             \
              msrp.measure.reset();                                   \
            }                                                         \
            vmrb = std::move(msrp);                                   \
            break;                                                    \
        }
#endif // HAVE_CXX17
          HYPERION_FOREACH_MCLASS(SET_VMRB)
#undef SET_VMRB
        default:
            assert(false);
          break;
        }
        ms.pop();
      }
    }
#if HAVE_CXX17
    assert(std::holds_alternative<std::unique_ptr<casacore::MRBase>>(vmrb));
    result[i] = std::move(std::get<std::unique_ptr<casacore::MRBase>>(vmrb));
#else // !HAVE_CXX17
    assert(!vmrb.is_measure);
    result[i] = std::move(vmrb.mrbase);
#endif // HAVE_CXX17

#undef PUSH_NEW
  }
  return result;
}

template <>
std::vector<std::unique_ptr<casacore::MRBase>>
instantiate<1>(MeasRef::DataRegions prs, Domain metadata_domain) {
  // TODO: this should never be called, it might be better to remove it from the
  // macro expansion in the switch statement in make() that's used to select
  // the call to instantiate<D>()
  assert(false);
  return {};
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

std::tuple<
  RegionRequirement,
  RegionRequirement,
  CXX_OPTIONAL_NAMESPACE::optional<RegionRequirement>>
MeasRef::requirements(
  Legion::PrivilegeMode mode,
  bool mapped) const {

  RegionRequirement mreq(metadata_lr, mode, EXCLUSIVE, metadata_lr);
  mreq.add_field(MEASURE_CLASS_FID, mapped);
  mreq.add_field(REF_TYPE_FID, mapped);
  mreq.add_field(NUM_VALUES_FID, mapped);
  RegionRequirement vreq(values_lr, mode, EXCLUSIVE, values_lr);
  vreq.add_field(0, mapped);
  CXX_OPTIONAL_NAMESPACE::optional<RegionRequirement> ireq;
  if (index_lr != LogicalRegion::NO_REGION) {
    RegionRequirement req(index_lr, mode, EXCLUSIVE, index_lr);
    req.add_field(M_CODE_FID, mapped);
    ireq = req;
  }

  return std::make_tuple(mreq, vreq, ireq);
}

std::tuple<
  RegionRequirement,
  RegionRequirement,
  CXX_OPTIONAL_NAMESPACE::optional<RegionRequirement>>
MeasRef::requirements(
  const MeasRef::DataRegions& drs,
  PrivilegeMode mode,
  bool mapped) {

  auto& md = drs.metadata;
  auto& vl = drs.values;
  auto& oidx = drs.index;
  auto metadata_lr = md.get_logical_region();
  RegionRequirement mreq(metadata_lr, mode, EXCLUSIVE, metadata_lr);
  mreq.add_field(MEASURE_CLASS_FID, mapped);
  mreq.add_field(REF_TYPE_FID, mapped);
  mreq.add_field(NUM_VALUES_FID, mapped);
  auto values_lr = vl.get_logical_region();
  RegionRequirement vreq(values_lr, mode, EXCLUSIVE, values_lr);
  vreq.add_field(0, mapped);
  CXX_OPTIONAL_NAMESPACE::optional<RegionRequirement> ireq;
  if (oidx) {
    auto index_lr = oidx.value().get_logical_region();
    RegionRequirement req(index_lr, mode, EXCLUSIVE, index_lr);
    req.add_field(M_CODE_FID, mapped);
    ireq = req;
  }

  return std::make_tuple(mreq, vreq, ireq);
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
      return (MClass)mc.read(p);                \
      break;                                    \
    }
    HYPERION_FOREACH_N_LESS_MAX(MC)
#undef MC
    default:
      assert(false);
      return M_NONE;
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
  if (is_empty() || other.is_empty())
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
  {
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
  {
    RegionRequirement
      req(other.values_lr, READ_ONLY, EXCLUSIVE, other.values_lr);
    req.add_field(0);
    pr_y.values = rt->map_region(ctx, req);
  }

  bool result = equiv(rt, pr_x, pr_y);

  rt->unmap_region(ctx, pr_x.metadata);
  rt->unmap_region(ctx, pr_x.values);
  rt->unmap_region(ctx, pr_y.metadata);
  rt->unmap_region(ctx, pr_y.values);
  return result;
}

bool
MeasRef::equiv(Runtime* rt, const DataRegions& x, const DataRegions& y) {

  bool result = true;
  IndexSpace is = x.metadata.get_logical_region().get_index_space();
  switch (is.get_dim()) {
#define CMP(D)                                                          \
    case D: {                                                           \
      const MeasureClassAccessor<READ_ONLY, D>                          \
        mclass_x(x.metadata, MEASURE_CLASS_FID);                        \
      const MeasureClassAccessor<READ_ONLY, D>                          \
        mclass_y(y.metadata, MEASURE_CLASS_FID);                        \
      const RefTypeAccessor<READ_ONLY, D>                               \
        rtype_x(x.metadata, REF_TYPE_FID);                              \
      const RefTypeAccessor<READ_ONLY, D>                               \
        rtype_y(y.metadata, REF_TYPE_FID);                              \
      const NumValuesAccessor<READ_ONLY, D>                             \
        nvals_x(x.metadata, NUM_VALUES_FID);                            \
      const NumValuesAccessor<READ_ONLY, D>                             \
        nvals_y(y.metadata, NUM_VALUES_FID);                            \
      for (PointInDomainIterator<D> pid(rt->get_index_space_domain(is)); \
           result && pid();                                             \
           pid++)                                                       \
        result =                                                        \
          mclass_x[*pid] == mclass_y[*pid]                              \
          && rtype_x[*pid] == rtype_y[*pid]                             \
          && nvals_x[*pid] == nvals_y[*pid];                            \
      const ValueAccessor<READ_ONLY, D + 1> values_x(x.values, 0);      \
      const ValueAccessor<READ_ONLY, D + 1> values_y(y.values, 0);      \
      for (PointInDomainIterator<D + 1>                                 \
             pid(                                                       \
               rt->get_index_space_domain(                              \
                 x.values.get_logical_region().get_index_space()));     \
           result && pid();                                             \
           pid++)                                                       \
        result = values_x[*pid] == values_y[*pid];                      \
                                                                       \
      break;                                                            \
    }
    HYPERION_FOREACH_N_LESS_MAX(CMP);
#undef CMP
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
  {
    RegionRequirement req(values_lr, READ_ONLY, EXCLUSIVE, values_lr);
    req.add_field(0);
    drs.values = rt->map_region(ctx, req);
  }
  if (index_lr != LogicalRegion::NO_REGION) {
    RegionRequirement req(index_lr, READ_ONLY, EXCLUSIVE, index_lr);
    req.add_field(M_CODE_FID);
    drs.index = rt->map_region(ctx, req);
  }
  auto result = clone(ctx, rt, drs);
  rt->unmap_region(ctx, drs.metadata);
  rt->unmap_region(ctx, drs.values);
  if (drs.index)
    rt->unmap_region(ctx, drs.index.value());
  return result;
}

MeasRef
MeasRef::clone(Context ctx, Runtime* rt, const DataRegions& drs) {
  MeasRef result;
  std::vector<std::tuple<casacore::MRBase*, unsigned>> pmrbs;
  switch (mclass(drs.metadata)) {
#define CLONE(MC)                                                 \
    case MC: {                                                    \
      auto mrbs = std::get<0>(make<MClassT<MC>::type>(rt, drs));  \
      pmrbs.reserve(mrbs.size());                                 \
      for (auto& mrb : mrbs)                                      \
        pmrbs.emplace_back(mrb.get(), mrb->getType());            \
      result = create(ctx, rt, pmrbs, MC, false);                 \
      break;                                                      \
    }
    HYPERION_FOREACH_MCLASS(CLONE);
#undef CLONE
    default:
      assert(false);
      break;
  }
  return result;
}

std::array<LogicalRegion, 3>
MeasRef::create_regions(
  Context ctx,
  Runtime* rt,
  const IndexTreeL& metadata_tree,
  const IndexTreeL& value_tree,
  bool no_index) {

  LogicalRegion metadata_lr;
  {
    IndexSpace is = tree_index_space(metadata_tree, ctx, rt);
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    fa.allocate_field(sizeof(MEASURE_CLASS_TYPE), MEASURE_CLASS_FID);
    rt->attach_name(fs, MEASURE_CLASS_FID, "MeasRef::measure_class");
    fa.allocate_field(sizeof(REF_TYPE_TYPE), REF_TYPE_FID);
    rt->attach_name(fs, REF_TYPE_FID, "MeasRef::ref_type");
    fa.allocate_field(sizeof(NUM_VALUES_TYPE), NUM_VALUES_FID);
    rt->attach_name(fs, NUM_VALUES_FID, "MeasRef::num_values");
    metadata_lr = rt->create_logical_region(ctx, is, fs);
  }

  LogicalRegion values_lr;
  {
    IndexSpace is = tree_index_space(value_tree, ctx, rt);
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    fa.allocate_field(sizeof(VALUE_TYPE), 0);
    rt->attach_name(fs, 0, "MeasRef::value_type");
    values_lr = rt->create_logical_region(ctx, is, fs);
  }

  LogicalRegion index_lr;
  if (!no_index) {
    auto d = rt->get_index_space_domain(metadata_lr.get_index_space());
    Rect<1> id(d.lo()[0], d.hi()[0]);
    IndexSpace is = rt->create_index_space(ctx, id);
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    fa.allocate_field(sizeof(M_CODE_TYPE), M_CODE_FID);
    rt->attach_name(fs, M_CODE_FID, "MeasRef::m_code");
    index_lr = rt->create_logical_region(ctx, is, fs);
  }
  return {metadata_lr, values_lr, index_lr};
}

MeasRef
MeasRef::create(
  Context ctx,
  Runtime *rt,
  const std::vector<std::tuple<casacore::MRBase*, unsigned>>& mrbs,
  MClass klass,
  bool no_index) {

  MeasureIndexTrees index_trees;
  {
    std::vector<MeasureIndexTrees> itrees;
#if HAVE_CXX17
    for (auto& mrb : mrbs)
      itrees.push_back(measure_index_trees(std::get<0>(mrb)));
#else // !HAVE_CXX17
    for (auto& mrb : mrbs) {
      measure_index_trees_arg_t arg;
      arg.is_measure = false;
      arg.mrbase = std::get<0>(mrb);
      itrees.push_back(measure_index_trees(arg));
    }
#endif // HAVE_CXX17
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
    index_trees.value_tree =
      extend_index_ranks(
        IndexTreeL(val_v),//index_trees.value_tree,
        index_trees.metadata_tree.value().rank().value() + 1);
  }

  std::array<LogicalRegion, 3> regions =
    create_regions(
      ctx,
      rt,
      index_trees.metadata_tree.value(),
      index_trees.value_tree.value(),
      no_index);

  LogicalRegion metadata_lr = regions[0];
  LogicalRegion values_lr = regions[1];
  LogicalRegion index_lr = regions[2];
  {
    RegionRequirement
      values_req(values_lr, WRITE_ONLY, EXCLUSIVE, values_lr);
    values_req.add_field(0);
    auto values_pr = rt->map_region(ctx, values_req);

    RegionRequirement
      metadata_req(metadata_lr, WRITE_ONLY, EXCLUSIVE, metadata_lr);
    metadata_req.add_field(MEASURE_CLASS_FID);
    metadata_req.add_field(REF_TYPE_FID);
    metadata_req.add_field(NUM_VALUES_FID);
    auto metadata_pr = rt->map_region(ctx, metadata_req);

    std::vector<casacore::MRBase*> mrs;
    mrs.reserve(mrbs.size());
    for (auto& mrb_tp : mrbs)
      mrs.push_back(std::get<0>(mrb_tp));
    switch (metadata_lr.get_dim()) {
#define INIT(D)                                                         \
      case D: {                                                         \
        initialize<D>(DataRegions{metadata_pr, values_pr}, mrs, klass); \
        break;                                                          \
      }
      HYPERION_FOREACH_N_LESS_MAX(INIT);
#undef INIT
    default:
      assert(false);
      break;
    }

    rt->unmap_region(ctx, metadata_pr);
    rt->unmap_region(ctx, values_pr);
  }

  if (!no_index) {
    RegionRequirement req(index_lr, WRITE_ONLY, EXCLUSIVE, index_lr);
    req.add_field(M_CODE_FID);
    auto pr = rt->map_region(ctx, req);
    const MCodeAccessor<WRITE_ONLY> mcodes(pr, M_CODE_FID);
    for (size_t i = 0; i < mrbs.size(); ++i)
      mcodes[i] = std::get<1>(mrbs[i]);
    rt->unmap_region(ctx, pr);
  }

  return MeasRef(metadata_lr, values_lr, index_lr);
}

std::tuple<
  std::vector<std::unique_ptr<casacore::MRBase>>,
  std::unordered_map<unsigned, unsigned>>
MeasRef::make(Context ctx, Runtime* rt) const {

  RegionRequirement
    values_req(values_lr, READ_ONLY, EXCLUSIVE, values_lr);
  values_req.add_field(0);
  auto values_pr = rt->map_region(ctx, values_req);

  RegionRequirement
    metadata_req(metadata_lr, READ_ONLY, EXCLUSIVE, metadata_lr);
  metadata_req.add_field(MEASURE_CLASS_FID);
  metadata_req.add_field(REF_TYPE_FID);
  metadata_req.add_field(NUM_VALUES_FID);
  auto metadata_pr = rt->map_region(ctx, metadata_req);

  CXX_OPTIONAL_NAMESPACE::optional<PhysicalRegion> index_pr;
  if (index_lr != LogicalRegion::NO_REGION) {
    RegionRequirement req(index_lr, READ_ONLY, EXCLUSIVE, index_lr);
    req.add_field(M_CODE_FID);
    index_pr = rt->map_region(ctx, req);
  }
  auto result = make(rt, DataRegions{metadata_pr, values_pr, index_pr});

  rt->unmap_region(ctx, metadata_pr);
  rt->unmap_region(ctx, values_pr);
  if (index_pr)
    rt->unmap_region(ctx, index_pr.value());

  return result;
}

std::tuple<
  std::vector<std::unique_ptr<casacore::MRBase>>,
  std::unordered_map<unsigned, unsigned>>
MeasRef::make(Legion::Runtime* rt, const DataRegions& prs) {

  std::vector<std::unique_ptr<casacore::MRBase>> mrbs;
  switch (prs.metadata.get_logical_region().get_dim()) {
#define INST(D)                                                    \
    case D:                                                        \
      mrbs =                                                       \
        instantiate<D>(                                            \
          prs,                                                     \
          rt->get_index_space_domain(                              \
            prs.metadata.get_logical_region().get_index_space())); \
      break;
    HYPERION_FOREACH_N_LESS_MAX(INST);
#undef INST
  default:
    assert(false);
    break;
  }
  std::unordered_map<unsigned, unsigned> rmap;
  if (prs.index) {
    const MCodeAccessor<READ_ONLY> mcodes(prs.index.value(), M_CODE_FID);
    for (PointInDomainIterator<1>
           pid(
             rt->get_index_space_domain(
               prs.index.value().get_logical_region().get_index_space()));
         pid();
         pid++) {
      rmap[mcodes[*pid]] = *pid;
    }
  }
  return std::make_tuple(std::move(mrbs), std::move(rmap));
}

void
MeasRef::destroy(Context ctx, Runtime* rt) {
  for (auto&lr : {&metadata_lr, &values_lr, &index_lr}) {
    if (*lr != LogicalRegion::NO_REGION) {
      auto fs = lr->get_field_space();
      auto is = lr->get_index_space();
      rt->destroy_logical_region(ctx, *lr);
      rt->destroy_field_space(ctx, fs);
      rt->destroy_index_space(ctx, is);
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
