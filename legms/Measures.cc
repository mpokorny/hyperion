#include "Measures.h"

#ifdef LEGMS_USE_CASACORE

#include "IndexTree.h"
#include "tree_index_space.h"
#include "utility.h"

#include <cassert>
#include <stack>
#include <unordered_map>

using namespace legms;
using namespace Legion;

#define MCLASS_NAME(M) \
  const std::string MClassT<M>::name = MClassT<M>::type::showMe();
FOREACH_MCLASS(MCLASS_NAME)
#undef MCLASS_NAME

static IndexTreeL
expand_tree(const IndexTreeL& tree, unsigned rank) {
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
measure_index_trees(const casacore::Measure& measure, bool with_reference) {
  std::unordered_map<MeasureRegion::ArrayComponent, const casacore::Measure*>
    components;
  components[MeasureRegion::ArrayComponent::VALUE] = &measure;
  if (with_reference){
    auto ref_base = measure.getRefPtr();
    auto offset = ref_base->offset();
    components[MeasureRegion::ArrayComponent::OFFSET] = offset;
    auto frame = ref_base->getFrame();
    auto epoch = frame.epoch();
    components[MeasureRegion::ArrayComponent::EPOCH] = epoch;
    auto position = frame.position();
    components[MeasureRegion::ArrayComponent::POSITION] = position;
    auto direction = frame.direction();
    components[MeasureRegion::ArrayComponent::DIRECTION] = direction;
    auto radial_velocity = frame.radialVelocity();
    components[MeasureRegion::ArrayComponent::RADIAL_VELOCITY] = radial_velocity;
    auto comet = frame.comet();
    assert(comet == nullptr);
  }

  MeasureIndexTrees result;
  for (auto& c_m : components) {
    auto& [c, m] = c_m;
    MeasureIndexTrees ctrees;
    switch (c) {
    case MeasureRegion::ArrayComponent::VALUE:
      ctrees.metadata_tree = IndexTreeL();
      ctrees.value_tree =
        IndexTreeL(
          components[MeasureRegion::ArrayComponent::VALUE]
          ->getData()
          ->getVector()
          .size());
      break;
    default:
      if (m != nullptr)
        ctrees = measure_index_trees(*m, with_reference);
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
    result.metadata_tree = expand_tree(md, md.height() + 1);
  }
  if (result.value_tree && !result.value_tree.value().rank()) {
    auto v = result.value_tree.value();
    result.value_tree = expand_tree(v, v.height() + 1);
  }
  return result;
}

template <int D>
static void
initialize(
  PhysicalRegion value_pr,
  PhysicalRegion metadata_pr,
  const casacore::Measure& measure,
  bool with_reference) {

  // TODO: remove bounds check on the following accessors
  const MeasureRegion::ValueAccessor<WRITE_ONLY, D+1, true>
    vals(value_pr, 0);
  // rtypes will not be used if with_reference is false, but statically creating
  // the accessor conditionally would require another template parameter, so
  // instead we create a duplicate accessor for another field...hacky!
  const MeasureRegion::RefTypeAccessor<WRITE_ONLY, D, true>
    rtypes(
      metadata_pr,
      with_reference
      ? MeasureRegion::REF_TYPE_FID
      : MeasureRegion::NUM_VALUES_FID);
  const MeasureRegion::MeasureClassAccessor<WRITE_ONLY, D, true>
    mclasses(metadata_pr, MeasureRegion::MEASURE_CLASS_FID);
  const MeasureRegion::NumValuesAccessor<WRITE_ONLY, D, true>
    numvals(metadata_pr, MeasureRegion::NUM_VALUES_FID);

  Point<D + 1> p1;
  Point<D> p;

  std::stack<
    std::tuple<const casacore::Measure*, MeasureRegion::ArrayComponent>> ms;
  ms.push(std::make_tuple(&measure, MeasureRegion::ArrayComponent::VALUE));
  while (!ms.empty()) {
    auto& [m, c] = ms.top();

    auto ms_size = ms.size();
    auto level = ms_size - 1;
    for (unsigned j = level + 1; j < D; ++j)
      p[j] = p1[j] = 0;
    p1[D] = 0;

    if (c == MeasureRegion::ArrayComponent::VALUE) {
      // the measure value itself
      p[level] = p1[level] = MeasureRegion::ArrayComponent::VALUE;
      auto mvals = m->getData()->getVector();
      for (unsigned j = 0; j < mvals.size(); ++j) {
        p1[level + 1] = j;
        vals[p1] = mvals[j];
      }
      if (with_reference)
        rtypes[p] = m->getRefPtr()->getType();
      numvals[p] = mvals.size();
      std::string name = m->tellMe();
      if (name == "") assert(false);
#define MCLASS(M)                               \
      else if (name == MClassT<M>::name)        \
        mclasses[p] = M;
      FOREACH_MCLASS(MCLASS)
#undef MCLASS
      else assert(false);
      c = MeasureRegion::ArrayComponent::OFFSET;
    }

    if (with_reference) {
      auto ref_base = m->getRefPtr();
      auto frame = ref_base->getFrame();

      while (ms_size == ms.size()
             && c < MeasureRegion::ArrayComponent::NUM_COMPONENTS) {
        const casacore::Measure* cm;
        switch (c) {
        case MeasureRegion::ArrayComponent::VALUE:
          assert(false);
          cm = nullptr;
          break;
        case MeasureRegion::ArrayComponent::OFFSET:
          cm = ref_base->offset();
          break;
        case MeasureRegion::ArrayComponent::EPOCH:
          cm = frame.epoch();
          break;
        case MeasureRegion::ArrayComponent::POSITION:
          cm = frame.position();
          break;
        case MeasureRegion::ArrayComponent::DIRECTION:
          cm = frame.direction();
          break;
        case MeasureRegion::ArrayComponent::RADIAL_VELOCITY:
          cm = frame.radialVelocity();
          break;
        default:
          assert(false);
          break;
        }
        p[level] = p1[level] = c;
        c = (MeasureRegion::ArrayComponent)((unsigned)c + 1);
        if (cm != nullptr)
          ms.push(std::make_tuple(cm, MeasureRegion::ArrayComponent::VALUE));
      }
    }
    if (ms_size == ms.size())
      ms.pop();
  }
}

template <int D>
static std::unique_ptr<casacore::Measure>
instantiate(
  PhysicalRegion value_pr,
  PhysicalRegion metadata_pr,
  Domain metadata_domain) {

  // TODO: remove bounds check on the following accessors
  const MeasureRegion::ValueAccessor<READ_ONLY, D+1, true>
    vals(value_pr, 0);
  const MeasureRegion::RefTypeAccessor<READ_ONLY, D, true>
    rtypes(metadata_pr, MeasureRegion::REF_TYPE_FID);
  const MeasureRegion::MeasureClassAccessor<READ_ONLY, D, true>
    mclasses(metadata_pr, MeasureRegion::MEASURE_CLASS_FID);
  const MeasureRegion::NumValuesAccessor<READ_ONLY, D, true>
    numvals(metadata_pr, MeasureRegion::NUM_VALUES_FID);

  Point<D + 1> p1;
  Point<D> p;

  std::unique_ptr<casacore::Measure> result;
  std::stack<
    std::tuple<
      std::unique_ptr<casacore::MeasValue>,
      std::unique_ptr<casacore::MRBase>,
      MClass,
      MeasureRegion::ArrayComponent>> ms;
#define PUSH_NEW(s) (s).push(                   \
    std::make_tuple(                            \
      std::unique_ptr<casacore::MeasValue>(),   \
      std::unique_ptr<casacore::MRBase>(),      \
      MClass::M_NONE,                           \
      MeasureRegion::ArrayComponent::VALUE))

  PUSH_NEW(ms);
  while (!ms.empty()) {
    auto& [v, r, k, c] = ms.top();

    auto ms_size = ms.size();
    auto level = ms_size - 1;
    for (unsigned j = level + 1; j < D; ++j)
      p[j] = p1[j] = 0;
    p1[D] = 0;

    if (c == MeasureRegion::ArrayComponent::VALUE) {
      // the measure value itself
      p[level] = p1[level] = MeasureRegion::ArrayComponent::VALUE;
      k = (MClass)mclasses[p];
      casacore::Vector<MeasureRegion::VALUE_TYPE> mvals(numvals[p]);
      for (unsigned i = 0; i < mvals.size(); ++i) {
        p1[level + 1] = i;
        mvals[i] = vals[p1];
      }
      switch (k) {
#define VR(M)                                                     \
        case M:                                                   \
          v = std::make_unique<MClassT<M>::type::MVType>(mvals);  \
          r = std::make_unique<MClassT<M>::type::Ref>(rtypes[p]); \
          break;
        FOREACH_MCLASS(VR);
#undef VR
      default:
        assert(false);
        break;
      }
      c = MeasureRegion::ArrayComponent::OFFSET;
      p[level] = p1[level] = c;
      if (metadata_domain.contains(p))
        PUSH_NEW(ms);
    }

    while (ms_size == ms.size()
           && c < MeasureRegion::ArrayComponent::NUM_COMPONENTS) {
      std::unique_ptr<casacore::Measure> cm = std::move(result);
      switch (c) {
      case MeasureRegion::ArrayComponent::VALUE:
        assert(false);
        break;
      case MeasureRegion::ArrayComponent::OFFSET:
        if (cm) {
          switch (k) {
#define SET_OFFSET(M)                                               \
            case M:                                                 \
              dynamic_cast<MClassT<M>::type::Ref*>(r.get())         \
                ->set(*dynamic_cast<MClassT<M>::type*>(cm.get()));  \
              break;
            FOREACH_MCLASS(SET_OFFSET);
#undef SET_OFFSET
          default:
            assert(false);
            break;
          }
        }
        break;
      case MeasureRegion::ArrayComponent::EPOCH:
        if (cm)
          r->getFrame().resetEpoch(*cm);
        break;
      case MeasureRegion::ArrayComponent::POSITION:
        if (cm)
          r->getFrame().resetPosition(*cm);
        break;
      case MeasureRegion::ArrayComponent::DIRECTION:
        if (cm)
          r->getFrame().resetDirection(*cm);
        break;
      case MeasureRegion::ArrayComponent::RADIAL_VELOCITY:
        if (cm)
          r->getFrame().resetRadialVelocity(*cm);
        break;
      default:
        assert(false);
        break;
      }
      c = (MeasureRegion::ArrayComponent)((unsigned)c + 1);
      p[level] = p1[level] = c;
      if (metadata_domain.contains(p))
        PUSH_NEW(ms);
    }

    if (ms_size == ms.size()) {
      switch (k) {
#define SET_RESULT(M)                                             \
        case M:                                                   \
          result =                                                \
            std::make_unique<MClassT<M>::type>(                   \
              *dynamic_cast<MClassT<M>::type::MVType*>(v.get()),  \
              *dynamic_cast<MClassT<M>::type::Ref*>(r.get()));    \
          break;
        FOREACH_MCLASS(SET_RESULT);
#undef SET_RESULT
      default:
        assert(false);
        break;
      }
      ms.pop();
    }
  }
  return result;
#undef PUSH_NEW
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

MeasureRegion
MeasureRegion::create_like(
  Context ctx,
  Runtime* rt,
  const casacore::Measure& measure,
  bool with_reference) {

  auto index_trees = measure_index_trees(measure, with_reference);

  LogicalRegion metadata_region;
  {
    IndexSpace is =
      tree_index_space(index_trees.metadata_tree.value(), ctx, rt);
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    fa.allocate_field(sizeof(MEASURE_CLASS_TYPE), MEASURE_CLASS_FID);
    if (with_reference)
      fa.allocate_field(sizeof(REF_TYPE_TYPE), REF_TYPE_FID);
    fa.allocate_field(sizeof(NUM_VALUES_TYPE), NUM_VALUES_FID);
    metadata_region = rt->create_logical_region(ctx, is, fs);
  }

  LogicalRegion value_region;
  {
    IndexSpace is =
      tree_index_space(index_trees.value_tree.value(), ctx, rt);
    FieldSpace fs = rt->create_field_space(ctx);
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    fa.allocate_field(sizeof(double), 0);
    value_region = rt->create_logical_region(ctx, is, fs);
  }
  return MeasureRegion(value_region, metadata_region);
}

MeasureRegion
MeasureRegion::create_from(
  Context ctx,
  Runtime* rt,
  const casacore::Measure& measure,
  bool with_reference) {

  MeasureRegion result = create_like(ctx, rt, measure, with_reference);
  RegionRequirement
    value_req(result.value_region, WRITE_ONLY, EXCLUSIVE, result.value_region);
  value_req.add_field(0);
  auto value_pr = rt->map_region(ctx, value_req);
  RegionRequirement
    metadata_req(
      result.metadata_region,
      WRITE_ONLY,
      EXCLUSIVE,
      result.metadata_region);
  metadata_req.add_field(MEASURE_CLASS_FID);
  if (with_reference)
    metadata_req.add_field(REF_TYPE_FID);
  metadata_req.add_field(NUM_VALUES_FID);
  auto metadata_pr = rt->map_region(ctx, metadata_req);

  switch (result.metadata_region.get_dim()) {
#define INIT(D)                                                       \
    case D:                                                           \
      initialize<D>(value_pr, metadata_pr, measure, with_reference);  \
      break;
    LEGMS_FOREACH_N_LESS_MAX(INIT);
#undef INIT
  default:
    assert(false);
    break;
  }

  rt->unmap_region(ctx, metadata_pr);
  rt->unmap_region(ctx, value_pr);

  return result;
}

std::unique_ptr<casacore::Measure>
MeasureRegion::make(Context ctx, Runtime* rt) const {

  RegionRequirement
    value_req(value_region, READ_ONLY, EXCLUSIVE, value_region);
  value_req.add_field(0);
  auto value_pr = rt->map_region(ctx, value_req);
  RegionRequirement
    metadata_req(metadata_region, READ_ONLY, EXCLUSIVE, metadata_region);
  metadata_req.add_field(MEASURE_CLASS_FID);
  metadata_req.add_field(REF_TYPE_FID);
  metadata_req.add_field(NUM_VALUES_FID);
  auto metadata_pr = rt->map_region(ctx, metadata_req);

  std::unique_ptr<casacore::Measure> result;
  switch (metadata_region.get_dim()) {
#define INST(D)                                   \
    case D:                                       \
      result =                                    \
        instantiate<D>(                           \
          value_pr,                               \
          metadata_pr,                            \
          rt->get_index_space_domain(             \
            metadata_region.get_index_space()));  \
      break;
    LEGMS_FOREACH_N_LESS_MAX(INST);
#undef INST
  default:
    assert(false);
    break;
  }

  rt->unmap_region(ctx, metadata_pr);
  rt->unmap_region(ctx, value_pr);

  return result;
}

void
MeasureRegion::destroy(Context ctx, Runtime* rt) {
  for (auto&lr : {&metadata_region, &value_region}) {
    if (*lr != LogicalRegion::NO_REGION) {
      rt->destroy_field_space(ctx, lr->get_field_space());
      rt->destroy_index_space(ctx, lr->get_index_space());
      *lr = LogicalRegion::NO_REGION;
    }
  }
}

#endif // LEGMS_USE_CASACORE

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
