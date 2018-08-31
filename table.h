#ifndef GRDLY_TABLE_H_
#define GRDLY_TABLE_H_

#include "legion.h"
#include "Tree.h"

namespace grdly {

// class Axis {
// public:
//   typedef ... COORDINATE_T;
//   size_t MAX_COORDINATE_SIZE=...;
//   const std::vector<COORDINATE_T>& coordinates() const {}
// };

typedef Legion::coord_t coord_t;
typedef Tree<coord_t> TreeL;

template <typename AXIS, typename ...AXES>
class Table {
public:

};

template <typename AXIS0, typename AXIS1, typename AXIS2>
class Table<AXIS0, AXIS1, AXIS2> {
public:

  Table(const TreeL& tree, AXIS0 axis0, AXIS1 axis1, AXIS2 axis2)
    : m_axis0(axis0)
    , m_axis1(axis1)
    , m_axis2(axis2) {

    auto rank = tree.rank();
    assert(rank && rank.value() == 3);
    auto env = tree.envelope();
    size_t hi;
    std::tie(std::ignore, hi) = env[0];
    assert(hi < axis0.coordinates().size());
    std::tie(std::ignore, hi) = env[1];
    assert(hi < axis1.coordinates().size());
    std::tie(std::ignore, hi) = env[2];
    assert(hi < axis2.coordinates().size());
  }

  std::tuple<
    typename AXIS0::COORDINATE_T,
    typename AXIS1::COORDINATE_T,
    typename AXIS2::COORDINATE_T>
  coordinates(coord_t i0, coord_t i1, coord_t i2) const {
    return {m_axis0.coordinates()[i0],
            m_axis1.coordinates()[i1],
            m_axis2.coordinates()[i2]};
  }

private:

  AXIS0 m_axis0;
  AXIS1 m_axis1;
  AXIS2 m_axis2;
};

template <typename AXIS0, typename AXIS1>
class Table<AXIS0, AXIS1> {
public:

  Table(const TreeL& tree, AXIS0 axis0, AXIS1 axis1)
    : m_axis0(axis0)
    , m_axis1(axis1) {

    auto rank = tree.rank();
    assert(rank && rank.value() == 2);
    auto env = tree.envelope();
    size_t hi;
    std::tie(std::ignore, hi) = env[0];
    assert(hi < axis0.coordinates().size());
    std::tie(std::ignore, hi) = env[1];
    assert(hi < axis1.coordinates().size());
  }

  std::tuple<typename AXIS0::COORDINATE_T, typename AXIS1::COORDINATE_T>
  coordinates(coord_t i0, coord_t i1) const {
    return {m_axis0.coordinates()[i0], m_axis1.coordinates()[i1]};
  }

private:

  AXIS0 m_axis0;
  AXIS1 m_axis1;
};

template <typename AXIS0>
class Table<AXIS0> {
public:

  Table(const TreeL& tree, const AXIS0& axis0)
    : m_axis0(axis0) {

    auto rank = tree.rank();
    assert(rank && rank.value() == 1);
    size_t hi;
    std::tie(std::ignore, hi) = tree.envelope()[0];
    assert(hi < axis0.coordinates().size());
  }

  const typename AXIS0::COORDINATE_T
  coordinates(coord_t i) const {
    return m_axis0.coordinates()[i];
  }

  Legion::LogicalRegionT<1>
  lr_axis0(Legion::Context ctx, Legion::Runtime* runtime) {
    Legion::IndexSpaceT<1> is =
      runtime->create_index_space(
        ctx,
        Legion::Rect<1>(0, m_axis0.coordinates.size() - 1));

    Legion::FieldSpace fs = runtime->create_field_space(ctx);
    {
      auto fa = runtime->create_field_allocator(ctx, fs);
      fa.allocate_field(AXIS0::MAX_COORDINATE_SIZE, m_coord0_id);
      fa.allocate_field(sizeof(Legion::Rect<1>), m_subspace0_id);
    }
    Legion::LogicalRegionT<1> result =
      runtime->create_logical_region(ctx, is, fs);
    Legion::RegionRequirement req(result, WRITE_DISCARD, EXCLUSIVE, result);
    req.add_field(m_coord0_id);
    req.add_field(m_subspace0_id);
    Legion::InlineLauncher launcher(req);
    Legion::PhysicalRegion region = runtime->map_region(ctx, launcher);
    const Legion::FieldAccessor<
      WRITE_DISCARD,
      typename AXIS0::COORDINATE_T,
      1,
      coord_t,
      Realm::AffineAccessor<typename AXIS0::COORDINATE_T,1,coord_t>,
      false> acc_coord(region, m_coord0_id);
    const Legion::FieldAccessor<
      WRITE_DISCARD,
      Legion::Rect<1>,
      1,
      coord_t,
      Realm::AffineAccessor<Legion::Rect<1>,1,coord_t>,
      false> acc_subspace(region, m_subspace0_id);
    for (Legion::PointInRectIterator<1> pir(is); pir(); pir++) {
      acc_coord[*pir] = m_axis0.coordinates[*pir];
      acc_subspace[*pir] = Legion::Rect<1>(*pir, *pir);
    }
    runtime->unmap_region(ctx, region);
    return result;
  }

  static const int m_coord0_id = 1;
  static const int m_subspace0_id = 2;
private:

  AXIS0 m_axis0;
};

} // end namespace grdly

#endif // GRDLY_TABLE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
