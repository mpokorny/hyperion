#ifndef LEGMS_TABLE_H_
#define LEGMS_TABLE_H_

#include <vector>

#include "legion.h"
#include "Tree.h"

namespace legms {


class ColumnSpace {

public:

  static void
  add_fields(
    const Legion::FieldSpace& fs,
    Legion::Context ctx,
    Legion::Runtime *runtime);
};

// class Axis {
// public:
//   typedef ... COORDINATE_T;
//   const std::vector<COORDINATE_T>& coordinates() const {}
// };

typedef Legion::coord_t coord_t;
typedef Tree<coord_t> TreeL;

template <int DIM, typename... AXES>
class RowSpace {

public:

  Legion::IndexSpaceT<DIM>
  index_space(Legion::Context ctx, Legion::Runtime* runtime);

protected:

  TreeL m_tree;

};

template <
  typename AXIS0,
  typename AXIS1,
  typename AXIS2,
  typename AXIS3,
  typename AXIS4,
  typename AXIS5,
  typename AXIS6,
  typename AXIS7>
class RowSpace<8, AXIS0, AXIS1, AXIS2, AXIS3, AXIS4, AXIS5, AXIS6, AXIS7> {

public:

  RowSpace(
    const TreeL& tree,
    AXIS0&& axis0,
    AXIS1&& axis1,
    AXIS2&& axis2,
    AXIS3&& axis3,
    AXIS4&& axis4,
    AXIS5&& axis5,
    AXIS6&& axis6,
    AXIS7&& axis7)
    : m_tree(tree)
    , m_axis0(std::forward<AXIS0>(axis0))
    , m_axis1(std::forward<AXIS1>(axis1))
    , m_axis2(std::forward<AXIS2>(axis2))
    , m_axis3(std::forward<AXIS3>(axis3))
    , m_axis4(std::forward<AXIS4>(axis4))
    , m_axis5(std::forward<AXIS5>(axis5))
    , m_axis6(std::forward<AXIS6>(axis6))
    , m_axis7(std::forward<AXIS7>(axis7)){

    assert(is_valid());
  }

  RowSpace(
    TreeL&& tree,
    AXIS0&& axis0,
    AXIS1&& axis1,
    AXIS2&& axis2,
    AXIS3&& axis3,
    AXIS4&& axis4,
    AXIS5&& axis5,
    AXIS6&& axis6,
    AXIS7&& axis7)
    : m_tree(std::move(tree))
    , m_axis0(std::forward<AXIS0>(axis0))
    , m_axis1(std::forward<AXIS1>(axis1))
    , m_axis2(std::forward<AXIS2>(axis2))
    , m_axis3(std::forward<AXIS3>(axis3))
    , m_axis4(std::forward<AXIS4>(axis4))
    , m_axis5(std::forward<AXIS5>(axis5))
    , m_axis6(std::forward<AXIS6>(axis6))
    , m_axis7(std::forward<AXIS7>(axis7)){

    assert(is_valid());
  }

  std::tuple<
    typename AXIS0::COORDINATE_T,
    typename AXIS1::COORDINATE_T,
    typename AXIS2::COORDINATE_T,
    typename AXIS3::COORDINATE_T,
    typename AXIS4::COORDINATE_T,
    typename AXIS5::COORDINATE_T,
    typename AXIS6::COORDINATE_T,
    typename AXIS7::COORDINATE_T>
  coordinates(
    coord_t i0,
    coord_t i1,
    coord_t i2,
    coord_t i3,
    coord_t i4,
    coord_t i5,
    coord_t i6,
    coord_t i7) const {
    return {m_axis0.coordinates()[i0],
        m_axis1.coordinates()[i1],
        m_axis2.coordinates()[i2],
        m_axis3.coordinates()[i3],
        m_axis4.coordinates()[i4],
        m_axis5.coordinates()[i5],
        m_axis6.coordinates()[i6],
        m_axis7.coordinates()[i7]};
  }

  Legion::IndexSpaceT<8>
  index_space(Legion::Context ctx, Legion::Runtime* runtime) {
    return tree_index_space(m_tree, ctx, runtime);
  }

protected:

  bool
  is_valid() const {
    auto rank = m_tree.rank();
    if (rank && rank.value() != 8)
      return false;
    auto env = m_tree.envelope();
    size_t hi;
    std::tie(std::ignore, hi) = env[0];
    if (hi >= m_axis0.coordinates().size())
      return false;
    std::tie(std::ignore, hi) = env[1];
    if (hi >= m_axis1.coordinates().size())
      return false;
    std::tie(std::ignore, hi) = env[2];
    if (hi >= m_axis2.coordinates().size())
      return false;
    std::tie(std::ignore, hi) = env[3];
    if (hi >= m_axis3.coordinates().size())
      return false;
    std::tie(std::ignore, hi) = env[4];
    if (hi >= m_axis4.coordinates().size())
      return false;
    std::tie(std::ignore, hi) = env[5];
    if (hi >= m_axis5.coordinates().size())
      return false;
    std::tie(std::ignore, hi) = env[6];
    if (hi >= m_axis6.coordinates().size())
      return false;
    std::tie(std::ignore, hi) = env[7];
    return hi < m_axis7.coordinates().size();
  }

  TreeL m_tree;
  AXIS0 m_axis0;
  AXIS1 m_axis1;
  AXIS2 m_axis2;
  AXIS3 m_axis3;
  AXIS4 m_axis4;
  AXIS5 m_axis5;
  AXIS6 m_axis6;
  AXIS7 m_axis7;
};

template <
  typename AXIS0,
  typename AXIS1,
  typename AXIS2,
  typename AXIS3,
  typename AXIS4,
  typename AXIS5,
  typename AXIS6>
class RowSpace<7, AXIS0, AXIS1, AXIS2, AXIS3, AXIS4, AXIS5, AXIS6> {

public:

  RowSpace(
    const TreeL& tree,
    AXIS0&& axis0,
    AXIS1&& axis1,
    AXIS2&& axis2,
    AXIS3&& axis3,
    AXIS4&& axis4,
    AXIS5&& axis5,
    AXIS6&& axis6)
    : m_tree(tree)
    , m_axis0(std::forward<AXIS0>(axis0))
    , m_axis1(std::forward<AXIS1>(axis1))
    , m_axis2(std::forward<AXIS2>(axis2))
    , m_axis3(std::forward<AXIS3>(axis3))
    , m_axis4(std::forward<AXIS4>(axis4))
    , m_axis5(std::forward<AXIS5>(axis5))
    , m_axis6(std::forward<AXIS6>(axis6)) {

    assert(is_valid());
  }

  RowSpace(
    TreeL&& tree,
    AXIS0&& axis0,
    AXIS1&& axis1,
    AXIS2&& axis2,
    AXIS3&& axis3,
    AXIS4&& axis4,
    AXIS5&& axis5,
    AXIS6&& axis6)
    : m_tree(std::move(tree))
    , m_axis0(std::forward<AXIS0>(axis0))
    , m_axis1(std::forward<AXIS1>(axis1))
    , m_axis2(std::forward<AXIS2>(axis2))
    , m_axis3(std::forward<AXIS3>(axis3))
    , m_axis4(std::forward<AXIS4>(axis4))
    , m_axis5(std::forward<AXIS5>(axis5))
    , m_axis6(std::forward<AXIS6>(axis6)) {

    assert(is_valid());
  }

  std::tuple<
    typename AXIS0::COORDINATE_T,
    typename AXIS1::COORDINATE_T,
    typename AXIS2::COORDINATE_T,
    typename AXIS3::COORDINATE_T,
    typename AXIS4::COORDINATE_T,
    typename AXIS5::COORDINATE_T,
    typename AXIS6::COORDINATE_T>
  coordinates(
    coord_t i0,
    coord_t i1,
    coord_t i2,
    coord_t i3,
    coord_t i4,
    coord_t i5,
    coord_t i6) const {
    return {m_axis0.coordinates()[i0],
        m_axis1.coordinates()[i1],
        m_axis2.coordinates()[i2],
        m_axis3.coordinates()[i3],
        m_axis4.coordinates()[i4],
        m_axis5.coordinates()[i5],
        m_axis6.coordinates()[i6]};
  }

  Legion::IndexSpaceT<7>
  index_space(Legion::Context ctx, Legion::Runtime* runtime) {
    return tree_index_space(m_tree, ctx, runtime);
  }

protected:

  bool
  is_valid() const {
    auto rank = m_tree.rank();
    if (rank && rank.value() != 7)
      return false;
    auto env = m_tree.envelope();
    size_t hi;
    std::tie(std::ignore, hi) = env[0];
    if (hi >= m_axis0.coordinates().size())
      return false;
    std::tie(std::ignore, hi) = env[1];
    if (hi >= m_axis1.coordinates().size())
      return false;
    std::tie(std::ignore, hi) = env[2];
    if (hi >= m_axis2.coordinates().size())
      return false;
    std::tie(std::ignore, hi) = env[3];
    if (hi >= m_axis3.coordinates().size())
      return false;
    std::tie(std::ignore, hi) = env[4];
    if (hi >= m_axis4.coordinates().size())
      return false;
    std::tie(std::ignore, hi) = env[5];
    if (hi >= m_axis5.coordinates().size())
      return false;
    std::tie(std::ignore, hi) = env[6];
    return hi < m_axis6.coordinates().size();
  }

  TreeL m_tree;
  AXIS0 m_axis0;
  AXIS1 m_axis1;
  AXIS2 m_axis2;
  AXIS3 m_axis3;
  AXIS4 m_axis4;
  AXIS5 m_axis5;
  AXIS6 m_axis6;
};

template <
  typename AXIS0,
  typename AXIS1,
  typename AXIS2,
  typename AXIS3,
  typename AXIS4,
  typename AXIS5>
class RowSpace<6, AXIS0, AXIS1, AXIS2, AXIS3, AXIS4, AXIS5> {

public:

  RowSpace(
    const TreeL& tree,
    AXIS0&& axis0,
    AXIS1&& axis1,
    AXIS2&& axis2,
    AXIS3&& axis3,
    AXIS4&& axis4,
    AXIS5&& axis5)
    : m_tree(tree)
    , m_axis0(std::forward<AXIS0>(axis0))
    , m_axis1(std::forward<AXIS1>(axis1))
    , m_axis2(std::forward<AXIS2>(axis2))
    , m_axis3(std::forward<AXIS3>(axis3))
    , m_axis4(std::forward<AXIS4>(axis4))
    , m_axis5(std::forward<AXIS5>(axis5)) {

    assert(is_valid());
  }

  RowSpace(
    TreeL&& tree,
    AXIS0&& axis0,
    AXIS1&& axis1,
    AXIS2&& axis2,
    AXIS3&& axis3,
    AXIS4&& axis4,
    AXIS5&& axis5)
    : m_tree(std::move(tree))
    , m_axis0(std::forward<AXIS0>(axis0))
    , m_axis1(std::forward<AXIS1>(axis1))
    , m_axis2(std::forward<AXIS2>(axis2))
    , m_axis3(std::forward<AXIS3>(axis3))
    , m_axis4(std::forward<AXIS4>(axis4))
    , m_axis5(std::forward<AXIS5>(axis5)) {

    assert(is_valid());
  }

  std::tuple<
    typename AXIS0::COORDINATE_T,
    typename AXIS1::COORDINATE_T,
    typename AXIS2::COORDINATE_T,
    typename AXIS3::COORDINATE_T,
    typename AXIS4::COORDINATE_T,
    typename AXIS5::COORDINATE_T>
  coordinates(
    coord_t i0,
    coord_t i1,
    coord_t i2,
    coord_t i3,
    coord_t i4,
    coord_t i5) const {
    return {m_axis0.coordinates()[i0],
        m_axis1.coordinates()[i1],
        m_axis2.coordinates()[i2],
        m_axis3.coordinates()[i3],
        m_axis4.coordinates()[i4],
        m_axis5.coordinates()[i5]};
  }

  Legion::IndexSpaceT<6>
  index_space(Legion::Context ctx, Legion::Runtime* runtime) {
    return tree_index_space(m_tree, ctx, runtime);
  }

protected:

  bool
  is_valid() const {
    auto rank = m_tree.rank();
    if (rank && rank.value() != 6)
      return false;
    auto env = m_tree.envelope();
    size_t hi;
    std::tie(std::ignore, hi) = env[0];
    if (hi >= m_axis0.coordinates().size())
      return false;
    std::tie(std::ignore, hi) = env[1];
    if (hi >= m_axis1.coordinates().size())
      return false;
    std::tie(std::ignore, hi) = env[2];
    if (hi >= m_axis2.coordinates().size())
      return false;
    std::tie(std::ignore, hi) = env[3];
    if (hi >= m_axis3.coordinates().size())
      return false;
    std::tie(std::ignore, hi) = env[4];
    if (hi >= m_axis4.coordinates().size())
      return false;
    std::tie(std::ignore, hi) = env[5];
    return hi < m_axis5.coordinates().size();
  }

  TreeL m_tree;
  AXIS0 m_axis0;
  AXIS1 m_axis1;
  AXIS2 m_axis2;
  AXIS3 m_axis3;
  AXIS4 m_axis4;
  AXIS5 m_axis5;
};

template <
  typename AXIS0,
  typename AXIS1,
  typename AXIS2,
  typename AXIS3,
  typename AXIS4>
class RowSpace<5, AXIS0, AXIS1, AXIS2, AXIS3, AXIS4> {

public:

  RowSpace(
    const TreeL& tree,
    AXIS0&& axis0,
    AXIS1&& axis1,
    AXIS2&& axis2,
    AXIS3&& axis3,
    AXIS4&& axis4)
    : m_tree(tree)
    , m_axis0(std::forward<AXIS0>(axis0))
    , m_axis1(std::forward<AXIS1>(axis1))
    , m_axis2(std::forward<AXIS2>(axis2))
    , m_axis3(std::forward<AXIS3>(axis3))
    , m_axis4(std::forward<AXIS4>(axis4)) {

    assert(is_valid());
  }

  RowSpace(
    TreeL&& tree,
    AXIS0&& axis0,
    AXIS1&& axis1,
    AXIS2&& axis2,
    AXIS3&& axis3,
    AXIS4&& axis4)
    : m_tree(std::move(tree))
    , m_axis0(std::forward<AXIS0>(axis0))
    , m_axis1(std::forward<AXIS1>(axis1))
    , m_axis2(std::forward<AXIS2>(axis2))
    , m_axis3(std::forward<AXIS3>(axis3))
    , m_axis4(std::forward<AXIS4>(axis4)) {

    assert(is_valid());
  }

  std::tuple<
    typename AXIS0::COORDINATE_T,
    typename AXIS1::COORDINATE_T,
    typename AXIS2::COORDINATE_T,
    typename AXIS3::COORDINATE_T,
    typename AXIS4::COORDINATE_T>
  coordinates(
    coord_t i0,
    coord_t i1,
    coord_t i2,
    coord_t i3,
    coord_t i4) const {
    return {m_axis0.coordinates()[i0],
        m_axis1.coordinates()[i1],
        m_axis2.coordinates()[i2],
        m_axis3.coordinates()[i3],
        m_axis4.coordinates()[i4]};
  }

  Legion::IndexSpaceT<5>
  index_space(Legion::Context ctx, Legion::Runtime* runtime) {
    return tree_index_space(m_tree, ctx, runtime);
  }

protected:

  bool
  is_valid() const {
    auto rank = m_tree.rank();
    if (rank && rank.value() != 5)
      return false;
    auto env = m_tree.envelope();
    size_t hi;
    std::tie(std::ignore, hi) = env[0];
    if (hi >= m_axis0.coordinates().size())
      return false;
    std::tie(std::ignore, hi) = env[1];
    if (hi >= m_axis1.coordinates().size())
      return false;
    std::tie(std::ignore, hi) = env[2];
    if (hi >= m_axis2.coordinates().size())
      return false;
    std::tie(std::ignore, hi) = env[3];
    if (hi >= m_axis3.coordinates().size())
      return false;
    std::tie(std::ignore, hi) = env[4];
    return hi < m_axis4.coordinates().size();
  }

  TreeL m_tree;
  AXIS0 m_axis0;
  AXIS1 m_axis1;
  AXIS2 m_axis2;
  AXIS3 m_axis3;
  AXIS4 m_axis4;
};

template <typename AXIS0, typename AXIS1, typename AXIS2, typename AXIS3>
class RowSpace<4, AXIS0, AXIS1, AXIS2, AXIS3> {

public:

  RowSpace(
    const TreeL& tree,
    AXIS0&& axis0,
    AXIS1&& axis1,
    AXIS2&& axis2,
    AXIS3&& axis3)
    : m_tree(tree)
    , m_axis0(std::forward<AXIS0>(axis0))
    , m_axis1(std::forward<AXIS1>(axis1))
    , m_axis2(std::forward<AXIS2>(axis2))
    , m_axis3(std::forward<AXIS3>(axis3)) {

    assert(is_valid());
  }

  RowSpace(
    TreeL&& tree,
    AXIS0&& axis0,
    AXIS1&& axis1,
    AXIS2&& axis2,
    AXIS3&& axis3)
    : m_tree(std::move(tree))
    , m_axis0(std::forward<AXIS0>(axis0))
    , m_axis1(std::forward<AXIS1>(axis1))
    , m_axis2(std::forward<AXIS2>(axis2))
    , m_axis3(std::forward<AXIS3>(axis3)) {

    assert(is_valid());
  }

  std::tuple<
    typename AXIS0::COORDINATE_T,
    typename AXIS1::COORDINATE_T,
    typename AXIS2::COORDINATE_T,
    typename AXIS3::COORDINATE_T>
  coordinates(coord_t i0, coord_t i1, coord_t i2, coord_t i3) const {
    return {m_axis0.coordinates()[i0],
        m_axis1.coordinates()[i1],
        m_axis2.coordinates()[i2],
        m_axis3.coordinates()[i3]};
  }

  Legion::IndexSpaceT<4>
  index_space(Legion::Context ctx, Legion::Runtime* runtime) {
    return tree_index_space(m_tree, ctx, runtime);
  }

protected:

  bool
  is_valid() const {
    auto rank = m_tree.rank();
    if (rank && rank.value() != 4)
      return false;
    auto env = m_tree.envelope();
    size_t hi;
    std::tie(std::ignore, hi) = env[0];
    if (hi >= m_axis0.coordinates().size())
      return false;
    std::tie(std::ignore, hi) = env[1];
    if (hi >= m_axis1.coordinates().size())
      return false;
    std::tie(std::ignore, hi) = env[2];
    if (hi >= m_axis2.coordinates().size())
      return false;
    std::tie(std::ignore, hi) = env[3];
    return hi < m_axis3.coordinates().size();
  }

  TreeL m_tree;
  AXIS0 m_axis0;
  AXIS1 m_axis1;
  AXIS2 m_axis2;
  AXIS3 m_axis3;
};

template <typename AXIS0, typename AXIS1, typename AXIS2>
class RowSpace<3, AXIS0, AXIS1, AXIS2> {

public:

  RowSpace(const TreeL& tree, AXIS0&& axis0, AXIS1&& axis1, AXIS2&& axis2)
    : m_tree(tree)
    , m_axis0(std::forward<AXIS0>(axis0))
    , m_axis1(std::forward<AXIS1>(axis1))
    , m_axis2(std::forward<AXIS2>(axis2)) {

    assert(is_valid());
  }

  RowSpace(TreeL&& tree, AXIS0&& axis0, AXIS1&& axis1, AXIS2&& axis2)
    : m_tree(std::move(tree))
    , m_axis0(std::forward<AXIS0>(axis0))
    , m_axis1(std::forward<AXIS1>(axis1))
    , m_axis2(std::forward<AXIS2>(axis2)) {

    assert(is_valid());
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

  Legion::IndexSpaceT<3>
  index_space(Legion::Context ctx, Legion::Runtime* runtime) {
    return tree_index_space(m_tree, ctx, runtime);
  }

protected:

  bool
  is_valid() const {
    auto rank = m_tree.rank();
    if (rank && rank.value() != 3)
      return false;
    auto env = m_tree.envelope();
    size_t hi;
    std::tie(std::ignore, hi) = env[0];
    if (hi >= m_axis0.coordinates().size())
      return false;
    std::tie(std::ignore, hi) = env[1];
    if (hi >=  m_axis1.coordinates().size())
      return false;
    std::tie(std::ignore, hi) = env[2];
    return hi < m_axis2.coordinates().size();
  }

  TreeL m_tree;
  AXIS0 m_axis0;
  AXIS1 m_axis1;
  AXIS2 m_axis2;
};

template <typename AXIS0, typename AXIS1>
class RowSpace<2, AXIS0, AXIS1> {

public:

  RowSpace(const TreeL& tree, AXIS0&& axis0, AXIS1&& axis1)
    : m_tree(tree)
    , m_axis0(std::forward<AXIS0>(axis0))
    , m_axis1(std::forward<AXIS1>(axis1)) {

    assert(is_valid());
  }

  RowSpace(TreeL&& tree, AXIS0&& axis0, AXIS1&& axis1)
    : m_tree(std::move(tree))
    , m_axis0(std::forward<AXIS0>(axis0))
    , m_axis1(std::forward<AXIS1>(axis1)) {

    assert(is_valid());
  }

  std::tuple<typename AXIS0::COORDINATE_T, typename AXIS1::COORDINATE_T>
  coordinates(coord_t i0, coord_t i1) const {
    return {m_axis0.coordinates()[i0], m_axis1.coordinates()[i1]};
  }

  Legion::IndexSpaceT<3>
  index_space(Legion::Context ctx, Legion::Runtime* runtime) {
    return tree_index_space(m_tree, ctx, runtime);
  }

protected:

  bool
  is_valid() const {
    auto rank = m_tree.rank();
    if (rank && rank.value() != 2)
      return false;
    auto env = m_tree.envelope();
    size_t hi;
    std::tie(std::ignore, hi) = env[0];
    if (hi >= m_axis0.coordinates().size())
      return false;
    std::tie(std::ignore, hi) = env[1];
    return hi < m_axis1.coordinates().size();
  }

  TreeL m_tree;
  AXIS0 m_axis0;
  AXIS1 m_axis1;
};

template <typename AXIS0>
class RowSpace<1, AXIS0> {
public:

  RowSpace(const TreeL& tree, AXIS0&& axis0)
    : m_tree(tree)
    , m_axis0(std::forward<AXIS0>(axis0)) {

    assert(is_valid());
  }

  RowSpace(TreeL&& tree, AXIS0&& axis0)
    : m_tree(std::move(tree))
    , m_axis0(std::forward<AXIS0>(axis0)) {

    assert(is_valid());
  }

  const typename AXIS0::COORDINATE_T
  coordinates(coord_t i) const {
    return m_axis0.coordinates()[i];
  }

  Legion::IndexSpaceT<1>
  index_space(Legion::Context ctx, Legion::Runtime* runtime) {
    return tree_index_space(m_tree, ctx, runtime);
  }

protected:

  bool
  is_valid() const {
    auto rank = m_tree.rank();
    if (rank && rank.value() != 1)
      return false;
    size_t hi;
    std::tie(std::ignore, hi) = m_tree.envelope()[0];
    return hi < m_axis0.coordinates().size();
  }

  TreeL m_tree;
  AXIS0 m_axis0;
};

template <int DIM, typename COLSPACE, typename ROWSPACE>
class Table
  : public COLSPACE
  , public ROWSPACE {

public:

  using ROWSPACE::ROWSPACE;

  Legion::LogicalRegionT<DIM>
  logicalRegion(Legion::Context ctx, Legion::Runtime* runtime) {
    Legion::IndexSpaceT<DIM> is = ROWSPACE::index_space(ctx, runtime);
    auto fs = runtime->create_field_space(ctx);
    COLSPACE::add_fields(fs, ctx, runtime);
    return runtime->create_logical_region(ctx, is, fs);
  }

};

class MSMainColumnSpace
  : public ColumnSpace {

public:

  static void
  add_fields(
    const Legion::FieldSpace& fs,
    Legion::Context ctx,
    Legion::Runtime *runtime) {

    auto fa = runtime->create_field_allocator(ctx, fs);
    // TODO: add fields
  }

};

template <int DIM, typename ROWSPACE>
class MSMainTable
  : public Table<DIM, MSMainColumnSpace, ROWSPACE> {

public:

  using Table<DIM, MSMainColumnSpace, ROWSPACE>::Table;
};

} // end namespace legms

#endif // LEGMS_TABLE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
