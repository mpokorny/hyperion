#include <vector>

#include "table.h"

using namespace legms;

class A0 {
public:

  typedef double COORDINATE_T;

  A0(const std::vector<double>& c)
    : m_coordinates(c) {
  }

  const std::vector<double>&
  coordinates() const {
    return m_coordinates;
  }

private:

  std::vector<double> m_coordinates;
};

class A1 {
public:

  typedef unsigned COORDINATE_T;

  A1(const std::vector<unsigned>& c)
    : m_coordinates(c) {
  }

  const
  std::vector<unsigned>&
  coordinates() const {
    return m_coordinates;
  }

private:

  std::vector<unsigned> m_coordinates;
};

class AR
  : public RowSpace<1, A0> {
public:

  using RowSpace<1, A0>::RowSpace;
};

class AA
  : public RowSpace<2, A0, A1> {
public:

  using RowSpace<2, A0, A1>::RowSpace;
};

class MSAA
  : public MSMainTable<2, AA> {
public:

  using MSMainTable<2, AA>::MSMainTable;
};

class MSA
  : public MSMainTable<1, AR> {
public:

  using MSMainTable<1, AR>::MSMainTable;
};

int
main() {

  TreeL t0(2);
  MSA msa(t0, A0({22.1, 23.2}));
  TreeL t1({{2, t0}});
  MSAA msaa(t1, A0({22.1, 23.2}), A1({12, 14}));

}
// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
