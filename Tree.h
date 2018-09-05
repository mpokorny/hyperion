#ifndef LEGMS_TREE_H_
#define LEGMS_TREE_H_

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace legms {


template <typename COORD_T=int>
class Tree {

public:

  Tree() {}

  Tree(COORD_T n)
    : m_children({{0, n, Tree()}}) {
  }

  Tree(const std::vector<std::tuple<COORD_T, Tree> >& ch) {
    COORD_T end = 0;
    std::vector<std::tuple<COORD_T, COORD_T, Tree>> ch1;
    std::transform(
      ch.begin(),
      ch.end(),
      std::back_inserter(ch1),
      [&end](auto& c) {
        COORD_T n;
        Tree t;
        std::tie(n, t) = c;
        COORD_T last_end = end;
        end += n;
        return std::make_tuple(last_end, n, t);
      });
    m_children = std::move(compact(ch1));
  }

  Tree(const std::vector<std::tuple<COORD_T, COORD_T, Tree> >& ch)
    : m_children(compact(ch)) {
  }

  bool
  operator==(const Tree& other) const {
    return m_children == other.m_children;
  }

  bool
  operator!=(const Tree& other) const {
    return !operator==(other);
  }

  const std::vector<std::tuple<COORD_T, COORD_T, Tree>>&
  children() const {
    return m_children;
  }

  bool
  is_array() const {
    COORD_T i, n;
    Tree t0;
    std::tie(i, n, t0)= m_children[0];
    if (!(t0 == Tree() || t0.is_array()))
      return false;
    COORD_T prev_end = i;
    for (auto& ch : m_children) {
      COORD_T lo, n;
      Tree t;
      std::tie(lo, n, t) = ch;
      if (lo != prev_end || t != t0)
        return false;
      prev_end = lo + n;
    }
    return true;
  }

  bool
  is_leaf() const {
    return std::all_of(
      m_children.begin(),
      m_children.end(),
      [](const auto& ch) {
        return std::get<2>(ch) == Tree();
      });
  }

  size_t
  size() const {
    if (m_children.size() == 0)
      return 1;
    else
      return std::accumulate(
        m_children.begin(),
        m_children.end(),
        0,
        [](size_t acc, const auto& ch) {
          COORD_T n;
          Tree t;
          std::tie(std::ignore, n, t) = ch;
          return acc + n * t.size();
        });
  }

  std::tuple<COORD_T, COORD_T>
  index_range() const {
    // inclusive
    return std::accumulate(
      m_children.begin(),
      m_children.end(),
      std::make_tuple(
        std::numeric_limits<COORD_T>::max(),
        std::numeric_limits<COORD_T>::min()),
      [](const auto& acc, const auto& ch) {
        COORD_T min, max;
        std::tie(min, max) = acc;
        COORD_T lo, n, hi;
        std::tie(lo, n, std::ignore) = ch;
        hi = lo + n - 1;
        if (lo < min)
          min = lo;
        if (hi > max)
          max = hi;
        return std::make_tuple(min, max);
      });
  }

  unsigned
  height() const {
    if (is_leaf())
      return 0;
    else
      return
        1 + std::accumulate(
          m_children.begin(),
          m_children.end(),
          0,
          [](unsigned hmax, const auto& ch) {
            Tree t;
            std::tie(std::ignore, std::ignore, t) = ch;
            return std::max(hmax, t.height());
          });
  }

  std::optional<unsigned>
  rank() const {
    if (is_leaf())
      return 1;
    auto ch = m_children.begin();
    auto end = m_children.end();
    Tree t;
    std::tie(std::ignore, std::ignore, t) = *ch;
    std::optional<unsigned> result = t.rank();
    while (result && ++ch != end) {
      std::tie(std::ignore, std::ignore, t) = *ch;
      if (result != t.rank())
        result.reset();
    }
    return (result ? (result.value() + 1) : result);
  }

  std::vector<std::tuple<COORD_T, COORD_T>>
  envelope() const {
    std::vector<std::tuple<COORD_T, COORD_T>> result;
    if (is_array()) {
      auto shp = array_shape(m_children);
      std::transform(
        shp.begin(),
        shp.end(),
        std::back_inserter(result),
        [](auto& d) {
          return std::make_tuple(0, d - 1);
        });
    } else {
      result.emplace_back(
        std::get<0>(*m_children.begin()),
        std::get<0>(*m_children.rbegin()));
      std::for_each(
        m_children.begin(),
        m_children.end(),
        [&result](const auto& ch){
          COORD_T lo, n, hi;
          Tree t;
          std::tie(lo, n, t) = ch;
          hi = lo + n - 1;
          COORD_T rlo, rhi;
          std::tie(rlo, rhi) = result[0];
          result[0] = {std::min(rlo, lo), std::max(rhi, hi)};
          extend_envelope(t, result);
        });
    }
    return result;
  }

  size_t
  serialized_size() const {
    size_t result = sizeof(size_t);
    for (auto& ch : m_children) {
      result += 2 * sizeof(COORD_T);
      const Tree& t = std::get<2>(ch);
      result += t.serialized_size();
    }
    return result;
  }

  size_t
  serialize(char *buff) const {
    char* start = buff;
    buff += sizeof(size_t);
    for (auto& ch : m_children) {
      COORD_T i, n;
      Tree t;
      std::tie(i, n, t) = ch;
      *reinterpret_cast<COORD_T *>(buff) = i;
      buff += sizeof(COORD_T);
      *reinterpret_cast<COORD_T *>(buff) = n;
      buff += sizeof(COORD_T);
      buff += t.serialize(buff);
    }
    *reinterpret_cast<size_t *>(start) = buff - start;
    return *reinterpret_cast<size_t *>(start);
  }

  static Tree
  deserialize(const char *buff) {
    size_t sz = *reinterpret_cast<const size_t *>(buff);
    const void *end = buff + sz;
    buff += sizeof(size_t);
    std::vector<std::tuple<COORD_T, COORD_T, Tree>> children;
    while (buff != end) {
      COORD_T i = *reinterpret_cast<const COORD_T *>(buff);
      buff += sizeof(COORD_T);
      COORD_T n = *reinterpret_cast<const COORD_T *>(buff);
      buff += sizeof(COORD_T);
      const char *next = buff + *reinterpret_cast<const size_t *>(buff);
      children.emplace_back(i, n, deserialize(buff));
      buff = next;
    }
    return Tree(children);
  }

  std::string
  show(bool with_linebreaks=false, unsigned indent=0) const {
    std::ostringstream oss;
    to_sstream(oss, with_linebreaks, indent);
    return oss.str();
  }

  void
  to_sstream(
    std::ostringstream& oss,
    bool with_linebreaks,
    unsigned indent) const {

    oss << "{";
    ++indent;
    if (is_leaf()) {
      COORD_T i, n;
      auto ch = m_children.begin();
      std::tie(i, n, std::ignore) = *ch;
      oss << i << "~" << i + n - 1;
      ++ch;
      std::for_each(
        ch,
        m_children.end(),
        [&oss](const auto& c) {
          COORD_T i, n;
          std::tie(i, n, std::ignore) = c;
          oss << "," << i << "~" << i + n - 1;
        });
    } else {
      const char* sep = "";
      std::for_each(
        m_children.begin(),
        m_children.end(),
        [&sep, &oss, &with_linebreaks, &indent](const auto& ch) {
          COORD_T i, n;
          Tree t;
          std::tie(i, n, t) = ch;
          oss << sep;
          if (with_linebreaks && *sep != '\0') {
            oss << std::endl;
            oss << std::string(indent, ' ');
          }
          std::string si = std::to_string(i) + "~" + std::to_string(i + n - 1);
          oss << "(" << si << ",";
          t.to_sstream(oss, with_linebreaks, indent + si.size() + 2);
          oss << ")";
          sep = ",";
        });
    }
    oss << "}";
  }

protected:

  void
  swap(Tree& other) {
    std::swap(m_children, other.m_children);
  }

  static std::vector<std::tuple<COORD_T, COORD_T, Tree>>
  compact(const std::vector<std::tuple<COORD_T, COORD_T, Tree>>& seq) {
    std::vector<std::tuple<COORD_T, COORD_T, Tree>> result;
    if (seq.size() > 0)
      return
        std::accumulate(
          seq.begin(),
          seq.end(),
          result,
          [](auto& compacted, const auto& ch) {
            if (compacted.empty()) {
              compacted.push_back(ch);
            } else {
              COORD_T i = 0, n = 0;
              Tree t;
              std::tie(i, n, t) = ch;
              auto& lastc = compacted.back();
              COORD_T* lasti = &std::get<0>(lastc);
              COORD_T* lastn = &std::get<1>(lastc);
              Tree* lastt = &std::get<2>(lastc);
              if (i < *lasti + *lastn)
                throw std::invalid_argument("Child blocks are overlapping");
              if (i == *lasti + *lastn
                  && (lastt->m_children.size() == 0 || lastt->is_array())
                  && *lastt == t)
                *lastn += n;
              else
                compacted.push_back(ch);
            }
            return compacted;
          });
    return result;
  }

  static std::tuple<COORD_T, Tree>
  peel_top(const std::vector<std::tuple<COORD_T, COORD_T, Tree>>& ch) {
    COORD_T last_n;
    Tree last_t;
    std::tie(std::ignore, last_n, last_t) = *ch.rbegin();
    return std::make_tuple(last_n, last_t);
  }

  static void
  array_dims(
    const std::vector<std::tuple<COORD_T, COORD_T, Tree>>& ch,
    std::back_insert_iterator<std::vector<COORD_T>> shape) {

    COORD_T n;
    Tree t;
    std::tie(n, t) = peel_top(ch);
    *shape = n;
    while (t != Tree()) {
      std::tie(n, t) = peel_top(t.children());
      ++shape;
      *shape = n;
    }
  }

  static std::vector<COORD_T>
  array_shape(const std::vector<std::tuple<COORD_T, COORD_T, Tree>>& ch) {

    std::vector<COORD_T> result;
    array_dims(ch, std::back_inserter(result));
    return result;
  }

  static void
  extend_envelope(
    const Tree& t,
    std::vector<std::tuple<COORD_T, COORD_T>>& env) {

    auto r = env.begin();
    auto r_end = env.end();
    ++r;
    auto tenv = t.envelope();
    auto te = tenv.begin();
    auto te_end = tenv.end();
    while (r != r_end && te != te_end) {
      COORD_T rlo, rhi, telo, tehi;
      std::tie(rlo, rhi) = *r;
      std::tie(telo, tehi) = *te;
      *r = {std::min(rlo, telo), std::max(rhi, tehi)};
      ++r;
      ++te;
    }
    while (te != te_end) {
      env.push_back(*te);
      ++te;
    }
  }

private:

  std::vector<std::tuple<COORD_T, COORD_T, Tree>> m_children;
};

} // end namespace legms

#endif // LEGMS_TREE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
