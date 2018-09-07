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
        auto& [n, t] = c;
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
    if (m_children.size() == 0)
      return false;
    COORD_T i;
    Tree t0;
    std::tie(i, std::ignore, t0)= m_children[0];
    if (!(t0 == Tree() || t0.is_array()))
      return false;
    COORD_T prev_end = i;
    for (auto& ch : m_children) {
      auto& [lo, n, t] = ch;
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
        auto& [min, max] = acc;
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
      result = array_shape(m_children);
    } else if (m_children.size() > 0) {
      result.emplace_back(
        std::numeric_limits<COORD_T>::max(),
        std::numeric_limits<COORD_T>::min());
      std::for_each(
        m_children.begin(),
        m_children.end(),
        [&result](const auto& ch){
          auto& [lo, n, t] = ch;
          COORD_T hi = lo + n - 1;
          auto& [rlo, rhi] = result[0];
          result[0] = {std::min(rlo, lo), std::max(rhi, hi)};
          extend_envelope(t, result);
        });
    }
    return result;
  }

  template <
    typename F,
    class = std::enable_if_t<
      std::is_same_v<
        std::optional<Tree<COORD_T>>,
        std::invoke_result_t<F, const std::vector<COORD_T>&>>>>
  Tree
  grow_leaves_at(F&& fn) const {
    return grow_at({}, std::forward<F>(fn));
  }

  Tree
  grow_leaves(const Tree& sprout) const {
    // this could be implemented in terms using grow_leaves_at(), however the
    // following implementation is more efficient since the 'sprout' value is
    // fixed
    std::vector<std::tuple<COORD_T, COORD_T, Tree>> sprouted;
    std::transform(
        m_children.begin(),
        m_children.end(),
        std::back_inserter(sprouted),
        [&sprout](auto& ch) {
          auto& [i, n, t] = ch;
          if (t != Tree())
            return std::make_tuple(i, n, t.grow_leaves(sprout));
          else
            return std::make_tuple(i, n, sprout);
        });
    return Tree(sprouted);
  }

  Tree
  pruned(size_t to_height) const {
    if (height() <= to_height)
      return *this;
    std::vector<std::tuple<COORD_T, COORD_T, Tree>> trimmed;
    std::transform(
      m_children.begin(),
      m_children.end(),
      std::back_inserter(trimmed),
      [&to_height](auto& ch) {
        auto& [i, n, t] = ch;
        if (to_height > 0)
          return std::make_tuple(i, n, t.pruned(to_height - 1));
        return std::make_tuple(i, n, Tree());
      });
    return Tree(trimmed);
  }

  Tree
  merged_with(const Tree& tree) const {
    if (tree == Tree() || *this == tree)
      return *this;
    if (*this == Tree())
      return tree;
    std::vector<std::tuple<COORD_T, COORD_T, Tree>> newch;
    auto ch = m_children.begin();
    auto ch_end = m_children.end();
    auto tch = tree.children().begin();
    auto tch_end = tree.children().end();
    COORD_T i, n, it, nt;
    Tree t, tt;
    if (ch != ch_end)
      std::tie(i, n, t) = *ch;
    if (tch != tch_end)
      std::tie(it, nt, tt) = *tch;
    while (ch != ch_end || tch != tch_end) {
      if (ch == ch_end || (tch != tch_end && it + nt <= i)) {
        newch.emplace_back(it, nt, tt);
        ++tch;
        if (tch != tch_end)
          std::tie(it, nt, tt) = *tch;
      } else if (tch == tch_end || (ch != ch_end && i + n <= it)) {
        newch.emplace_back(i, n, t);
        ++ch;
        if (ch != ch_end)
          std::tie(i, n, t) = *ch;
      } else {
        if (i < it) {
          newch.emplace_back(i, it - i, t);
          n -= it - i;
          i = it;
        } else if (it < i) {
          newch.emplace_back(it, i - it, tt);
          nt -= i - it;
          it = i;
        }
        COORD_T bi = i;
        COORD_T bn = std::min(n, nt);
        newch.emplace_back(bi, bn, t.merged_with(tt));
        i = bi + bn;
        n -= bn;
        it = bi + bn;
        nt -= bn;
        if (n == 0) {
          ++ch;
          if (ch != ch_end)
            std::tie(i, n, t) = *ch;
        } else {
          assert(nt == 0);
          ++tch;
          if (tch != tch_end)
            std::tie(it, nt, tt) = *tch;
        }
      }
    }
    return Tree(newch);
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
      auto& [i, n, t] = ch;
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
    if (*this != Tree()) {
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
            auto& [i, n, t] = ch;
            oss << sep;
            if (with_linebreaks && *sep != '\0') {
              oss << std::endl;
              oss << std::string(indent, ' ');
            }
            std::string si =
              std::to_string(i) + "~" + std::to_string(i + n - 1);
            oss << "(" << si;
            if (t != Tree()) {
              oss << ",";
              t.to_sstream(oss, with_linebreaks, indent + si.size() + 2);
            }
            oss << ")";
            sep = ",";
          });
      }
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
    if (seq.size() > 0) {
      std::vector<std::tuple<COORD_T, COORD_T, Tree>> sorted = seq;
      std::sort(
        sorted.begin(),
        sorted.end(),
        [](auto& b0, auto& b1) {
          return std::get<0>(b0) < std::get<1>(b1);
        });
      return
        std::accumulate(
          sorted.begin(),
          sorted.end(),
          result,
          [](auto& compacted, const auto& ch) {
            if (compacted.empty()) {
              compacted.push_back(ch);
            } else {
              auto& [i, n, t] = ch;
              auto& lastc = compacted.back();
              COORD_T* lasti = &std::get<0>(lastc);
              COORD_T* lastn = &std::get<1>(lastc);
              Tree* lastt = &std::get<2>(lastc);
              if (i < *lasti + *lastn)
                throw std::invalid_argument("Child blocks are overlapping");
              if (i == *lasti + *lastn
                  && *lastt == t)
                *lastn += n;
              else
                compacted.push_back(ch);
            }
            return compacted;
          });
    }
    return result;
  }

  static void
  array_dims(
    const std::vector<std::tuple<COORD_T, COORD_T, Tree>>& ch,
    std::back_insert_iterator<
    std::vector<std::tuple<COORD_T, COORD_T>>> shape) {

    COORD_T i, n;
    Tree t;
    std::tie(i, n, t) = ch[0];
    *shape = std::make_tuple(i, i + n - 1);
    while (t != Tree()) {
      std::tie(i, n, t) = t.children()[0];
      ++shape;
      *shape = std::make_tuple(i, i + n - 1);
    }
  }

  static std::vector<std::tuple<COORD_T, COORD_T>>
  array_shape(const std::vector<std::tuple<COORD_T, COORD_T, Tree>>& ch) {

    std::vector<std::tuple<COORD_T, COORD_T>> result;
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
      auto& [rlo, rhi] = *r;
      auto& [telo, tehi] = *te;
      *r = {std::min(rlo, telo), std::max(rhi, tehi)};
      ++r;
      ++te;
    }
    while (te != te_end) {
      env.push_back(*te);
      ++te;
    }
  }

  template <
    typename F,
    class = std::enable_if_t<
      std::is_same_v<
        std::optional<Tree<COORD_T>>,
        std::invoke_result_t<F, const std::vector<COORD_T>&>>>>
  Tree
  grow_at(const std::vector<COORD_T>& here, F&& fn) const {
    std::vector<std::tuple<COORD_T, COORD_T, Tree>> sprouted;
    std::for_each(
      m_children.begin(),
      m_children.end(),
      [&here, &fn, &sprouted](auto& ch) {
        auto& [i, n, t] = ch;
        std::vector<COORD_T> coords = here;
        if (t == Tree()) {
          for (COORD_T j = 0; j < n; ++j) {
            coords.push_back(i + j);
            std::optional<Tree> ch = fn(coords);
            if (ch)
              sprouted.push_back({i + j, 1, ch.value()});
            else
              sprouted.push_back({i + j, 1, t});
            coords.pop_back();
          }
        } else {
          for (COORD_T j = 0; j < n; ++j) {
            coords.push_back(i + j);
            sprouted.push_back({i + j, 1, t.grow_at(coords, fn)});
            coords.pop_back();
          }
        }
      });
    return Tree(sprouted);
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
