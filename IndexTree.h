#ifndef LEGMS_INDEX_TREE_H_
#define LEGMS_INDEX_TREE_H_

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
class IndexTree {

public:

  IndexTree() {}

  IndexTree(COORD_T n)
    : m_children({{0, n, IndexTree()}}) {
  }

  IndexTree(const std::vector<std::tuple<COORD_T, IndexTree> >& ch) {
    COORD_T end = 0;
    std::vector<std::tuple<COORD_T, COORD_T, IndexTree>> ch1;
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

  IndexTree(const std::vector<std::tuple<COORD_T, COORD_T, IndexTree> >& ch)
    : m_children(compact(ch)) {
  }

  bool
  operator==(const IndexTree& other) const {
    return m_children == other.m_children;
  }

  bool
  operator!=(const IndexTree& other) const {
    return !operator==(other);
  }

  bool
  less_than(const IndexTree& other) const {
    bool other_empty = other == IndexTree();
    bool result;
    if (*this == IndexTree())
      result = !other_empty;
    else if (other_empty)
      result = false;
    else {
      auto& [ie, ne, te] = m_children[m_children.size() - 1];
      auto jch = other.children();
      auto& [j0, jn0, jt0]= jch[0];
      if (ie + ne <= j0)
        result = true;
      else if (jn0 == 1 && jch.size() == 1 && ie + ne - 1 == j0)
        result = te.less_than(jt0);
      else
        result = false;
    }
    return result;
  }

  const std::vector<std::tuple<COORD_T, COORD_T, IndexTree>>&
  children() const {
    return m_children;
  }

  bool
  is_array() const {
    if (m_children.size() == 0)
      return false;
    COORD_T i;
    IndexTree t0;
    std::tie(i, std::ignore, t0)= m_children[0];
    if (!(t0 == IndexTree() || t0.is_array()))
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
        return std::get<2>(ch) == IndexTree();
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
          IndexTree t;
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
            IndexTree t;
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
    IndexTree t;
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
        std::optional<IndexTree<COORD_T>>,
        std::invoke_result_t<F, const std::vector<COORD_T>&>>>>
  IndexTree
  grow_leaves_at(F&& fn) const {
    return grow_at({}, std::forward<F>(fn));
  }

  IndexTree
  grow_leaves(const IndexTree& sprout) const {
    // this could be implemented in terms using grow_leaves_at(), however the
    // following implementation is more efficient since the 'sprout' value is
    // fixed
    std::vector<std::tuple<COORD_T, COORD_T, IndexTree>> sprouted;
    std::transform(
        m_children.begin(),
        m_children.end(),
        std::back_inserter(sprouted),
        [&sprout](auto& ch) {
          auto& [i, n, t] = ch;
          if (t != IndexTree())
            return std::make_tuple(i, n, t.grow_leaves(sprout));
          else
            return std::make_tuple(i, n, sprout);
        });
    return IndexTree(sprouted);
  }

  IndexTree
  pruned(size_t to_height) const {
    if (height() <= to_height)
      return *this;
    std::vector<std::tuple<COORD_T, COORD_T, IndexTree>> trimmed;
    std::transform(
      m_children.begin(),
      m_children.end(),
      std::back_inserter(trimmed),
      [&to_height](auto& ch) {
        auto& [i, n, t] = ch;
        if (to_height > 0)
          return std::make_tuple(i, n, t.pruned(to_height - 1));
        return std::make_tuple(i, n, IndexTree());
      });
    return IndexTree(trimmed);
  }

  IndexTree
  merged_with(const IndexTree& tree) const {
    if (tree == IndexTree() || *this == tree)
      return *this;
    if (*this == IndexTree())
      return tree;
    std::vector<std::tuple<COORD_T, COORD_T, IndexTree>> newch;
    if (less_than(tree)) {
      newch = m_children;
      std::copy(
        tree.children().begin(),
        tree.children().end(),
        std::back_inserter(newch));
    } else {
      auto ch = m_children.begin();
      auto ch_end = m_children.end();
      auto tch = tree.children().begin();
      auto tch_end = tree.children().end();
      COORD_T i, n, it, nt;
      IndexTree t, tt;
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
    }
    return IndexTree(newch);
  }

  size_t
  serialized_size() const {
    size_t result = sizeof(size_t);
    for (auto& ch : m_children) {
      result += 2 * sizeof(COORD_T);
      const IndexTree& t = std::get<2>(ch);
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

  static IndexTree
  deserialize(const char *buff) {
    size_t sz = *reinterpret_cast<const size_t *>(buff);
    const void *end = buff + sz;
    buff += sizeof(size_t);
    std::vector<std::tuple<COORD_T, COORD_T, IndexTree>> children;
    while (buff != end) {
      COORD_T i = *reinterpret_cast<const COORD_T *>(buff);
      buff += sizeof(COORD_T);
      COORD_T n = *reinterpret_cast<const COORD_T *>(buff);
      buff += sizeof(COORD_T);
      const char *next = buff + *reinterpret_cast<const size_t *>(buff);
      children.emplace_back(i, n, deserialize(buff));
      buff = next;
    }
    return IndexTree(children);
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
    if (*this != IndexTree()) {
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
            if (t != IndexTree()) {
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
  swap(IndexTree& other) {
    std::swap(m_children, other.m_children);
  }

  static std::vector<std::tuple<COORD_T, COORD_T, IndexTree>>
  compact(const std::vector<std::tuple<COORD_T, COORD_T, IndexTree>>& seq) {
    std::vector<std::tuple<COORD_T, COORD_T, IndexTree>> result;
    if (seq.size() > 0) {
      std::vector<std::tuple<COORD_T, COORD_T, IndexTree>> sorted = seq;
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
              IndexTree* lastt = &std::get<2>(lastc);
              if (i < *lasti + *lastn)
                throw std::invalid_argument("Child blocks are overlapping");
              if (i == *lasti + *lastn && *lastt == t)
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
    const std::vector<std::tuple<COORD_T, COORD_T, IndexTree>>& ch,
    std::back_insert_iterator<
    std::vector<std::tuple<COORD_T, COORD_T>>> shape) {

    COORD_T i, n;
    IndexTree t;
    std::tie(i, n, t) = ch[0];
    *shape = std::make_tuple(i, i + n - 1);
    while (t != IndexTree()) {
      std::tie(i, n, t) = t.children()[0];
      ++shape;
      *shape = std::make_tuple(i, i + n - 1);
    }
  }

  static std::vector<std::tuple<COORD_T, COORD_T>>
  array_shape(const std::vector<std::tuple<COORD_T, COORD_T, IndexTree>>& ch) {

    std::vector<std::tuple<COORD_T, COORD_T>> result;
    array_dims(ch, std::back_inserter(result));
    return result;
  }

  static void
  extend_envelope(
    const IndexTree& t,
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
        std::optional<IndexTree<COORD_T>>,
        std::invoke_result_t<F, const std::vector<COORD_T>&>>>>
  IndexTree
  grow_at(const std::vector<COORD_T>& here, F&& fn) const {
    std::vector<std::tuple<COORD_T, COORD_T, IndexTree>> sprouted;
    std::for_each(
      m_children.begin(),
      m_children.end(),
      [&here, &fn, &sprouted](auto& ch) {
        auto& [i, n, t] = ch;
        std::vector<COORD_T> coords = here;
        if (t == IndexTree()) {
          for (COORD_T j = 0; j < n; ++j) {
            coords.push_back(i + j);
            std::optional<IndexTree> ch = fn(coords);
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
    return IndexTree(sprouted);
  }

private:

  std::vector<std::tuple<COORD_T, COORD_T, IndexTree>> m_children;
};

template <typename COORD_T>
class IndexTreeIterator {
public:

  IndexTreeIterator(const IndexTree<COORD_T>& tree, COORD_T stride = 0)
    : m_current(0) {
    m_tree_coords.push_back({});
    gather_coords(tree, m_tree_coords);
    m_tree_coords.pop_back();
    COORD_T i, n;
    std::tie(i, n, std::ignore) = tree.children().back();
    m_stride = std::max(i + n, stride);
  }

  IndexTreeIterator&
  operator++() {
    ++m_current;
    if (m_current == m_tree_coords.size()) {
      m_current = 0;
      std::for_each(
        m_tree_coords.begin(),
        m_tree_coords.end(),
        [this](auto& cs) {
          cs[0] += m_stride;
        });
    }
    return *this;
  }

  IndexTreeIterator
  operator++(int) {
    IndexTreeIterator result = *this;
    ++m_current;
    if (m_current == m_tree_coords.size()) {
      m_current = 0;
      std::for_each(
        m_tree_coords.begin(),
        m_tree_coords.end(),
        [this](auto& cs) {
          cs[0] += m_stride;
        });
    }
    return result;
  }

  const std::vector<COORD_T>&
  operator*() const {
    return m_tree_coords[m_current];
  }

private:

  static void
  gather_coords(
    const IndexTree<COORD_T>& tree,
    std::vector<std::vector<COORD_T>>& coords) {

    std::for_each(
      tree.children().begin(),
      tree.children().end(),
      [&coords](auto& ch) {
        auto& [i, n, t] = ch;
        if (t == IndexTree<COORD_T>()) {
          std::vector<COORD_T> last = coords.back();
          coords.pop_back();
          last.push_back(i);
          COORD_T& b = last.back();
          for (COORD_T j = 0; j < n; ++j) {
            coords.push_back(last);
            ++b;
          }
          last.pop_back();
          coords.push_back(last);
        } else {
          coords.back().push_back(i);
          for (COORD_T j = 0; j < n; ++j) {
            gather_coords(t, coords);
            ++coords.back().back();
          }
          coords.back().pop_back();
        }
      });
  }

  std::vector<std::vector<COORD_T>> m_tree_coords;

  size_t m_current;

  COORD_T m_stride;
};

} // end namespace legms

#endif // LEGMS_INDEX_TREE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:

