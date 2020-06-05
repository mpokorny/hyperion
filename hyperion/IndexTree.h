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
#ifndef HYPERION_INDEX_TREE_H_
#define HYPERION_INDEX_TREE_H_

#include <hyperion/hyperion.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include CXX_OPTIONAL_HEADER
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace hyperion {

template <typename COORD_T=int>
class IndexTree {

public:

  IndexTree() {}

  IndexTree(COORD_T n)
    : m_children(compact({{0, n, IndexTree()}})) {
  }

  IndexTree(const std::vector<std::tuple<COORD_T, IndexTree> >& ch) {
    COORD_T end = 0;
    std::vector<std::tuple<COORD_T, COORD_T, IndexTree>> ch1;
    std::transform(
      ch.begin(),
      ch.end(),
      std::back_inserter(ch1),
      [&end](auto& c) {
#if HAVE_CXX17
        auto& [n, t] = c;
#else // !HAVE_CXX17
        auto& n = std::get<0>(c);
        auto& t = std::get<1>(c);
#endif // HAVE_CXX17
        COORD_T last_end = end;
        end += n;
        return std::make_tuple(last_end, n, t);
      });
    m_children = std::move(compact(ch1));
  }

  IndexTree(const std::vector<std::tuple<COORD_T, COORD_T, IndexTree> >& ch)
    : m_children(compact(ch)) {
  }

  IndexTree(const IndexTree& pattern, size_t num) {
    std::vector<std::tuple<COORD_T, COORD_T, IndexTree>> ch;
    auto pattern_n = pattern.size();
    auto pattern_rep = num / pattern_n;
    auto pattern_rem = num % pattern_n;
    assert(std::get<0>(pattern.index_range()) == 0);
    auto stride = std::get<1>(pattern.index_range()) + 1;
    COORD_T offset = 0;
    if (pattern.children().size() == 1) {
      COORD_T i;
      IndexTree t;
      std::tie(i, std::ignore, t) = pattern.children()[0];
      offset += pattern_rep * stride;
      ch.emplace_back(i, offset, t);
    } else {
      for (size_t r = 0; r < pattern_rep; ++r) {
        std::transform(
          pattern.children().begin(),
          pattern.children().end(),
          std::back_inserter(ch),
          [&offset](auto& c) {
#if __cplusplus >= 201703L
            auto& [i, n, t] = c;
#else
            auto& i = std::get<0>(c);
            auto& n = std::get<1>(c);
            auto& t = std::get<2>(c);
#endif
            return std::make_tuple(i + offset, n, t);
          });
        offset += stride;
      }
    }
    auto rch = pattern.children().begin();
    auto rch_end = pattern.children().end();
    while (pattern_rem > 0 && rch != rch_end) {
#if __cplusplus >= 201703L
      auto& [i, n, t] = *rch;
#else
      auto& i = std::get<0>(*rch);
      auto& n = std::get<1>(*rch);
      auto& t = std::get<2>(*rch);
#endif
      auto tsz = t.size();
      if (pattern_rem >= tsz) {
        auto nt = std::min(pattern_rem / tsz, static_cast<size_t>(n));
        ch.emplace_back(i + offset, nt, t);
        pattern_rem -= nt * tsz;
        if (nt == static_cast<size_t>(n))
          ++rch;
      } else /* pattern_rem < tsz */ {
        auto pt = IndexTree(t, pattern_rem);
        pattern_rem = 0;
        ch.emplace_back(i + offset, 1, pt);
      }
    }
    m_children = std::move(compact(ch));
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
      auto jch = other.children();
#if __cplusplus >= 201703L
      auto& [ie, ne, te] = m_children.back();
      auto& [j0, jn0, jt0]= jch.front();
#else
      COORD_T ie, ne, j0, jn0;
      IndexTree te, jt0;
      std::tie(ie, ne, te) = m_children.back();
      std::tie(j0, jn0, jt0) = jch.front();
#endif
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
#if __cplusplus >= 201703L
      auto& [lo, n, t] = ch;
#else
      auto& lo = std::get<0>(ch);
      auto& n = std::get<1>(ch);
      auto& t = std::get<2>(ch);
#endif
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
        (size_t)0,
        [](size_t acc, const auto& ch) {
          COORD_T n;
          IndexTree t;
          std::tie(std::ignore, n, t) = ch;
          if (n == 0)
            return acc;
          return acc + (size_t)n * t.size();
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

  CXX_OPTIONAL_NAMESPACE::optional<unsigned>
  rank() const {
    if (m_children.size() == 0)
      return 0;
    if (is_leaf())
      return 1;
    auto ch = m_children.begin();
    auto end = m_children.end();
    IndexTree t;
    std::tie(std::ignore, std::ignore, t) = *ch;
    CXX_OPTIONAL_NAMESPACE::optional<unsigned> result = t.rank();
    while (result && ++ch != end) {
      std::tie(std::ignore, std::ignore, t) = *ch;
      if (result != t.rank())
        result = CXX_OPTIONAL_NAMESPACE::nullopt;
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
#if __cplusplus >= 201703L
          auto& [lo, n, t] = ch;
          auto& [rlo, rhi] = result[0];
#else
          COORD_T lo, n, rlo, rhi;
          IndexTree t;
          std::tie(lo, n, t) = ch;
          std::tie(rlo, rhi) = result[0];
#endif
          COORD_T hi = lo + n - 1;
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
        CXX_OPTIONAL_NAMESPACE::optional<IndexTree<COORD_T>>,
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
#if __cplusplus >= 201703L
        auto& [i, n, t] = ch;
#else
        auto& i = std::get<0>(ch);
        auto& n = std::get<1>(ch);
        auto& t = std::get<2>(ch);
#endif
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
#if __cplusplus >= 201703L
        auto& [i, n, t] = ch;
#else
        auto& i = std::get<0>(ch);
        auto& n = std::get<1>(ch);
        auto& t = std::get<2>(ch);
#endif
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
    if (less_than(tree)
        && (std::get<0>(m_children.back()) + std::get<1>(m_children.back())
            < std::get<0>(tree.children().front()))) {
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

  CXX_OPTIONAL_NAMESPACE::optional<size_t>
  num_repeats(const IndexTree& pattern) const {
    return nr(pattern, true);
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
#if __cplusplus >= 201703L
      auto& [i, n, t] = ch;
#else
      auto& i = std::get<0>(ch);
      auto& n = std::get<1>(ch);
      auto& t = std::get<2>(ch);
#endif
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
#if __cplusplus >= 201703L
            auto& [i, n, t] = ch;
#else
            auto& i = std::get<0>(ch);
            auto& n = std::get<1>(ch);
            auto& t = std::get<2>(ch);
#endif
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

  static std::vector<std::tuple<COORD_T, COORD_T, IndexTree>>
  compact(const std::vector<std::tuple<COORD_T, COORD_T, IndexTree>>& seq) {
    std::vector<std::tuple<COORD_T, COORD_T, IndexTree>> result;
    if (seq.size() > 0) {
      std::vector<std::tuple<COORD_T, COORD_T, IndexTree>> sorted;
      std::copy_if(
        seq.begin(),
        seq.end(),
        std::back_inserter(sorted),
        [](auto& ch) {
          return std::get<1>(ch) > 0;
        });
      std::sort(
        sorted.begin(),
        sorted.end(),
        [](auto& b0, auto& b1) {
          return std::get<0>(b0) < std::get<0>(b1);
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
#if __cplusplus >= 201703L
              auto& [i, n, t] = ch;
#else
              auto& i = std::get<0>(ch);
              auto& n = std::get<1>(ch);
              auto& t = std::get<2>(ch);
#endif
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
#if __cplusplus >= 201703L
      auto& [rlo, rhi] = *r;
      auto& [telo, tehi] = *te;
#else
      COORD_T rlo, rhi, telo, tehi;
      std::tie(rlo, rhi) = *r;
      std::tie(telo, tehi) = *te;
#endif
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
        CXX_OPTIONAL_NAMESPACE::optional<IndexTree<COORD_T>>,
        std::invoke_result_t<F, const std::vector<COORD_T>&>>>>
  IndexTree
  grow_at(const std::vector<COORD_T>& here, F&& fn) const {
    std::vector<std::tuple<COORD_T, COORD_T, IndexTree>> sprouted;
    std::for_each(
      m_children.begin(),
      m_children.end(),
      [&here, &fn, &sprouted](auto& ch) {
#if __cplusplus >= 201703L
        auto& [i, n, t] = ch;
#else
        auto& i = std::get<0>(ch);
        auto& n = std::get<1>(ch);
        auto& t = std::get<2>(ch);
#endif
        std::vector<COORD_T> coords = here;
        if (t == IndexTree()) {
          for (COORD_T j = 0; j < n; ++j) {
            coords.push_back(i + j);
            CXX_OPTIONAL_NAMESPACE::optional<IndexTree> ch = fn(coords);
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

  CXX_OPTIONAL_NAMESPACE::optional<size_t>
  nr(const IndexTree& pattern, bool cycle) const {

    if (pattern.rank().value() > rank().value())
      return CXX_OPTIONAL_NAMESPACE::nullopt;
    auto pruned_shape = pruned(pattern.rank().value() - 1);
    auto p_iter = pruned_shape.children().begin();
    auto p_end = pruned_shape.children().end();
    COORD_T i0 = std::get<0>(pattern.index_range());
    size_t result = 0;
    COORD_T pi, pn;
    IndexTree pt;
    std::tie(pi, pn, pt) = *p_iter;
    while (p_iter != p_end) {
      auto r_iter = pattern.children().begin();
      auto r_end = pattern.children().end();
      COORD_T i, n;
      IndexTree t;
      while (p_iter != p_end && r_iter != r_end) {
        std::tie(i, n, t) = *r_iter;
        if (i + i0 != pi) {
          return CXX_OPTIONAL_NAMESPACE::nullopt;
        } else if (t == pt) {
          auto m = std::min(n, pn);
          result += m * t.size();
          pi += m;
          pn -= m;
          if (pn == 0) {
            ++p_iter;
            if (p_iter != p_end)
              std::tie(pi, pn, pt) = *p_iter;
          }
        } else {
          ++p_iter;
          if (p_iter != p_end)
            return CXX_OPTIONAL_NAMESPACE::nullopt;
          auto chnr = pt.nr(t, false);
          if (chnr)
            result += chnr.value();
          else
            return CXX_OPTIONAL_NAMESPACE::nullopt;
        }
        ++r_iter;
      }
      i0 = pi;
      if (!cycle && p_iter != p_end && r_iter == r_end)
        return CXX_OPTIONAL_NAMESPACE::nullopt;
    }
    return result;
  }
};

template <typename COORD_T>
class IndexTreeIterator {
public:

  IndexTreeIterator(const IndexTree<COORD_T>& tree)
    : m_current(0) {
    m_tree_coords.push_back({});
    gather_coords(tree, m_tree_coords);
    m_tree_coords.pop_back();
  }

  IndexTreeIterator&
  operator++() {
    ++m_current;
    return *this;
  }

  IndexTreeIterator
  operator++(int) {
    IndexTreeIterator result = *this;
    ++m_current;
    return result;
  }

  operator bool() const {
    return m_current != m_tree_coords.size();
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
#if __cplusplus >= 201703L
        auto& [i, n, t] = ch;
#else
        auto& i = std::get<0>(ch);
        auto& n = std::get<1>(ch);
        auto& t = std::get<2>(ch);
#endif
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

} // end namespace hyperion

template <typename COORD_T>
std::ostream&
operator<<(std::ostream& stream, const hyperion::IndexTree<COORD_T>& it) {
  stream << it.show();
  return stream;
}

#endif // HYPERION_INDEX_TREE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
