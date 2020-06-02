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
#include <hyperion/tree_index_space.h>

using namespace hyperion;
using namespace Legion;

TreeIndexSpaceTask::TreeIndexSpaceTask(const IndexTreeL& it) {
  assert(it.rank());
  assert(it.rank().value() > 0);
  std::for_each(
    it.children().begin(),
    it.children().end(),
    [this](auto& ch) {
#if __cplusplus >= 201703L
      auto& [o, n, t] = ch;
#else
      auto& o = std::get<0>(ch);
      auto& n = std::get<1>(ch);
      auto& t = std::get<2>(ch);
#endif
      m_blocks.push_back(o);
      m_blocks.push_back(n);
      m_trees.push_back(t);
    });
}

Future /* IndexSpace */
TreeIndexSpaceTask::dispatch(Context context, Runtime* runtime) {

  Future result;
  if (m_trees[0] != IndexTreeL()) {
    TaskLauncher
      launcher(
        TASK_ID,
        TaskArgument(m_blocks.data(), m_blocks.size() * sizeof(coord_t)));
    std::for_each(
      m_trees.begin(),
      m_trees.end(),
      [&launcher, &context, runtime](auto& t) {
        if (t != IndexTreeL()) {
          TreeIndexSpaceTask ti(t);
          launcher.add_future(ti.dispatch(context, runtime));
        }
      });
    result = runtime->execute_task(context, launcher);
  } else {
    auto num_blocks = m_blocks.size() / 2;
    std::vector<Rect<1>> rects;
    rects.reserve(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) {
      auto lo = m_blocks[2 * i];
      auto hi = m_blocks[2 * i + 1] + lo - 1;
      rects.emplace_back(lo, hi);
    }
    result =
      Future::from_value(runtime, runtime->create_index_space(context, rects));
  }
  return result;
}

IndexSpace
TreeIndexSpaceTask::base_impl(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context context,
  Runtime *runtime) {

  auto num_blocks = task->futures.size();
  const coord_t* blocks = static_cast<const coord_t *>(task->args);

  std::vector<Domain> domains;
  domains.reserve(num_blocks);

  switch (task->futures[0].get_result<IndexSpace>().get_dim()) {
#define IS(DIM)                                                         \
    case DIM: {                                                         \
      std::vector<DomainT<DIM>> child_doms;                             \
      child_doms.reserve(num_blocks);                                   \
      std::transform(                                                   \
        task->futures.begin(),                                          \
        task->futures.end(),                                            \
        std::back_inserter(child_doms),                                 \
        [&context, runtime](const Future& f) {                          \
          IndexSpace is(f.get_result<IndexSpace>());                    \
          DomainT<DIM>                                                  \
            result(runtime->get_index_space_domain(IndexSpaceT<DIM>(is))); \
          runtime->destroy_index_space(context, is);                    \
          return result;                                                \
        });                                                             \
      for (size_t b = 0; b < num_blocks; ++b) {                         \
        Rect<DIM + 1> r;                                                \
        r.lo[0] = blocks[2 * b];                                        \
        r.hi[0] = r.lo[0] + blocks[2 * b + 1] - 1;                      \
        for (RectInDomainIterator<DIM> rid(child_doms[b]); rid(); rid++) { \
          for (size_t i = 0; i < DIM; ++i) {                            \
            r.lo[i + 1] = rid->lo[i];                                   \
            r.hi[i + 1] = rid->hi[i];                                   \
          }                                                             \
          domains.push_back(r);                                         \
        }                                                               \
      }                                                                 \
      break;                                                            \
    }
    HYPERION_FOREACH_N_LESS_MAX(IS);
  default:
    assert(false);
    break;
  }
  return runtime->create_index_space(context, domains);
}

void
TreeIndexSpaceTask::preregister_task() {
  TASK_ID = Runtime::generate_static_task_id();
  TaskVariantRegistrar registrar(TASK_ID, TASK_NAME, false);
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  registrar.set_idempotent();
  //registrar.set_replicable();
  Runtime::preregister_task_variant<IndexSpace,base_impl>(registrar, TASK_NAME);
}

TaskID TreeIndexSpaceTask::TASK_ID;

IndexSpace
hyperion::tree_index_space(
  const IndexTreeL& tree,
  Context context,
  Runtime* runtime) {

  TreeIndexSpaceTask task(tree);
  return task.dispatch(context, runtime).get_result<IndexSpace>();
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
