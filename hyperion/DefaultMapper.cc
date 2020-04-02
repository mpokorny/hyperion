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
#include <hyperion/DefaultMapper.h>

using namespace hyperion;
using namespace Legion;

static constexpr unsigned
cgroup(unsigned tag) {
  return
    ((tag >> DefaultMapper::cgroup_shift)
     & ((1 << DefaultMapper::cgroup_bits) - 1));
}

DefaultMapper::DefaultMapper(
  Machine machine,
  Runtime* rt,
  Processor local)
  : Mapping::DefaultMapper(rt->get_mapper_runtime(), machine, local) {
}

void
DefaultMapper::premap_task(
  const Mapping::MapperContext ctx,
  const Task& task,
  const Mapping::Mapper::PremapTaskInput& input,
  Mapping::Mapper::PremapTaskOutput& output) {

  // Iterate over the premap regions
  bool has_variant_info = false;
  VariantInfo info;
  bool has_restricted_regions = false;
  for (auto it = input.valid_instances.begin();
       it != input.valid_instances.end();
       ++it) {
    // If this region requirements is restricted, then we can just
    // copy over the instances because we know we have to use them
    if (task.regions[it->first].is_restricted()) {
      output.premapped_instances.insert(*it);
      runtime->acquire_instances(ctx, it->second);
      has_restricted_regions = true;
      continue;
    }
    // These are non-restricted regions which means they have to be
    // shared by everyone in this task
    // TODO: some caching here
    if (total_nodes > 1) {
      // multi-node, see how big the index space is, if it is big
      // enough to span more than our node, put it in gasnet memory
      // otherwise we can fall through to the single node case
      Memory target_memory = Memory::NO_MEMORY;
      Machine::MemoryQuery visible_memories(machine);
      visible_memories
        .has_affinity_to(task.target_proc)
        .only_kind(Memory::GLOBAL_MEM);
      Memory global_memory = Memory::NO_MEMORY;
      if (visible_memories.count() > 0)
        global_memory = visible_memories.first();
      switch (task.target_proc.kind()) {
      case Processor::IO_PROC: {
        if (task.index_domain.get_volume() > local_ios.size()) {
          if (!global_memory.exists()) {
            std::cerr
              << "hyperion default mapper failure. "
              << "No memory found for I/O task " << task.get_task_name()
              << " (ID " << task.get_unique_id()
              << ") which is visible for all points in the index space."
              << std::endl;
            assert(false);
          }
          else {
            target_memory = global_memory;
          }
        }
        break;
      }
      case Processor::PY_PROC: {
        if (task.index_domain.get_volume() > local_pys.size()) {
          if (!global_memory.exists()) {
            std::cerr
              << "hyperion default mapper failure. "
              << "No memory found for Python task " << task.get_task_name()
              << " (ID " << task.get_unique_id()
              << ") which is visible for all points in the index space."
              << std::endl;
            assert(false);
          }
          else {
            target_memory = global_memory;
          }
        }
        break;
      }
      case Processor::LOC_PROC: {
        if (task.index_domain.get_volume() > local_cpus.size()) {
          if (!global_memory.exists()) {
            std::cerr
              << "hyperion default mapper failure. "
              << "No memory found for CPU task " << task.get_task_name()
              << " (ID " << task.get_unique_id()
              << ") which is visible for all point in the index space."
              << std::endl;
            assert(false);
          }
          else {
            target_memory = global_memory;
          }
        }
        break;
      }
      case Processor::TOC_PROC: {
        if (task.index_domain.get_volume() > local_gpus.size()) {
          std::cerr
            << "hyperion default mapper failure. "
            << "No memory found for GPU task "<< task.get_task_name()
            << " (ID " << task.get_unique_id()
            << ") which is visible for all points in the index space."
            << std::endl;
          assert(false);
        }
        break;
      }
      case Processor::PROC_SET: {
        if (task.index_domain.get_volume() > local_procsets.size()) {
          if (!global_memory.exists()) {
            std::cerr
              << "hyperion default mapper failure."
              << " No memory found for ProcessorSet task " << task.get_task_name()
              << " (ID " << task.get_unique_id()
              << ") which is visible for all point in the index space."
              << std::endl;
            assert(false);
          }
          else {
            target_memory = global_memory;
          }
        }
        break;
      }
      case Processor::OMP_PROC: {
        if (task.index_domain.get_volume() > local_omps.size()) {
          if (!global_memory.exists()) {
            std::cerr
              << "hyperion default mapper failure."
              << " No memory found for OMP task " << task.get_task_name()
              << " (ID " << task.get_unique_id()
              << ") which is visible for all point in the index space."
              << std::endl;
            assert(false);
          }
          else {
            target_memory = global_memory;
          }
        }
        break;
      }
      default:
        assert(false); // unrecognized processor kind
      }
      if (target_memory.exists()) {
        if (!has_variant_info) {
          info =
            default_find_preferred_variant(
              task,
              ctx,
              true/*needs tight bound*/,
              true/*cache*/,
              task.target_proc.kind());
          has_variant_info = true;
        }
        // Map into the target memory and we are done
        std::set<FieldID> needed_fields =
          task.regions[it->first].privilege_fields;
        const TaskLayoutConstraintSet &layout_constraints =
          runtime->find_task_layout_constraints(
            ctx,
            task.task_id,
            info.variant);
        size_t footprint;
        if (!default_create_custom_instances(
              ctx,
              task.target_proc,
              target_memory,
              task.regions[it->first],
              cgroup(task.regions[it->first].tag),
              needed_fields,
              layout_constraints,
              true/*needs check*/,
              output.premapped_instances[it->first],
              &footprint)) {
          default_report_failed_instance_creation(
            task,
            it->first,
            task.target_proc,
            target_memory,
            footprint);
        }
        continue;
      }
    }
    // should be local to a node
    // see where we are mapping
    Memory target_memory = Memory::NO_MEMORY;
    Machine::MemoryQuery visible_memories(machine);
    visible_memories.has_affinity_to(task.target_proc);
    switch (task.target_proc.kind()) {
    case Processor::LOC_PROC:
    case Processor::IO_PROC:
    case Processor::PROC_SET:
    case Processor::OMP_PROC:
    case Processor::PY_PROC: {
      visible_memories.only_kind(Memory::SYSTEM_MEM);
      if (visible_memories.count() == 0) {
        std::cerr
          << "hyperion default mapper error."
          << " No memory found for CPU task " << task.get_task_name()
          << " (ID " << task.get_unique_id()
          << ") which is visible for all points in the index space."
          << std::endl;
        assert(false);
      }
      target_memory = visible_memories.first();
      break;
    }
    case Processor::TOC_PROC: {
      // Otherwise for GPUs put the instance in zero-copy memory
      visible_memories.only_kind(Memory::Z_COPY_MEM);
      if (visible_memories.count() == 0) {
        std::cerr
          << "hyperion default mapper error. "
          << "No memory found for GPU task " << task.get_task_name()
          << " (ID " << task.get_unique_id()
          << ") which is visible for all points in the index space."
          << std::endl;
        assert(false);
      }
      target_memory = visible_memories.first();
      break;
    }
    default:
      assert(false); // unknown processor kind
    }
    assert(target_memory.exists());
    if (!has_variant_info) {
      info =
        default_find_preferred_variant(
          task,
          ctx,
          true/*needs tight bound*/,
          true/*cache*/,
          task.target_proc.kind());
      has_variant_info = true;
    }
    // Map into the target memory and we are done
    std::set<FieldID> needed_fields =
      task.regions[it->first].privilege_fields;
    const TaskLayoutConstraintSet &layout_constraints =
      runtime->find_task_layout_constraints(
        ctx,
        task.task_id,
        info.variant);
    size_t footprint;
    if (!default_create_custom_instances(
          ctx,
          task.target_proc,
          target_memory,
          task.regions[it->first],
          cgroup(task.regions[it->first].tag),
          needed_fields,
          layout_constraints,
          true/*needs check*/,
          output.premapped_instances[it->first],
          &footprint)) {
      default_report_failed_instance_creation(
        task,
        it->first,
        task.target_proc,
        target_memory,
        footprint);
    }
  }
  // If we have any restricted regions, put the task
  // back on the origin processor
  if (has_restricted_regions)
    output.new_target_proc = task.orig_proc;
}

void
DefaultMapper::map_task(
  const Mapping::MapperContext ctx,
  const Task& task,
  const Mapping::Mapper::MapTaskInput&input,
  Mapping::Mapper::MapTaskOutput& output) {

  Processor::Kind target_kind = task.target_proc.kind();
  // Get the variant that we are going to use to map this task
  VariantInfo chosen =
    default_find_preferred_variant(
      task,
      ctx,
      true/*needs tight bound*/,
      true/*cache*/,
      target_kind);
  output.chosen_variant = chosen.variant;
  output.task_priority = default_policy_select_task_priority(ctx, task);
  output.postmap_task = false;
  // Figure out our target processors
  default_policy_select_target_processors(ctx, task, output.target_procs);

  // See if we have an inner variant, if we do virtually map all the regions
  // We don't even both caching these since they are so simple
  if (chosen.is_inner) {
    // Check to see if we have any relaxed coherence modes in which
    // case we can no longer do virtual mappings so we'll fall through
    bool has_relaxed_coherence = false;
    for (unsigned idx = 0; idx < task.regions.size(); idx++) {
      if (task.regions[idx].prop != EXCLUSIVE) {
        has_relaxed_coherence = true;
        break;
      }
    }
    if (!has_relaxed_coherence) {
      std::vector<unsigned> reduction_indexes;
      for (unsigned idx = 0; idx < task.regions.size(); idx++) {
        // As long as this isn't a reduction-only region requirement
        // we will do a virtual mapping, for reduction-only instances
        // we will actually make a physical instance because the runtime
        // doesn't allow virtual mappings for reduction-only privileges
        if (task.regions[idx].privilege == REDUCE)
          reduction_indexes.push_back(idx);
        else
          output.chosen_instances[idx].push_back(
            Legion::Mapping::PhysicalInstance::get_virtual_instance());
      }
      if (!reduction_indexes.empty()) {
        const TaskLayoutConstraintSet& layout_constraints =
          runtime->find_task_layout_constraints(
            ctx,
            task.task_id,
            output.chosen_variant);
        for (auto it = reduction_indexes.begin();
             it != reduction_indexes.end();
             ++it) {
          Memory target_memory =
            default_policy_select_target_memory(
              ctx,
              task.target_proc,
              task.regions[*it]);
          std::set<FieldID> copy = task.regions[*it].privilege_fields;
          size_t footprint;
          if (!default_create_custom_instances(
                ctx,
                task.target_proc,
                target_memory,
                task.regions[*it],
                cgroup(task.regions[*it].tag),
                copy,
                layout_constraints,
                false/*needs constraint check*/,
                output.chosen_instances[*it],
                &footprint)) {
            default_report_failed_instance_creation(
              task,
              *it,
              task.target_proc,
              target_memory,
              footprint);
          }
        }
      }
      return;
    }
  }
  // Should we cache this task?
  CachedMappingPolicy cache_policy =
    default_policy_select_task_cache_policy(ctx, task);

  // First, let's see if we've cached a result of this task mapping
  const unsigned long long task_hash = compute_task_hash(task);
  std::pair<TaskID,Processor> cache_key(task.task_id, task.target_proc);
  auto finder = cached_task_mappings.find(cache_key);
  // This flag says whether we need to recheck the field constraints,
  // possibly because a new field was allocated in a region, so our old
  // cached physical instance(s) is(are) no longer valid
  bool needs_field_constraint_check = false;
  if (cache_policy == DEFAULT_CACHE_POLICY_ENABLE
      && finder != cached_task_mappings.end()) {
    bool found = false;
    bool has_reductions = false;
    // Iterate through and see if we can find one with our variant and hash
    for (auto it = finder->second.begin(); it != finder->second.end(); ++it) {
      if ((it->variant == output.chosen_variant) &&
          (it->task_hash == task_hash)) {
        // Have to copy it before we do the external call which
        // might invalidate our iterator
        output.chosen_instances = it->mapping;
        has_reductions = it->has_reductions;
        found = true;
        break;
      }
    }
    if (found) {
      // If we have reductions, make those instances now since we
      // never cache the reduction instances
      if (has_reductions) {
        const TaskLayoutConstraintSet &layout_constraints =
          runtime->find_task_layout_constraints(
            ctx,
            task.task_id,
            output.chosen_variant);
        for (unsigned idx = 0; idx < task.regions.size(); idx++) {
          if (task.regions[idx].privilege == REDUCE) {
            Memory target_memory =
              default_policy_select_target_memory(
                ctx,
                task.target_proc,
                task.regions[idx]);
            std::set<FieldID> copy = task.regions[idx].privilege_fields;
            size_t footprint;
            if (!default_create_custom_instances(
                  ctx,
                  task.target_proc,
                  target_memory,
                  task.regions[idx],
                  cgroup(task.regions[idx].tag),
                  copy,
                  layout_constraints,
                  needs_field_constraint_check,
                  output.chosen_instances[idx],
                  &footprint)) {
              default_report_failed_instance_creation(
                task,
                idx,
                task.target_proc,
                target_memory,
                footprint);
            }
          }
        }
      }
      // See if we can acquire these instances still
      if (runtime->acquire_and_filter_instances(
            ctx,
            output.chosen_instances))
        return;
      // We need to check the constraints here because we had a
      // prior mapping and it failed, which may be the result
      // of a change in the allocated fields of a field space
      needs_field_constraint_check = true;
      // If some of them were deleted, go back and remove this entry
      // Have to renew our iterators since they might have been
      // invalidated during the 'acquire_and_filter_instances' call
      default_remove_cached_task(
        ctx,
        output.chosen_variant,
        task_hash,
        cache_key,
        output.chosen_instances);
    }
  }
  // We didn't find a cached version of the mapping so we need to
  // do a full mapping, we already know what variant we want to use
  // so let's use one of the acceleration functions to figure out
  // which instances still need to be mapped.
  std::vector<std::set<FieldID>> missing_fields(task.regions.size());
  runtime->filter_instances(
    ctx,
    task,
    output.chosen_variant,
    output.chosen_instances,
    missing_fields);
  // Track which regions have already been mapped
  std::vector<bool> done_regions(task.regions.size(), false);
  if (!input.premapped_regions.empty())
    for (auto it = input.premapped_regions.begin();
         it != input.premapped_regions.end();
         ++it)
      done_regions[*it] = true;
  const TaskLayoutConstraintSet &layout_constraints =
    runtime->find_task_layout_constraints(
      ctx,
      task.task_id,
      output.chosen_variant);
  // Now we need to go through and make instances for any of our
  // regions which do not have space for certain fields
  bool has_reductions = false;
  for (unsigned idx = 0; idx < task.regions.size(); idx++) {
    if (done_regions[idx])
      continue;
    // Skip any empty regions
    if ((task.regions[idx].privilege == NO_ACCESS) ||
        (task.regions[idx].privilege_fields.empty()) ||
        missing_fields[idx].empty())
      continue;
    // See if this is a reduction
    Memory target_memory =
      default_policy_select_target_memory(
        ctx,
        task.target_proc,
        task.regions[idx]);
    if (task.regions[idx].privilege == REDUCE) {
      has_reductions = true;
      size_t footprint;
      if (!default_create_custom_instances(
            ctx,
            task.target_proc,
            target_memory,
            task.regions[idx],
            cgroup(task.regions[idx].tag),
            missing_fields[idx],
            layout_constraints,
            needs_field_constraint_check,
            output.chosen_instances[idx],
            &footprint)) {
        default_report_failed_instance_creation(
          task,
          idx,
          task.target_proc,
          target_memory,
          footprint);
      }
      continue;
    }
    // Did the application request a virtual mapping for this requirement?
    if ((task.regions[idx].tag & DefaultMapper::VIRTUAL_MAP) != 0) {
      Legion::Mapping::PhysicalInstance virt_inst =
        Legion::Mapping::PhysicalInstance::get_virtual_instance();
      output.chosen_instances[idx].push_back(virt_inst);
      continue;
    }
    // Check to see if any of the valid instances satisfy this requirement
    {
      std::vector<Legion::Mapping::PhysicalInstance> valid_instances;

      for (auto it = input.valid_instances[idx].begin(),
             ie = input.valid_instances[idx].end();
           it != ie;
           ++it) {
        if (it->get_location() == target_memory)
          valid_instances.push_back(*it);
      }

      std::set<FieldID> valid_missing_fields;
      runtime->filter_instances(
        ctx,
        task,
        idx,
        output.chosen_variant,
        valid_instances,
        valid_missing_fields);

#ifndef NDEBUG
      bool check =
#endif
        runtime->acquire_and_filter_instances(ctx, valid_instances);
      assert(check);

      output.chosen_instances[idx] = valid_instances;
      missing_fields[idx] = valid_missing_fields;

      if (missing_fields[idx].empty())
        continue;
    }
    // Otherwise make normal instances for the given region
    size_t footprint;
    if (!default_create_custom_instances(
          ctx,
          task.target_proc,
          target_memory,
          task.regions[idx],
          cgroup(task.regions[idx].tag),
          missing_fields[idx],
          layout_constraints,
          needs_field_constraint_check,
          output.chosen_instances[idx],
          &footprint)) {
      default_report_failed_instance_creation(
        task,
        idx,
        task.target_proc,
        target_memory,
        footprint);
    }
  }
  if (cache_policy == DEFAULT_CACHE_POLICY_ENABLE) {
    // Now that we are done, let's cache the result so we can use it later
    std::list<CachedTaskMapping> &map_list = cached_task_mappings[cache_key];
    map_list.push_back(CachedTaskMapping());
    CachedTaskMapping &cached_result = map_list.back();
    cached_result.task_hash = task_hash;
    cached_result.variant = output.chosen_variant;
    cached_result.mapping = output.chosen_instances;
    cached_result.has_reductions = has_reductions;
    // We don't ever save reduction instances in our cache
    if (has_reductions) {
      for (unsigned idx = 0; idx < task.regions.size(); idx++) {
        if (task.regions[idx].privilege != REDUCE)
          continue;
        cached_result.mapping[idx].clear();
      }
    }
  }
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
