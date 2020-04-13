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
#ifndef HYPERION_TABLE_MAPPER_H_
#define HYPERION_TABLE_MAPPER_H_

#include <hyperion/hyperion.h>
#include <hyperion/utility.h>

#pragma GCC visibility push(default)
# include <mappers/default_mapper.h>
# include <legion/legion_mapping.h>
#pragma GCC visibility pop

namespace hyperion {

class HYPERION_API TableMapper
  : public Legion::Mapping::DefaultMapper {
public:

  TableMapper(
    Legion::Machine machine,
    Legion::Runtime* rt,
    Legion::Processor local,
    const char* name);

  // layout tag ids saved to RegionRequirement are always in a bitfield left
  // shifted by "layout_tag_shift" bits in order to maintain the MappingTagID
  // values used by the Legion DefaultMapper as well as allow functionality like
  // the Legion default mapper when the number of RegionRequirements is small
  // enough (<= 1 << layout_tag_shift)
  static const constexpr unsigned layout_tag_shift = 8;
  // number of bits allocated to column group ids
  static const constexpr unsigned layout_tag_bits = 8;

  typedef unsigned LayoutTag;

  static const constexpr LayoutTag table_layout_tag = 0;

  static const constexpr LayoutTag default_column_layout_tag = 1;

  static constexpr Legion::MappingTagID
  to_mapping_tag(LayoutTag ltag, unsigned flags = 0) {
    return
      (static_cast<Legion::MappingTagID>(ltag + 1) << layout_tag_shift) | flags;
  }

  static Legion::TaskVariantRegistrar
  add_table_layout_constraint(Legion::TaskVariantRegistrar& registrar);

  virtual void
  premap_task(
    const Legion::Mapping::MapperContext ctx,
    const Legion::Task& task,
    const Legion::Mapping::Mapper::PremapTaskInput& input,
    Legion::Mapping::Mapper::PremapTaskOutput& output) override;

  virtual void
  map_task(
    const Legion::Mapping::MapperContext ctx,
    const Legion::Task& task,
    const Legion::Mapping::Mapper::MapTaskInput& input,
    Legion::Mapping::Mapper::MapTaskOutput& output) override;

};

} // end namespace hyperion

#endif // HYPERION_TABLE_MAPPER_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
