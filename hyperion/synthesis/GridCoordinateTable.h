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
#ifndef HYPERION_SYNTHESIS_GRID_COORDINATE_TABLE_H_
#define HYPERION_SYNTHESIS_GRID_COORDINATE_TABLE_H_

#include <hyperion/hyperion.h>
#include <hyperion/synthesis/CFTable.h>

#include <array>
#include <vector>

namespace hyperion {
namespace synthesis {

class HYPERION_EXPORT GridCoordinateTable
  : public CFTable<CF_PARALLACTIC_ANGLE> {
public:

  GridCoordinateTable(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const size_t& grid_size,
    const std::vector<typename cf_table_axis<CF_PARALLACTIC_ANGLE>::type>&
      parallactic_angles);

  static const constexpr unsigned d_pa = 0;

  /**
   * columns of 2-d coordinates
   */
  static const constexpr unsigned coord_rank = d_y + 1;
  static const constexpr Legion::FieldID COORD_X_FID = 88;
  static const constexpr Legion::FieldID COORD_Y_FID = 89;
  static const constexpr char* COORD_X_NAME = "COORD_X";
  static const constexpr char* COORD_Y_NAME = "COORD_Y";
  typedef double coord_t; // type used by casacore::GridCoordinate
  template <Legion::PrivilegeMode MODE, bool CHECK_BOUNDS=HYPERION_CHECK_BOUNDS>
  using coord_accessor_t =
    Legion::FieldAccessor<
      MODE,
      coord_t,
      coord_rank,
      Legion::coord_t,
      Legion::AffineAccessor<coord_t, coord_rank, Legion::coord_t>,
      CHECK_BOUNDS>;
  template <
    template <typename, int, typename> typename A = Legion::GenericAccessor,
    typename COORD_T = Legion::coord_t>
  using CoordColumn =
    PhysicalColumnTD<
      ValueType<coord_t>::DataType,
      index_rank,
      coord_rank,
      A,
      COORD_T>;

  void
  compute_coordinates(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const casacore::Coordinate& cf_coordinates,
    const double& cf_radius,
    const ColumnSpacePartition& partition = ColumnSpacePartition()) const;

  static const constexpr char* compute_coordinates_task_name =
    "GridCoordinateTable::compute_coordinates";

  static Legion::TaskID compute_coordinates_task_id;

  struct ComputeCoordinatesTaskArgs {
    Table::Desc desc;
    bool is_linear_coordinate;
    union {
      std::array<char, linear_coordinate_serdez::MAX_SERIALIZED_SIZE> lc;
      std::array<char, direction_coordinate_serdez::MAX_SERIALIZED_SIZE> dc;
    };
  };

  static void
  compute_coordinates_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* rt);

  static void
  preregister_tasks();

  /**
   * CF domain origin in continuous extension of grid index space, always at the
   * center of the grid
   */
  std::array<double, 2>
    domain_origin() const {
    return {
      static_cast<double>(m_grid_size) / 2,
      static_cast<double>(m_grid_size) / 2};
  }

protected:

  size_t m_grid_size;
};

} // end namespace synthesis
} // end namespace hyperion

#endif // HYPERION_SYNTHESIS_GRID_COORDINATE_TABLE_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
