import "regent"

local c = regentlib.c

local utility = terralib.includec("utility_c.h")
local grids = terralib.includec("Grids_c.h")
local tab = terralib.includec("Table_c.h")
local column = terralib.includec("Column_c.h")
local mstable = terralib.includec("MSTable_c.h")
local colpart = terralib.includec("ColumnPartion_c.h")
local msread = terralib.includec("TableReadTask_c.h")
terralib.linklibrary("liblegms")

-- single precision complex value datatype
struct complexf {
  real: float,
  imag: float
                }

-- derived values indexed by row
fspace derived_r {
  dphase: double[3],
                 }

-- derived values indexed by row x channel
fspace derived_rch {
  phasor: complexf[3],
  loc: int[2],
  off: int[2],
  freq: float,
  channel: int
                   }

-- derived values indexed by row x polarization
fspace derived_rpol {
  stokes: int
}

-- gridded visibilities
vis_t = complexf

-- gridding convolution kernel
ck_t = complexf

--local terra conv()

local main_columns = {}
main_columns["DATA"] = 0
main_columns["WEIGHT_SPECTRUM"] = 1
main_columns["UVW"] = 2
main_columns["ANTENNA1"] = 3
main_columns["ANTENNA2"] = 4
main_columns["DATA_DESC_ID"] = 5
local num_main_columns = 6

task rotate_uvw(uvw: region(uvw_t), dphase: region(float))
where reads writes(uvw, dphase) do
    for i in uvw.ispace do
      -- TODO: replace this with a real implementation
      uvw[i] = uvw[i]
      dphase[i] = 1.0
    end
end

task main()
  -- main table
  var colnames: (&int8)[num_main_columns + 1]
  colnames[main_columns.DATA] = "DATA"
  colnames[main_columns.WEIGHT_SPECTRUM] = "WEIGHT_SPECTRUM"
  colnames[main_columns.UVW ]= "UVW"
  colnames[main_columns.ANTENNA1] = "ANTENNA1"
  colnames[main_columns.ANTENNA2] = "ANTENNA2"
  colnames[main_columns.DATA_DESC_ID] = "DATA_DESC_ID"
  colnames[num_main_columns] = 0 -- NULL-value?

  var ms_main =
    tab.table_from_ms(__context(), __runtime(), "FIXME", colnames)
  -- initialize main table blockwise
  msread.table_block_read_task("FIXME", ms_main, colnames, 10000)

  var main_row_rank = tab.table_row_rank(ms_main)
  var all_rows_cp = tab.table_all_rows_partition(ms_main)
  var all_rows_ip = colpart.column_partition_index_partition(all_rows_cp)
  var all_rows_is = all_rows_ip[all_rows_ip.colors[0]]

  --
  -- main table columns
  --
  var main_cols: column.column_t[num_main_columns]
  for i = 0, num_main_columns do
    main_cols[i] = tab.table_column(ms_main, colnames[i])
  end

  --
  -- column index spaces
  --
  var main_iss: c.legion_index_space_t[num_main_columns]
  for i = 0, num_main_columns do
    main_iss[i] = column.column_index_space(main_cols[i])
  end

  --
  -- column logical regions
  --
  var main_lrs: c.legion_logical_region_t[num_main_columns]
  for i = 0, num_main_columns do
    main_lrs[i] = column.column_logical_region(main_cols[i])
  end

  --
  -- output grid
  --
  var u_min
  var u_max
  var v_min
  var v_max
  var nch = 64
  var nsto = 4
  var grid_is =
    ispace(
      int4d,
      {u_max - u_min + 1, v_max - v_min + 1, nch, nsto},
      {u_min, v_min, 0, 0})

  -- negate uv
  __demand(__parallel)
  for i in all_rows_is do
    main_lrs[main_columns.UVW][i][0] *= -1
    main_lrs[main_columns.UVW][i][1] *= -1
  end

  __demand(__parallel)
  for i in main_ips[main_columns.UVW].colors do
    rotate_uvw(main_lps[main_columns.UVW][i])
  end
  --
  -- zero-centered convolution kernel
  --
  var ck_u_r
  var ck_v_r
  var ck_ch_r
  var ck_dim
  var ck_is
  if ck_ch_r > 0 then
    ck_dim = int3d
    ck_is =
      ispace(
        int3d,
        {2 * ck_u_r + 1, 2 * ck_v_r + 1, 2 * ck_ch_r + 1},
        {-ck_u_r, -ck_v_r, -ck_ch_r})
  else
    ck_dim = int2d
    ck_is =
      ispace(
        int2d,
        {2 * ck_u_r + 1, 2 * ck_v_r + 1},
        {-ck_u_r, -ck_v_r})
  end

  -- initialize convolution kernel
  var ck = region(ck_dim, ck_t)
  __demand(__parallel)
  for i in ck.ispace do
    ck[i] = conv(i)
  end

  --
  -- grid partitions: disjoint blocks, and blocks with halo regions
  --
  var block_size: legion_point_4d
  block_size.x[0] = FIXME
  block_size.x[1] = FIXME
  block_size.x[2] = 1
  block_size.x[3] = 1
  var halo_size: legion_point_4d
  halo_size.x[0] = ck_u_r + 1
  halo_size.x[1] = ck_v_r + 1
  if ck_dim == int3d then
    halo_size.x[2] = ck_ch_r + 1
  else
    halo_size.x[2] = 0
  end
  halo.size.x[3] = 0
  var block_ip
  var halo_ip
  grids.block_and_halo_partitions_4d(
    __context(),
    __runtime(),
    grid_is,
    block_size,
    halo_size,
      &block_ip,
      &halo_ip)
end

regentlib.start(main)
