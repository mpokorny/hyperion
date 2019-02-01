import "regent"

local c = regentlib.c

local mstable = terralib.includec("Table_c.h")

local grids = terralib.includec("Grids_c.h")

-- single precision complex value datatype
local struct complexf() {
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

local terra conv()

local main_columns = {}
main_columns["DATA"] = 0
main_columns["WEIGHT_SPECTRUM"] = 1
main_columns["UVW"] = 2
main_columns["ANTENNA1"] = 3
main_columns["ANTENNA2"] = 4
main_columns["DATA_DESC_ID"] = 5
local num_main_columns = 6

-- UVW column data type
uvw_t = float[3]

task negate_uv(uvw: region(uvw_t))
where reads writes(uvw) do
    for p in uvw do
      p[0] = -p[0]
      p[1] = -p[1]
    end
end

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
  var ms_main
  -- read MS blockwise
  var main_is = table_index_space(ms_main)
  var main_ip

  var ms_spw

  -- initialize mapping from column index to name
  var main_column_names: (&int8)[num_main_columns]
  main_column_names[main_columns.DATA] = "DATA"
  main_column_names[main_columns.WEIGHT_SPECTRUM] = "WEIGHT_SPECTRUM"
  main_column_names[main_columns.UVW ]= "UVW"
  main_column_names[main_columns.ANTENNA1] = "ANTENNA1"
  main_column_names[main_columns.ANTENNA2] = "ANTENNA2"
  main_column_names[main_columns.DATA_DESC_ID] = "DATA_DESC_ID"

  --
  -- logical regions (columns)
  --
  var main_lrs = table_logical_regions(ms_main, main_column_names)

  --
  -- index partitions
  --
  var main_ips = table_index_partitions(ms_main, main_ip, main_column_names)


  --
  -- logical partitions
  --
  var main_lps: legion_logical_partition_t[num_main_columns]
  for i = 0, num_main_columns do
    main_lps[i]  =
      c.legion_logical_partition_create(
        __runtime(),
        __context(),
        main_lrs[i],
        main_ips[i])
  end

  --
  -- output grid
  --
  var u_min
  var u_max
  var v_min
  var v_max
  var nch
  var nsto
  var grid_is =
    ispace(
      int4d,
      {u_max - u_min + 1, v_max - v_min + 1, nch, nsto},
      {u_min, v_min, 0, 0})

  -- negate uv
  __demand(__parallel)
  for i in main_ips[main_columns.UVW].colors do
    negate_uv(main_lps[main_columns.UVW][i])
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
