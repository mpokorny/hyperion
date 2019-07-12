import "regent"

local c = regentlib.c
local stdlib = terralib.includec("stdlib.h")
local cstring = terralib.includec("string.h")
terralib.linklibrary("liblegms.so")
local legms = terralib.includecstring([[
#include "legms_c.h"
#include "utility_c.h"
#include "MSTable_c.h"
#include "Table_c.h"
#include "Column_c.h"
]])

local h5_path = "t0.h5"
local main_tbl_path = "/MAIN"

fspace cplx_acc {
  sum_re: float,
  sum_im: float,
  num: uint
}

task accumulate(
    visibilities: region(ispace(int3d), complex32),
    vis_acc: region(ispace(int1d), cplx_acc))
where
  reads(visibilities, vis_acc),
  writes(vis_acc)
do
  var t = vis_acc.bounds.lo
  var a = &vis_acc[t]
  for v in visibilities do
    a.sum_re = a.sum_re + v.real
    a.sum_im = a.sum_im + v.imag
    a.num = a.num + 1
  end
end

task normalize(
    vis_acc: region(ispace(int1d), cplx_acc))
where
  reads(vis_acc),
  writes(vis_acc)
do
  var t = vis_acc.bounds.lo
  var a = &vis_acc[t]
  a.sum_re = a.sum_re / a.num
  a.sum_im = a.sum_im / a.num
end

task avg_by_time(
    main_table: legms.table_t,
    visibilities: region(ispace(int3d), complex32),
    times: region(ispace(int1d), double))
where
  reads(visibilities, times)
do
  var paxes = array(legms.MAIN_TIME)
  var cn: rawstring[2]
  var lp: c.legion_logical_partition_t[2]
  legms.table_partition_by_value(
    __context(), __runtime(), &main_table, 1, paxes, cn, lp)
  var ts = __import_ispace(
    int1d,
    c.legion_index_partition_get_color_space(
      __runtime(),
      lp[0].index_partition))
  var data_idx: int
  if (c.strcmp(cn[0], "DATA") == 0) then
    data_idx = 0
  else
    data_idx = 1
  end
  var time_idx = 1 - data_idx
  var visibilities_partition =
    __import_partition(disjoint, visibilities, ts, lp[data_idx])
  var times_partition =
    __import_partition(disjoint, times, ts, lp[time_idx])

  var vis_acc = region(ts, cplx_acc)
  fill(vis_acc, cplx_acc{0.0, 0.0, 0})
  var vis_acc_partition = partition(equal, vis_acc, ts)

  __demand(__spmd)
  for t in ts do
    accumulate(visibilities_partition[t], vis_acc_partition[t])
  end

  __demand(__parallel)
  for t in ts do
    normalize(vis_acc_partition[t])
  end
  -- __demand(__vectorize)
  -- for v in vis_acc do
  --   v.sum_re = v.sum_re / v.num
  --   v.sum_im = v.sum_im / v.num
  -- end

  __forbid(__parallel)
  for t in ts do
    var v = vis_acc[t]
    var ti = times_partition[t].ispace.bounds.lo
    c.printf("%u: %13.3f %g %g\n", t, times[ti], v.sum_re, v.sum_im)
  end
end

task attach_and_avg(
    main_table: legms.table_t,
    visibilities: region(ispace(int3d), complex32),
    times: region(ispace(int1d), double))
where
  reads(visibilities, times),
  writes(visibilities, times)
do
  var data_h5_path: rawstring
  legms.table_column_value_path(
    __context(), __runtime(), &main_table, "DATA", &data_h5_path)
  attach(hdf5, visibilities, h5_path, regentlib.file_read_only, array(data_h5_path))
  acquire(visibilities)

  var time_h5_path: rawstring
  legms.table_column_value_path(
    __context(), __runtime(), &main_table, "TIME", &time_h5_path)
  attach(hdf5, times, h5_path, regentlib.file_read_only, array(time_h5_path))
  acquire(times)

  avg_by_time(main_table, visibilities, times)

  release(times)
  detach(hdf5, times)
  release(visibilities)
  detach(hdf5, visibilities)
end

local function import_column(column, isdim, etype)
  return rexpr
      __import_region(
        __import_ispace(isdim, legms.column_index_space(column)),
        etype,
        legms.column_values_region(column),
        array(legms.column_value_fid()))
  end
end

__forbid(__inner)
task main()
  legms.register_tasks(__runtime())

  var main_table =
    legms.table_from_h5(
      __context(), __runtime(),
      h5_path, main_tbl_path, 2, array("DATA", "TIME"))

  var data_column = legms.table_column(&main_table, "DATA")
  var visibilities = [import_column(data_column, int3d, complex32)]

  var time_column = legms.table_column(&main_table, "TIME")
  var times = [import_column(time_column, int1d, double)]

  attach_and_avg(main_table, visibilities, times)

end

if os.getenv("SAVEOBJ") == "1" then
  local lib_dir = os.getenv("LIBDIR") or "../legion_build/lib64"
  local link_flags = terralib.newlist(
    {"-L"..lib_dir,
     "-L../legms",
     "-llegms",
     "-Wl,-rpath="..lib_dir,
     "-Wl,-rpath=../legms"})
  regentlib.saveobj(main, "vavg", "executable", legms.preregister_all, link_flags)
else
  legms.preregister_all()
  regentlib.start(main)
end
