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

fspace acc {
  sum_re: float,
  sum_im: float,
  num: uint
}

task accumulate(
    data: region(ispace(int3d), complex32),
    vacc: region(ispace(int1d), acc))
where
  reads(data, vacc),
  writes(vacc)
do
  var t = vacc.ispace.bounds.lo
  var v = &vacc[t]
  for d in data do
    v.sum_re = v.sum_re + d.real
    v.sum_im = v.sum_im + d.imag
    v.num = v.num + 1
  end
end

task normalize(
    vacc: region(ispace(int1d), acc))
where
  reads(vacc),
  writes(vacc)
do
  var t = vacc.ispace.bounds.lo
  var v = &vacc[t]
  v.sum_re = v.sum_re / vacc[t].num
  v.sum_im = v.sum_im / vacc[t].num
end

task vavg(
    main_table: legms.table_t,
    data: region(ispace(int3d), complex32),
    time: region(ispace(int1d), double))
where
  reads(data, time)
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
  var data_lp = __import_partition(disjoint, data, ts, lp[data_idx])
  var time_lp = __import_partition(disjoint, time, ts, lp[time_idx])

  var vacc = region(ts, acc)
  var vacc_lp = partition(equal, vacc, ts)

  __demand(__spmd)
  for t in ts do
    accumulate(data_lp[t], vacc_lp[t])
  end

  __demand(__parallel)
  for t in ts do
    normalize(vacc_lp[t])
  end
  -- __demand(__vectorize)
  -- for v in vacc do
  --   v.sum_re = v.sum_re / v.num
  --   v.sum_im = v.sum_im / v.num
  -- end

  __forbid(__parallel)
  for t in ts do
    var v = vacc[t]
    c.printf("%u: %13.3f %g %g\n",
             t, time[time_lp[t].ispace.bounds.lo], v.sum_re, v.sum_im)
  end
end

task attach_and_avg(
    main_table: legms.table_t,
    data: region(ispace(int3d), complex32),
    time: region(ispace(int1d), double))
where
  reads(data, time),
  writes(data, time)
do
  var data_h5_path: rawstring
  legms.table_column_value_path(
    __context(), __runtime(), &main_table, "DATA", &data_h5_path)
  attach(hdf5, data, h5_path, regentlib.file_read_only, array(data_h5_path))
  acquire(data)

  var time_h5_path: rawstring
  legms.table_column_value_path(
    __context(), __runtime(), &main_table, "TIME", &time_h5_path)
  attach(hdf5, time, h5_path, regentlib.file_read_only, array(time_h5_path))
  acquire(time)

  vavg(main_table, data, time)

  release(time)
  detach(hdf5, time)
  release(data)
  detach(hdf5, data)
end

__forbid(__inner)
task main()
  legms.register_tasks(__runtime())

  var main_table =
    legms.table_from_h5(
      __context(), __runtime(), h5_path, main_tbl_path, 2, array("DATA", "TIME"))

  var data_column = legms.table_column(&main_table, "DATA")
  var data_is = __import_ispace(int3d, legms.column_index_space(data_column))
  var data =
    __import_region(data_is, complex32, legms.column_values_region(data_column),
                    array(legms.column_value_fid()))

  var time_column = legms.table_column(&main_table, "TIME")
  var time_is = __import_ispace(int1d, legms.column_index_space(time_column))
  var time =
    __import_region(time_is, double, legms.column_values_region(time_column),
                    array(legms.column_value_fid()))

  attach_and_avg(main_table, data, time)

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
