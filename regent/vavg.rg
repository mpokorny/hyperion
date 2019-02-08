import "regent"

local c = regentlib.c
local stdlib = terralib.includec("stdlib.h")
local legms = terralib.includec("legms_c.h")
terralib.linklibrary("liblegms")

-- single precision complex value datatype
struct complexf {
  real: float,
  imag: float
}

function Array(T)
  struct Array {
    size: uint
    capacity: uint
    data: &T
  }

  terra Array:get(i: unsigned)
    return self.data[i]
  end

  terra Array:set(i: unsigned, t: T)
    self.data[i] = t
  end

  terra Array:push(t: T)
    if size == capacity then
      capacity = 2 * capacity
      data = [&T] stdlib.realloc(data, capacity * sizeof(T))
    end
    self.set(size, t)
    size = size + 1
  end

  terra Array:destroy()
    stdlib.free(data)
  end

  local d = stdlib.calloc(8, sizeof(T))
  return Array { 0, 8, d }
end

struct RowsAtT {
  time: float,
  rows: Array(legms.column_row_number_t)
}

struct Times {
  rows_at_t: Array(RowsAtT)
}

terra Times:find(time: float)
  for i = 0, self.rows_at_t.size do
    if self.rows_at_t.get(i).time == time then
      return i
    end
  end
  return -1
end

terra Times:add_row(time: float, row: legms.column_row_number_t)
  var i = find(time)
  if i >= 0 then
    self.rows_at_t.get(i).rows.push(row)
  else
    var a = Array(legms.column_row_number_t)
    a.push(row)
    self.rows_at_t.push(RowsAtT { time, a })
  end
end

terra Times:destroy()
  for i = 0, self.rows_at_t.size do
    self.rows_at_t(i).rows.destroy()
  end
end

terra Times:row_partition()
  var result = [&&legms.column_row_number_t] stdlib.calloc(
    self.rows_at_t.size + 1, sizeof(&legms.column_row_number_t))
  for i = 0, self.rows_at_t.size do
    var rt = self.rows_at_t.get(i)
    var rs = [&legms.column_row_number_t] stdlib.calloc(
      rt.rows.size + 1, sizeof(legms.column_row_number_t))
    rs[0] = rt.rows.size
    for j = 0, rt.rows.size do
      rs[j + 1] = rt.rows.get(j)
    end
    result[i] = rs
  end
  result[self.rows_at_t.size] = nil
  return result
end

task accumulate(is: ispace(int3d),
                vis: region(is, complexf),
                acc: region(is, complexf))
where reads(vis), reduces +(acc) do
  for i in vis.ispace do
    acc[i].real += vis[i].real
    acc[i].imag += vis[i].imag
  end
end

local val_fid = legms.column_value_fid()
local rn_fid = legms.column_row_number_fid()
local colnames = quote
    [&int8] stdlib.malloc(3 * sizeof(&int8))
  end
colnames[0], colnames[1], colnames[2] = "DATA", "TIME", nil

task main()
  -- main table
  var ms_main =
    legms.table_from_ms(__context(), __runtime(), "FIXME", colnames)
  -- initialize main table blockwise
  legms.table_block_read_task("FIXME", ms_main, colnames, 10000)

  var time_col = legms.table_column(ms_main, colnames[1])
  var time_is = legms.column_index_space(time_col)
  var time_lr = legms.column_logical_region(time_col)
  var times = Times -- leaked
  for i in time_is do
    times.add_row(time_lr[i].[val_fid], time_lr[i].[rn_fid])
  end
  -- leaking times.row_partition() result
  var time_p = legms.table_row_partition(ms_main, times.row_partition(), 0, 1)

  var data_col = legms.table_column(ms_main, colnames[0])
  var data_lr = legms.column_logical_region(data_col)
  var data_time_p = legms.column_projected_column_partition(data_col, time_p)
  var data_lp = partition(
    data_lr, legms.column_partition_index_partition(data_time_p))
  var acc = region(data_time_p[data_time_p.colors[0]].ispace, complexf)
  fill(acc, complexf {0, 0})
  __demand(__parallel)
  for t in data_time_p do
    accumulate(data_lp[t].[val_fid], acc)
  end
end

regentlib.start(main)
