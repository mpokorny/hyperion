import "regent"

local c = regentlib.c
local stdlib = terralib.includec("stdlib.h")
local cstring = terralib.includec("string.h")
local legms = terralib.includec("legms_c.h")
terralib.linklibrary("liblegms.so")

-- single precision complex value datatype
struct complexf {
  real: float,
  imag: float
}

function MakeArray(T)
  local struct Array {
    size: uint
    capacity: uint
    data: &T
  }

  terra Array:init(capacity: uint)
    self.size = 0
    self.capacity = capacity
    self.data = [&T](stdlib.calloc(self.capacity, sizeof(T)))
  end

  terra Array:get(i: uint)
    return self.data[i]
  end

  terra Array:set(i: uint, t: T)
    self.data[i] = t
  end

  terra Array:push(t: T)
    if self.size == self.capacity then
      self.capacity = 2 * self.capacity
      self.data = [&T](stdlib.realloc(self.data, self.capacity * sizeof(T)))
    end
    self:set(self.size, t)
    self.size = self.size + 1
  end

  terra Array:__add(other: Array)
    if self.capacity < self.size + other.size then
      self.capacity = self.capacity + other.size
      self.data = [&T](stdlib.realloc(self.data, self.capacity * sizeof(T)))
    end
    cstring.memcpy(self.data + self.size, other.data, other.size * sizeof(T))
    self.size = self.size + other.size
  end

  terra Array:destroy()
    stdlib.free(self.data)
  end

  return Array
end

RowNumberArray = MakeArray(legms.column_row_number_t)

struct RowsAtT {
  time: float,
  rows: RowNumberArray
}

terra RowsAtT:init(time: float)
  self.time = time
  self.rows:init(8)
end

terra RowsAtT:add(other: RowsAtT)
  
end

RowTimeArray = MakeArray(RowsAtT)

struct Times {
  rows_at_t: RowTimeArray
}

terra Times:init()
  self.rows_at_t:init(8)
end

terra Times:find(time: float)
  for i = 0, self.rows_at_t.size do
    if self.rows_at_t:get(i).time == time then
      return i
    end
  end
  return -1
end

terra Times:add_row(time: float, row: legms.column_row_number_t)
  var i = self:find(time)
  if i >= 0 then
    self.rows_at_t:get(i).rows:push(row)
  else
    var r: RowsAtT
    r:init(time)
    r.rows:push(row)
    self.rows_at_t:push(r)
  end
end

terra Times:destroy()
  for i = 0, self.rows_at_t.size do
    self.rows_at_t:get(i).rows:destroy()
  end
end

terra Times:row_partition()
  var result = [&&legms.column_row_number_t](
    stdlib.calloc(self.rows_at_t.size + 1, sizeof([&legms.column_row_number_t])))
  for i = 0, self.rows_at_t.size do
    var rt = self.rows_at_t:get(i)
    var rs = [&legms.column_row_number_t](
      stdlib.calloc(rt.rows.size + 1, sizeof(legms.column_row_number_t)))
    rs[0] = rt.rows.size
    for j = 0, rt.rows.size do
      rs[j + 1] = rt.rows:get(j)
    end
    result[i] = rs
  end
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

fspace time_fs {
  time: double,
  rownr: legms.column_row_number_t
}

fspace data_fs {
  vis: complexf,
  rownr: legms.column_row_number_t
}

local rsz = sizeof(rawstring)

task main()
  var val_fid = legms.column_value_fid()
  var rn_fid = legms.column_row_number_fid()

  -- main table
  var colnames = [&rawstring](stdlib.calloc(3, rsz)) -- FIXME: want sizeof(rawstring)
  colnames[0], colnames[1] = "DATA", "TIME"
  var ms_main =
    legms.table_from_ms(__context(), __runtime(), "FIXME", colnames)
  -- initialize main table blockwise
  legms.table_block_read_task("FIXME", ms_main, colnames, 10000)

  var time_col = legms.table_column(ms_main, "TIME")
  var time_is = __import_ispace(int1d, legms.column_index_space(time_col))
  var time_lr =
    __import_region(time_is, time_fs, legms.column_logical_region(time_col),
                    array(val_fid, rn_fid))
  var times: Times
  times:init()
  for i in time_is do
    times:add_row(time_lr[i].time, time_lr[i].rownr)
  end
  -- leaking times.row_partition() result
  var time_p = legms.table_row_partition(ms_main, times:row_partition(), 0, 1)
  var time_cs = ispace(int1d, times.rows_at_t.size)
  times:destroy()

  var data_col = legms.table_column(ms_main, "DATA")
  var data_is = __import_ispace(int3d, legms.column_index_space(data_col))
  var data_lr =
    __import_region(data_is, data_fs, legms.column_logical_region(data_col),
                    array(val_fid, rn_fid))

  var data_time_p = legms.column_projected_column_partition(data_col, time_p)
  var data_lp = __import_partition(disjoint, data_lr, time_cs, __raw(data_time_p))
  var acc = region(data_time_p[data_time_p.colors[0]].ispace, complexf)
  fill(acc, complexf {0, 0})
  __demand(__parallel)
  for t in data_time_p do
    accumulate(data_lp[t].DATA, acc)
  end
end

regentlib.start(main)
