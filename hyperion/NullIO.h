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
#ifndef HYPERION_NULL_IO_H_
#define HYPERION_NULL_IO_H_

#include <hyperion/hyperion.h>
#include <casacore/casa/IO/ByteIO.h>

namespace hyperion {

class HYPERION_EXPORT NullIO
  : public casacore::ByteIO {

public:

  NullIO();

  void
  write(casacore::Int64 size, const void *) override;

  void
  pwrite(casacore::Int64 size, casacore::Int64 offset, const void *) override;

  casacore::Int64 read(casacore::Int64 size, void *, casacore::Bool) override;

  casacore::Int64
  pread(
    casacore::Int64 size,
    casacore::Int64 offset,
    void *, casacore::Bool) override;

  casacore::Int64
  length() override;

  casacore::Bool
  isReadable() const override;

  casacore::Bool
  isWritable() const override;

  casacore::Bool
  isSeekable() const override;

protected:

  casacore::Int64
  doSeek(casacore::Int64 offset, casacore::ByteIO::SeekOption option) override;

private:

  casacore::Int64 m_position;
  casacore::Int64 m_min;
  casacore::Int64 m_max;

  void update_range(casacore::Int64 offset, casacore::Int64 size);
};

} // end namespace hyperion

#endif // HYPERION_NULL_IO_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
