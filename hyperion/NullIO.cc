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
#include <hyperion/NullIO.h>

namespace cc = casacore;

hyperion::NullIO::NullIO()
  : m_position(0)
  , m_min(0)
  , m_max(0) {
}

void
hyperion::NullIO::write(cc::Int64 size, const void *) {
  update_range(m_position, size);
  m_position += size;
}

void
hyperion::NullIO::pwrite(cc::Int64 size, cc::Int64 offset, const void *) {
  update_range(offset, size);
}

cc::Int64
hyperion::NullIO::read(cc::Int64 size, void *, cc::Bool) {
  update_range(m_position, size);
  m_position += size;
  return 0;
}

cc::Int64
hyperion::NullIO::pread(cc::Int64 size, cc::Int64 offset, void *, cc::Bool) {
  update_range(offset, size);
  return 0;
}

cc::Int64
hyperion::NullIO::length() {
  return m_max - m_min;
}

cc::Bool
hyperion::NullIO::isReadable() const {
  return false;
}

cc::Bool
hyperion::NullIO::isWritable() const {
  return true;
}

cc::Bool
hyperion::NullIO::isSeekable() const {
  return true;
}

cc::Int64
hyperion::NullIO::doSeek(cc::Int64 offset, cc::ByteIO::SeekOption option) {
  switch (option) {
  case ByteIO::SeekOption::Current:
    offset += m_position;
    break;
  case ByteIO::SeekOption::End:
    offset += m_max;
    break;
  case ByteIO::SeekOption::Begin:
    offset += m_min;
    break;
  default:
    break;
  }
  update_range(offset, 0);
  m_position = offset;
  return offset;
}

void
hyperion::NullIO::update_range(cc::Int64 offset, cc::Int64 size) {
  m_min = std::min(offset, m_min);
  m_max = std::max(offset + size, m_max);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
