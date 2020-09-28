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
#ifndef HYPERION_SYNTHESIS_FFT_H_
#define HYPERION_SYNTHESIS_FFT_H_
#include <hyperion/hyperion.h>

#include <cstring>
#include <functional>
#include <map>
#include <memory>

#include <fftw3.h>
#ifdef HYPERION_USE_CUDA
# include <cufft.h>
#endif

namespace hyperion {
namespace synthesis {

/**
 * Mutex for guarding global variables in a Legion program. This uses a
 * spin-lock and periodically yields back to the Legion runtime while waiting to
 * acquire the lock.
 */
struct HYPERION_EXPORT Mutex {

  Mutex()
    : m_count(default_count) {
  }

  Mutex(size_t count)
    : m_count(count) {
  }

  void
  lock(Legion::Context ctx, Legion::Runtime* rt) noexcept {
    if (!acquire())
      rt->yield(ctx);
  }

  void
  unlock() noexcept {
    release();
  }

  static const constexpr size_t default_count = 1000;

private:
  std::atomic_flag m_lock = ATOMIC_FLAG_INIT; // a spin-lock
  size_t m_count;

  bool
  acquire() noexcept {
    // this must be called when m_lock is not held by the caller, but there's no
    // way to check that for a spin-lock
    decltype(m_count) cnt = m_count;
    while (cnt > 0 && m_lock.test_and_set(std::memory_order_acquire))
      --cnt;
    return cnt > 0;
  }

  void
  release() noexcept {
    // this must be called when m_lock is held by the caller, but there's no way
    // to check that for a spin-lock
    m_lock.clear(std::memory_order_release);
  }
};

extern Mutex fftw_mutex;
extern Mutex fftwf_mutex;

class HYPERION_EXPORT FFT {
public:

  /**
   * FFT type
   *
   * Only complex-complex types are currently supported
   */
  enum class Type {
    C2C
    /*,R2R*/
  };

  /**
   * FFT array element precision
   */
  enum class Precision {
    SINGLE,
    DOUBLE
  };

  /**
   * FFT logical descriptor
   */
  struct Desc {
    unsigned rank; /**< rank of array to transform */
    Type transform; /**< FFT transform type */
    Precision precision; /**< FFT array element precision */
    int sign; /**< sign of exponent in transform */
  };

  /**
   * union of FFTW and cuFFT plan representations
   */
  struct Plan {
    Desc desc;
    void* buffer;
    union {
      fftw_plan fftw;
      fftwf_plan fftwf;
#ifdef HYPERION_USE_CUDA
      cufftHandle cufft;
#endif
    } handle;
  };

  /**
   * FFT planning and execution arguments for in_place_fft()
   */
  struct Args {
    Desc desc; /**< FFT descriptor */
    Legion::FieldID fid; /**< field in region */
    /** true iff array half^n-sections should be rotated before FFT */
    bool rotate_in;
    /** true iff array half^n-sections should be rotated after FFT */
    bool rotate_out;
    unsigned flags; /**< FFTW planner flags */
    double seconds; /**< FFTW planner time limit */
  };

  /**
   * task for in-place FFT on field of region
   *
   * Acts on a single field of a region, which must have READ_WRITE
   * privileges. This task can be used to go through the entire process of
   * creating an FFT plan, executing it, and then destroying it
   */
  static const constexpr char* in_place_task_name = "FFT::in_place_task";
  static Legion::TaskID in_place_task_id;

  static void
  fftw_in_place(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* rt);

#ifdef HYPERION_USE_CUDA
  static void
  cufft_in_place(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* rt);
#endif

  /**
   * task for creating an FFT plan
   */
  static const constexpr char* create_plan_task_name =
    "FFT::create_plan_task";
  static Legion::TaskID create_plan_task_id;

  static Plan
  fftw_create_plan(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* rt);

#ifdef HYPERION_USE_CUDA
  static Plan
  cufft_create_plan(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* rt);
#endif

  /**
   * task for executing an FFT plan
   */
  static const constexpr char* execute_fft_task_name =
    "FFT::execute_fft_task";
  static Legion::TaskID execute_fft_task_id;

  static int
  fftw_execute(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* rt);

#ifdef HYPERION_USE_CUDA
  static int
  cufft_execute(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* rt);
#endif

  /**
   * task for destroying an FFT plan
   */
  static const constexpr char* destroy_plan_task_name =
    "FFT::destroy_plan_task";
  static Legion::TaskID destroy_plan_task_id;

  static void
  fftw_destroy_plan(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* rt);

#ifdef HYPERION_USE_CUDA
  static void
  cufft_destroy_plan(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* rt);
#endif

  /**
   * task for rotatiing array values
   */
  static const constexpr char* rotate_arrays_task_name =
    "FFT::rotate_arrays_task";
  static Legion::TaskID rotate_arrays_task_id;

  static void
  rotate_arrays_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* rt);

  static void
  preregister_tasks();
};

} // end namespace synthesis
} // end namespace hyperion

#endif // HYPERION_SYNTHESIS_FFT_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
