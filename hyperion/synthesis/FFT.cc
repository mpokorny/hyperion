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
#include <hyperion/synthesis/FFT.h>
#include <hyperion/utility.h>
#include <mappers/default_mapper.h>
#include <limits>

using namespace hyperion;
using namespace hyperion::synthesis;
using namespace Legion;

Mutex hyperion::synthesis::fftw_mutex;
Mutex hyperion::synthesis::fftwf_mutex;

#if !HAVE_CXX17
const size_t Mutex::default_count;
const constexpr char* FFT::in_place_task_name;
const constexpr char* FFT::create_plan_task_name;
const constexpr char* FFT::execute_fft_task_name;
const constexpr char* FFT::destroy_plan_task_name;
#endif

TaskID FFT::in_place_task_id;
TaskID FFT::create_plan_task_id;
TaskID FFT::execute_fft_task_id;
TaskID FFT::destroy_plan_task_id;

struct Params {
  std::vector<int> n;
  std::vector<int> nembed;
  int dist;
  int stride;
  int howmany;
  void* buffer;
};

/**
 * check for matching values in first n dimensions of to points
 */
template <unsigned N>
static bool
prefixes_match(const Point<N>& p0, const Point<N>& p1, unsigned n) {
  for (size_t i = 0; i < std::min(n, N); ++i)
    if (p0[i] != p1[i])
      return false;
  return true;
}

/**
 * get parameters for FFTW or cuFFT for an array of type T, rank array_rank in
 * a field fid of a region with rank N
 */
template <typename F, unsigned N>
static Params
get_paramsN(
  Runtime* rt,
  const RegionRequirement& req,
  const PhysicalRegion& region,
  const FieldID& fid,
  unsigned array_rank) {

  assert(array_rank > 0);

  Params result;
  Rect<N> rect(rt->get_index_space_domain(req.region.get_index_space()));
  Rect<N> region_rect(region);
  result.howmany = 1;
  for (size_t i = 0; i < N; ++i) {
    auto len = rect.hi[i] - rect.lo[i] + 1;
    if (i < N - array_rank) {
      result.howmany *= len;
    } else {
      result.n.push_back(len);
      result.nembed.push_back(region_rect.hi[i] - region_rect.lo[i] + 1);
    }
  }
  const FieldAccessor<
    READ_ONLY,
    F,
    N,
    coord_t,
    AffineAccessor<F, N, coord_t>,
    HYPERION_CHECK_BOUNDS> acc(region, fid);
  PointInRectIterator<N> pir(rect, false);
  Point<N> pt0 = *pir;
  const F* f0 = acc.ptr(*pir);
  result.buffer = const_cast<F*>(f0);
  pir++;
  if (pir() && prefixes_match<N>(pt0, *pir, N - array_rank)) {
    result.stride = acc.ptr(*pir) - f0;
    pir++;
    while (pir() && prefixes_match<N>(pt0, *pir, N - array_rank))
      pir++;
    if (pir())
      result.dist = acc.ptr(*pir) - f0;
    else
      result.dist = 0;
  } else {
    result.stride = 0;
    result.dist = 0;
  }
  return result;
}

/**
 * get parameters for FFTW or cuFFT for an array of type T, rank array_rank in
 * a field fid of a region with variable rank
 */
template <typename F>
static Params
get_params(
  Runtime* rt,
  const RegionRequirement& req,
  const PhysicalRegion& region,
  const FieldID& fid,
  unsigned array_rank) {

  switch (req.region.get_index_space().get_dim()) {
#define GET_PARAMSN(N)                                    \
  case N: {                                                    \
    return get_paramsN<F, N>(rt, req, region, fid, array_rank); \
    break;                                                      \
  }
  HYPERION_FOREACH_N(GET_PARAMSN);
  default:
    assert(false);
    return Params();
    break;
  }
}

/**
 * coordinate the computation of an FFT through sub-tasks that create a plan,
 * execute the plan, and finally destroy the plan
 */
static void
in_place(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt,
  bool enable_inlined_planner_subtasks) {

  auto req = task->regions[0];
  req.tag = Mapping::DefaultMapper::MappingTags::SAME_ADDRESS_SPACE;

  // create the FFT plan
  const FFT::Args& args = *static_cast<const FFT::Args*>(task->args);
  TaskLauncher
    plan_creator(FFT::create_plan_task_id, TaskArgument(&args, sizeof(args)));
  plan_creator.add_region_requirement(req);
  plan_creator.enable_inlining = enable_inlined_planner_subtasks;
  auto plan = rt->execute_task(ctx, plan_creator);

  // execute the FFT plan, dependency on plan future for sequencing
  TaskLauncher executor(
    FFT::execute_fft_task_id,
    TaskArgument(
      &enable_inlined_planner_subtasks,
      sizeof(enable_inlined_planner_subtasks)));
  executor.add_region_requirement(req);
  executor.add_future(plan);
  auto rc = rt->execute_task(ctx, executor);

  // destroy the FFT plan, dependency on rc (and plan) future for sequencing
  if (!enable_inlined_planner_subtasks) {
    TaskLauncher plan_destroyer(FFT::destroy_plan_task_id, TaskArgument());
    plan_destroyer.add_future(plan);
    plan_destroyer.add_future(rc);
    rt->execute_task(ctx, plan_destroyer);
  }
}

void
FFT::fftw_in_place(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  in_place(task, regions, ctx, rt, false);
}

#ifdef HYPERION_USE_CUDA
void
FFT::cufft_in_place(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  in_place(task, regions, ctx, rt, true);
}
#endif

FFT::Plan
FFT::fftw_create_plan(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  const Args& args = *static_cast<const Args*>(task->args);

  assert(args.desc.transform == Type::C2C);
  Plan result;
  result.desc = args.desc;
  if (args.desc.precision == Precision::SINGLE) {
    auto params =
      get_params<complex<float>>(
        rt,
        task->regions[0],
        regions[0],
        args.fid,
        args.desc.rank);
    fftwf_mutex.lock(ctx, rt);
    if (args.seconds >= 0)
      fftwf_set_timelimit(args.seconds);
    // when creating a plan, if FFTW does not yet have wisdom for that plan, it
    // will overwrite the array; thus when necessary, create a similar plan
    // initially with a different buffer
    auto make_plan =
      [&params, &args](fftwf_complex* buffer, unsigned flags) {
        return
          fftwf_plan_many_dft(
            params.n.size(), params.n.data(), params.howmany,
            buffer, params.nembed.data(), params.stride, params.dist,
            buffer, params.nembed.data(), params.stride, params.dist,
            args.desc.sign,
            flags);
      };
    result.handle.fftwf =
      make_plan(
        static_cast<fftwf_complex*>(params.buffer),
        args.flags | FFTW_WISDOM_ONLY);
    if (result.handle.fftwf == NULL && (args.flags & FFTW_WISDOM_ONLY) == 0) {
      // no existing wisdom for plan, and caller accepts generation of new plan
      std::cout << "new FFTWF plan" << std::endl;
      auto buff = fftwf_alloc_complex(params.howmany * params.dist);
      auto p = make_plan(buff, args.flags);
      ::fftwf_destroy_plan(p);
      fftwf_free(buff);
      result.handle.fftwf =
        make_plan(
          static_cast<fftwf_complex*>(params.buffer),
          args.flags | FFTW_WISDOM_ONLY);
      assert(result.handle.fftwf != NULL);
    }
    fftwf_mutex.unlock();
  } else {
    auto params =
      get_params<complex<double>>(
        rt,
        task->regions[0],
        regions[0],
        args.fid,
        args.desc.rank);
    fftw_mutex.lock(ctx, rt);
    if (args.seconds >= 0)
      fftw_set_timelimit(args.seconds);
    // when creating a plan, if FFTW does not yet have wisdom for that plan, it
    // will overwrite the array; thus when necessary, create a similar plan
    // initially with a different buffer
    auto make_plan =
      [&params, &args](fftw_complex* buffer, unsigned flags) {
        return
          fftw_plan_many_dft(
            params.n.size(), params.n.data(), params.howmany,
            buffer, params.nembed.data(), params.stride, params.dist,
            buffer, params.nembed.data(), params.stride, params.dist,
            args.desc.sign,
            flags);
      };
    result.handle.fftw =
      make_plan(
        static_cast<fftw_complex*>(result.buffer),
        args.flags | FFTW_WISDOM_ONLY);
    if (result.handle.fftw == NULL && (args.flags & FFTW_WISDOM_ONLY) == 0) {
      std::cout << "new FFTW plan" << std::endl;
      auto buff = fftw_alloc_complex(params.howmany * params.dist);
      auto p = make_plan(buff, args.flags);
      ::fftw_destroy_plan(p);
      fftw_free(buff);
      result.handle.fftw =
        make_plan(
          static_cast<fftw_complex*>(result.buffer),
          args.flags | FFTW_WISDOM_ONLY);
      assert(result.handle.fftw != NULL);
    }
    fftw_mutex.unlock();
  }
#undef DESC
  return result;
}

#ifdef HYPERION_USE_CUDA
FFT::Plan
FFT::cufft_create_plan(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  const Args& args = *static_cast<const Args*>(task->args);

  assert(args.desc.transform == Type::C2C);
  Plan result;
  result.desc = args.desc;
  if (args.desc.precision == Precision::SINGLE) {
    auto params =
      get_params<complex<float>>(
        rt,
        task->regions[0],
        regions[0],
        args.fid,
        args.desc.rank);
    auto rc =
      cufftPlanMany(
        &result.handle.cufft, params.n.size(), params.n.data(),
        params.nembed.data(), params.stride, params.dist,
        params.nembed.data(), params.stride, params.dist,
        CUFFT_C2C,
        params.howmany);
    if (rc != CUFFT_SUCCESS)
      result.handle.cufft = 0;
  } else {
    auto params =
      get_params<complex<double>>(
        rt,
        task->regions[0],
        regions[0],
        args.fid,
        args.desc.rank);
    auto rc =
      cufftPlanMany(
        &result.handle.cufft, params.n.size(), params.n.data(),
        params.nembed.data(), params.stride, params.dist,
        params.nembed.data(), params.stride, params.dist,
        CUFFT_Z2Z,
        params.howmany);
    if (rc != CUFFT_SUCCESS)
      result.handle.cufft = 0;
  }
#undef DESC
  return result;
}
#endif

int
FFT::fftw_execute(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  auto plan = task->futures[0].get_result<Plan>();
  int result;
  if (plan.desc.precision == Precision::SINGLE) {
    if (plan.handle.fftwf != NULL) {
      ::fftwf_execute(plan.handle.fftwf);
      result = 0;
    } else {
      auto lr = task->regions[0].region;
      complex<float> nan(
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN());
      rt->fill_field(ctx, lr, lr, task->regions[0].instance_fields[0], nan);
      result = 1;
    }
  } else {
    if (plan.handle.fftw != NULL) {
      ::fftw_execute(plan.handle.fftw);
      result = 0;
    } else {
      auto lr = task->regions[0].region;
      complex<double> nan(
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN());
      rt->fill_field(ctx, lr, lr, task->regions[0].instance_fields[0], nan);
      result = 1;
    }
  }
  if (*static_cast<bool*>(task->args))
    fftw_destroy_plan(task, regions, ctx, rt);
  return result;
}

#ifdef HYPERION_USE_CUDA
int
FFT::cufft_execute(
  const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx,
  Runtime* rt) {

  auto plan = task->futures[0].get_result<Plan>();
  int result;
  if (plan.handle.cufft != 0) {
    if (plan.desc.precision == Precision::SINGLE)
      result =
        cufftExecC2C(
          plan.handle.cufft,
          static_cast<cufftComplex*>(plan.buffer),
          static_cast<cufftComplex*>(plan.buffer),
          plan.desc.sign);
    else
      result =
        cufftExecZ2Z(
          plan.handle.cufft,
          static_cast<cufftDoubleComplex*>(plan.buffer),
          static_cast<cufftDoubleComplex*>(plan.buffer),
          plan.desc.sign);
  } else {
    result = -1;
  }
  if (*static_cast<bool*>(task->args))
    cufft_destroy_plan(task, regions, ctx, rt);
  return result;
}
#endif

void
FFT::fftw_destroy_plan(
  const Task* task,
  const std::vector<PhysicalRegion>&,
  Context ctx,
  Runtime* rt) {

  auto plan = task->futures[0].get_result<Plan>();
  if (plan.desc.precision == Precision::SINGLE) {
    if (plan.handle.fftwf != NULL) {
      fftwf_mutex.lock(ctx, rt);
      ::fftwf_destroy_plan(plan.handle.fftwf);
      fftwf_mutex.unlock();
    }
  } else {
    if (plan.handle.fftw != NULL) {
      fftw_mutex.lock(ctx, rt);
      ::fftw_destroy_plan(plan.handle.fftw);
      fftw_mutex.unlock();
    }
  }
}

#ifdef HYPERION_USE_CUDA
void
FFT::cufft_destroy_plan(
  const Task* task,
  const std::vector<PhysicalRegion>&,
  Context,
  Runtime*) {

  auto plan = task->futures[0].get_result<Plan>();
  if (plan.handle.cufft != 0)
    cufftDestroy(plan.handle.cufft);
}
#endif

void
FFT::preregister_tasks() {

  LayoutConstraintRegistrar
    fftw_constraints(FieldSpace::NO_SPACE, "FFT::fftw_constraints");
  add_soa_right_ordering_constraint(fftw_constraints);
  fftw_constraints.add_constraint(
    SpecializedConstraint(LEGION_AFFINE_SPECIALIZE));
  auto fftw_layout_id = Runtime::preregister_layout(fftw_constraints);

#ifdef HYPERION_USE_CUDA
  LayoutConstraintRegistrar
    cufft_constraints(FieldSpace::NO_SPACE, "FFT::cufft_constraints");
  add_soa_right_ordering_constraint(cufft_constraints);
  cufft_constraints.add_constraint(
    SpecializedConstraint(LEGION_AFFINE_SPECIALIZE));
  auto cufft_layout_id = Runtime::preregister_layout(cufft_constraints);
#endif

  //
  // in_place_task
  //
  {
    in_place_task_id = Runtime::generate_static_task_id();
    // fftw variant
    //
    // FIXME: remove assumption that FFTW is using OpenMP
    {
      TaskVariantRegistrar
        registrar(in_place_task_id, in_place_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_inner();
      registrar.add_layout_constraint_set(0, fftw_layout_id);
      Runtime::preregister_task_variant<fftw_in_place>(
        registrar,
        in_place_task_name);
    }
#ifdef HYPERION_USE_CUDA
    // cufft variant
    {
      TaskVariantRegistrar
        registrar(in_place_task_id, in_place_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_inner();
      registrar.add_layout_constraint_set(0, cufft_layout_id);
      Runtime::preregister_task_variant<cufft_in_place>(
        registrar,
        in_place_task_name);
    }
#endif
  }

  //
  // create_plan_task
  //
  {
    create_plan_task_id = Runtime::generate_static_task_id();
    // fftw variant
    //
    // FIXME: remove assumption that FFTW is using OpenMP
    {
      TaskVariantRegistrar
        registrar(create_plan_task_id, create_plan_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      registrar.add_layout_constraint_set(0, fftw_layout_id);
      Runtime::preregister_task_variant<Plan, fftw_create_plan>(
        registrar,
        create_plan_task_name);
    }
#ifdef HYPERION_USE_CUDA
    // cufft variant
    {
      TaskVariantRegistrar
        registrar(create_plan_task_id, create_plan_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      registrar.add_layout_constraint_set(0, cufft_layout_id);
      Runtime::preregister_task_variant<Plan, cufft_create_plan>(
        registrar,
        create_plan_task_name);
    }
#endif
  }
  //
  // execute_fft_task
  //
  {
    execute_fft_task_id = Runtime::generate_static_task_id();
    // fftw variant
    //
    // FIXME: remove assumption that FFTW is using OpenMP
    {
      TaskVariantRegistrar
        registrar(execute_fft_task_id, execute_fft_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();
      registrar.add_layout_constraint_set(0, fftw_layout_id);
      Runtime::preregister_task_variant<int, fftw_execute>(
        registrar,
        execute_fft_task_name);
    }
#ifdef HYPERION_USE_CUDA
    // cufft variant
    {
      TaskVariantRegistrar
        registrar(execute_fft_task_id, execute_fft_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      registrar.set_idempotent();
      registrar.add_layout_constraint_set(0, cufft_layout_id);
      Runtime::preregister_task_variant<int, cufft_execute>(
        registrar,
        execute_fft_task_name);
    }
#endif
  }
  //
  // destroy_plan_task
  //
  {
    destroy_plan_task_id = Runtime::generate_static_task_id();
    // fftw variant
    //
    // FIXME: remove assumption that FFTW is using OpenMP
    {
      TaskVariantRegistrar
        registrar(destroy_plan_task_id, destroy_plan_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      registrar.add_layout_constraint_set(0, fftw_layout_id);
      Runtime::preregister_task_variant<fftw_destroy_plan>(
        registrar,
        destroy_plan_task_name);
    }
#ifdef HYPERION_USE_CUDA
    // cufft variant
    {
      TaskVariantRegistrar
        registrar(destroy_plan_task_id, destroy_plan_task_name);
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      registrar.add_layout_constraint_set(0, cufft_layout_id);
      Runtime::preregister_task_variant<cufft_destroy_plan>(
        registrar,
        destroy_plan_task_name);
    }
#endif
  }
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
