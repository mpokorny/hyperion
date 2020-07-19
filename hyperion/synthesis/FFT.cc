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
  std::vector<int> n;
  int howmany;
  int dist;
  auto req = task->regions[0];
#define DESC(FT, N) do {                                          \
    const FieldAccessor<                                          \
      READ_WRITE,                                                 \
      FT,                                                         \
      N,                                                          \
      coord_t,                                                    \
      AffineAccessor<FT, N, coord_t>,                             \
      false> acc(regions[0], req.instance_fields[0]);             \
    Rect<N> rect(                                                 \
      rt->get_index_space_domain(req.region.get_index_space()));  \
    PointInRectIterator<N> pir(rect, false);                      \
    result.buffer = acc.ptr(*pir);                                \
    coord_t tsize = 1;                                            \
    coord_t rsize = 1;                                            \
    for (size_t i = 0; i < N; ++i) {                              \
      if (i < N - args.desc.rank) {                               \
        rsize *= rect.hi[i] - rect.lo[i] + 1;                     \
      } else {                                                    \
        n.push_back(rect.hi[i]- rect.lo[i] + 1);                  \
        tsize *= n.back();                                        \
      }                                                           \
    }                                                             \
    dist = tsize;                                                 \
    howmany = rsize;                                              \
  } while (0)

  auto region_rank = req.region.get_index_space().get_dim();
  if (args.desc.precision == Precision::SINGLE) {
    switch (region_rank) {
#define DESC_SINGLE(N)                          \
      case N: DESC(complex<float>, N); break;
      HYPERION_FOREACH_N(DESC_SINGLE);
#undef DESC_SINGLE
    default:
      assert(false);
      break;
    }
    fftwf_mutex.lock(ctx, rt);
    if (args.seconds >= 0)
      fftwf_set_timelimit(args.seconds);
    // when creating a plan, if FFTW does not yet have wisdom for that plan, it
    // will overwrite the array; thus when necessary, create a similar plan
    // initially with a different buffer
    result.handle.fftwf =
      fftwf_plan_many_dft(
        n.size(), n.data(), howmany,
        static_cast<fftwf_complex*>(result.buffer), NULL, 1, dist,
        static_cast<fftwf_complex*>(result.buffer), NULL, 1, dist,
        args.desc.sign,
        args.flags | FFTW_WISDOM_ONLY);
    if (result.handle.fftwf == NULL && (args.flags & FFTW_WISDOM_ONLY) == 0) {
      std::cout << "new FFTWF plan" << std::endl;
      auto buff = fftwf_alloc_complex(howmany * dist);
      auto p =
        fftwf_plan_many_dft(
          n.size(), n.data(), howmany,
          buff, NULL, 1, dist,
          buff, NULL, 1, dist,
          args.desc.sign,
          args.flags);
      ::fftwf_destroy_plan(p);
      fftwf_free(buff);
      result.handle.fftwf =
        fftwf_plan_many_dft(
          n.size(), n.data(), howmany,
          static_cast<fftwf_complex*>(result.buffer), NULL, 1, dist,
          static_cast<fftwf_complex*>(result.buffer), NULL, 1, dist,
          args.desc.sign,
          args.flags | FFTW_WISDOM_ONLY);
      assert(result.handle.fftwf != NULL);
    }
    fftwf_mutex.unlock();
  } else {
    switch (region_rank) {
#define DESC_DOUBLE(N)                          \
      case N: DESC(complex<double>, N); break;
      HYPERION_FOREACH_N(DESC_DOUBLE);
#undef DESC_DOUBLE
    default:
      assert(false);
      break;
    }
    fftw_mutex.lock(ctx, rt);
    if (args.seconds >= 0)
      fftw_set_timelimit(args.seconds);
    result.handle.fftw =
      fftw_plan_many_dft(
        n.size(), n.data(), howmany,
        static_cast<fftw_complex*>(result.buffer), NULL, 1, dist,
        static_cast<fftw_complex*>(result.buffer), NULL, 1, dist,
        args.desc.sign,
        args.flags | FFTW_WISDOM_ONLY);
    if (result.handle.fftw == NULL && (args.flags & FFTW_WISDOM_ONLY) == 0) {
      std::cout << "new FFTW plan" << std::endl;
      auto buff = fftw_alloc_complex(howmany * dist);
      auto p =
        fftw_plan_many_dft(
          n.size(), n.data(), howmany,
          buff, NULL, 1, dist,
          buff, NULL, 1, dist,
          args.desc.sign,
          args.flags);
      ::fftw_destroy_plan(p);
      fftw_free(buff);
      result.handle.fftw =
        fftw_plan_many_dft(
          n.size(), n.data(), howmany,
          static_cast<fftw_complex*>(result.buffer), NULL, 1, dist,
          static_cast<fftw_complex*>(result.buffer), NULL, 1, dist,
          args.desc.sign,
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
  std::vector<int> n;
  int batch;
  int dist;
  auto req = task->regions[0];
#define DESC(FT, N) do {                                          \
    const FieldAccessor<                                          \
      READ_WRITE,                                                 \
      FT,                                                         \
      N,                                                          \
      coord_t,                                                    \
      AffineAccessor<FT, N, coord_t>,                             \
      false> acc(regions[0], req.instance_fields[0]);             \
    Rect<N> rect(                                                 \
      rt->get_index_space_domain(req.region.get_index_space()));  \
    PointInRectIterator<N> pir(rect, false);                      \
    result.buffer = acc.ptr(*pir);                                \
    coord_t tsize = 1;                                            \
    coord_t rsize = 1;                                            \
    for (size_t i = 0; i < N; ++i) {                              \
      if (i < N - args.desc.rank) {                               \
        rsize *= rect.hi[i] - rect.lo[i] + 1;                     \
      } else {                                                    \
        n.push_back(rect.hi[i]- rect.lo[i] + 1);                  \
        tsize *= n.back();                                        \
      }                                                           \
    }                                                             \
    dist = tsize;                                                 \
    batch = rsize;                                                \
  } while (0)

  auto region_rank = req.region.get_index_space().get_dim();
  if (args.desc.precision == Precision::SINGLE) {
    switch (region_rank) {
#define DESC_SINGLE(N)                          \
      case N: DESC(complex<float>, N); break;
      HYPERION_FOREACH_N(DESC_SINGLE);
#undef DESC_SINGLE
    default:
      assert(false);
      break;
    }
    auto rc =
      cufftPlanMany(
        &result.handle.cufft, n.size(), n.data(),
        NULL, 1, dist,
        NULL, 1, dist,
        CUFFT_C2C,
        batch);
    if (rc != CUFFT_SUCCESS)
      result.handle.cufft = 0;
  } else {
    switch (region_rank) {
#define DESC_DOUBLE(N)                          \
      case N: DESC(complex<double>, N); break;
      HYPERION_FOREACH_N(DESC_DOUBLE);
#undef DESC_DOUBLE
    default:
      assert(false);
      break;
    }
    auto rc =
      cufftPlanMany(
        &result.handle.cufft, n.size(), n.data(),
        NULL, 1, dist,
        NULL, 1, dist,
        CUFFT_Z2Z,
        batch);
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
