#ifndef LEGMS_UTILITY_H_
#define LEGMS_UTILITY_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>

#include <casacore/casa/aipstype.h>
#include <casacore/casa/Arrays/IPosition.h>
#include <casacore/casa/BasicSL/String.h>
#include <casacore/casa/Utilities/DataType.h>
#include "legion.h"
#include "IndexTree.h"

namespace legms {

typedef IndexTree<Legion::coord_t> IndexTreeL;

template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

template <typename D>
bool
has_unique_values(const std::vector<D>& axes) {
  std::vector<D> ax = axes;
  std::sort(ax.begin(), ax.end());
  std::unique(ax.begin(), ax.end());
  return ax.size() == axes.size();
}

template <typename D>
std::vector<int>
dimensions_map(const std::vector<D>& from, const std::vector<D>& to) {
  std::vector<int> result(from.size());
  for (size_t i = 0; i < from.size(); ++i) {
    result[i] = -1;
    for (size_t j = 0; result[i] == -1 && j < to.size(); ++j)
      if (from[i] == to[j])
        result[i] = j;
  }
  return result;
}

template <typename S>
class string_serdez {
  // this class only works for fixed size character encodings!

public:

  typedef S FIELD_TYPE;

  typedef typename S::size_type size_type;
  typedef typename S::value_type value_type;

  static const size_t MAX_SERIALIZED_SIZE =
    std::numeric_limits<size_type>::max();

  static size_t
  serialized_size(const S& val) {
    return sizeof(size_type) + val.size() * sizeof(value_type);
  }

  static size_t
  serialize(const S& val, void *buffer) {
    size_t result = serialized_size(val);
    size_type nch = val.size();
    memcpy(static_cast<size_type *>(buffer), &nch, sizeof(nch));
    if (nch > 0) {
      value_type* chbuf =
        reinterpret_cast<value_type *>(static_cast<size_type *>(buffer) + 1);
      memcpy(chbuf, val.data(), result - sizeof(size_type));
    }
    return result;
  }

  static size_t
  deserialize(S& val, const void *buffer) {
    size_type nch = *static_cast<const size_type *>(buffer);
    val.clear();
    if (nch > 0) {
      val.reserve(nch);
      val.append(
        reinterpret_cast<const value_type*>(
          static_cast<const size_type *>(buffer) + 1),
        nch);
    }
    return serialized_size(val);
  }

  static void
  destroy(S&) {
  }
};

class index_tree_serdez {
public:
  typedef IndexTreeL FIELD_TYPE;

  static const size_t MAX_SERIALIZED_SIZE = std::numeric_limits<size_t>::max();

  static size_t
  serialized_size(const IndexTreeL& val) {
    return val.serialized_size();
  }

  static size_t
  serialize(const IndexTreeL& val, void *buffer) {
    return val.serialize(reinterpret_cast<char*>(buffer));
  }

  static size_t
  deserialize(IndexTreeL& val, const void *buffer) {
    val = IndexTreeL::deserialize(static_cast<const char*>(buffer));
    return *reinterpret_cast<const size_t *>(buffer);
  }

  static void
  destroy(IndexTreeL&) {
  }
};

class bool_or_redop {
public:
  typedef bool LHS;
  typedef bool RHS;

  static void
  combine(LHS& lhs, RHS rhs) {
    lhs = lhs || rhs;
  }

  template <bool EXCL>
  static void
  apply(LHS& lhs, RHS rhs) {
    combine(lhs, rhs);
  }

  static constexpr const RHS identity = false;

  template <bool EXCL>
  static void
  fold(RHS& rhs1, RHS rhs2) {
    combine(rhs1, rhs2);
  }
};

class SerdezManager {
public:

  static void
  register_ops() {
    std::call_once(
      initialized,
      []() {
        Legion::Runtime::register_custom_serdez_op<index_tree_serdez>(
          INDEX_TREE_SID);
        Legion::Runtime::register_custom_serdez_op<
          string_serdez<casacore::String>>(CASACORE_STRING_SID);
        Legion::Runtime::register_reduction_op<bool_or_redop>(
          BOOL_OR_REDOP);
      });
  }

  enum {
    INDEX_TREE_SID = 1,
    CASACORE_STRING_SID
  };

  enum {
    BOOL_OR_REDOP = 1,
  };

private:

  static std::once_flag initialized;
};

Legion::FieldID
add_field(
  casacore::DataType datatype,
  Legion::FieldAllocator fa,
  Legion::FieldID field_id = AUTO_GENERATE_ID);

template <casacore::DataType T>
struct DataType {

  //typedef X ValueType;
};

template <>
struct DataType<casacore::DataType::TpBool> {
  typedef casacore::Bool ValueType;
  constexpr static int serdez_id = 0;
};

template <>
struct DataType<casacore::DataType::TpChar> {
  typedef casacore::Char ValueType;
  constexpr static int serdez_id = 0;
};

template <>
struct DataType<casacore::DataType::TpUChar> {
  typedef casacore::uChar ValueType;
  constexpr static int serdez_id = 0;
};

template <>
struct DataType<casacore::DataType::TpShort> {
  typedef casacore::Short ValueType;
  constexpr static int serdez_id = 0;
};

template <>
struct DataType<casacore::DataType::TpUShort> {
  typedef casacore::uShort ValueType;
  constexpr static int serdez_id = 0;
};

template <>
struct DataType<casacore::DataType::TpInt> {
  typedef casacore::Int ValueType;
  constexpr static int serdez_id = 0;
};

template <>
struct DataType<casacore::DataType::TpUInt> {
  typedef casacore::uInt ValueType;
  constexpr static int serdez_id = 0;
};

template <>
struct DataType<casacore::DataType::TpFloat> {
  typedef casacore::Float ValueType;
  constexpr static int serdez_id = 0;
};

template <>
struct DataType<casacore::DataType::TpDouble> {
  typedef casacore::Double ValueType;
  constexpr static int serdez_id = 0;
};

template <>
struct DataType<casacore::DataType::TpComplex> {
  typedef casacore::Complex ValueType;
  constexpr static int serdez_id = 0;
};

template <>
struct DataType<casacore::DataType::TpDComplex> {
  typedef casacore::DComplex ValueType;
  constexpr static int serdez_id = 0;
};

template <>
struct DataType<casacore::DataType::TpString> {
  typedef casacore::String ValueType;
  constexpr static int serdez_id = SerdezManager::CASACORE_STRING_SID;
};

template <typename T>
struct ValueType {
  // constexpr static const casacore::DataType DataType;
};

template <>
struct ValueType<casacore::Bool> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpBool;
};

template <>
struct ValueType<casacore::Char> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpChar;
};

template <>
struct ValueType<casacore::uChar> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpUChar;
};

template <>
struct ValueType<casacore::Short> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpShort;
};

template <>
struct ValueType<casacore::uShort> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpUShort;
};

template <>
struct ValueType<casacore::Int> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpInt;
};

template <>
struct ValueType<casacore::uInt> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpUInt;
};

template <>
struct ValueType<casacore::Float> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpFloat;
};

template <>
struct ValueType<casacore::Double> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpDouble;
};

template <>
struct ValueType<casacore::Complex> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpComplex;
};

template <>
struct ValueType<casacore::DComplex> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpDComplex;
};

template <>
struct ValueType<casacore::String> {
  constexpr static casacore::DataType DataType = casacore::DataType::TpString;
};

class ProjectedIndexPartitionTask
  : Legion::IndexTaskLauncher {
public:

  enum { IMAGE_RANGES_FID };

  struct args {
    Legion::Domain bounds;
    int prjdim;
    int dmap[];
  };

  constexpr static const char * const TASK_NAME =
    "legms::ProjectedIndexPartitionTask";
  static Legion::TaskID TASK_ID;

  ProjectedIndexPartitionTask(
    Legion::IndexSpace launch_space,
    Legion::LogicalPartition lp,
    Legion::LogicalRegion lr,
    args* global_arg);

  void
  dispatch(Legion::Context ctx, Legion::Runtime* runtime);

  static void
  base_impl(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime *runtime);

  static void
  register_task(Legion::Runtime* runtime);
};

template <int IPDIM, int PRJDIM>
Legion::IndexPartitionT<PRJDIM>
projected_index_partition(
  Legion::Context ctx,
  Legion::Runtime* runtime,
  Legion::IndexPartitionT<IPDIM> ip,
  Legion::IndexSpaceT<PRJDIM> prj_is,
  const std::array<int, PRJDIM>& dmap) {

  assert(
    std::all_of(
      dmap.begin(),
      dmap.end(),
      [](auto d) { return -1 <= d && d < IPDIM; }));

  std::unique_ptr<ProjectedIndexPartitionTask::args> args(
    static_cast<ProjectedIndexPartitionTask::args*>(
      operator new(sizeof(ProjectedIndexPartitionTask::args)
                   + PRJDIM * sizeof(dmap[0]))));
  auto prj_domain = runtime->get_index_space_domain(ctx, prj_is);
  args->bounds = Legion::Rect<PRJDIM>(prj_domain.lo(), prj_domain.hi());
  args->prjdim = PRJDIM;
  memcpy(args->dmap, dmap.data(), PRJDIM * sizeof(dmap[0]));

  Legion::FieldSpace images_fs = runtime->create_field_space(ctx);
  {
    auto fa = runtime->create_field_allocator(ctx, images_fs);
    fa.allocate_field(
      sizeof(Legion::Rect<PRJDIM>),
      ProjectedIndexPartitionTask::IMAGE_RANGES_FID);
  }
  Legion::LogicalRegionT<IPDIM> images_lr(
    runtime->create_logical_region(
      ctx,
      runtime->get_parent_index_space(ip),
      images_fs));
  Legion::LogicalPartitionT<IPDIM> images_lp(
    runtime->get_logical_partition(ctx, images_lr, ip));

  Legion::IndexSpace ip_cs =
    runtime->get_index_partition_color_space_name(ctx, ip);

  ProjectedIndexPartitionTask
    fill_images(ip_cs, images_lp, images_lr, args.get());
  fill_images.dispatch(ctx, runtime);

  Legion::IndexPartitionT<PRJDIM> result(
    runtime->create_partition_by_image_range(
      ctx,
      prj_is,
      images_lp,
      images_lr,
      0,
      ip_cs));

  runtime->destroy_logical_partition(ctx, images_lp);
  runtime->destroy_logical_region(ctx, images_lr);
  runtime->destroy_field_space(ctx, images_fs);
  return result;
}

Legion::IndexPartition
projected_index_partition(
  Legion::Context ctx,
  Legion::Runtime* runtime,
  Legion::IndexPartition ip,
  Legion::IndexSpace prj_is,
  const std::vector<int>& dmap);

Legion::IndexPartition
create_partition_on_axes(
  Legion::Context ctx,
  Legion::Runtime* runtime,
  Legion::IndexSpace is,
  const std::vector<int>& dims);

void
register_tasks(Legion::Runtime* runtime);

} // end namespace legms

#endif // LEGMS_UTILITY_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
