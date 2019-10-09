#ifndef LEGMS_COLUMN_BUILDER_H_
#define LEGMS_COLUMN_BUILDER_H_

#ifdef LEGMS_USE_CASACORE

#pragma GCC visibility push(default)
#include <any>
#include <cassert>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>
#pragma GCC visibility pop

#include <legms/legms.h>
#include <legms/utility.h>
#include <legms/KeywordsBuilder.h>
#include <legms/IndexTree.h>
#include <legms/Column.h>
#include <legms/MSTable.h>

#include <casacore/casa/BasicSL/String.h>
#include <casacore/casa/Containers/Record.h>
#include <casacore/measures/Measures/MeasureHolder.h>

namespace legms {

template <MSTables D>
class ColumnBuilder
  : public KeywordsBuilder {
public:

  typedef typename MSTable<D>::Axes Axes;

  ColumnBuilder(
    const std::string& name,
    TypeTag datatype,
    const std::vector<Axes>& axes)
    : KeywordsBuilder()
    , m_name(name)
    , m_datatype(datatype)
    , m_axes(axes)
    , m_num_rows(0) {

    assert(axes[0] == MSTable<D>::ROW_AXIS);
    assert(has_unique_values(m_axes));
  }

  virtual ~ColumnBuilder() {}

  const std::string&
  name() const {
    return m_name;
  }

  TypeTag
  datatype() const {
    return m_datatype;
  }

  const std::vector<Axes>&
  axes() const {
    return m_axes;
  }

  const IndexTreeL&
  index_tree() const {
    return m_index_tree;
  }

  unsigned
  rank() const {
    return m_axes.size();
  }

  size_t
  num_rows() const {
    return m_num_rows;
  }

  bool
  empty() const {
    return index_tree().size() == 0;
  }

  void
  add_meas_record(const casacore::Record& rec) {
    m_meas_records.push_back(rec);
  }

  virtual void
  add_row(const std::any&) = 0;

  Column
  column(
    Legion::Context ctx,
    Legion::Runtime* rt,
    const std::string& name_prefix,
    const MeasRefContainer& inherited_meas_ref) const {

    IndexTreeL itree;
    auto itrank = index_tree().rank();
    if (itrank && itrank.value() == rank())
      itree = index_tree();
    std::vector<MeasRef> meas_refs;
    std::transform(
      m_meas_records.begin(),
      m_meas_records.end(),
      std::back_inserter(meas_refs),
      [&ctx, rt](const casacore::RecordInterface& rec) {
        casacore::MeasureHolder mh;
        casacore::String err;
        auto converted = mh.fromType(err, rec);
        if (converted) {
          if (mh.isMBaseline()) {
            auto baseline = mh.asMBaseline();
            return
              MeasRef::create<casacore::MBaseline>(
                ctx,
                rt,
                MClassT<MClass::M_BASELINE>::name,
                baseline.getRef());
          } else if (mh.isMDirection()) {
            auto direction = mh.asMDirection();
            return
              MeasRef::create<casacore::MDirection>(
                ctx,
                rt,
                MClassT<MClass::M_DIRECTION>::name,
                direction.getRef());
          } else if (mh.isMDoppler()) {
            auto doppler = mh.asMDoppler();
            return
              MeasRef::create<casacore::MDoppler>(
                ctx,
                rt,
                MClassT<MClass::M_DOPPLER>::name,
                doppler.getRef());
          } else if (mh.isMEarthMagnetic()) {
            auto earth_magnetic = mh.asMEarthMagnetic();
            return
              MeasRef::create<casacore::MEarthMagnetic>(
                ctx,
                rt,
                MClassT<MClass::M_EARTH_MAGNETIC>::name,
                earth_magnetic.getRef());
          } else if (mh.isMEpoch()) {
            auto epoch = mh.asMEpoch();
            return
              MeasRef::create<casacore::MEpoch>(
                ctx,
                rt,
                MClassT<MClass::M_EPOCH>::name,
                epoch.getRef());
          } else if (mh.isMFrequency()) {
            auto frequency = mh.asMFrequency();
            return
              MeasRef::create<casacore::MFrequency>(
                ctx,
                rt,
                MClassT<MClass::M_FREQUENCY>::name,
                frequency.getRef());
          } else if (mh.isMPosition()) {
            auto position = mh.asMPosition();
            return
              MeasRef::create<casacore::MPosition>(
                ctx,
                rt,
                MClassT<MClass::M_POSITION>::name,
                position.getRef());
          } else if (mh.isMRadialVelocity()) {
            auto radial_velocity = mh.asMRadialVelocity();
            return
              MeasRef::create<casacore::MRadialVelocity>(
                ctx,
                rt,
                MClassT<MClass::M_RADIAL_VELOCITY>::name,
                radial_velocity.getRef());
          } else if (mh.isMuvw()) {
            auto uvw = mh.asMuvw();
            return
              MeasRef::create<casacore::Muvw>(
                ctx,
                rt,
                MClassT<MClass::M_UVW>::name,
                uvw.getRef());
          } else {
            assert(false);
          }
        } else {
          assert(false);
        }
      });
    return
      Column::create(
        ctx,
        rt,
        name(),
        axes(),
        datatype(),
        itree,
#ifdef LEGMS_USE_CASACORE
        MeasRefContainer::create(ctx, rt, meas_refs, inherited_meas_ref),
#endif
        keywords(),
        name_prefix);
  }

protected:

  void
  set_next_row(const IndexTreeL& element_tree) {
    auto row_index = m_num_rows++;
    m_index_tree =
      m_index_tree.merged_with(IndexTreeL({{row_index, 1, element_tree}}));
  }

private:

  std::string m_name;

  TypeTag m_datatype;

  std::vector<Axes> m_axes;

  size_t m_num_rows;

  IndexTreeL m_index_tree;

  std::vector<casacore::Record> m_meas_records;
};

template <MSTables D>
class ScalarColumnBuilder
  : public ColumnBuilder<D> {
public:

  ScalarColumnBuilder(
    const std::string& name,
    TypeTag datatype)
    : ColumnBuilder<D>(name, datatype, {MSTable<D>::ROW_AXIS}) {
  }

  template <typename T>
  static auto
  generator(const std::string& name) {
    return
      [=]() {
        return std::make_unique<ScalarColumnBuilder<D>>(
          name,
          ValueType<T>::DataType);
      };
  }

  virtual ~ScalarColumnBuilder() {}

  void
  add_row(const std::any&) override {
    ColumnBuilder<D>::set_next_row(IndexTreeL());
  }
};

template <MSTables D, int ARRAYDIM>
class ArrayColumnBuilder
  : public ColumnBuilder<D> {
public:

  typedef typename MSTable<D>::Axes Axes;

  ArrayColumnBuilder(
    const std::string& name,
    TypeTag datatype,
    const std::vector<Axes>& axes,
    std::function<std::array<size_t, ARRAYDIM>(const std::any&)> element_shape)
    : ColumnBuilder<D>(name, datatype, axes)
    , m_element_shape(element_shape) {

    assert(axes.size() > 0);
  }

  virtual ~ArrayColumnBuilder() {}

  template <typename T>
  static auto
  generator(
    const std::string& name,
    std::function<std::array<size_t, ARRAYDIM>(const std::any&)>
    element_shape) {

    return
      [=](const std::vector<Axes>& axes) {
        return std::make_unique<ArrayColumnBuilder<D, ARRAYDIM>>(
          name,
          ValueType<T>::DataType,
          axes,
          element_shape);
      };
  }

  void
  add_row(const std::any& args) override {
    auto ary = m_element_shape(args);
    IndexTreeL t =
      std::accumulate(
        ary.rbegin(),
        ary.rend(),
        IndexTreeL(),
        [](const auto& t, const auto& d) {
          return IndexTreeL({{d, t}});
        });
    ColumnBuilder<D>::set_next_row(t);
  }

private:

  std::function<std::array<size_t, ARRAYDIM>(const std::any&)> m_element_shape;
};

} // end namespace legms

#endif // LEGMS_USE_CASACORE

#endif // LEGMS_COLUMN_BUILDER_H_

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
