#define GENERATORS(T)                                                   \
  template <> inline                                                    \
  auto                                                                  \
  ScalarColumnBuilder::generator<T>(                                    \
    const std::string& name) {                                          \
                                                                        \
    return                                                              \
      [=](const IndexTreeL& row_index_pattern) {                        \
        return std::make_unique<ScalarColumnBuilder>(                   \
          name,                                                         \
          ValueType<T>::DataType,                                       \
          row_index_pattern);                                           \
      };                                                                \
  }                                                                     \
  template <> inline                                                    \
  auto                                                                  \
  ScalarColumnBuilder::generator<std::vector<T>>(                       \
    const std::string& name) {                                          \
                                                                        \
    return                                                              \
      [=](const IndexTreeL& row_index_pattern) {                        \
        return std::make_unique<ScalarColumnBuilder>(                   \
          name,                                                         \
          ValueType<std::vector<T>>::DataType,                          \
          row_index_pattern);                                           \
      };                                                                \
  }                                                                     \
  template <> template<> inline                                         \
  auto                                                                  \
  ArrayColumnBuilder<1>::generator<T>(                                  \
    const std::string& name,                                            \
    std::function<std::array<size_t, 1>(const std::any&)> row_dimensions) { \
                                                                        \
    return                                                              \
      [=](const IndexTreeL& row_index_pattern) {                        \
        return std::make_unique<ArrayColumnBuilder<1>>(                 \
          name,                                                         \
          ValueType<T>::DataType,                                       \
          row_index_pattern,                                            \
          row_dimensions);                                              \
      };                                                                \
  }                                                                     \
  template <> template<> inline                                         \
  auto                                                                  \
  ArrayColumnBuilder<1>::generator<std::vector<T>>(                     \
    const std::string& name,                                            \
    std::function<std::array<size_t, 1>(const std::any&)> row_dimensions) { \
                                                                        \
    return                                                              \
      [=](const IndexTreeL& row_index_pattern) {                        \
        return std::make_unique<ArrayColumnBuilder<1>>(                 \
          name,                                                         \
          ValueType<std::vector<T>>::DataType,                          \
          row_index_pattern,                                            \
          row_dimensions);                                              \
      };                                                                \
  }                                                                     \
  template <> template<> inline                                         \
  auto                                                                  \
  ArrayColumnBuilder<2>::generator<T>(                                  \
    const std::string& name,                                            \
    std::function<std::array<size_t, 2>(const std::any&)> row_dimensions) { \
                                                                        \
    return                                                              \
      [=](const IndexTreeL& row_index_pattern) {                        \
        return std::make_unique<ArrayColumnBuilder<2>>(                 \
          name,                                                         \
          ValueType<T>::DataType,                                       \
          row_index_pattern,                                            \
          row_dimensions);                                              \
      };                                                                \
  }                                                                     \
  template <> template<> inline                                         \
  auto                                                                  \
  ArrayColumnBuilder<2>::generator<std::vector<T>>(                     \
    const std::string& name,                                            \
    std::function<std::array<size_t, 2>(const std::any&)> row_dimensions) { \
                                                                        \
    return                                                              \
      [=](const IndexTreeL& row_index_pattern) {                        \
        return std::make_unique<ArrayColumnBuilder<2>>(                 \
          name,                                                         \
          ValueType<std::vector<T>>::DataType,                          \
          row_index_pattern,                                            \
          row_dimensions);                                              \
      };                                                                \
  }                                                                     \
  template <> template<> inline                                         \
  auto                                                                  \
  ArrayColumnBuilder<3>::generator<T>(                                  \
    const std::string& name,                                            \
    std::function<std::array<size_t, 3>(const std::any&)> row_dimensions) { \
                                                                        \
    return                                                              \
      [=](const IndexTreeL& row_index_pattern) {                        \
        return std::make_unique<ArrayColumnBuilder<3>>(                 \
          name,                                                         \
          ValueType<T>::DataType,                                       \
          row_index_pattern,                                            \
          row_dimensions);                                              \
      };                                                                \
  }                                                                     \
  template <> template<> inline                                         \
  auto                                                                  \
  ArrayColumnBuilder<3>::generator<std::vector<T>>(                     \
    const std::string& name,                                            \
    std::function<std::array<size_t, 3>(const std::any&)> row_dimensions) { \
                                                                        \
    return                                                              \
      [=](const IndexTreeL& row_index_pattern) {                        \
        return std::make_unique<ArrayColumnBuilder<3>>(                 \
          name,                                                         \
          ValueType<std::vector<T>>::DataType,                          \
          row_index_pattern,                                            \
          row_dimensions);                                              \
      };                                                                \
  }


GENERATORS(casacore::Bool)
GENERATORS(casacore::Char)
GENERATORS(casacore::uChar)
GENERATORS(casacore::Short)
GENERATORS(casacore::uShort)
GENERATORS(casacore::Int)
GENERATORS(casacore::uInt)
GENERATORS(casacore::Float)
GENERATORS(casacore::Double)
GENERATORS(casacore::Complex)
GENERATORS(casacore::DComplex)
GENERATORS(casacore::String)

#undef GENERATORS

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
