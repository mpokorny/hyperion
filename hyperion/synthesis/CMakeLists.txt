#------------------------------------------------------------------------------#
# hyperion synthesis library
#------------------------------------------------------------------------------#

target_sources(hyperion PRIVATE
  CFTableBase.h
  CFTableBase.cc
  CFTable.h
  CFPhysicalTable.h
  PSTermTable.h
  PSTermTable.cc
  WTermTable.h
  WTermTable.cc
  Zernike.cc
  ATermTable.h
  ATermTable.cc
  ATermZernikeModel.h
  ATermZernikeModel.cc
  ATermIlluminationFunction.h
  ATermIlluminationFunction.cc
  GridCoordinateTable.h
  GridCoordinateTable.cc
  FFT.h
  FFT.cc
  ProductCFTable.h)
