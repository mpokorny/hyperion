#------------------------------------------------------------------------------#
# hyperion synthesis library
#------------------------------------------------------------------------------#

target_sources(hyperion PRIVATE
  CFTableBase.h
  CFTableBase.cc
  CFTable.h
  CFTable.cc
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
  DirectionCoordinateTable.h
  DirectionCoordinateTable.cc
  FFT.h
  FFT.cc)
