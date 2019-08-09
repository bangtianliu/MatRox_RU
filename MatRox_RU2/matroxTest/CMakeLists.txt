cmake_minimum_required(VERSION 2.7)
project(MatRoxHGEMM)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_COMPILER /opt/intel/bin/icpc)
set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -qopenmp")

set(OMP /opt/intel/lib/intel64/)
set(MKL_INCLUDE_DIRS /opt/intel/mkl/include/)
set(MKL_LIBRARIES /opt/intel/mkl/lib/intel64/)

include_directories(${MKL_INCLUDE_DIRS} "../symGen/")
link_directories(${MKL_LIBRARIES} ${OMP})

FILE(GLOB srcode "../sympiler/HMatrix.cpp" "../sympiler/CDS.cpp" "../sympiler/boundingbox.cpp" "../sympiler/nUtil.cpp" "../sympiler/Matrix.cpp" "../sympiler/ClusterTree.cpp" "../sympiler/IR.cpp" "../sympiler/Util.cpp")
set(SOURCE_FILES main.cpp Util.h ${srcode})

add_executable(MatRoxHGEMM ${SOURCE_FILES})
target_link_libraries(MatRoxHGEMM mkl_intel_lp64 mkl_gnu_thread mkl_core iomp5 pthread m dl)