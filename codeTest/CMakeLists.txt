cmake_minimum_required(VERSION 2.7)
project(MatRoxHGEMM)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_COMPILER /opt/intel/bin/icpc)
set(OMP /opt/intel/lib/intel64/)

set(CMAKE_CXX_COMPILER /share/apps/compute/intel/intelmpi2018/compilers_and_libraries/linux/bin/intel64/icpc)
set(OMP /share/apps/compute/intel/intelmpi2018/compilers_and_libraries/linux/lib/intel64/)

set(MKL_INCLUDE_DIRS ${MKL_ROOT}/include/)
set(MKL_LIBRARIES ${MKL_ROOT}/lib/intel64/)


include_directories(${MKL_INCLUDE_DIRS} "../codeGen/")
link_directories(${MKL_LIBRARIES} ${OMP})

FILE(GLOB srcode "../sympiler/HMatrix.cpp" "../sympiler/CDS.cpp" "../sympiler/boundingbox.cpp" "../sympiler/nUtil.cpp" "../sympiler/Matrix.cpp" "../sympiler/ClusterTree.cpp" "../sympiler/IR.cpp" "../sympiler/Util.cpp")
set(SOURCE_FILES main.cpp Util.h ${srcode})

add_executable(testHGEMM ${SOURCE_FILES})
target_link_libraries(testHGEMM mkl_intel_lp64 mkl_gnu_thread mkl_core iomp5 pthread m dl)
