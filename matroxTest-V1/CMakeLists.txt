cmake_minimum_required(VERSION 2.7)
project(FMatRoxHGEMM)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -qopenmp")

set(OMP "${MKL_LIB2}")
set(CMAKE_CXX_COMPILER "$ENV{CMAKE_CXX_COMPILER}")
#set(CMAKE_CXX_COMPILER /opt/intel/parallel_studio_xe_2019/compilers_and_libraries_2019/linux/bin/intel64/icpc)
#set(OMP /opt/intel/compilers_and_libraries/linux/lib/intel64/)

set(MKL_INCLUDE_DIRS ${MKL_ROOT}/include/)
set(MKL_LIBRARIES ${MKL_ROOT}/lib/intel64/)

include_directories(${MKL_INCLUDE_DIRS} "../symGen/")
link_directories(${MKL_LIBRARIES} ${OMP})

FILE(GLOB srcode "../sympiler/HMatrix.cpp" "../sympiler/CDS.cpp" "../sympiler/boundingbox.cpp" "../sympiler/nUtil.cpp" "../sympiler/Matrix.cpp" "../sympiler/ClusterTree.cpp" "../sympiler/IR.cpp" "../sympiler/Util.cpp")


set(SOURCE_FILES main.cpp Util.h main.cpp HTree.h ${srcode} compression.h hcds.h)

add_executable(FMatRoxHGEMM ${SOURCE_FILES})
target_link_libraries(FMatRoxHGEMM mkl_intel_lp64 mkl_gnu_thread mkl_core iomp5 pthread m dl)
