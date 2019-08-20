cmake_minimum_required(VERSION 2.7)
project(FMatRoxHGEMM)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-fno-rtti -O3 -Wl,-rpath,${MKLROOT}/lib -mkl=parallel  -fpic -qopenmp -std=c++11")

set(CMAKE_CXX_COMPILER /opt/intel/compilers_and_libraries_2019.4.233/mac/bin/intel64/icpc)
set(OMP /opt/intel/compilers_and_libraries_2019/mac/lib)



set(MKL_ROOT /opt/intel/mkl)
set(MKL_INCLUDE_DIRS ${MKL_ROOT}/include/)
set(MKL_LIBRARIES ${MKL_ROOT}/lib/intel64/)

include_directories(${MKL_INCLUDE_DIRS} "../symGen/")
link_directories(${MKL_LIBRARIES} ${OMP})

FILE(GLOB srcode "../sympiler/HMatrix.cpp" "../sympiler/CDS.cpp" "../sympiler/boundingbox.cpp" "../sympiler/nUtil.cpp" "../sympiler/Matrix.cpp" "../sympiler/ClusterTree.cpp" "../sympiler/IR.cpp" "../sympiler/Util.cpp")


set(SOURCE_FILES main.cpp Util.h main.cpp HTree.h ${srcode} compression.h hcds.h)

add_executable(FMatRoxHGEMM ${SOURCE_FILES})
target_link_libraries(FMatRoxHGEMM iomp5 pthread m dl)
