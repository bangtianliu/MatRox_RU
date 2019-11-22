cmake_minimum_required(VERSION 2.7)

#set(MKLROOT "/opt/intel/mkl/")



set(MKL_INC "$ENV{MKLROOT}/include")
message(is:"${MKL_INC}")
if(APPLE)
    SET(CMAKE_C_COMPILER /usr/local/Cellar/gcc/9.1.0/bin/gcc-9)
    SET(CMAKE_CXX_COMPILER /usr/local/Cellar/gcc/9.1.0/bin/g++-9)
    set(MKL_LIB "$ENV{MKLROOT}/lib")
    set(MKL_LIB2 "$ENV{MKLROOT}/../compiler/lib")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp")
endif()
if(UNIX AND NOT APPLE)
    set(MKL_LIB "$ENV{MKLROOT}/lib/intel64")
    # for iomp lib
    set(MKL_LIB2 "$ENV{MKLROOT}/../compiler/lib/intel64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()



set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -DCHOL -DMKL -DVERIFY -DCOLDCACHE   -std=gnu++11  -m64  ")

add_subdirectory(sympiler)
add_subdirectory(matroxTest-V1)
add_subdirectory(codeTest)
