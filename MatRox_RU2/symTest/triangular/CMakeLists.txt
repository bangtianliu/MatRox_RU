cmake_minimum_required(VERSION 2.7)
project(SymTriang)

message(is:"${MKL_INC}")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-DMYBLAS -std=gnu++11" )

set(MKL_INC /opt/intel/mkl/include/)

include_directories(${MKL_INC} "../../symGen/" "../../util" )

set(SOURCE_FILES main.cpp)
add_executable(SymTriang ${SOURCE_FILES})

target_link_libraries(SymTriang   )