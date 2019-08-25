cmake_minimum_required(VERSION 2.8)

project(sympiler_proj)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -O3 -fpic -Wno-dev -qopenmp -std=c++11")
if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
endif()

set(CMAKE_CXX_COMPILER /share/apps/compute/intel/intelmpi2018/compilers_and_libraries/linux/bin/intel64/icpc)
set(OMP /share/apps/compute/intel/intelmpi2018/compilers_and_libraries/linux/lib/intel64/)

set(MKL_INCLUDE_DIRS ${MKL_ROOT}/include/)
set(MKL_LIBRARIES ${MKL_ROOT}/lib/intel64/)

include_directories(${MKL_INCLUDE_DIRS})
link_directories(${MKL_LIBRARIES} ${OMP})

include_directories(${MKL_INCLUDE_DIRS})
link_directories(${MKL_LIBRARIES} ${OMP})
add_executable(sympiler_proj MatRox.cpp Func.cpp Func.h Var.cpp Var.h Module.cpp Module.h
        CodeGen_C.cpp CodeGen_C.h Target.cpp Target.h Output.cpp Output.h IR.cpp IR.h Expr.cpp Expr.h
        IROperator.cpp IROperator.h Schedule.cpp Schedule.h IRPrinter.h IRPrinter.cpp IRVisitor.h IRVisitor.cpp
        Type.h Type.cpp Optimization.h Optimization.cpp IntrusivePtr.h IRMutator.h IRMutator.cpp ScheduleCompute.cpp
        ScheduleCompute.h Kernel.cpp Kernel.h Inspector.cpp Inspector.h Matrix.cpp Matrix.h Util.cpp Util.h Scope.h
        Lower.cpp Lower.h Argument.h VIPrune.cpp VIPrune.h Substitute.cpp Substitute.h IREquality.cpp IREquality.h Function.cpp Function.h Factorization.cpp Factorization.h NumericalUtils.h FactorizationInspector.cpp FactorizationInspector.h NumericalUtils.cpp Triangular.cpp Triangular.h VIUnroll.h VIUnroll.cpp MatRox.cpp HMatrix.cpp HMatrix.h HGEMM.cpp HGEMM.h DSPattern.h ClusterTree.cpp ClusterTree.h boundingbox.cpp boundingbox.h config.h nUtil.h nUtil.cpp CDS.cpp CDS.h DSInspector.cpp DSInspector.h VICoarsen.cpp VICoarsen.h rstree.cpp rstree.h ReNearFar.cpp ReNearFar.h HTree.h HTree.cpp time_util.h NearSearch.h NearSearch.cpp NNtree.cpp NNtree.h)
#add_executable(Substitute.cpp Substitute.h) #TODO will add later

target_link_libraries(sympiler_proj
        mkl_intel_lp64 mkl_gnu_thread mkl_core iomp5 pthread m dl
        )
