cmake_minimum_required(VERSION 3.22)
project(parallel)

set (CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

project(parallel LANGUAGES CXX CUDA)
enable_language(CUDA)

#find_package(Boost REQUIRED COMPONENTS filesystem)




if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(TBB REQUIRED)

add_executable(parallel main.cpp
        UndirectedGraph.cpp
        UndirectedGraph.h
        Vertex.cpp
        Vertex.h
        iMBEA.cpp
        iMBEA.h
        ParLmbc.cpp
        ParLmbc.h
        utils.h
        GMBE.cpp
        GMBE.h
        matrixMul.cu
        matrixMul.cuh
        BipartiteGraph.cpp
        BipartiteGraph.h
        CliqueEnum.cpp
        CliqueEnum.h
        cuda/mat.cu
        cuda/mat.cuh)
#        cuda/bfsCUDA.cu
#        cuda/bfsCUDA.cuh
#        cuda/run.cpp
#        cuda/Graph.cpp
#        cuda/Graph.h)

target_link_libraries(parallel PRIVATE TBB::tbb)
#target_link_libraries(parallel Boost::filesystem)

