cmake_minimum_required(VERSION 3.5)
project(nanmath)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 11)

set(CMAKE_CXX_COMPILER g++-7)
set(CMAKE_C_COMPILER gcc-7)
set(CMAKE_CXX_FLAGS "-Wall -O3 -fopenmp -pthread")
set(CMAKE_C_FLAGS "-Wall -O3 -mavx -mfma")

#set(CMAKE_CXX_COMPILER icpc)
#set(CMAKE_C_COMPILER icc)
#set(CMAKE_CXX_FLAGS "-Wall -O3 -fopenmp -pthread")
#set(CMAKE_C_FLAGS "-Wall -O3 -march=core-avx2")

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

include_directories(/opt/armadillo-8.100.1/include)
link_directories(/opt/armadillo-8.100.1/lib64)
set(CMAKE_INSTALL_RPATH /opt/armadillo-8.100.1/lib64 \${ORIGIN})

add_library(nanmath SHARED nanmath.c)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark nanmath armadillo)

