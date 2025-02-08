cmake_minimum_required(VERSION 3.5)

project(CUDA_KERNELS)

# Libtorch
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")
find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Development REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})

# header file
include_directories(${CMAKE_SOURCE_DIR} ${TORCH_INCLUDE_DIRS})

# Define a variable to specify which day to build
set(DAY "day001" CACHE STRING "Specify the day to build (e.g., day001, day002, etc.)")

# add_executable(execut day001/vector_addition.cu)
add_subdirectory(${DAY})