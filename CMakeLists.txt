cmake_minimum_required(VERSION 3.15)

project(CUDA_KERNELS LANGUAGES C CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA package
# find_package(CUDA REQUIRED)

# Define a variable to specify which day to build
set(DAY "day001" CACHE STRING "Specify the day to build (e.g., day001, day002, etc.)")

# add_executable(execut day001/vector_addition.cu)
add_subdirectory(${DAY})