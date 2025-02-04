# find all .cu files in the current directory
file(GLOB CUDA_SOURCES "*.cu")

# Add executable
add_executable(main ${CUDA_SOURCES})