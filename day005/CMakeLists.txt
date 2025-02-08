# Find all .cu files in the current directory
file(GLOB CUDA_SOURCES "*.cu")

# Add executable
add_executable(main ${CUDA_SOURCES})

# Link against LibTorch
target_link_libraries(main "${TORCH_LIBRARIES}")

# Add include directories for LibTorch
target_include_directories(main PRIVATE ${TORCH_INCLUDE_DIRS})

# Ensure the correct C++ standard is used
set_property(TARGET main PROPERTY CXX_STANDARD 17)
