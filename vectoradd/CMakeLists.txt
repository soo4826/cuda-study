cmake_minimum_required(VERSION 3.8)

project(vectoradd)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA architecture (change it according to your GPU)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_75)

# CUDA kernel files
file(GLOB CUDA_SOURCES src/*.cu)

# Include directories
include_directories(include ${CUDA_INCLUDE_DIRS})

# Create a C++ executable using the CUDA library
cuda_add_executable(vectoradd_cuda ${CUDA_SOURCES})

# Link the CUDA library and required libraries
target_link_libraries(vectoradd_cuda ${CUDA_LIBRARIES})
