# Minimal CMakeLists.txt for the cube renderer (single .cu file).
# Mirrors the lab pattern: cmake -B build && cmake --build build
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(cube LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Single executable from one .cu file
add_executable(cube main.cu)

# Optimize and let nvcc auto-pick the host compiler / arch
target_compile_options(cube PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# Drop the binary in build/release/ to mirror your other labs
set_target_properties(cube PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/release
)
