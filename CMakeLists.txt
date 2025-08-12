cmake_minimum_required(VERSION 3.16)
project(volumetric_renderer LANGUAGES CXX)

set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_C_COMPILER clang)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add Externals
include_directories("extern/eigen-3.4.0")
include_directories("extern/gif-h")

# Include folders
include_directories(include)

# Add executable early
add_executable(vol_render "tests/main.cpp")

# Use CMake's built-in OpenMP support
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_INCLUDE_DIRS}")
    target_link_libraries(vol_render PRIVATE OpenMP::OpenMP_CXX)
endif()
