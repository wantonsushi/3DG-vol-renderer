cmake_minimum_required(VERSION 3.16)
project(volumetric_renderer LANGUAGES CXX)

# Use Clang
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_C_COMPILER clang)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
if(DEFINED ENV{OPENMP_DIR})
    message(STATUS "Using OpenMP from: $ENV{OPENMP_DIR}")
    include_directories("$ENV{OPENMP_DIR}/include")
    link_directories("$ENV{OPENMP_DIR}/lib")
else()
    message(FATAL_ERROR "OPENMP_DIR not set! Please define it in your system environment variables.")
endif()

# Add OpenMP flags manually
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")

# Find Embree
find_package(Embree 4.4 REQUIRED)
if(Embree_FOUND)
    message(STATUS "Found Embree: ${Embree_INCLUDE_DIR}")
    include_directories(${Embree_INCLUDE_DIR})
else()
    message(FATAL_ERROR "Embree not found!")
endif()

# Add Eigen
include_directories("extern/eigen-3.4.0")

# My include/src folders
include_directories(include)
file(GLOB SOURCES "src/*.cpp")

add_executable(vol_render ${SOURCES})
