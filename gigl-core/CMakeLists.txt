cmake_minimum_required(VERSION 3.18)
project(gigl_core CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CMP0104: CMake 3.18+ warns when enable_language(CUDA) is called without
# CMAKE_CUDA_ARCHITECTURES being set. Set it to OFF — Torch provides a
# comprehensive arch list via TORCH_LIBRARIES.
cmake_policy(SET CMP0104 NEW)
set(CMAKE_CUDA_ARCHITECTURES OFF)

# Enable CUDA only when the toolkit is present; allows the same CMakeLists.txt
# to build on CPU-only machines without requiring nvcc. find_package(Torch)
# internally calls enable_language(CUDA) when torch was built with CUDA support,
# which requires CMAKE_CUDA_COMPILER to be set — hence the nvcc hint below.
include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    find_program(CMAKE_CUDA_COMPILER nvcc HINTS /usr/local/cuda/bin)
endif()
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Locate pybind11 and torch cmake configs by scanning CMAKE_PREFIX_PATH, which
# scikit-build-core sets to the active environment's site-packages directory.
foreach(_prefix IN LISTS CMAKE_PREFIX_PATH)
    if(NOT pybind11_DIR AND EXISTS "${_prefix}/pybind11/share/cmake/pybind11")
        set(pybind11_DIR "${_prefix}/pybind11/share/cmake/pybind11")
    endif()
    if(NOT TORCH_CMAKE_PREFIX AND EXISTS "${_prefix}/torch/share/cmake")
        set(TORCH_CMAKE_PREFIX "${_prefix}/torch/share/cmake")
    endif()
endforeach()

if(NOT pybind11_DIR)
    message(FATAL_ERROR "Cannot find pybind11 cmake config in CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
endif()
if(NOT TORCH_CMAKE_PREFIX)
    message(FATAL_ERROR "Cannot find torch cmake config in CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
endif()

find_package(pybind11 CONFIG REQUIRED)
find_package(Torch REQUIRED PATHS "${TORCH_CMAKE_PREFIX}")

# torch_python provides the pybind11 type casters for at::Tensor. It is not
# included in TORCH_LIBRARIES but is required for extensions that pass tensors
# across the Python/C++ boundary.
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)

set(GIGL_COMPILE_FLAGS -O3 -g -Wall -Wextra -Wno-unused-parameter)
# nvcc does not accept bare -Wall/-Wextra; wrap with -Xcompiler for CUDA sources.
set(GIGL_COMPILE_FLAGS_CUDA -O3 -Xcompiler=-g,-Wall,-Wextra,-Wno-unused-parameter)

pybind11_add_module(_core
    src/add_one.cpp
    src/python_add_one.cpp
)
target_link_libraries(_core PRIVATE "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}")
target_compile_options(_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${GIGL_COMPILE_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${GIGL_COMPILE_FLAGS_CUDA}>
)
# TORCH_EXTENSION_NAME is used in PYBIND11_MODULE() to name the module.
# PyTorch's own build system sets this; we must define it explicitly here.
target_compile_definitions(_core PRIVATE TORCH_EXTENSION_NAME=_core)
install(TARGETS _core DESTINATION gigl_core)

option(GIGL_CORE_BUILD_TESTS "Build C++ unit tests" OFF)
if(GIGL_CORE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
