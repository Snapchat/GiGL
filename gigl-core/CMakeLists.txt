cmake_minimum_required(VERSION 3.18)
project(gigl_core CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CMP0104: CMake 3.18+ warns when enable_language(CUDA) is called without
# CMAKE_CUDA_ARCHITECTURES being set. Set it to OFF so CMake does not inject
# any -gencode flags (torch provides its own arch list via TORCH_LIBRARIES).
cmake_policy(SET CMP0104 NEW)
set(CMAKE_CUDA_ARCHITECTURES OFF)

# Enable CUDA only when the toolkit is present; allows the same CMakeLists.txt
# to build on CPU-only machines without requiring nvcc.
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

# ---------------------------------------------------------------------------
# Extension modules — auto-discovered.
# Files named python_*.cpp under csrc/ are compiled as pybind11 extension
# modules. The companion <name>.cpp (without the "python_" prefix) is included
# automatically when present. Add a new extension by dropping source files
# here; no changes to this CMakeLists.txt are needed.
# ---------------------------------------------------------------------------
if(CMAKE_CUDA_COMPILER)
    file(GLOB_RECURSE _PYTHON_SRCS CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc/python_*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc/python_*.cu"
    )
else()
    file(GLOB_RECURSE _PYTHON_SRCS CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc/python_*.cpp"
    )
endif()

if(_PYTHON_SRCS)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

    # Locate pybind11 and torch cmake configs by scanning CMAKE_PREFIX_PATH,
    # which scikit-build-core sets to the active environment's site-packages.
    foreach(_prefix IN LISTS CMAKE_PREFIX_PATH)
        if(NOT pybind11_DIR AND EXISTS "${_prefix}/pybind11/share/cmake/pybind11")
            set(pybind11_DIR "${_prefix}/pybind11/share/cmake/pybind11")
        endif()
        if(NOT TORCH_CMAKE_PREFIX AND EXISTS "${_prefix}/torch/share/cmake")
            set(TORCH_CMAKE_PREFIX "${_prefix}/torch/share/cmake")
        endif()
    endforeach()

    if(NOT pybind11_DIR)
        message(FATAL_ERROR "Cannot find pybind11 cmake config in CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}. Run: make install_dev_deps")
    endif()
    if(NOT TORCH_CMAKE_PREFIX)
        message(FATAL_ERROR "Cannot find torch cmake config in CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}. Run: make install_dev_deps")
    endif()

    find_package(pybind11 CONFIG REQUIRED)
    find_package(Torch REQUIRED PATHS "${TORCH_CMAKE_PREFIX}")

    # torch_python provides the pybind11 type casters for at::Tensor. It is not
    # included in TORCH_LIBRARIES but is required for extensions that pass
    # tensors across the Python/C++ boundary.
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)

    set(GIGL_COMPILE_FLAGS -O3 -g -Wall -Wextra -Wno-unused-parameter)
    # nvcc does not accept bare -Wall/-Wextra; wrap with -Xcompiler for CUDA.
    set(GIGL_COMPILE_FLAGS_CUDA -O3 -Xcompiler=-g,-Wall,-Wextra,-Wno-unused-parameter)

    foreach(_py_src IN LISTS _PYTHON_SRCS)
        get_filename_component(_dir  "${_py_src}" DIRECTORY)
        get_filename_component(_stem "${_py_src}" NAME_WE)
        string(REGEX REPLACE "^python_" "" _name "${_stem}")

        set(_sources "${_py_src}")
        if(EXISTS "${_dir}/${_name}.cpp")
            list(APPEND _sources "${_dir}/${_name}.cpp")
        endif()
        if(CMAKE_CUDA_COMPILER AND EXISTS "${_dir}/${_name}.cu")
            list(APPEND _sources "${_dir}/${_name}.cu")
        endif()

        pybind11_add_module("${_name}" ${_sources})
        target_link_libraries("${_name}" PRIVATE "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}")
        target_compile_options("${_name}" PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:${GIGL_COMPILE_FLAGS}>
            $<$<COMPILE_LANGUAGE:CUDA>:${GIGL_COMPILE_FLAGS_CUDA}>
        )
        # TORCH_EXTENSION_NAME is used in PYBIND11_MODULE() to name the module.
        # PyTorch's own build system sets this; we must define it explicitly here.
        target_compile_definitions("${_name}" PRIVATE "TORCH_EXTENSION_NAME=${_name}")
        # Install into gigl_core/ so the .so is importable as gigl_core.<name>.
        install(TARGETS "${_name}" DESTINATION gigl_core)
    endforeach()
endif()

option(GIGL_CORE_BUILD_TESTS "Build C++ unit tests (used by make unit_test_cpp)" OFF)
if(GIGL_CORE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
