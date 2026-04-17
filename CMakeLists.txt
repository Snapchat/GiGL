cmake_minimum_required(VERSION 3.18)
project(gigl_csrc CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Locate pybind11's cmake config via the Python interpreter so this works
# whether pybind11 is installed system-wide or via pip.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
        "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)

# Locate PyTorch headers via its own cmake prefix path (works for both CPU and CUDA torch).
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
        "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# CUDA torch's CMake unconditionally calls enable_language(CUDA), which requires nvcc.
# nvcc may not be on PATH even when the CUDA toolkit is installed; find it explicitly.
if(NOT CMAKE_CUDA_COMPILER)
    find_program(CMAKE_CUDA_COMPILER nvcc PATHS "/usr/local/cuda/bin")
endif()

find_package(Torch REQUIRED PATHS "${TORCH_CMAKE_PREFIX}")

set(GIGL_COMPILE_FLAGS -O3 -Wall -Wextra -Wno-unused-parameter)

# ---------------------------------------------------------------------------
# Extension modules — auto-discovered.
# Files named python_*.cpp under gigl/csrc/ are compiled as pybind11 extension
# modules. The companion <name>.cpp (without the "python_" prefix) is included
# automatically when present. This mirrors the convention from the old
# build_cpp_extensions.py: no changes here are needed to add new extensions.
# ---------------------------------------------------------------------------

file(GLOB_RECURSE _PYTHON_SRCS CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/gigl/csrc/**/python_*.cpp"
    "${CMAKE_SOURCE_DIR}/gigl/csrc/**/python_*.cu"
)

foreach(_py_src IN LISTS _PYTHON_SRCS)
    get_filename_component(_dir  "${_py_src}" DIRECTORY)
    get_filename_component(_stem "${_py_src}" NAME_WE)
    string(REGEX REPLACE "^python_" "" _name "${_stem}")

    set(_sources "${_py_src}")
    foreach(_ext IN ITEMS cpp cu)
        if(EXISTS "${_dir}/${_name}.${_ext}")
            list(APPEND _sources "${_dir}/${_name}.${_ext}")
        endif()
    endforeach()

    file(RELATIVE_PATH _rel_dir "${CMAKE_SOURCE_DIR}" "${_dir}")

    pybind11_add_module("${_name}" ${_sources})
    target_link_libraries("${_name}" PRIVATE "${TORCH_LIBRARIES}")
    target_compile_options("${_name}" PRIVATE ${GIGL_COMPILE_FLAGS})
    install(TARGETS "${_name}" DESTINATION "${_rel_dir}")
endforeach()
