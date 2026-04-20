cmake_minimum_required(VERSION 3.18)
project(gigl_csrc CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enable CUDA only when the toolkit is present; allows the same CMakeLists.txt
# to build on CPU-only machines without requiring nvcc.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

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

find_package(Torch REQUIRED PATHS "${TORCH_CMAKE_PREFIX}")

# torch_python provides the pybind11 type casters for at::Tensor. It is not
# included in TORCH_LIBRARIES but is required for extensions that pass tensors
# across the Python/C++ boundary.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
        "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
    OUTPUT_VARIABLE TORCH_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_LIB_DIR}" REQUIRED)

# Preserve torch lib dir in the installed .so RPATH so the extension loads
# libtorch.so without requiring `import torch` to pre-populate the symbol table.
set(CMAKE_INSTALL_RPATH "${TORCH_LIB_DIR}")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

set(GIGL_COMPILE_FLAGS -O3 -Wall -Wextra -Wno-unused-parameter)

# ---------------------------------------------------------------------------
# Extension modules — auto-discovered.
# Files named python_*.cpp under gigl/csrc/ are compiled as pybind11 extension
# modules. The companion <name>.cpp (without the "python_" prefix) is included
# automatically when present. This mirrors the convention from the old
# build_cpp_extensions.py: no changes here are needed to add new extensions.
# ---------------------------------------------------------------------------

if(CMAKE_CUDA_COMPILER)
    file(GLOB_RECURSE _PYTHON_SRCS CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/gigl/csrc/**/python_*.cpp"
        "${CMAKE_SOURCE_DIR}/gigl/csrc/**/python_*.cu"
    )
else()
    file(GLOB_RECURSE _PYTHON_SRCS CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/gigl/csrc/**/python_*.cpp"
    )
endif()

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

    file(RELATIVE_PATH _rel_dir "${CMAKE_SOURCE_DIR}" "${_dir}")

    pybind11_add_module("${_name}" ${_sources})
    target_link_libraries("${_name}" PRIVATE "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}")
    target_compile_options("${_name}" PRIVATE ${GIGL_COMPILE_FLAGS})
    # TORCH_EXTENSION_NAME is used in PYBIND11_MODULE() to name the module.
    # PyTorch's own build system sets this; we must define it explicitly here.
    target_compile_definitions("${_name}" PRIVATE "TORCH_EXTENSION_NAME=${_name}")
    install(TARGETS "${_name}" DESTINATION "${_rel_dir}")
endforeach()
