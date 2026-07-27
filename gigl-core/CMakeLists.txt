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
# Files named python_*.cpp under core/ are compiled as pybind11 extension
# modules. The companion <name>.cpp (without the "python_" prefix) is included
# automatically when present. Add a new extension by dropping source files
# here; no changes to this CMakeLists.txt are needed.
# ---------------------------------------------------------------------------
if(CMAKE_CUDA_COMPILER)
    file(GLOB_RECURSE _PYTHON_SRCS CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/core/python_*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/core/python_*.cu"
    )
else()
    file(GLOB_RECURSE _PYTHON_SRCS CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/core/python_*.cpp"
    )
endif()

# Declared here, before the GLT block below, because that block must run for the test build too
# (tests/ compiles the same core/ sources into gigl_core_kernels).
option(GIGL_CORE_BUILD_TESTS "Build C++ unit tests (used by make unit_test_cpp)" OFF)

# ---------------------------------------------------------------------------
# GraphLearn-for-PyTorch (GLT) headers.
#
# gigl-core compiles against GLT's C++ headers, but GLT's installed wheel ships only its Python tree
# (its setup.py sets package_dir={'graphlearn_torch': 'graphlearn_torch/python'}), so there are no
# headers in site-packages to point at. Nor can we discover them from the installed package:
# requirements/install_py_deps.sh builds gigl-core *before* it installs GLT, so GLT is not even
# importable at this point. We therefore fetch the pinned source ourselves.
#
# The commit comes from GLT_PIN.cmake and must match the commit gigl/scripts/install_glt.sh builds the
# wheel from -- tests/config_tests/dep_vars_check.py enforces that, in `make precondition_tests`.
#
# Guarded so the metadata-only image builds (containers/Dockerfile.{cpu,cuda,dataflow}.base and
# Dockerfile.builder copy only pyproject.toml/CMakeLists.txt/GLT_PIN.cmake/README.md, no core/) do not
# pay for a download they cannot use.
# ---------------------------------------------------------------------------
if(_PYTHON_SRCS OR GIGL_CORE_BUILD_TESTS)
    include("${CMAKE_CURRENT_SOURCE_DIR}/GLT_PIN.cmake")

    # Escape hatch for offline/air-gapped builds and for iterating against a locally modified GLT.
    if(NOT GIGL_GLT_SOURCE_DIR AND DEFINED ENV{GIGL_GLT_SOURCE_DIR})
        set(GIGL_GLT_SOURCE_DIR "$ENV{GIGL_GLT_SOURCE_DIR}")
    endif()

    if(GIGL_GLT_SOURCE_DIR)
        set(_glt_source_dir "${GIGL_GLT_SOURCE_DIR}")
        set(_glt_is_override TRUE)
        message(STATUS "GLT: using local checkout ${_glt_source_dir} (GIGL_GLT_SOURCE_DIR override)")
    else()
        # FetchContent keys _deps off CMAKE_BINARY_DIR, so the wheel build (.cache/cmake_build) and the
        # test build (.cache/cpp_tests) each populate their own copy of GLT.
        #
        # Do NOT "fix" this by pointing both at a shared FETCHCONTENT_BASE_DIR. That also shares
        # glt_src-subbuild, which records the generator, and the two trees use different ones:
        # scikit-build-core configures the wheel build with Ninja while gigl-core/Makefile configures
        # the test build with Unix Makefiles. The second configure then dies with
        # "generator : Unix Makefiles Does not match the generator used previously: Ninja".
        #
        # The duplication costs one extra shallow clone at configure time and nothing on rebuilds
        # (each tree caches its own populated source under .cache/, which is gitignored). Sharing would
        # not improve availability anyway: a cache only helps a machine that already fetched
        # successfully, and fresh CI runners, container builds, and the release wheel build all start
        # empty. Integrity comes from git's content-addressing below, not from any cache.
        include(FetchContent)

        FetchContent_Declare(
            glt_src
            GIT_REPOSITORY ${GIGL_GLT_GIT_REPOSITORY}
            GIT_TAG        ${GIGL_GLT_COMMIT}
            # Deliberately NOT GIT_SHALLOW. CMake does not fetch GIGL_GLT_COMMIT directly: the script
            # it generates runs `git clone --no-checkout --depth 1 --no-single-branch` and then
            # `git checkout <commit>`. With --depth 1 the clone contains only current ref *tips*, so
            # the checkout succeeds only while the pin happens to be one. GIGL_GLT_COMMIT is a fixed
            # historical commit, so as soon as upstream advances the branch it was the tip of, every
            # fresh configure would fail with "Failed to checkout tag". Verified: a shallow clone of
            # this repo cannot even resolve `<commit>^`. A full clone of GLT is a few MB and happens
            # once per build tree, which is the right trade for not breaking on upstream activity.
            # GLT's own third_party/ submodules are not needed: we consume its headers, and the one
            # .cc compiled into test binaries (csrc/shm_queue.cc) has no submodule dependencies.
            GIT_SUBMODULES ""
            # GLT's repo root has its own CMakeLists.txt which defaults WITH_CUDA=ON and
            # BUILD_TESTS=ON; letting MakeAvailable add_subdirectory() it would drag GLT's entire CUDA
            # build into ours. Pointing SOURCE_SUBDIR at a path that does not exist makes
            # MakeAvailable populate the sources without configuring them. (Using the deprecated
            # single-argument FetchContent_Populate() instead would warn under CMP0169.)
            SOURCE_SUBDIR  gigl-does-not-build-glt
        )
        FetchContent_MakeAvailable(glt_src)
        # The populated tree is the GLT repo root, which is the include root GLT itself uses (its
        # build_glt.py appends root_path, and its headers self-include as
        # "graphlearn_torch/include/...").
        set(_glt_source_dir "${glt_src_SOURCE_DIR}")
        set(_glt_is_override FALSE)
        message(STATUS "GLT: fetched commit ${GIGL_GLT_COMMIT} into ${_glt_source_dir}")
    endif()

    # Validate whichever tree we ended up with. For a git fetch this is belt-and-braces on top of
    # git's object verification; for an override it is the only check there is.
    set(_glt_header "${_glt_source_dir}/${GIGL_GLT_SHM_QUEUE_H_RELPATH}")
    if(NOT EXISTS "${_glt_header}")
        message(FATAL_ERROR
            "GLT source at ${_glt_source_dir} does not contain ${GIGL_GLT_SHM_QUEUE_H_RELPATH}. "
            "If GIGL_GLT_SOURCE_DIR is set, point it at the root of a graphlearn-for-pytorch checkout.")
    endif()
    # tests/CMakeLists.txt compiles this one GLT translation unit into the test binaries. Checked here
    # so a partial checkout fails with an actionable message instead of CMake's much later and more
    # cryptic "No SOURCES given to target: glt_shm_queue" at generate time.
    if(GIGL_CORE_BUILD_TESTS AND NOT EXISTS "${_glt_source_dir}/graphlearn_torch/csrc/shm_queue.cc")
        message(FATAL_ERROR
            "GLT source at ${_glt_source_dir} is missing graphlearn_torch/csrc/shm_queue.cc, which the "
            "C++ tests compile. Point GIGL_GLT_SOURCE_DIR at a complete graphlearn-for-pytorch "
            "checkout, or unset it to fetch the pinned commit.")
    endif()

    file(SHA256 "${_glt_header}" _glt_header_sha256)
    if(NOT _glt_header_sha256 STREQUAL GIGL_GLT_SHM_QUEUE_H_SHA256)
        set(_glt_drift_message
            "GLT header drift: ${_glt_header}\n"
            "  expected SHA256 ${GIGL_GLT_SHM_QUEUE_H_SHA256} (commit ${GIGL_GLT_COMMIT})\n"
            "  actual   SHA256 ${_glt_header_sha256}\n"
            "gigl-core reads graphlearn_torch::ShmQueueMeta's field offsets out of a shared-memory "
            "segment created by the *installed* GLT wheel. Compiling against a different layout "
            "produces silently wrong queue sizes -- no crash, no error.")
        if(_glt_is_override)
            string(APPEND _glt_drift_message
                "\nEither check out ${GIGL_GLT_COMMIT}, or unset GIGL_GLT_SOURCE_DIR to fetch the "
                "pinned commit.")
        else()
            string(APPEND _glt_drift_message
                "\nThis is the fetched tree, so either GIGL_GLT_SHM_QUEUE_H_SHA256 in GLT_PIN.cmake is "
                "stale relative to GIGL_GLT_COMMIT, or .cache/_deps holds a stale checkout -- try "
                "`make clean_cpp`.")
        endif()
        if(GIGL_ALLOW_UNPINNED_GLT)
            message(WARNING
                ${_glt_drift_message}
                "\nProceeding anyway because GIGL_ALLOW_UNPINNED_GLT=ON. Do not ship this build.")
        else()
            message(FATAL_ERROR
                ${_glt_drift_message}
                "\nTo override deliberately (e.g. while developing an upstream GLT patch), configure "
                "with -DGIGL_ALLOW_UNPINNED_GLT=ON.")
        endif()
    endif()

    add_library(glt_headers INTERFACE)
    # SYSTEM is a correctness requirement, not warning hygiene: shm_queue.h forward-declares
    # `struct ShmQueueMeta` and then defines `class ShmQueueMeta`, which clang rejects as a hard error
    # under our -Wall -Wextra plus -Werror via -Wmismatched-tags. It also keeps .clang-tidy's
    # WarningsAsErrors: '*' from firing on GLT's headers.
    target_include_directories(glt_headers SYSTEM INTERFACE "${_glt_source_dir}")

    # Exported for tests/CMakeLists.txt, which compiles a GLT .cc (csrc/shm_queue.cc) into the test
    # binaries so they can drive a real ShmQueue. Only tests do this: the shipped extensions use GLT
    # headers only, so they never duplicate GLT's compiled symbols.
    set(GIGL_GLT_RESOLVED_SOURCE_DIR "${_glt_source_dir}")
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

    # pybind11 is provided by the scikit-build-core isolated build env when building a
    # wheel. For direct cmake invocations (make unit_test_cpp), fall back to FetchContent
    # so pybind11 does not need to be pre-installed in the dev venv.
    if(pybind11_DIR)
        find_package(pybind11 CONFIG REQUIRED)
    else()
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v2.13.6
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(pybind11)
    endif()

    if(NOT TORCH_CMAKE_PREFIX)
        message(FATAL_ERROR "Cannot find torch cmake config in CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}. Run: make install_dev_deps")
    endif()

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
        # glt_headers is an INTERFACE target: its include directory only reaches sources that link it,
        # so every module must be linked against it explicitly.
        target_link_libraries("${_name}" PRIVATE glt_headers "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}")
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

# GIGL_CORE_BUILD_TESTS is declared above, before the GLT block that the test build also needs.
if(GIGL_CORE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
