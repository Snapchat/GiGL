include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# ---------------------------------------------------------------------------
# GoogleTest via FetchContent
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        58d77fa8070e8cec2dc1ed015d66b454c8d78850  # v1.14.0
    GIT_SHALLOW    TRUE
)
# Prevent GoogleTest from overriding the compiler's runtime on Windows
# (no-op on Linux/Mac, but required for portable CMake config).
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Required for add_test() to register tests with CTest.
enable_testing()

# ---------------------------------------------------------------------------
# Torch + kernel library (required by tests that use C++ kernels)
# ---------------------------------------------------------------------------
foreach(_prefix IN LISTS CMAKE_PREFIX_PATH)
    if(NOT TORCH_CMAKE_PREFIX AND EXISTS "${_prefix}/torch/share/cmake")
        set(TORCH_CMAKE_PREFIX "${_prefix}/torch/share/cmake")
    endif()
endforeach()
find_package(Torch REQUIRED PATHS "${TORCH_CMAKE_PREFIX}")

# Auto-discover all kernel sources under core/. python_*.cpp are pybind11
# extension entry points that belong to the wheel, not to the test library.
if(CMAKE_CUDA_COMPILER)
    file(GLOB_RECURSE _KERNEL_SRCS "${CMAKE_SOURCE_DIR}/core/*.cpp" "${CMAKE_SOURCE_DIR}/core/*.cu")
else()
    file(GLOB_RECURSE _KERNEL_SRCS "${CMAKE_SOURCE_DIR}/core/*.cpp")
endif()
list(FILTER _KERNEL_SRCS EXCLUDE REGEX ".*/python_[^/]*$")

add_library(gigl_core_kernels STATIC ${_KERNEL_SRCS})
target_include_directories(gigl_core_kernels PUBLIC "${CMAKE_SOURCE_DIR}/core")
# glt_headers is defined once in the top-level CMakeLists.txt (which this file is a subdirectory of),
# so there is no second GLT acquisition here. PUBLIC so test binaries that include GLT headers
# themselves inherit the include path.
target_link_libraries(gigl_core_kernels PUBLIC glt_headers "${TORCH_LIBRARIES}")

# ---------------------------------------------------------------------------
# GLT's ShmQueue implementation, for tests only.
#
# The shipped extensions consume GLT headers only. Tests additionally need GLT's *compiled* ShmQueue so
# they can enqueue and dequeue against a real queue rather than asserting on a hand-built struct.
# Compiling one .cc here (rather than in the wheel) keeps a second copy of GLT's symbols out of the
# shipped .so.
#
# WITH_CUDA is deliberately left undefined: csrc/shm_queue.cc only reaches for cuda_runtime.h and
# common.cuh under that macro, and PinMemory() becomes a no-op, which is all tests need.
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)
add_library(glt_shm_queue STATIC
    "${GIGL_GLT_RESOLVED_SOURCE_DIR}/graphlearn_torch/csrc/shm_queue.cc"
)
target_link_libraries(glt_shm_queue PUBLIC glt_headers Threads::Threads)

# ---------------------------------------------------------------------------
# Auto-discover test targets
# ---------------------------------------------------------------------------
# Any file named *_test.cpp in this directory (or subdirectories) is
# automatically compiled into its own test binary and registered with CTest.
# To add a new test suite, drop a *_test.cpp file here — no changes to this
# file required.  This matches the *_test.py convention used for Python tests.
if(CMAKE_CUDA_COMPILER)
    file(GLOB_RECURSE TEST_SOURCES "*_test.cpp" "*_test.cu")
else()
    file(GLOB_RECURSE TEST_SOURCES "*_test.cpp")
endif()

foreach(test_source ${TEST_SOURCES})
    # Derive a unique binary name from the path relative to this directory, e.g.:
    #   foo_test.cpp         → foo_test
    #   sampling/foo_test.cpp → sampling_foo_test
    file(RELATIVE_PATH _rel "${CMAKE_CURRENT_SOURCE_DIR}" "${test_source}")
    string(REPLACE "/" "_" test_name "${_rel}")
    string(REGEX REPLACE "\\.[^.]+$" "" test_name "${test_name}")
    add_executable(${test_name} ${test_source})
    # glt_shm_queue is linked into every test binary rather than special-cased for the one test that
    # needs it, preserving the "drop in a *_test.cpp and it just builds" convention. It is a static
    # library, so tests that do not reference ShmQueue pull in nothing.
    target_link_libraries(${test_name} GTest::gtest_main gigl_core_kernels glt_shm_queue)
    # add_test registers the binary with CTest. Each *_test binary is one
    # CTest entry; GoogleTest itself reports individual TEST() results inside it.
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
