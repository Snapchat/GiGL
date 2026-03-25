cmake_minimum_required(VERSION 3.18)
project(GiGLCppTests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# GoogleTest via FetchContent
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    # TODO: pin URL_HASH once infrastructure is validated, e.g.:
    #   URL_HASH SHA256=<hash>
)
# Prevent GoogleTest from overriding the compiler's runtime on Windows
# (no-op on Linux/Mac, but required for portable CMake config).
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Required for add_test() to register tests with CTest.
enable_testing()

# ---------------------------------------------------------------------------
# Auto-discover test targets
# ---------------------------------------------------------------------------
# Any file named *_test.cpp in this directory (or subdirectories) is
# automatically compiled into its own test binary and registered with CTest.
# To add a new test suite, drop a *_test.cpp file here — no changes to this
# file required.  This matches the *_test.py convention used for Python tests.
file(GLOB_RECURSE TEST_SOURCES "*_test.cpp")

foreach(test_source ${TEST_SOURCES})
    # Derive the binary name from the filename, e.g.:
    #   ppr_forward_push_test.cpp → ppr_forward_push_test
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} GTest::gtest_main)
    # add_test registers the binary with CTest. Each *_test binary is one
    # CTest entry; GoogleTest itself reports individual TEST() results inside it.
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
