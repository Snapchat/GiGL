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
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE
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
if(CMAKE_CUDA_COMPILER)
    file(GLOB_RECURSE TEST_SOURCES "*_test.cpp" "*_test.cu")
else()
    file(GLOB_RECURSE TEST_SOURCES "*_test.cpp")
endif()

foreach(test_source ${TEST_SOURCES})
    # Derive a unique binary name from the path relative to this directory, e.g.:
    #   ppr_forward_push_test.cpp       → ppr_forward_push_test
    #   sampling/ppr_forward_push_test.cpp → sampling_ppr_forward_push_test
    file(RELATIVE_PATH _rel "${CMAKE_CURRENT_SOURCE_DIR}" "${test_source}")
    string(REPLACE "/" "_" test_name "${_rel}")
    string(REGEX REPLACE "\\.[^.]+$" "" test_name "${test_name}")
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} GTest::gtest_main)
    # add_test registers the binary with CTest. Each *_test binary is one
    # CTest entry; GoogleTest itself reports individual TEST() results inside it.
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
