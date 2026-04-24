# GoogleTest via FetchContent. Pinned to v1.14.0 (stable, C++17 compatible).
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

# Any file named *_test.cpp (or *_test.cu when CUDA is enabled) in this
# directory is compiled into its own CTest binary. To add a new test suite,
# drop a *_test.cpp file here — no changes to this file required.
if(CMAKE_CUDA_COMPILER)
    file(GLOB_RECURSE TEST_SOURCES "*_test.cpp" "*_test.cu")
else()
    file(GLOB_RECURSE TEST_SOURCES "*_test.cpp")
endif()

foreach(test_source ${TEST_SOURCES})
    # Derive a unique binary name from the relative path, e.g.:
    #   add_one_test.cpp              → add_one_test
    #   sampling/ppr_forward_push_test.cpp → sampling_ppr_forward_push_test
    file(RELATIVE_PATH _rel "${CMAKE_CURRENT_SOURCE_DIR}" "${test_source}")
    string(REPLACE "/" "_" test_name "${_rel}")
    string(REGEX REPLACE "\\.[^.]+$" "" test_name "${test_name}")
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} GTest::gtest_main)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
