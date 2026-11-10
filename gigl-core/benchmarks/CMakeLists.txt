# ---------------------------------------------------------------------------
# C++ kernel microbenchmarks (used by make benchmark_cpp)
# ---------------------------------------------------------------------------
# Kernels are compiled with the same optimization flags as the Python
# extension modules, so timings match the shipped wheel rather than the
# unoptimized unit-test build.
foreach(_prefix IN LISTS CMAKE_PREFIX_PATH)
  if(NOT TORCH_CMAKE_PREFIX AND EXISTS "${_prefix}/torch/share/cmake")
    set(TORCH_CMAKE_PREFIX "${_prefix}/torch/share/cmake")
  endif()
endforeach()
find_package(Torch REQUIRED PATHS "${TORCH_CMAKE_PREFIX}")

set(GIGL_BENCHMARK_COMPILE_FLAGS -O3 -g -Wall -Wextra -Wno-unused-parameter)

file(GLOB_RECURSE _BENCHMARK_KERNEL_SRCS "${CMAKE_SOURCE_DIR}/core/*.cpp")
list(FILTER _BENCHMARK_KERNEL_SRCS EXCLUDE REGEX ".*/python_[^/]*$")

add_library(gigl_core_benchmark_kernels STATIC ${_BENCHMARK_KERNEL_SRCS})
target_include_directories(gigl_core_benchmark_kernels
                           PUBLIC "${CMAKE_SOURCE_DIR}/core")
target_link_libraries(gigl_core_benchmark_kernels PUBLIC "${TORCH_LIBRARIES}")
target_compile_options(gigl_core_benchmark_kernels
                       PRIVATE ${GIGL_BENCHMARK_COMPILE_FLAGS})

# Any file named *_benchmark.cpp in this directory becomes its own binary.
file(GLOB_RECURSE BENCHMARK_SOURCES "*_benchmark.cpp")
foreach(benchmark_source ${BENCHMARK_SOURCES})
  get_filename_component(benchmark_name "${benchmark_source}" NAME_WE)
  add_executable(${benchmark_name} ${benchmark_source})
  target_link_libraries(${benchmark_name} gigl_core_benchmark_kernels)
  target_compile_options(${benchmark_name}
                         PRIVATE ${GIGL_BENCHMARK_COMPILE_FLAGS})
endforeach()
