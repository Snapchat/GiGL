# C++ build, test, format, and clean targets for gigl-core.
# Invoked from the GiGL repo root via: $(MAKE) -C gigl-core <target>
# All paths below are relative to gigl-core/.

CPP_SOURCES := $(shell find csrc \( -name "*.cpp" -o -name "*.cu" \) 2>/dev/null)
# clang-tidy 15 does not fully support CUDA syntax (e.g. <<<...>>>, __global__).
# Exclude .cu files from tidy targets; clang-format and clangd handle them fine.
CPP_SOURCES_NO_CUDA := $(filter-out %.cu,$(CPP_SOURCES))

# Stamp-file guard: uv pip install -e gigl-core/ triggers a full CMake configure-and-build
# cycle even when nothing changed. By making the stamp file depend on C++ sources and
# pyproject.toml, make skips the reinstall unless something actually changed.
# We cd to the repo root so that no-build-isolation-package in the root pyproject.toml
# is respected by uv pip install.
.cache/cmake_build/CMakeInit.txt: $(shell find csrc \( -name '*.cpp' -o -name '*.cu' -o -name '*.h' -o -name '*.cuh' \) 2>/dev/null) CMakeLists.txt pyproject.toml
	cd $(abspath $(CURDIR)/..) && uv pip install -e gigl-core/

build_cpp_extensions: .cache/cmake_build/CMakeInit.txt

.cache/cpp_tests/.configured: CMakeLists.txt tests/CMakeLists.txt .cache/cmake_build/CMakeInit.txt
	cmake -C .cache/cmake_build/CMakeInit.txt -S . -B .cache/cpp_tests -DGIGL_CORE_BUILD_TESTS=ON
	touch .cache/cpp_tests/.configured

unit_test_cpp: .cache/cpp_tests/.configured
	cmake --build .cache/cpp_tests --parallel
	ctest --test-dir .cache/cpp_tests --output-on-failure

# TODO: Remove the $(if ...) guards once C++ source files are permanently present in the
# repo. The guards exist to silently no-op on branches that have no python_*.cpp files yet.
check_format_cpp:
	$(if $(CPP_SOURCES),clang-format-15 --dry-run --Werror --style=file $(CPP_SOURCES))

format_cpp:
	$(if $(CPP_SOURCES),clang-format-15 -i --style=file $(CPP_SOURCES))

# Wipe cmake build caches. Use this if cmake's cached state becomes inconsistent
# after switching between branches with substantially different CMakeLists.txt structure.
clean_cpp:
	rm -rf .cache/cpp_tests .cache/cmake_build

clean_build_files_cpp:
	rm -rf .cache/cpp_tests

# Declare targets as phony so make always runs their recipes, even if a file or
# directory with the same name happens to exist on disk.
.PHONY: build_cpp_extensions unit_test_cpp check_format_cpp format_cpp clean_cpp clean_build_files_cpp
