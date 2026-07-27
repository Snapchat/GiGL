# C++ build, test, format, and clean targets for gigl-core.
# Invoked from the GiGL repo root via: $(MAKE) -C gigl-core <target>
# All paths below are relative to gigl-core/.

CPP_SOURCES := $(shell find core \( -name "*.cpp" -o -name "*.cu" \) 2>/dev/null)
# clang-tidy 15 does not fully support CUDA syntax (e.g. <<<...>>>, __global__).
# Exclude .cu files from tidy targets; clang-format and clangd handle them fine.
CPP_SOURCES_NO_CUDA := $(filter-out %.cu,$(CPP_SOURCES))
# Formatting covers headers as well as definitions, and tests/ as well as core/ — a test file that is
# never format-checked drifts silently.
# Headers are deliberately kept out of CPP_SOURCES (and so out of the lint targets): those drive
# clangd/clang-tidy from compile_commands.json, which lists only translation units, so linting a bare
# header would fall back to a guessed compile command and report spurious diagnostics.
CPP_FORMAT_SOURCES := $(shell find core tests \( -name "*.cpp" -o -name "*.cu" -o -name "*.h" -o -name "*.cuh" \) 2>/dev/null)

# Stamp-file guard: uv pip install -e gigl-core/ triggers a full CMake configure-and-build
# cycle even when nothing changed. By making the stamp file depend on C++ sources and
# pyproject.toml, make skips the reinstall unless something actually changed.
# We cd to the repo root so that no-build-isolation-package in the root pyproject.toml
# is respected by uv pip install.
# GLT_PIN.cmake is a prerequisite because it selects which graphlearn_torch headers the extensions
# compile against. Omitting it would let a pin bump leave a stale .so in place, whose compiled field
# offsets no longer match the installed GLT wheel's shared-memory layout.
.cache/cmake_build/CMakeInit.txt: $(shell find core \( -name '*.cpp' -o -name '*.cu' -o -name '*.h' -o -name '*.cuh' \) 2>/dev/null) CMakeLists.txt GLT_PIN.cmake pyproject.toml
	cd $(abspath $(CURDIR)/..) && uv pip install -e gigl-core/

# Also depend on the test cmake configure so that .cache/cpp_tests/compile_commands.json
# is generated automatically, giving clangd visibility into test files and gtest headers.
build_cpp_extensions: .cache/cmake_build/CMakeInit.txt .cache/cpp_tests/.configured

.cache/cpp_tests/.configured: CMakeLists.txt tests/CMakeLists.txt GLT_PIN.cmake .cache/cmake_build/CMakeInit.txt
	cmake -C .cache/cmake_build/CMakeInit.txt -S . -B .cache/cpp_tests -DGIGL_CORE_BUILD_TESTS=ON
	touch .cache/cpp_tests/.configured

unit_test_cpp: .cache/cpp_tests/.configured
	cmake --build .cache/cpp_tests --parallel
	ctest --test-dir .cache/cpp_tests --output-on-failure

check_format_cpp:
	clang-format-15 --dry-run --Werror --style=file $(CPP_FORMAT_SOURCES)

format_cpp:
	clang-format-15 -i --style=file $(CPP_FORMAT_SOURCES)

# Not part of `make format`: clang-tidy --fix rewrites logic (renames identifiers,
# changes expressions, adds/removes keywords), not just style. Run manually and
# review the diff before committing. Note: --fix cannot auto-repair every check;
# some violations require manual edits.
# --extra-arg=-Wno-ignored-optimization-argument suppresses GCC-specific LTO flags
# (-fno-fat-lto-objects, -flto=auto) that cmake writes into compile_commands.json.
# clang-tidy forwards compiler warnings via clang-diagnostic-*, and .clang-tidy sets
# WarningsAsErrors: '*', so the warning must be silenced at the compiler level before
# clang-tidy ever sees it.
check_lint_cpp: build_cpp_extensions
	cd $(abspath $(CURDIR)/..) && uv run python gigl-core/scripts/run_cpp_lint.py $(addprefix gigl-core/,$(CPP_SOURCES_NO_CUDA))

fix_lint_cpp: build_cpp_extensions
	clang-tidy-15 --fix --extra-arg=-Wno-ignored-optimization-argument -p .cache/cmake_build/compile_commands.json $(CPP_SOURCES_NO_CUDA)

# Wipe cmake build caches. Use this if cmake's cached state becomes inconsistent
# after switching between branches with substantially different CMakeLists.txt structure.
# This also drops each tree's _deps/, so FetchContent re-fetches GLT and googletest from the pin.
clean_cpp:
	rm -rf .cache/cpp_tests .cache/cmake_build

clean_build_files_cpp:
	rm -rf .cache/cpp_tests

# Declare targets as phony so make always runs their recipes, even if a file or
# directory with the same name happens to exist on disk.
.PHONY: build_cpp_extensions unit_test_cpp check_format_cpp format_cpp check_lint_cpp fix_lint_cpp clean_cpp clean_build_files_cpp
