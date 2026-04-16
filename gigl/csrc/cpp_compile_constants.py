"""C++ compiler flags used by the pybind11 extension build and clangd compile commands generation."""

COMPILE_ARGS: list[str] = [
    "-O3",
    "-std=c++17",
    "-Wall",
    "-Wextra",
    "-Wno-unused-parameter",
]
