# motion_correction

Standalone `correct_motion` implementation packaged as a small C++ motion trajectory correction library with Python bindings.

## Installation Guide

### Prerequisites

Ensure you have a C++17 compatible compiler (GCC 7.0+, Clang 5.0+, or MSVC 2017+) and CMake 3.15+. On Windows, install MinGW-w64 or Visual Studio with C++ tools. On Linux, install `build-essential` and `cmake`.

This bundled build never downloads source. Install CMake, pybind11 and Eigen3 first. Example: `python -m pip install "cmake>=3.15" "pybind11>=2.11.1"`; install Eigen3 with your system package manager (e.g. libeigen3-dev on Debian/Ubuntu), or point CMAKE_PREFIX_PATH to an existing Eigen installation. Missing dependencies produce a build error rather than a Git clone. Use `python -m pip install --no-deps --no-build-isolation ./native/MotionCorrection` from the server directory. This extension is optional; normal motion inference works with post_processing=false.

### Build & Install

#### Standard Installation
```bash
pip install .
```

#### Development Installation
```bash
pip install -e .
```

### Verify Installation

```python
import motion_correction
print("Installation successful!")
```
You can also run `python run_test.py` for a simple test.
