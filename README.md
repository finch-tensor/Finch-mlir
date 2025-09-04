# This repository is archived, the current effort to productionize finch is [finch-tensor-lite](https://github.com/finch-tensor/finch-tensor-lite)


# An out-of-tree MLIR dialect

This is an example of an out-of-tree [MLIR](https://mlir.llvm.org/) dialect along with a finch `opt`-like tool to operate on that dialect.

## Install

0. Do a recursive clone
```sh
git clone --recursive https://github.com/finch-tensor/Finch-mlir.git
```

Or if you have already cloned this repository, run
```sh
git submodule init --update --recursive
```

This will take some time as it perform a shallow-clone of LLVM.

1. Install prerequisites to build the dialect and the Python bindings:

1a. Use your system package manager or `conda` to install a C/C++ compiler and CCache.

1b. Install the build-time Python requirements:

```sh
# Generate the build requirements dynamically from `pyproject.toml` and install them.
./scripts/install_build_reqs.sh
```

2. Build the wheel or install the package:

```sh
python -m build --no-isolation
```

OR

```sh
pip install --no-build-isolation .
```
In theory, the `--no-[build-]isolation` flag is unnecessary, but it caches the LLVM build which can greatly speed-up the buid process on repeated runs.

## Run Tests
One can run the tests once the wheel is built and/or the package is installed with

```sh
cmake --build build --target check-finch
```

Assuming the CMake configure step doesn't need to run.
