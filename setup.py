import sys, os
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


LLVM_SOURCE_DIR = os.environ["LLVM_SOURCE_DIR"]
FINCH_MLIR_SOURCE_DIR = os.environ["FINCH_MLIR_SOURCE_DIR"]
PYTHON_EXECUTABLE = str(Path(sys.executable))


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()
        install_dir = extdir

        cmake_args = [
            "-G Ninja",
            f"-B{build_temp}",
            "-DLLVM_ENABLE_PROJECTS=mlir;llvm;clang;lld",
            "-DLLVM_TARGETS_TO_BUILD=Native",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            "-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
            f"-DPython3_EXECUTABLE={PYTHON_EXECUTABLE}",
            "-DLLVM_INSTALL_UTILS=ON",
            f"-DMLIR_BINARY_DIR={build_temp}",
            f"-DCMAKE_MODULE_PATH={LLVM_SOURCE_DIR}/mlir/cmake/modules",
            "-DLLVM_EXTERNAL_PROJECTS=finch-dialect",
            f"-DLLVM_EXTERNAL_FINCH_DIALECT_SOURCE_DIR={FINCH_MLIR_SOURCE_DIR}",
        ]

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True,
        )

        subprocess.run(
            ["cmake", "--build", ".", "--target", "install"],
            cwd=build_temp,
            check=True,
        )


build_temp = Path.cwd() / "build" / "temp"
if not build_temp.exists():
    build_temp.mkdir(parents=True)

setup(
    name="mlir_finch",
    version="0.0.1",
    include_package_data=True,
    description="MLIR Finch distribution as wheel.",
    long_description="MLIR Finch distribution as wheel.",
    long_description_content_type="text/markdown",
    ext_modules=[CMakeExtension("mlir_finch_ext", sourcedir=f"{LLVM_SOURCE_DIR}/llvm")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
