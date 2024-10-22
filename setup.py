import os
import shutil
import sys
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

        # BUILD LLVM
        llvm_cmake_args = [
            "-G Ninja",
            f"-B{llvm_build_dir}",
            "-DLLVM_ENABLE_PROJECTS=mlir;llvm;clang;lld",
            "-DLLVM_TARGETS_TO_BUILD=Native",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
            f"-DPython3_EXECUTABLE={PYTHON_EXECUTABLE}",
            "-DLLVM_INSTALL_UTILS=ON",
        ]

        subprocess.run(
            ["cmake", ext.sourcedir, *llvm_cmake_args], cwd=llvm_build_dir, check=True,
        )
        subprocess.run(["ninja"], cwd=llvm_build_dir, check=True)

        # INSTALL LLVM
        subprocess.run(
            ["cmake", f"-DCMAKE_INSTALL_PREFIX={llvm_install_dir}", "-Pcmake_install.cmake"],
            cwd=llvm_build_dir,
            check=True,
        )

        # BUILD FINCH DIALECT
        dialect_cmake_args = [
            "-G Ninja",
            f"-B{finch_build_dir}",
            f"-DMLIR_DIR={llvm_install_dir}/lib/cmake/mlir",
            f"-DLLVM_EXTERNAL_LIT={llvm_build_dir}/bin/llvm-lit",
        ]

        subprocess.run(
            ["cmake", FINCH_MLIR_SOURCE_DIR, *dialect_cmake_args], cwd=finch_build_dir, check=True,
        )
        subprocess.run(["ninja"], cwd=finch_build_dir, check=True)

        # INSTALL FINCH DIALECT
        subprocess.run(
            ["cmake", f"-DCMAKE_INSTALL_PREFIX={install_dir}", "-Pcmake_install.cmake"],
            cwd=finch_build_dir,
            check=True,
        )

        # We need unversioned copies of `mlir_c_runner_utils` and `mlir_float16_utils`.
        install_dir_lib = install_dir / 'lib'
        files = [
            (f, f.replace(".20.0git", "")) for f in os.listdir(install_dir_lib) if f.startswith("libmlir_")
        ]
        for file, new_file in files:
            shutil.copy(install_dir_lib / file, install_dir_lib / new_file)


def create_dir(name: str) -> Path:
    path = Path.cwd() / "build" / name
    if not path.exists():
        path.mkdir(parents=True)
    return path


llvm_build_dir = create_dir("llvm-build")
llvm_install_dir = create_dir("llvm-install")
finch_build_dir = create_dir("finch-build")


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
