import os
import shutil
import sys
import subprocess
from pathlib import Path

import ninja
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


PYTHON_EXECUTABLE = str(Path(sys.executable))


class CMakeExtension(Extension):
    def __init__(
        self,
        name: str,
        llvm_source_dir: str,
        finch_source_dir: str,
    ) -> None:
        super().__init__(name, sources=[])
        self.llvm_source_dir = os.fspath(Path(llvm_source_dir).resolve())
        self.finch_source_dir = os.fspath(Path(finch_source_dir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()
        install_dir = extdir
        ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"

        extra_flags = []
        if sys.platform.startswith("darwin"):
            extra_flags.append("-DCMAKE_OSX_DEPLOYMENT_TARGET=11.0")

        # BUILD LLVM
        llvm_cmake_args = [
            "-G Ninja",
            f"-B{llvm_build_dir}",
            "-DLLVM_ENABLE_PROJECTS=mlir",
            "-DLLVM_TARGETS_TO_BUILD=Native",
            "-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
            f"-DPython3_EXECUTABLE={PYTHON_EXECUTABLE}",
            "-DLLVM_INSTALL_UTILS=ON",
            "-DLLVM_CCACHE_BUILD=ON",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON",
            f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
            *extra_flags,
        ]

        subprocess.run(
            ["cmake", ext.llvm_source_dir, *llvm_cmake_args], cwd=llvm_build_dir, check=True,
        )
        subprocess.run([ninja_executable_path], cwd=llvm_build_dir, check=True)

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
            "-DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON",
            f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
            *extra_flags,
        ]

        subprocess.run(
            ["cmake", ext.finch_source_dir, *dialect_cmake_args], cwd=finch_build_dir, check=True,
        )
        subprocess.run([ninja_executable_path], cwd=finch_build_dir, check=True)

        # INSTALL FINCH DIALECT
        subprocess.run(
            ["cmake", f"-DCMAKE_INSTALL_PREFIX={install_dir}", "-Pcmake_install.cmake"],
            cwd=finch_build_dir,
            check=True,
        )

        # Move Python package out of nested directories.
        python_package_dir = install_dir / "python_packages" / "finch" / "mlir_finch"
        shutil.copytree(python_package_dir, install_dir / "mlir_finch")
        shutil.rmtree(install_dir / "python_packages")


def create_dir(name: str) -> Path:
    path = Path.cwd() / "build" / name
    if not path.exists():
        path.mkdir(parents=True)
    return path


llvm_build_dir = create_dir("llvm-build")
llvm_install_dir = create_dir("llvm-install")
finch_build_dir = create_dir("finch-build")


setup(
    name="finch-mlir",
    version="0.0.1",
    include_package_data=True,
    description="Finch MLIR distribution as wheel.",
    long_description="Finch MLIR distribution as wheel.",
    long_description_content_type="text/markdown",
    ext_modules=[CMakeExtension(
        "mlir_finch_ext",
        llvm_source_dir=f"./llvm-project/llvm",
        finch_source_dir="./Finch-mlir",
    )],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
