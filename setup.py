import os
import platform
import shutil
import sys
import subprocess
from pathlib import Path

import ninja
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


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

        # p1 = os.fspath(Path("/tmp/st").absolute())
        # shutil.move(ext.llvm_source_dir, p1)
        # ext.llvm_source_dir = os.fspath(Path("/tmp/st/llvm").absolute())
        # p2 = os.fspath(Path("/tmp/sf").absolute())
        # shutil.move(ext.finch_source_dir, p2)
        # ext.finch_source_dir = p2

        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()
        install_dir = extdir
        ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
        PYTHON_EXECUTABLE = str(Path(sys.executable))


        extra_flags = []
        if sys.platform.startswith("darwin"):
            extra_flags.append("-DCMAKE_OSX_DEPLOYMENT_TARGET=11.0")
        elif platform.system() == "Windows":
            extra_flags += [
                "-DCMAKE_C_COMPILER=cl",
                "-DCMAKE_CXX_COMPILER=cl",
                "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded",
                "-DCMAKE_C_FLAGS=/MT",
                "-DCMAKE_CXX_FLAGS=/MT",
            ]

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
            "-DLLVM_ENABLE_ZLIB=OFF",
            "-DLLVM_ENABLE_ZSTD=OFF",
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

        llvm_lit = "llvm-lit.py" if platform.system() == "Windows" else "llvm-lit"

        # subprocess.run(
        #     ["dir", llvm_build_dir / 'bin'],
        #     cwd=finch_build_dir,
        #     check=True,
        # )

        # subprocess.run(
        #     ["dir", str(llvm_build_dir)],
        #     cwd=finch_build_dir,
        #     check=True,
        # )

        # if platform.system() == "Windows":
        #     # fatal error LNK1170: line in command file contains 131071 or more characters
        #     if Path("/tmp/m").exists():
        #         shutil.rmtree("/tmp/m")
        #     shutil.move(llvm_install_dir, "/tmp/m")
        #     llvm_install_dir = Path("/tmp/m").absolute()

        # BUILD FINCH DIALECT
        dialect_cmake_args = [
            "-G Ninja",
            f"-B{finch_build_dir}",
            f"-DMLIR_DIR={llvm_install_dir / 'lib' / 'cmake' / 'mlir'}",
            f"-DLLVM_EXTERNAL_LIT={llvm_build_dir / 'bin' / llvm_lit}",
            "-DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON",
            f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
            "-DLLVM_ENABLE_ZLIB=OFF",
            "-DLLVM_ENABLE_ZSTD=OFF",
            f"-DPython3_EXECUTABLE={PYTHON_EXECUTABLE}",
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

        subprocess.run(
            [
                "find",
                ".",
                "-exec",
                "touch",
                "-a",
                "-m",
                "-t",
                "197001010000",
                "{}",
                ";",
            ],
            cwd=install_dir,
            check=False,
        )


def create_dir(name: str) -> Path:
    path = Path.cwd() / "build" / name  # Path("/tmp").absolute()
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
        llvm_source_dir=f"./llvm-project/llvm", # /llvm
        finch_source_dir="./Finch-mlir",
    )],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
