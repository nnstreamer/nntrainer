#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build public-API norm lifetime checkers against a matching Android core.

This builds only; it never deploys or accesses a device. The core must have been
built from the supplied repository with ENABLE_HEXKL and the selected FP16 ABI.
Stage each checker with that same libnntrainer.so and DSP skel. On an idle device,
run the checker with default, stats, and profile arguments in separate processes.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--ndk", type=Path, required=True)
    parser.add_argument("--core-lib-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api", type=int, default=29)
    parser.add_argument("--fp16-abi", choices=("__fp16", "_Float16", "disabled"),
                        default="__fp16")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = args.repo_root.resolve() if args.repo_root else here.parents[1]
    systems = {"Linux": "linux-x86_64", "Darwin": "darwin-x86_64", "Windows": "windows-x86_64"}
    toolchain = args.ndk.resolve() / "toolchains/llvm/prebuilt" / systems[platform.system()]
    compiler = toolchain / "bin" / ("clang++.exe" if platform.system() == "Windows" else "clang++")
    libraries = args.core_lib_dir.resolve()
    core = libraries / "libnntrainer.so"
    if not compiler.is_file() or not core.is_file():
        parser.error("NDK compiler and matching libnntrainer.so must exist")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    flags = ["--target=aarch64-linux-android" + str(args.api),
             "--sysroot=" + str(toolchain / "sysroot"), "-std=c++17", "-O2",
             "-DENABLE_HEXKL=1", "-Wall", "-Wextra", "-Werror",
             "-Wno-unused-parameter", "-Wno-ignored-qualifiers"]
    if args.fp16_abi != "disabled":
        flags += ["-DENABLE_FP16=1"]
        if args.fp16_abi == "__fp16":
            flags += ["-DUSE__FP16=1"]
    flags += ["-I" + str(p) for p in
              (root / "api/ccapi/include", root / "nntrainer/tensor",
               root / "nntrainer/tensor/cpu_backend",
               root / "nntrainer/tensor/htp_backend")]
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = {"core_sha256": sha(core), "fp16_abi": args.fp16_abi,
              "compute_ops_sha256": sha(root / "nntrainer/tensor/cpu_backend/compute_ops.h"),
              "checkers": {}}
    for name in ("qkv_norm_lifetime_device_check", "router_norm_lifetime_device_check"):
        source = here / (name + ".cpp")
        binary = output / name
        command = [str(compiler)] + flags + [str(source), "-L" + str(libraries),
                  "-lnntrainer", "-llog", "-landroid", "-Wl,--allow-shlib-undefined",
                  "-Wl,-z,max-page-size=16384", "-o", str(binary)]
        subprocess.run(command, check=True)
        report["checkers"][name] = {"source_sha256": sha(source),
                                    "binary_sha256": sha(binary), "command": command}
    (output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
