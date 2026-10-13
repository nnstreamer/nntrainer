#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check the actual compact heap reader and unchanged compact arena reader.

Optional --baseline-root compares the selected reader against an immutable
earlier header, included directly under a separate namespace. No implementation
is mirrored in the fixture. O_DIRECT file checks run on Linux and Android.
"""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = args.repo_root.resolve() if args.repo_root else here.parents[2]
    backend = root / "nntrainer/tensor/htp_backend"
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    flags = ["-std=c++17", "-O1" if args.sanitize else "-O2", "-Wall",
             "-Wextra", "-Werror", "-pthread", "-I" + str(backend),
             "-I" + str(root / "nntrainer/utils")]
    env = os.environ.copy()
    if args.sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g1"]
        env["ASAN_OPTIONS"] = env.get("ASAN_OPTIONS", "") + ":halt_on_error=1:detect_leaks=1"
        env["UBSAN_OPTIONS"] = env.get("UBSAN_OPTIONS", "") + ":halt_on_error=1"
    with tempfile.TemporaryDirectory(prefix="nntr-qs2-inplace-") as path:
        temporary = Path(path)
        baseline_flags = []
        if args.baseline_root:
            original = (args.baseline_root.resolve() /
                        "nntrainer/tensor/htp_backend/htp_qs2_reader.h")
            if not original.is_file():
                raise ValueError("Missing baseline reader: " + str(original))
            wrapper = temporary / "baseline_reader.h"
            wrapper.write_text(
                "#define nntrainer nntrainer_qs2_baseline\n"
                '#include "' + str(original) + '"\n'
                "#undef nntrainer\n"
                "#undef __NNTRAINER_HTP_QS2_READER_H__\n")
            baseline_flags = ["-include", str(wrapper), "-DNNTR_QS2_BASELINE=1"]
        for name in ("qs2_inplace_reader_check", "qs2_reader_check"):
            binary = temporary / name
            extra = baseline_flags if name == "qs2_inplace_reader_check" else []
            subprocess.run(compiler + flags + extra +
                           [str(here / (name + ".cpp")), "-o", str(binary)],
                           check=True)
            subprocess.run([str(binary)], check=True, env=env, timeout=60)
        portable = temporary / "portable.cpp"
        portable.write_text('#include "htp_qs2_reader.h"\nint main() { return 0; }\n')
        subprocess.run(compiler + ["-std=c++17", "-Wall", "-Wextra", "-Werror",
                                  "-U__linux__", "-U__ANDROID__",
                                  "-I" + str(backend), "-fsyntax-only",
                                  str(portable)], check=True)
        print("PASS: compact reader header without Linux/Android macros")


if __name__ == "__main__":
    main()
