#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check bounded expert staging and the actual backend queue/read methods.

No device, SDK, or frozen implementation copy is used. The backend fixture
stubs the platform and registry boundary; the runner extracts queue, scope,
read, copy, and staging methods from the selected source root unchanged.
The file and thread fixtures require Linux or Android and otherwise report a
skip. The generic header is also compiled with both platform macros undefined.
"""

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


METHODS = (
    "begin_qs2_host_prefetch", "end_qs2_host_prefetch", "beginExpertPrefetch",
    "stageExpert", "readExpert", "readExpertPart", "hostExpertProfile",
    "hostExpertBank", "hostExpertLease", "waitHostExpertReads",
    "readHostExpert", "prefetchReaderLoop",
)


def body_end(source, opening):
    """Match a C++ body while ignoring braces in comments and literals."""
    depth = 0
    state = "code"
    i = opening
    while i < len(source):
        char = source[i]
        pair = source[i:i + 2]
        if state in ("string", "char"):
            if char == "\\":
                i += 2
                continue
            if char == ('"' if state == "string" else "'"):
                state = "code"
        elif state == "line":
            if char == "\n":
                state = "code"
        elif state == "block":
            if pair == "*/":
                state = "code"
                i += 2
                continue
        elif pair in ("//", "/*"):
            state = "line" if pair == "//" else "block"
            i += 2
            continue
        elif char in ('"', "'"):
            state = "string" if char == '"' else "char"
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("Unterminated backend C++ body")


def extract_method(source, name):
    pattern = r"(?m)^  (?=\S)[^\n]*\b" + re.escape(name) + r"\("
    match = re.search(pattern, source)
    if not match:
        raise ValueError("Missing backend method: " + name)
    start = match.start()
    if source[start:match.end()].lstrip().startswith(name + "("):
        start = source.rfind("\n", 0, start - 1) + 1
    opening = source.index("{", match.end())
    body = source[start:body_end(source, opening)]
    return re.sub(r"\boverride\b", "", body, count=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = args.repo_root.resolve() if args.repo_root else here.parents[2]
    backend = root / "nntrainer/tensor/htp_backend"
    source = (backend / "htp_compute_ops.cpp").read_text()
    match = re.search(r"(?m)^  struct StagedExpert\s*\{", source)
    if not match:
        raise ValueError("Missing backend StagedExpert type")
    opening = source.index("{", match.start())
    staged_type = source[match.start():body_end(source, opening)] + ";\n"
    methods = "\n\n".join(extract_method(source, name) for name in METHODS)
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    flags = ["-std=c++17", "-O1" if args.sanitize else "-O2", "-Wall",
             "-Wextra", "-Werror", "-Wno-unused-parameter", "-pthread"]
    has_tl2 = "ternaryMode()" in extract_method(source, "begin_qs2_host_prefetch")
    flags += ["-DNNTR_TEST_HOST_STAGE_HAS_TL2=" + str(int(has_tl2))]
    env = os.environ.copy()
    if args.sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g1"]
        env["ASAN_OPTIONS"] = env.get("ASAN_OPTIONS", "") + ":halt_on_error=1:detect_leaks=1"
        env["UBSAN_OPTIONS"] = env.get("UBSAN_OPTIONS", "") + ":halt_on_error=1"
    with tempfile.TemporaryDirectory(prefix="nntr-host-staging-") as path:
        temporary = Path(path)
        (temporary / "expert_host_staging_staged_type.inc").write_text(staged_type)
        (temporary / "expert_host_staging_backend_methods.inc").write_text(methods)
        includes = ["-I" + str(p) for p in (temporary, here, backend)]
        for name in ("expert_host_staging_host_check", "expert_host_staging_backend_check"):
            binary = temporary / name
            command = compiler + flags + includes + [str(here / (name + ".cpp")),
                                                      "-o", str(binary)]
            subprocess.run(command, check=True)
            subprocess.run([str(binary)], check=True, env=env, timeout=60)
        portable = temporary / "header_portable.cpp"
        portable.write_text('#include "htp_expert_host_staging.h"\nint main() { return 0; }\n')
        command = compiler + ["-std=c++17", "-Wall", "-Wextra", "-Werror",
                              "-U__linux__", "-U__ANDROID__", "-I" + str(backend),
                              "-fsyntax-only", str(portable)]
        subprocess.run(command, check=True)
        print("PASS: generic staging header without Linux/Android macros")


if __name__ == "__main__":
    main()
