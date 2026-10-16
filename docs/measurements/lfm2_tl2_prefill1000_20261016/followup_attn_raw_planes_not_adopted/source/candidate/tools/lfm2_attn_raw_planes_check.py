#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Actual raw-plane helpers and Qx pipeline, using SDK libnative and pthreads.

The v81 qualification is enabled explicitly after native headers. This does
not emulate HMX conversion/fences: the device fixture proves those separately.
No sanitizer checks are disabled.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def function(source, name):
    m = re.search(r"^static[^;{]*\b" + re.escape(name) + r"\([^;{]*\{", source, re.M)
    if not m:
        raise RuntimeError("missing actual function " + name)
    end, depth = m.end(), 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[m.start():end]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--hexagon-tools", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--sanitize", action="store_true")
    a = p.parse_args()
    if a.output is None:
        a.output = Path(tempfile.mkdtemp(prefix="lfm2-raw-planes-"))
    a.output.mkdir(parents=True, exist_ok=True)
    b = a.repo / "nntrainer/tensor/htp_backend"
    host = a.repo / "test/htp/host"
    fixtures = Path(__file__).resolve().parents[1] / "test/htp/host"
    src = a.source.read_text()
    start = src.index("typedef struct {\n  uint8_t *vb;")
    end = src.index("} attn_ctx;", start) + len("} attn_ctx;")
    common = '''#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <AEEStdErr.h>
#include "hexkl_attn_q.h"
#include "hexkl_acc_tile.h"
#include "hvx_convert.h"
#define TB HEXKL_ATTN_TILE_BYTES
#define TILE_VECS (TB / sizeof(HVX_Vector))
#define HEXKL_CVT_BLOCK_BYTES 1024u
#define HEXKL_CVT_PLANE_BYTES 2048u
#define HEXKL_CVT_PLANE_ALIGN 2048u
/* Explicit architecture stand-in after libnative headers; not real HMX. */
#define __hexagon__ 1
#define __HEXAGON_ARCH__ 81
static uint32_t hexkl_micro_hmx_config_size(void);
static int hexkl_micro_hmx_acc_read_int32(uint8_t *, uint32_t, uint32_t);
static void hexkl_cvt_issue(const void *, void *);
''' + src[start:end] + "\n"
    for name in ("qk_planes_available", "qk_read_acc", "qk_acc_four_rows"):
        common += function(src, name) + "\n"
    # Compile the exact guard separately with unsupported architecture macros.
    common += "#undef __HEXAGON_ARCH__\n#define __HEXAGON_ARCH__ 79\n"
    common += function(src, "qk_planes_available").replace("qk_planes_available", "guard_v79", 1) + "\n"
    common += "#undef __hexagon__\n#undef __HEXAGON_ARCH__\n"
    common += function(src, "qk_planes_available").replace("qk_planes_available", "guard_host", 1) + "\n"
    (a.output / "attn_raw_planes_under_test.h").write_text(common)
    jobend = src.index("} dequant_job;") + len("} dequant_job;")
    jobstart = src.rfind("typedef struct {", 0, jobend)
    h = common + src[jobstart:jobend] + "\n"
    h += function((b / "hvx/hvx_tile_f16.h").read_text(), "hvx_tile_load_u") + "\n"
    for name in ("q_tile", "kt_tile", "s_tile", "acc_rows", "unit_range",
                 "dequant_s_at", "dequant_s_planes_at", "dequant_s", "qx_dequant_available", "dequant_worker"):
        h += function(src, name) + "\n"
    h += '''
static uint64_t now_us(void);
static void softmax_state_init(attn_ctx *, uint32_t, uint32_t);
static void dma_issue(attn_ctx *, uint32_t, uint32_t, uint32_t);
static void dma_land(attn_ctx *, uint32_t, uint32_t);
static void softmax_start(attn_ctx *, uint32_t, uint32_t, uint32_t, uint32_t, int);
static void softmax_finish_pquant(attn_ctx *, uint32_t);
static int phase_pv(attn_ctx *, uint32_t, int);
static void phase_store(attn_ctx *, uint32_t, uint32_t);
static void hexkl_micro_hmx_acc_clear_int32(void);
static void hexkl_hmx_mm_u8i8(const uint8_t *, const uint8_t *);
static void hexkl_hmx_mm_u8i4(const uint8_t *, const uint8_t *);
static void checked_submit(hvx_worker_pool *, hvx_worker_pool_func, void *, uint32_t);
static void checked_wait(hvx_worker_pool *);
'''
    h += function(src, "phase_qk") + "\n"
    h += "#define hvx_worker_pool_submit checked_submit\n#define hvx_worker_pool_wait checked_wait\n"
    h += function(src, "run_q_block") + "\n"
    h += "#undef hvx_worker_pool_submit\n#undef hvx_worker_pool_wait\n"
    (a.output / "attn_raw_planes_overlap_under_test.h").write_text(h)
    native = a.hexagon_tools / "Tools/libnative"
    flags = ["-std=gnu11", "-O2", "-g", "-fno-strict-aliasing", "-ffp-contract=off",
             "-include", "malloc.h", "-include", str(host / "attn_qprep_native_unaligned.h"),
             "-pthread", "-Wall", "-Wextra", "-Werror"]
    if a.sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-sanitize-recover=all", "-fno-omit-frame-pointer"]
    includes = ["-isystem", str(native / "include")]
    for x in (host / "stub", b / "hvx", b / "hmx", a.output):
        includes += ["-I", str(x)]
    commands, results = [], {}
    def run(cmd, label):
        commands.append(cmd)
        (a.output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        env = dict(os.environ, ASAN_OPTIONS="halt_on_error=1:detect_leaks=1", UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
        q = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
        (a.output / (label + ".log")).write_text(q.stdout + q.stderr)
        if q.returncode:
            raise RuntimeError(label + " failed:\n" + q.stdout + q.stderr)
        return q.stdout
    pool = a.output / "pool.o"
    run([os.environ.get("CC", "gcc"), *[x for x in flags if x != "-Werror"], *includes,
         "-c", str(b / "hvx/hvx_worker_pool.c"), "-o", str(pool)], "pool-build")
    for name in ("attn_raw_planes_check", "attn_raw_planes_overlap_check"):
        obj, exe = a.output / (name + ".o"), a.output / name
        run([os.environ.get("CC", "gcc"), *flags, *includes, "-c", str(fixtures / (name + ".c")), "-o", str(obj)], name + "-build")
        run([os.environ.get("CXX", "g++"), *flags, str(obj), str(pool), str(native / "lib/libnative.a"), "-lm", "-o", str(exe)], name + "-link")
        results[name] = run([str(exe)], name)
        print(results[name], end="", flush=True)
    results["source_sha256"] = hashlib.sha256(a.source.read_bytes()).hexdigest()
    results["fixture_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in fixtures.glob("attn_raw_planes*.c")}
    results["sanitizers"] = "ASan/UBSan with halt_on_error, no suppression" if a.sanitize else "none"
    (a.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
