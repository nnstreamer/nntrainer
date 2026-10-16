#!/usr/bin/env python3
"""Build isolated production DN worker/HVX parity and simulator timing."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess


def function(source, name):
    m = re.search(r"^static[^;{]*\b" + name + r"\([^;{]*\{", source, re.M)
    if m is None:
        raise RuntimeError(name)
    end, depth = m.end(), 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[m.start():end]


p = argparse.ArgumentParser()
p.add_argument("--repo", type=Path, required=True)
p.add_argument("--hexagon-tools", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--simulate", action="store_true")
p.add_argument("--sanitize", action="store_true")
a = p.parse_args()
d = Path(__file__).resolve().parent
a.output.mkdir(parents=True, exist_ok=True)
fixture = a.output / "scatter_check.c"
fixture.write_bytes((d / "scatter_check.c").read_bytes())
backend = a.repo / "nntrainer/tensor/htp_backend"
host = a.repo / "test/htp/host"
before = (d / "before/hexkl_mm_u8i4_moe.c").read_text()
source = (d / "candidate/hexkl_mm_u8i4_moe.c").read_text()
end = source.index("} moe_dn_ctx;") + len("} moe_dn_ctx;")
start = source.rfind("typedef struct {", 0, end)
hdr = '''#include "hvx_dequant_i32.h"
#include "hvx_scale_add_f32.h"
#include "hexkl_acc_tile.h"
#include "hexkl_acc_i24.h"
#define HEXKL_HMX_INT8_BLOCK_N_ROW 64u
#define HEXKL_PROBE_T0(x) ((void)(x))
static inline void moe_worker_probe_add(uint64_t unused) {(void)unused;}
'''
hdr += source[start:end] + "\n"
hdr += function(source, "moe_dn_rows") + "\n"
hdr += "static void moe_dn_worker(uint32_t, uint32_t, void *);\n"
hdr += function(before, "moe_dn_fg_worker").replace("moe_dn_fg_worker", "reference_moe_dn_fg_worker") + "\n"
hdr += function(source, "moe_dn_fg_worker") + "\n"
(a.output / "scatter_under_test.h").write_text(hdr)
common = ["-std=gnu11", "-O3", "-ffp-contract=off", "-ffunction-sections", "-fdata-sections"]
includes = []
for x in [a.output, host / "stub", backend / "hvx", backend / "hmx"]:
    includes += ["-I", str(x)]
native = a.hexagon_tools / "Tools/libnative"
tools = a.hexagon_tools / "Tools/bin"
extra = []
if a.simulate:
    cc = str(tools / "hexagon-clang")
    common += ["-mv81", "-mhvx", "-mhvx-length=128B", "-G0"]
    hist = json.loads((d / "route_block_histogram.json").read_text())
    extra += ["-DSCATTER_SIM", "-DROUTE_HIST=" + ",".join(str(hist.get(str(i), 0)) for i in range(65))]
else:
    cc = os.environ.get("CC", "gcc")
    common += ["-g", "-fno-strict-aliasing", "-DHVX_UVector=HEXAGON_Vect1024"]
    includes += ["-isystem", str(native / "include")]
    if a.sanitize:
        common += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
commands = []


def run(cmd, name):
    commands.append(cmd)
    (a.output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    with (a.output / (name + ".log")).open("w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)


objs = []
sources = [fixture, backend / "hvx/hvx_dequant_i32.c", backend / "hvx/hvx_scale_add_f32.c"]
if not a.simulate:
    common += ["-pthread"]
    sources.append(backend / "hvx/hvx_worker_pool.c")
for f in sources:
    obj = a.output / (f.stem + ".o")
    flags = extra if f.name == "scatter_check.c" else []
    run([cc, *common, *flags, *includes, "-c", str(f), "-o", str(obj)], f.stem)
    objs.append(str(obj))
exe = a.output / "scatter_check"
if a.simulate:
    run([cc, "-mv81", "-mhvx", "-mhvx-length=128B", "-G0", *objs,
         "-Wl,--gc-sections", "-lhexagon", "-lm", "-o", str(exe)], "link")
    run([str(tools / "hexagon-sim"), "-mv81", str(exe)], "result")
else:
    flags = ["-fsanitize=address,undefined"] if a.sanitize else []
    run([os.environ.get("CXX", "g++"), *flags, "-pthread", *objs,
         str(native / "lib/libnative.a"), "-Wl,--gc-sections", "-lm", "-o", str(exe)], "link")
    run([str(exe)], "result")
(a.output / "manifest.json").write_text(json.dumps({
    "base_commit": "03d652fc0",
    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    "reference_source_sha256": hashlib.sha256(before.encode()).hexdigest(),
    "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
    "generated_sha256": hashlib.sha256(hdr.encode()).hexdigest(),
    "binary_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
    "requantization": "unchanged function mocked for lane ownership only",
}, indent=2) + "\n")
print((a.output / "result.log").read_text())
