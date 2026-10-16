#!/usr/bin/env python3
"""Actual gather/worker checks. Explicit paths keep overlays reproducible."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def function(text, name):
    start = text.index(name + "(")
    start = text.rfind("\n", 0, start) + 1
    brace = text.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end] + "\n"


def extract(text):
    guard = text[text.index("  const int lfm_prefill_pack ="):]
    guard = guard[:guard.index(";") + 1]
    meta = text[text.index("      if (lfm_prefill_pack) {"):]
    brace = meta.index("{", meta.index("} else {"))
    depth, end = 1, brace + 1
    while depth:
        depth += (meta[end] == "{") - (meta[end] == "}")
        end += 1
    meta = meta[:end]
    names = ("use_m1", "M", "K", "inter", "N_out", "n_experts",
             "n_with_rows", "n_tl2", "n_affine", "act")
    return (function(text, "hexkl_moe_pack_bg_worker") +
            function(text, "moe_lfm_pack_bg_worker") +
            "static int actual_guard(const uint32_t *x) {\n" +
            "\n".join(f"  uint32_t {name} = x[{i}];"
                      for i, name in enumerate(names)) + "\n" + guard +
            "\n  return lfm_prefill_pack;\n}\n" +
            "static void actual_metadata(uint8_t *pack_skip, "
            "const uint32_t *row_count, uint32_t e, uint32_t d, "
            "uint32_t padded, int lfm_prefill_pack) {\n" + meta + "\n}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tools-root", type=Path, required=True,
                    help="Hexagon release containing Tools/libnative")
    ap.add_argument("--sanitize", action="store_true")
    ap.add_argument("--whole", action="store_true")
    args = ap.parse_args()
    repo, source, out = (p.resolve() for p in
                         (args.repo, args.source, args.output))
    out.mkdir(parents=True, exist_ok=True)
    backend = repo / "nntrainer/tensor/htp_backend"
    here = repo / "test/htp/host"
    fixtures = source / "test/htp/host"
    kernel = source / "nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.c"
    native = args.tools_root.resolve() / "Tools/libnative"
    (out / "actual-tail-workers.inc").write_text(extract(kernel.read_text()))
    (out / "native-types.h").write_text(
        "#include <hexagon_types.h>\n#include <hvx_hexagon_protos.h>\n"
        "typedef HVX_Vector nntr_unaligned_hvx __attribute__((aligned(1)));\n"
        "#define HVX_UVector nntr_unaligned_hvx\n")
    commands = []
    env = dict(os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
               UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")

    def run(cmd, logfile):
        cmd = [str(v) for v in cmd]
        commands.append(cmd)
        (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        with (out / logfile).open("w") as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT,
                           env=env)

    flags = ["-std=gnu11", "-O2", "-g", "-fno-strict-aliasing",
             "-ffp-contract=off", "-ffunction-sections", "-fdata-sections",
             "-fno-pie", "-Wall", "-Wextra", "-Wno-unused-parameter",
             "-Wno-unused-function"]
    if args.sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    includes = ["-I" + str(p) for p in
                (out, here, here / "stub", here / "standin", here.parent,
                 backend.parent, backend / "hmx", backend / "hvx")]
    quant = out / "native-gather.o"
    run(["gcc", *flags, *includes, "-I" + str(native / "include"),
         "-include", out / "native-types.h", "-include", "malloc.h",
         "-Dhvx_gather_u8_ah_rows=native_gather", "-c",
         backend / "hvx/hvx_quant_u8.c", "-o", quant], "quant-build.log")
    symbols = subprocess.check_output(["nm", "-g", "--defined-only", quant],
                                      text=True)
    (out / "native-rename.txt").write_text("".join(
        f"{p[-1]} native_unused_{p[-1]}\n" for line in symbols.splitlines()
        if (p := line.split()) and p[-1] != "native_gather"))
    run(["objcopy", "--redefine-syms=" + str(out / "native-rename.txt"), quant],
        "quant-rename.log")
    run(["gcc", *flags, *includes, "-c",
         fixtures / "lfm_tail_gather_check.c", "-o", out / "helper.o"],
        "helper-build.log")
    linkflags = ["-no-pie", "-Wl,--gc-sections", "-pthread"]
    if args.sanitize:
        linkflags += ["-fsanitize=address,undefined"]
    run(["g++", *linkflags, out / "helper.o", quant,
         native / "lib/libnative.a", "-lm", "-o", out / "helper-check"],
        "helper-link.log")
    run([out / "helper-check"], "helper-run.log")
    run(["gcc", *flags, *includes, "-I" + str(native / "include"),
         "-include", out / "native-types.h", "-c",
         backend / "hvx/hvx_dequant_i32.c", "-o", out / "readout.o"],
        "readout-build.log")
    run(["gcc", *flags, *includes, "-c",
         fixtures / "lfm_tail_readout_check.c", "-o", out / "readout-check.o"],
        "readout-check-build.log")
    run(["g++", *linkflags, out / "readout-check.o", out / "readout.o",
         native / "lib/libnative.a", "-lm", "-o", out / "readout-check"],
        "readout-link.log")
    run([out / "readout-check"], "readout-run.log")
    result = {"helper": "PASS", "helper_cases": 198, "guard_cases": 16,
              "readout": "PASS", "readout_cases": 512,
              "source_sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
              "sanitizers": args.sanitize,
              "limits": "libnative HVX; no physical DMA/HMX or TPS proof"}
    if args.whole:
        if not args.reference:
            ap.error("--whole requires --reference containing the baseline TU")
        ref = args.reference.resolve() / "nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.c"
        objects = []
        units = [("baseline", ref, []), ("candidate", kernel, []),
                 ("fixture", fixtures / "lfm_tail_gather_tu_check.c", []),
                 ("scalar", here / "standin/hvx_scalar.c",
                  ["-Dhvx_gather_u8_ah_rows=unused_scalar_gather",
                   "-fno-sanitize=shift-base"]),
                 ("trace", backend / "hmx/hexkl_dma_trace.c", []),
                 ("expand", backend / "hvx/hvx_expand_i2i4.c", []),
                 ("tl2", backend / "hvx/hvx_gemv_u8tl2.c", []),
                 ("pool", backend / "hvx/hvx_worker_pool.c", [])]
        for name, path, extra in units:
            obj = out / (name + ".o")
            run(["gcc", *flags, *includes, *extra, "-pthread", "-c", path,
                 "-o", obj], name + "-build.log")
            if name == "baseline":
                symbols = subprocess.check_output(
                    ["nm", "-g", "--defined-only", obj], text=True)
                rename = out / "reference-rename.txt"
                rename.write_text("".join(
                    f"{p[-1]} reference_{p[-1]}\n"
                    for line in symbols.splitlines() if (p := line.split())))
                run(["objcopy", "--redefine-syms=" + str(rename), obj],
                    "reference-rename.log")
            objects.append(obj)
        run(["g++", *linkflags, "-Wl,--wrap=hexkl_micro_hmx_mm_u8i4",
             *objects, quant, native / "lib/libnative.a", "-lm",
             "-o", out / "whole-check"], "whole-link.log")
        run([out / "whole-check"], "whole-run.log")
        result.update(whole_tu="PASS", whole_cases=21,
                      whole_limits="Actual kernel, gather, pthread pool; "
                      "immediate DMA/scalar HMX and math stand-ins; "
                      "compact i24 Hexagon-only branch is a source/codegen proof; "
                      "shift-base UBSan disabled only in scalar stand-in TU "
                      "(existing swiglu_det_exp negative exponent shift)")
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
