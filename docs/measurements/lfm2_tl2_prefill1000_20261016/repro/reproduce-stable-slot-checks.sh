#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Usage: HEXAGON_SDK_ROOT=/path/to/sdk ANDROID_NDK_ROOT=/path/to/ndk \
#   bash reproduce-stable-slot-checks.sh REPO OUT
# Reads REPO only; all generated files and logs go into OUT. No adb/device calls.
set -euo pipefail
stable_repo="${1:?Pass the reviewed checkout path}"
stable_out="${2:?Pass an output directory outside the checkout}"
stable_sdk="${HEXAGON_SDK_ROOT:?Set HEXAGON_SDK_ROOT to the Hexagon SDK directory}"
stable_ndk="${ANDROID_NDK_ROOT:-${ANDROID_NDK_HOME:?Set ANDROID_NDK_ROOT or ANDROID_NDK_HOME to the Android NDK directory}}"
stable_cc="${CC:-gcc}"
stable_cxx="${CXX:-g++}"
stable_backend="$stable_repo/nntrainer/tensor/htp_backend"
stable_test="$stable_repo/test/htp"
stable_toolchain="$stable_ndk/toolchains/llvm/prebuilt/linux-x86_64/bin"
mkdir -p "$stable_out/host" "$stable_out/device"
python3 - "$stable_repo" "$stable_out" <<'PYBEFORE'
from pathlib import Path
import hashlib, json, subprocess, sys
repo, out = map(Path, sys.argv[1:])
files = [
    'nntrainer/tensor/htp_backend/htp_compute_ops.cpp',
    'nntrainer/tensor/htp_backend/htp_tl2_opts.h',
    'nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_dma.c',
    'nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_dma.h',
    'nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.c',
    'test/htp/nntr_hvx.idl', 'test/htp/nntr_hvx_mm_u8i4.c',
    'test/htp/host/swap_host_check.c',
    'test/htp/host/standin/hvx_scalar.c',
    'tools/tl2_stable_slots_device_check.cpp',
    'Applications/CausalLM/models/lfm2_moe/lfm2_moe_layer.cpp',
    'Applications/CausalLM/models/lfm2_moe/lfm2_htp_moe_causallm.cpp',
    'Applications/CausalLM/models/gemma4/gemma4_htp_moe_causallm.cpp',
]
if 'weight_swap_batch_tl2_stable_arena' not in (repo/'test/htp/nntr_hvx.idl').read_text():
    raise SystemExit('Reviewed stable-slot IDL is absent; restore the candidate before reproducing.')
record = {
    'repository': str(repo.resolve()),
    'head': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
    'sources': {name: hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in files},
}
(out/'source-before.json').write_text(json.dumps(record, indent=2)+'\n')
PYBEFORE

python3 "$stable_test/host/gen_nntr_hvx_h.py" "$stable_test/nntr_hvx.idl" \
  "$stable_out/host/nntr_hvx.h"
"$stable_cc" -std=c99 -O1 -g -fsanitize=address,undefined \
  -fno-omit-frame-pointer -Wall -Wextra -Wno-unused-parameter -include malloc.h \
  -I "$stable_out/host" -I "$stable_backend" -I "$stable_test/host/stub" \
  -I "$stable_test/host/standin" -I "$stable_test" -I "$stable_backend/.." \
  -I "$stable_backend/hmx" -I "$stable_backend/hvx" \
  "$stable_test/host/swap_host_check.c" "$stable_test/nntr_hvx_mm_u8i4.c" \
  "$stable_backend/hmx/hexkl_mm_u8i4_dma.c" \
  "$stable_backend/hmx/hexkl_dma_trace.c" \
  "$stable_test/host/hvx_scalar_stubs.c" "$stable_test/host/standin/hvx_scalar.c" \
  -lm -o "$stable_out/stable-swap-host-check" \
  > "$stable_out/host-build.log" 2>&1
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  "$stable_out/stable-swap-host-check" > "$stable_out/host-sanitizer.log" 2>&1
# Checks the non-Linux/Windows-preprocessor branch; this is not an MSVC build.
"$stable_cxx" -std=c++17 -Wall -Wextra -Werror -U__linux__ -D_WIN32 \
  -I "$stable_repo/nntrainer/utils" -fsyntax-only \
  "$stable_repo/nntrainer/utils/weight_file_reader.cpp" \
  > "$stable_out/nonlinux-syntax.log" 2>&1

"$stable_sdk/ipc/fastrpc/qaic/Ubuntu/qaic" \
  -I "$stable_sdk/incs" -I "$stable_sdk/incs/stddef" \
  -mdll -o "$stable_out/device" "$stable_test/nntr_hvx.idl" \
  > "$stable_out/qaic.log" 2>&1
"$stable_toolchain/aarch64-linux-android29-clang" -O2 -fPIC \
  -I "$stable_out/device" -I "$stable_sdk/incs" \
  -I "$stable_sdk/incs/stddef" -I "$stable_sdk/ipc/fastrpc/incs" \
  -c "$stable_out/device/nntr_hvx_stub.c" -o "$stable_out/device/rpc-stub.o" \
  > "$stable_out/stub-build.log" 2>&1
"$stable_toolchain/aarch64-linux-android29-clang" -std=c99 -O2 -fPIC \
  -I "$stable_backend/hvx" -c "$stable_backend/hvx/hvx_gemv_u8tl2.c" \
  -o "$stable_out/device/tl2-scalar.o" > "$stable_out/tl2-scalar-build.log" 2>&1
"$stable_toolchain/aarch64-linux-android29-clang++" -std=c++17 -O2 \
  -static-libstdc++ -DENABLE_HEXKL -I "$stable_out/device" \
  -I "$stable_backend" -I "$stable_backend/hvx" -I "$stable_sdk/incs" \
  -I "$stable_sdk/incs/stddef" -I "$stable_sdk/ipc/fastrpc/incs" \
  "$stable_repo/tools/tl2_stable_slots_device_check.cpp" \
  "$stable_out/device/rpc-stub.o" "$stable_out/device/tl2-scalar.o" \
  "$stable_sdk/ipc/fastrpc/remote/ship/android_aarch64/libcdsprpc.so" -ldl \
  -o "$stable_out/tl2_stable_slots_device_check" \
  > "$stable_out/checker-build.log" 2>&1

# Source identities distinguish a stale helper or IDL from the reviewed tree.
python3 - "$stable_repo" "$stable_out" <<'PYMETA'
from pathlib import Path
import hashlib, json, subprocess, sys
repo, out = map(Path, sys.argv[1:])
hash_file = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
record = json.loads((out/'source-before.json').read_text())
current = {name: hash_file(repo/name) for name in record['sources']}
if current != record['sources'] or subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() != record['head']:
    raise SystemExit('Source changed during build: this run is invalid; reproduce on a stable checkout.')
log = (out/'host-sanitizer.log').read_text()
for expected in ['STABLE TL2 SWAP OK: K=2048 I=1536 N=2048',
                 'STABLE TL2 GEMMA REJECT OK: no handle mutation']:
    if expected not in log:
        raise SystemExit('Missing stable fixture result: '+expected)
record.update({
    'artifacts': {name: hash_file(out/name) for name in ['stable-swap-host-check', 'tl2_stable_slots_device_check', 'host-sanitizer.log']},
    'host_result': 'PASS: ASan + UBSan + leak detection, LFM stable fixture and Gemma rejection guard',
    'nonlinux_result': 'PASS: GNU syntax check of non-Linux/Windows-preprocessor branch, not MSVC',
    'device_result': 'not run by this script; requires root-owned S26 Ultra execution',
})
(out/'source-and-build.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
PYMETA
cat "$stable_out/host-sanitizer.log"
