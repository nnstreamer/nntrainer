#!/usr/bin/env python3
"""Build the device correctness fixture against the existing seven runtimes.
No runtime, source tree, device or IDL is changed. NDK and SDK are external.
"""
import argparse, hashlib, json, subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--ndk',type=Path,required=True);p.add_argument('--sdk',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();D=Path(__file__).resolve().parent
a.output.mkdir(parents=True,exist_ok=True)
cmd=[str(a.ndk/'toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++'),'-std=c++17','-O2','-fno-fast-math','-ffp-contract=off','-Wall','-Wextra','-Werror','-DENABLE_HEXKL','-static-libstdc++','-Wl,-z,max-page-size=16384','-I'+str(a.repo/'nntrainer/tensor/htp_backend'),'-I'+str(a.repo/'test/htp/generated'),'-I'+str(a.sdk/'incs'),'-I'+str(a.sdk/'incs/stddef'),'-I'+str(a.sdk/'ipc/fastrpc/incs'),str(D/'attention_raw_planes_device_check.cpp'),'-L'+str(a.repo/'builddir/android_build_result/lib/arm64-v8a'),'-lnntrainer','-Wl,-rpath,$ORIGIN','-o',str(a.output/'attention_raw_planes_device_check')]
r=subprocess.run(cmd,capture_output=True,text=True);(a.output/'build.log').write_text(r.stdout+r.stderr);(a.output/'build-command.json').write_text(json.dumps({'command':cmd,'returncode':r.returncode},indent=2)+'\n')
if r.returncode:raise RuntimeError(r.stdout+r.stderr)
print(hashlib.sha256((a.output/'attention_raw_planes_device_check').read_bytes()).hexdigest())
