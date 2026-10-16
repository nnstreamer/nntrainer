Reproduce and verify the retained ARM-router evidence

1. Verify the complete archive using only Python's standard library:
   python3 -B validation/verify.py
   python3 -B validation/verify.py --json
This rehashes all payloads, checks all 12 normal runs and memory reports, recomputes paired summaries, reparses both diagnostics with the frozen parser, and verifies repeated-prompt/native results and runtime/source identities. It makes no device calls.

2. Materialize the exact frozen source and canonical tests to a NEW directory:
   python3 -B repro/reproduce.py --output /absolute/new/output
The output has source/, tests/test/htp/host/ and diagnostic/. Original absolute paths inside archived compiler commands are provenance, not required defaults in this helper.

3. Run the actual helper/caller host checks and ten mutations:
   python3 -B repro/reproduce.py --output /absolute/new/check-output --host --repo /absolute/nntrainer-checkout --simde-include /absolute/simde-0.8.2
Use the nntrainer source at f7b16e48ca01a650160d823141e86e67c7993a7c or an equivalent checked source for the unchanged HVX router/RMS and host lane-emulation headers. SIMDe 0.8.2 is an external test dependency, not vendored into this archive. GCC/G++, ASan and UBSan must be available. The runner itself records actual argv/source hashes in its new output. Host stand-ins are not hardware parity proof.

4. Build the actual native fixture, without deploying/running it:
   python3 tests/test/htp/host/run_lfm2_router_arm_native.py --source source --output /absolute/new/native --compiler /absolute/ndk/bin/aarch64-linux-android29-clang++ --nntrainer-include /absolute/core/include/nntrainer --nntrainer-lib-dir /absolute/core/lib/arm64-v8a
Run this from the materialized directory. The two compile argv, core link-time SHA and extracted helper SHA values are saved by the runner. The archived final native command is artifacts/native/commands.json. Root's actual four device commands and raw hashes are in artifacts/native/root-native-v3-results.json. Matching FastRPC runtime/DSP and libomp support are required; binaries are intentionally not bundled.

5. Diagnostic checks, from a materialized directory after host output exists:
   python3 -B diagnostic/parser_check.py
   python3 -B diagnostic/check.py --checked-output /absolute/host-output --production-tests tests/test/htp/host --source diagnostic/source --output /absolute/new/diagnostic-checks
For full-app diagnostic build, apply the archived diagnostic patch over the normal production source and define NNTR_LFM2_ROUTER_ARM_TRACE=1 only for that derived app TU. Root used a source-local first-line define, preserved in artifacts/root/root-diagnostic-source.cpp.gz; no production build flags changed. Runtime NNTR_M0_PROFILE=1 enables records. The frozen parse.py expects one child's stderr, --rows 1024 or 2048 --enabled 1 --expect-fast 38. It does not authenticate PID. No diagnostic TPS is a normal result.

6. Full application and model reproduction requires the repository build/toolchain setup, model files and a separately controlled S26 Ultra. The five frozen production source files and minimal patch are in artifacts/production/. Root's actual Meson compile command, app-link output, source hashes and seven deployed runtime hashes are preserved under artifacts/root/. Default remains OFF; use exactly NNTR_LFM2_ROUTER_ARM=1 to enable the candidate.

The original device orchestration scripts are retained under repro/device/ and artifacts/repeat/root-repeat.py for exact provenance. Some contain the original absolute workspace/device paths and must be configured in a new environment before explicit device use. This archive verifier/materializer never invokes them. bench.py uses ANDROID_SERIAL and NNTR_BENCH_DEVICE_ROOT or a local device.json, preserves raw commands/thermal records, requires C16/PAGE0/grow0/preload128, and can collect complete child memory samples. root-normal-jobs.json records the original balanced OFF/ON order. Complete invocation/environment/model-config hashes are in each raw result JSON.

Use the two repeat builders for their respective prompt lengths: the original 1024 checker deliberately rejects 2048, while repro/device/repeat2048/build_repeat_check.py supports the tested 2048 case. Each same-handle JSON/log and two 65-ID files is preserved under artifacts/repeat/. Do not substitute only the 64-ID legacy token trace for the 65 dumped IDs.

The archive can be regenerated with validation/curate_router_arm.py plus explicit --source, --evidence, --review, --prescreen, --shared and a fresh --output, provided the original input paths recorded in manifest.json are still available. Its sibling verify.py is included. Regeneration and offline verification are distinct: verification requires only this archive.
