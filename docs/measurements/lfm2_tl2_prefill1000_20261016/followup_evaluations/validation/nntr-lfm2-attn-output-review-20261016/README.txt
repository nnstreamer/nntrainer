# Independent review: dynamic-Q8 attention output reuse

Result: no correctness or ownership blocker found in the reviewed 39-line ARM change. This is approval for root-controlled device validation, not a measured performance or numerical-equivalence result.

Reviewed artifact: `/tmp/lfm2-gemma-prefill-port-20261016/attn-output-reuse/attn-output-reuse.patch`, SHA-256 `3191a26b381473e6b8e4c49c1235ff6643b7f4d4315dab5449e5b436c31ac5d4`.

## Ownership and allocation

- The eligible shape has 2..1024 rows, 32 Q heads, 8 KV heads, head dimension 64, no sinks and zero softcap. The existing public precondition validates contiguous Q/output strides. On this shape, q_len and the maximum 8 MiB output length are bounded safely.
- Selection requires an already allocated ION slot of the exact output size class and sufficient actual size. stage() is called only after that check, so its fit() cannot allocate or grow the buffer. A kept record in either holds or copied_out rejects reuse. Non-kept records already have complete host data and can be retired safely.
- The overflow-avoiding pointer-difference overlap check covers the entire existing allocation, including page padding, against live Q, appended K, appended V and host output. This is stronger than checking only the intended output prefix. A zero-row append correctly has no live K/V byte range.
- invoke_mutex_ protects lookup, clearing old records, the RPC, complete host copy and copied_out publication. These paths do not recursively acquire that mutex. unique_lock releases it on error/return and unwinding.
- An RPC error may partially overwrite the selected ION output but never publishes it or copies it to the caller. All selected input ranges remain disjoint. Previous non-kept tensors remain complete in their host memory; the caller can take the existing CPU fallback. Failure followed by retry is safe in the host fixture.

## Downstream consumer

- The LFM FP32 MHA commit is a no-op, so copyOutOf(...,true) is necessary and preserves CPU readers. It publishes a non-kept copied_out record only after the full copy succeeds.
- The output projection calls invokeFc -> invokeLayer. For 1024 rows/K2048 it is one call. invokeLayer calls stageInput before staging its own output. The exact-size copied_out match swaps the attention ION allocation into the input pool; the former input allocation becomes the next output. Input and output remain disjoint.
- The existing sampledSame mutation check remains unchanged. Like the existing staging design, it cannot detect a host mutation that preserves every sampled window; this patch does not offer a stronger immutable-alias guarantee. The graph's ordinary FP32 attention-to-FC path does not mutate that tensor between these operations.
- No new ionAlias is added. Existing kept/sampled alias rules and invalidation policy are unchanged.
- 2048 rows remain on the original path: the following FC consumes 1920+128 row chunks and would not satisfy a whole-output exact-size handoff. Missing/null/small/non-ION/overlapping slots and other shapes use the original RPC destination. Gemma fixed-Q8 uses a separate method and its geometry also fails this guard.

## Independent validation

1. Re-ran the author's portable fixture against the exact candidate source: ASan+UBSan, 29 cases PASS. It extracts the actual selection/publication and existing staging methods verbatim.
2. Added two independent cases and made the RPC stand-in partially write its output even when returning an error: ASan+UBSan, 31 cases PASS. These cover prior materialized holds/copied_out -> partial RPC failure -> retry -> FC swap, and eight consecutive attention/FC pool-swap cycles with the FC output overwritten between iterations. No additional HtpRpcBuffer allocations, unexpected host copies, input/output overlap or stale successful publication occurred.
3. Read the actual public guard, MHA commit, invokeFc chunk loop, invokeLayer ordering, stage/stageInput/copyOutOf/fit and alias helpers. The author's Android compile artifact was inspected; this reviewer did not repeat the full Android compile.

The fixture uses host allocation and RPC stand-ins. It does not prove real FastRPC/DMA mapping behavior, hardware output bits, whole-model generated IDs, measured memory peak or speed. Root must validate those on the device before adoption. The prior 35 ms wall-minus-DSP gap includes all non-DSP work and is not an output-copy saving estimate.

No device access or production/root file modification was performed by this review.

## Reproduction and artifacts

- `check/commands.json`: source/extracted hashes and original sanitizer build command.
- `original.log`: the 29-case rerun.
- `check/independent_check.cpp`: independent partial-failure/repeat fixture; includes verbatim extracted `.inc` methods from the same directory.
- `independent.log`: the 31-case result.
- `manifest.json`: reviewed hashes and independent build/run commands.

Rebuild independent cases:

```sh
c++ -std=c++17 -O2 -g -Wall -Wextra -pthread -fsanitize=address,undefined -fno-omit-frame-pointer check/independent_check.cpp -o check/independent_check
./check/independent_check
```
